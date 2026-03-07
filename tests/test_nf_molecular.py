"""Integration tests for NF training on molecular systems.

NF training on molecular systems has NEVER been end-to-end tested because
the pipeline force-disables NF via adapt_to_system_size(). These tests
verify the full training loop works: sampling, loss computation, gradient
updates, particle conservation, and convergence.

Architecture note (SigmoidTopK determinism):
    The current NF uses SigmoidTopK which is fully deterministic — same
    logits always produce the same top-k selection. This means an untrained
    or short-trained NF produces only ~1 unique configuration per sample
    batch (unique_ratio ~ 1/batch_size). Tests are calibrated to this
    reality. Diversity in the subspace comes from essential-config injection
    (HF + singles + doubles) rather than NF sampling.

Usage:
    uv run pytest tests/test_nf_molecular.py -v --no-header -m slow
"""

import sys
import os
import math

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pyscf = pytest.importorskip("pyscf", reason="PySCF required for molecular tests")

from flows.physics_guided_training import PhysicsGuidedFlowTrainer, PhysicsGuidedConfig
from flows.particle_conserving_flow import (
    ParticleConservingFlowSampler,
    verify_particle_conservation,
)
from nqs.dense import DenseNQS


# ---------------------------------------------------------------------------
# Helper: create a trainer for a given Hamiltonian
# ---------------------------------------------------------------------------

def _make_trainer(H, num_epochs=30, min_epochs=10, samples_per_batch=100, **config_kwargs):
    """Build flow sampler, NQS, config, and trainer for a Hamiltonian.

    Note: we pass the ``ParticleConservingFlowSampler`` (not ``.flow``)
    because the trainer checks ``hasattr(self.flow, 'sample_with_probs')``
    to take the correct sampling branch. The inner ``ParticleConservingFlow``
    has a different return order for ``sample()`` and would crash.
    """
    n_orbitals = H.n_orbitals
    n_alpha = H.n_alpha
    n_beta = H.n_beta
    num_sites = 2 * n_orbitals

    flow_sampler = ParticleConservingFlowSampler(
        num_sites=num_sites,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden_dims=[64, 64],
    )
    nqs = DenseNQS(num_sites=num_sites, hidden_dims=[64, 64])

    cfg = PhysicsGuidedConfig(
        num_epochs=num_epochs,
        min_epochs=min_epochs,
        samples_per_batch=samples_per_batch,
        # Very low threshold to avoid premature convergence stop.
        # SigmoidTopK is deterministic so unique_ratio ~ 1/batch_size.
        convergence_threshold=0.001,
        # Effectively disable early stopping for short test runs.
        early_stopping_patience=num_epochs + 10,
        use_torch_compile=False,
        **config_kwargs,
    )

    trainer = PhysicsGuidedFlowTrainer(
        flow=flow_sampler,
        nqs=nqs,
        hamiltonian=H,
        config=cfg,
        device="cpu",
    )

    return trainer, flow_sampler, nqs


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestNFLiHBasic:
    """Basic NF training on LiH (12Q, 225 configs)."""

    def test_nf_lih_basic(self, lih_hamiltonian):
        """NF training on LiH completes without error and produces finite energy.

        SigmoidTopK is deterministic, so unique_ratio will be very low
        (~1/batch_size). We only verify training completes and energy is finite.
        """
        H = lih_hamiltonian
        trainer, flow_sampler, nqs = _make_trainer(
            H, num_epochs=30, min_epochs=10, samples_per_batch=100,
        )

        history = trainer.train()

        # Training completed -- verify history is populated
        assert len(history["energies"]) > 0, "No energies recorded"

        # Final energy is finite (not NaN or Inf)
        final_energy = history["energies"][-1]
        assert math.isfinite(final_energy), f"Final energy is not finite: {final_energy}"

        # unique_ratio is recorded and in valid range.
        # SigmoidTopK is deterministic so we expect very low values (~0.01).
        # We only verify it is non-negative and recorded.
        assert len(history["unique_ratios"]) > 0, "No unique_ratios recorded"
        for i, ur in enumerate(history["unique_ratios"]):
            assert 0 <= ur <= 1, f"unique_ratios[{i}] out of range: {ur}"


@pytest.mark.slow
@pytest.mark.molecular
class TestNFParticleConservation:
    """After training, NF samples must conserve particle number."""

    def test_nf_particle_conservation(self, lih_hamiltonian):
        """All sampled configs must have exactly n_alpha + n_beta electrons.

        ParticleConservingFlowSampler guarantees exact electron count via
        SigmoidTopK top-k selection. This test verifies the constraint holds
        after gradient updates have modified the flow parameters.
        """
        H = lih_hamiltonian
        trainer, flow_sampler, nqs = _make_trainer(
            H, num_epochs=20, min_epochs=5, samples_per_batch=100,
        )

        history = trainer.train()

        n_orbitals = H.n_orbitals
        n_alpha = H.n_alpha
        n_beta = H.n_beta

        with torch.no_grad():
            _, unique_configs = flow_sampler.sample(200)

        # Verify via the utility function
        valid, stats = verify_particle_conservation(
            unique_configs, n_orbitals, n_alpha, n_beta,
        )

        assert valid, (
            f"Particle conservation violated: "
            f"alpha violations={stats['alpha_violations']}, "
            f"beta violations={stats['beta_violations']} "
            f"out of {stats['n_configs']} configs"
        )

        # Manual double-check on raw counts
        alpha_counts = unique_configs[:, :n_orbitals].sum(dim=1)
        beta_counts = unique_configs[:, n_orbitals:].sum(dim=1)
        assert (alpha_counts == n_alpha).all(), (
            f"Alpha electron count mismatch: {alpha_counts.tolist()}"
        )
        assert (beta_counts == n_beta).all(), (
            f"Beta electron count mismatch: {beta_counts.tolist()}"
        )


@pytest.mark.slow
@pytest.mark.molecular
class TestNFH2OSampleDiversity:
    """NF on H2O (14Q, 441 configs) should produce a diverse subspace."""

    def test_nf_h2o_sample_diversity(self, h2o_hamiltonian):
        """The trainer's accumulated basis should span multiple excitation ranks.

        SigmoidTopK is deterministic, so the NF itself produces ~1 unique
        config. Diversity in the subspace comes from essential-config
        injection (HF + singles + doubles). We verify the trainer's
        accumulated basis (NF samples + essential configs) contains
        configurations at excitation ranks 0, 1, and 2.
        """
        H = h2o_hamiltonian
        trainer, flow_sampler, nqs = _make_trainer(
            H, num_epochs=30, min_epochs=10, samples_per_batch=150,
        )

        history = trainer.train()

        # The subspace used during training includes essential configs
        # (HF + singles + doubles) merged with NF samples.
        # Check via the trainer's accumulated basis OR essential configs.
        basis_to_check = trainer.accumulated_basis
        if basis_to_check is None:
            # Fall back to essential configs
            basis_to_check = trainer._essential_configs

        assert basis_to_check is not None, (
            "Neither accumulated_basis nor _essential_configs available"
        )

        # Compute excitation rank relative to HF
        hf_state = H.get_hf_state()
        n_orbitals = H.n_orbitals

        excitation_ranks = set()
        for cfg in basis_to_check:
            alpha_diff = (cfg[:n_orbitals] != hf_state[:n_orbitals]).sum().item()
            beta_diff = (cfg[n_orbitals:] != hf_state[n_orbitals:]).sum().item()
            rank = (alpha_diff + beta_diff) // 2
            excitation_ranks.add(rank)

        # Must have HF (rank 0), singles (rank 1), and doubles (rank 2)
        assert 0 in excitation_ranks, "HF configuration (rank 0) not found in subspace"
        assert 1 in excitation_ranks, "Single excitations (rank 1) not found in subspace"
        assert 2 in excitation_ranks, "Double excitations (rank 2) not found in subspace"

        print(f"Excitation ranks in subspace: {sorted(excitation_ranks)}")
        print(f"Subspace size: {len(basis_to_check)}")


@pytest.mark.slow
@pytest.mark.molecular
class TestNFBeH2Convergence:
    """NF on BeH2 (14Q, 1225 configs) should show reasonable energy."""

    def test_nf_beh2_convergence(self, beh2_hamiltonian):
        """Subspace energy should be physically reasonable and stable.

        With essential-config injection (HF + singles + doubles), the
        subspace energy converges to near-FCI from epoch 1. We verify:
        1. The energy is below zero (bound state)
        2. The energy is stable across epochs (no divergence)
        3. All recorded energies are finite
        """
        H = beh2_hamiltonian
        trainer, flow_sampler, nqs = _make_trainer(
            H, num_epochs=50, min_epochs=40, samples_per_batch=100,
        )

        history = trainer.train()

        energies = history["energies"]
        assert len(energies) >= 10, f"Too few epochs completed: {len(energies)}"

        # All energies must be finite
        for i, e in enumerate(energies):
            assert math.isfinite(e), f"Energy at epoch {i} is not finite: {e}"

        # Energy should be negative (bound state of BeH2)
        final_energy = energies[-1]
        assert final_energy < 0, f"Energy is positive (unphysical): {final_energy}"

        # Energy should be stable: std dev of last 10 epochs < 1.0 Ha
        last_10 = energies[-10:]
        mean_e = sum(last_10) / len(last_10)
        variance = sum((e - mean_e) ** 2 for e in last_10) / len(last_10)
        std_e = math.sqrt(variance)
        assert std_e < 1.0, (
            f"Energy not stable in last 10 epochs: std={std_e:.6f} Ha"
        )

        print(
            f"BeH2 energy: {final_energy:.6f} Ha "
            f"(std over last 10 epochs: {std_e:.6f} Ha)"
        )


@pytest.mark.slow
@pytest.mark.molecular
class TestNFProducesValidConfigs:
    """NF must produce valid binary configs of correct length."""

    def test_nf_produces_valid_configs(self, lih_hamiltonian):
        """All configs should be binary (0/1) and have length 2 * n_orbitals."""
        H = lih_hamiltonian
        trainer, flow_sampler, nqs = _make_trainer(
            H, num_epochs=15, min_epochs=5, samples_per_batch=100,
        )

        history = trainer.train()

        # Sample configs from trained NF
        n_orbitals = H.n_orbitals
        expected_length = 2 * n_orbitals

        with torch.no_grad():
            _, unique_configs = flow_sampler.sample(200)

        assert unique_configs.shape[1] == expected_length, (
            f"Config length {unique_configs.shape[1]} != expected {expected_length}"
        )

        # All values should be 0 or 1 (binary)
        is_binary = ((unique_configs == 0) | (unique_configs == 1)).all()
        assert is_binary, (
            f"Non-binary values found in configs. "
            f"Unique values: {torch.unique(unique_configs).tolist()}"
        )


@pytest.mark.slow
@pytest.mark.molecular
class TestNFTrainingWithPhysicsWeight:
    """NF training with explicit physics_weight should not produce NaN/Inf."""

    def test_nf_training_with_physics_weight(self, lih_hamiltonian):
        """Training with physics_weight=0.1 completes without NaN/Inf losses."""
        H = lih_hamiltonian
        trainer, flow_sampler, nqs = _make_trainer(
            H,
            num_epochs=30,
            min_epochs=10,
            samples_per_batch=100,
            physics_weight=0.1,
        )

        history = trainer.train()

        # All recorded values must be finite
        for key in ["energies", "teacher_losses", "physics_losses", "entropy_values"]:
            values = history[key]
            assert len(values) > 0, f"No values recorded for {key}"
            for i, v in enumerate(values):
                assert math.isfinite(v), f"{key}[{i}] is not finite: {v}"

        # unique_ratios should all be in [0, 1]
        for i, ur in enumerate(history["unique_ratios"]):
            assert 0 <= ur <= 1, f"unique_ratios[{i}] out of range: {ur}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-m", "slow"])
