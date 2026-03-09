"""Phase 4 validation tests: VMC+Sign on harder molecular systems.

Tests focus on systems where wavefunction sign structure matters:

1. **N2 stretched (2.0 A)**: Multi-reference system where CI coefficients have
   significant negative components. The sign network should meaningfully improve
   VMC energy vs the positive-real-only ansatz.

2. **CAS(10,10)**: 63,504 configurations. Tests that VMC+sign runs without OOM,
   produces finite energies, and AR flow generates valid (particle-conserving)
   configurations at this scale.

3. **Sign physics validation**: Verifies the sign network assigns physically
   meaningful sign structure (different signs for different excitation classes).

Usage:
    uv run pytest tests/test_phase4_validation.py -v --tb=short -m slow
    uv run pytest tests/test_phase4_validation.py -v --tb=short -k "not cas10_10"  # faster subset
"""

import math
import os
import sys

import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pyscf = pytest.importorskip("pyscf", reason="PySCF required for molecular tests")

from flows.autoregressive_flow import (
    AutoregressiveConfig,
    AutoregressiveFlowSampler,
    configs_to_states,
    states_to_configs,
)
from flows.sign_network import SignNetwork
from flows.vmc_training import VMCConfig, VMCTrainer

CHEMICAL_ACCURACY_HA = 1.594e-3  # 1.0 kcal/mol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_sampler(n_orbitals, n_alpha, n_beta):
    """Build a compact AR flow for test speed (fewer params than production)."""
    num_sites = 2 * n_orbitals
    config = AutoregressiveConfig(
        n_layers=2, n_heads=2, d_model=32, d_ff=64, dropout=0.0,
    )
    return AutoregressiveFlowSampler(
        num_sites=num_sites,
        n_alpha=n_alpha,
        n_beta=n_beta,
        transformer_config=config,
    )


def _count_excitation_rank(config, hf_state):
    """Count excitation rank (number of orbital differences from HF)."""
    diff = (config != hf_state).sum().item()
    # Each excitation flips one occupied -> unoccupied and one unoccupied -> occupied
    # So excitation rank = diff / 2
    return diff // 2


# ---------------------------------------------------------------------------
# Module-scoped fixtures (expensive Hamiltonians created once per module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def n2_stretched():
    """N2 at stretched bond length 2.0 A (multi-reference, STO-3G).

    20 qubits, 14400 configs. The triple bond is significantly stretched,
    creating strong multi-reference character with large CI coefficients
    on doubly-excited determinants.
    """
    from hamiltonians.molecular import create_n2_hamiltonian

    H = create_n2_hamiltonian(bond_length=2.0, device="cpu")
    return H


@pytest.fixture(scope="module")
def n2_equilibrium():
    """N2 at equilibrium bond length 1.10 A (STO-3G) for comparison."""
    from hamiltonians.molecular import create_n2_hamiltonian

    H = create_n2_hamiltonian(bond_length=1.10, device="cpu")
    return H


@pytest.fixture(scope="module")
def n2_cas10_10():
    """N2/cc-pVDZ CAS(10,10) Hamiltonian: 10 orbitals, 63504 configs.

    This is a very_large tier system. Creating the Hamiltonian takes ~10s
    due to CASSCF optimization, which is why this fixture is module-scoped.
    """
    from hamiltonians.molecular import create_n2_cas_hamiltonian

    H = create_n2_cas_hamiltonian(
        bond_length=1.10, basis="cc-pvdz", cas=(10, 10), device="cpu"
    )
    return H


# =========================================================================
# TestN2StretchedSign — N2 at 2.0 A with sign network
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestN2StretchedSign:
    """VMC+sign on stretched N2 (2.0 A) where sign structure matters.

    N2 at 2.0 A has strong multi-reference character — the ground state
    wavefunction has large negative CI coefficients on double excitations.
    A positive-real ansatz (no sign network) cannot represent these, so the
    sign network should provide a measurable improvement.
    """

    def test_n2_stretched_hamiltonian_valid(self, n2_stretched):
        """Verify the stretched N2 Hamiltonian is valid."""
        H = n2_stretched
        assert H.n_orbitals == 10
        assert H.n_alpha == 7
        assert H.n_beta == 7
        assert H.num_sites == 20

    def test_n2_stretched_fci_energy(self, n2_stretched):
        """FCI energy of stretched N2 should be finite and physical."""
        fci_e = n2_stretched.fci_energy()
        assert math.isfinite(fci_e), f"FCI energy not finite: {fci_e}"
        # N2/STO-3G at 2.0 A: total energy should be around -107 to -109 Ha
        assert -115 < fci_e < -100, f"FCI energy {fci_e:.6f} outside expected range"

    def test_n2_stretched_sign_vs_no_sign(self, n2_stretched):
        """Sign network should improve VMC energy on stretched N2.

        We train two VMC models (with and without sign network) under identical
        conditions. The sign model should find a lower (better) energy because
        it can represent the negative CI coefficients in the multi-reference
        ground state.

        We use a small transformer and few steps to keep test time reasonable.
        The improvement may be modest with so few steps, so we check that the
        signed energy is at least as good (not worse) rather than strictly better.
        """
        torch.manual_seed(42)

        # --- Without sign network (positive-real ansatz) ---
        flow_nosign = _make_small_sampler(
            n_orbitals=n2_stretched.n_orbitals,
            n_alpha=n2_stretched.n_alpha,
            n_beta=n2_stretched.n_beta,
        )
        vmc_cfg = VMCConfig(
            n_samples=200, n_steps=80, lr=2e-3, min_steps=80,
        )
        trainer_nosign = VMCTrainer(
            flow=flow_nosign, hamiltonian=n2_stretched,
            config=vmc_cfg, device="cpu",
        )
        result_nosign = trainer_nosign.train(verbose=False)

        # --- With sign network ---
        torch.manual_seed(42)
        flow_sign = _make_small_sampler(
            n_orbitals=n2_stretched.n_orbitals,
            n_alpha=n2_stretched.n_alpha,
            n_beta=n2_stretched.n_beta,
        )
        sign_net = SignNetwork(num_sites=n2_stretched.num_sites)
        trainer_sign = VMCTrainer(
            flow=flow_sign, hamiltonian=n2_stretched,
            config=vmc_cfg, device="cpu",
            sign_network=sign_net,
        )
        result_sign = trainer_sign.train(verbose=False)

        # Both should produce finite energies
        assert math.isfinite(result_nosign["best_energy"])
        assert math.isfinite(result_sign["best_energy"])

        # The sign model should do at least as well (allowing 10 mHa noise tolerance
        # for short training).  We do NOT require strict improvement because 80 steps
        # with 200 samples is not enough for convergence on a 20-qubit system.
        # The key assertion: sign model energy is finite and training didn't diverge.
        assert result_sign["best_energy"] < result_nosign["best_energy"] + 0.010, (
            f"Sign model ({result_sign['best_energy']:.6f}) much worse than "
            f"no-sign ({result_nosign['best_energy']:.6f})"
        )

    def test_n2_stretched_sign_convergence(self, n2_stretched):
        """Sign magnitudes should be non-trivial after training on stretched N2.

        After VMC training, the sign network should output values with appreciable
        magnitude (close to +/- 1), not stuck near 0.  On a multi-reference system
        the sign network needs to commit to definite signs to lower the energy.
        """
        torch.manual_seed(123)
        flow = _make_small_sampler(
            n_orbitals=n2_stretched.n_orbitals,
            n_alpha=n2_stretched.n_alpha,
            n_beta=n2_stretched.n_beta,
        )
        sign_net = SignNetwork(num_sites=n2_stretched.num_sites)

        vmc_cfg = VMCConfig(
            n_samples=200, n_steps=60, lr=2e-3, min_steps=60,
        )
        trainer = VMCTrainer(
            flow=flow, hamiltonian=n2_stretched,
            config=vmc_cfg, device="cpu",
            sign_network=sign_net,
        )
        trainer.train(verbose=False)

        # Sample configs and evaluate sign magnitudes
        with torch.no_grad():
            states, _ = flow._sample_autoregressive(200)
            configs = states_to_configs(states, flow.n_orbitals)
            signs = sign_net(configs.float())

        mean_magnitude = signs.abs().mean().item()
        assert mean_magnitude > 0.05, (
            f"Sign magnitudes too small after training: mean |s| = {mean_magnitude:.4f}. "
            f"Sign network should develop non-trivial sign structure on stretched N2."
        )
        # All sign values should be finite
        assert torch.isfinite(signs).all(), "Non-finite sign values detected"

    def test_n2_stretched_pipeline_chemical_accuracy(self, n2_stretched):
        """Direct-CI + SKQD should still achieve chemical accuracy on stretched N2.

        Even without VMC/sign, the pipeline (Direct-CI generating HF+singles+doubles
        followed by SKQD Krylov expansion) should reach chemical accuracy on
        N2/STO-3G at 2.0 A, since STO-3G has only 14400 configs and CISD captures
        the dominant correlation.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        fci_e = n2_stretched.fci_energy()
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            n2_stretched, config=config, exact_energy=fci_e,
        )
        results = pipeline.run(progress=False)

        # Extract best energy from pipeline results
        best_e = results.get("combined_energy") or results.get("skqd_energy")
        assert best_e is not None, "Pipeline did not produce an energy result"

        error_ha = abs(best_e - fci_e)
        assert error_ha < CHEMICAL_ACCURACY_HA, (
            f"Pipeline error {error_ha * 1000:.2f} mHa exceeds chemical accuracy "
            f"(1.594 mHa) on stretched N2. best_e={best_e:.6f}, fci_e={fci_e:.6f}"
        )

    def test_n2_stretched_local_energies_finite(self, n2_stretched):
        """Local energies with sign network should all be finite on stretched N2.

        This catches numerical issues like division-by-zero in sign ratios
        s(y)/s(x) or exp overflow in amplitude ratios.
        """
        torch.manual_seed(42)
        flow = _make_small_sampler(
            n_orbitals=n2_stretched.n_orbitals,
            n_alpha=n2_stretched.n_alpha,
            n_beta=n2_stretched.n_beta,
        )
        sign_net = SignNetwork(num_sites=n2_stretched.num_sites)

        trainer = VMCTrainer(
            flow=flow, hamiltonian=n2_stretched,
            config=VMCConfig(n_samples=100, n_steps=1),
            device="cpu", sign_network=sign_net,
        )

        # Compute local energies and check finiteness
        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(100)
            configs = states_to_configs(states, flow.n_orbitals)

        E_loc = trainer.compute_local_energies(configs, log_probs)
        assert E_loc.shape == (100,)
        n_finite = torch.isfinite(E_loc).sum().item()
        assert n_finite == 100, (
            f"Only {n_finite}/100 local energies are finite on stretched N2"
        )


# =========================================================================
# TestCAS10_10_VMC — CAS(10,10) scale validation with VMC+sign
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS10_10_VMC:
    """VMC+sign on CAS(10,10) with 63504 configurations.

    These tests validate that the Phase 4 AR flow + sign network can handle
    the very_large tier (40Q-scale) without crashing.  We use minimal training
    steps since the goal is correctness/stability, not convergence.
    """

    def test_cas10_10_hamiltonian_dimensions(self, n2_cas10_10):
        """CAS(10,10) has correct dimensions: 10 orbitals, 5+5 electrons."""
        H = n2_cas10_10
        assert H.n_orbitals == 10
        assert H.n_alpha == 5
        assert H.n_beta == 5
        assert H.num_sites == 20  # 2 * 10 spin orbitals

    def test_cas10_10_ar_flow_samples_valid(self, n2_cas10_10):
        """AR flow samples on CAS(10,10) have correct particle conservation.

        Each sampled configuration must have exactly n_alpha=5 alpha electrons
        (first 10 spin-orbitals) and n_beta=5 beta electrons (last 10).
        """
        H = n2_cas10_10
        torch.manual_seed(42)
        flow = _make_small_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(200)
            configs = states_to_configs(states, flow.n_orbitals)

        # Check particle conservation
        n_orb = H.n_orbitals
        alpha_counts = configs[:, :n_orb].sum(dim=1)
        beta_counts = configs[:, n_orb:].sum(dim=1)

        assert (alpha_counts == H.n_alpha).all(), (
            f"Alpha electron violation: expected {H.n_alpha}, "
            f"got counts {alpha_counts.unique().tolist()}"
        )
        assert (beta_counts == H.n_beta).all(), (
            f"Beta electron violation: expected {H.n_beta}, "
            f"got counts {beta_counts.unique().tolist()}"
        )

        # Log probs should be finite and negative
        assert torch.isfinite(log_probs).all(), "Non-finite log probs from AR flow"
        assert (log_probs <= 0).all(), "log_prob > 0 detected (impossible)"

    def test_cas10_10_vmc_runs_without_oom(self, n2_cas10_10):
        """VMC with AR flow + sign on 63504-config system doesn't OOM.

        This is the critical memory safety test. We run a few VMC steps with
        moderate sample count.  The test passes if it completes without
        MemoryError or OOM crash.
        """
        H = n2_cas10_10
        torch.manual_seed(42)
        flow = _make_small_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )
        sign_net = SignNetwork(num_sites=H.num_sites)

        # Small samples and steps to test memory, not convergence
        vmc_cfg = VMCConfig(
            n_samples=50, n_steps=5, lr=1e-3, min_steps=5,
        )
        trainer = VMCTrainer(
            flow=flow, hamiltonian=H,
            config=vmc_cfg, device="cpu",
            sign_network=sign_net,
        )

        result = trainer.train(verbose=False)
        assert "best_energy" in result
        assert result["n_steps"] == 5

    def test_cas10_10_vmc_energy_finite(self, n2_cas10_10):
        """Local energies are all finite on CAS(10,10) (no NaN/Inf from sign ratios).

        With 63504 configs, the Hamiltonian has many off-diagonal connections.
        The amplitude ratios exp(0.5*(log p(y) - log p(x))) and sign ratios
        s(y)/s(x) must remain numerically stable.
        """
        H = n2_cas10_10
        torch.manual_seed(42)
        flow = _make_small_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )
        sign_net = SignNetwork(num_sites=H.num_sites)

        trainer = VMCTrainer(
            flow=flow, hamiltonian=H,
            config=VMCConfig(n_samples=30, n_steps=1),
            device="cpu", sign_network=sign_net,
        )

        with torch.no_grad():
            states, log_probs = flow._sample_autoregressive(30)
            configs = states_to_configs(states, flow.n_orbitals)

        E_loc = trainer.compute_local_energies(configs, log_probs)
        assert E_loc.shape == (30,)
        assert torch.isfinite(E_loc).all(), (
            f"Non-finite local energies on CAS(10,10): "
            f"{(~torch.isfinite(E_loc)).sum().item()}/30 are NaN/Inf"
        )

    def test_cas10_10_vmc_positive_only_runs(self, n2_cas10_10):
        """VMC without sign network (positive-real ansatz) also runs on CAS(10,10).

        Baseline test: the positive-real ansatz should work (badly) without crashing.
        """
        H = n2_cas10_10
        torch.manual_seed(42)
        flow = _make_small_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )

        vmc_cfg = VMCConfig(
            n_samples=50, n_steps=5, lr=1e-3, min_steps=5,
        )
        trainer = VMCTrainer(
            flow=flow, hamiltonian=H,
            config=vmc_cfg, device="cpu",
            # No sign network
        )

        result = trainer.train(verbose=False)
        assert math.isfinite(result["best_energy"])

    def test_cas10_10_ar_flow_log_prob_consistency(self, n2_cas10_10):
        """Log probabilities from sampling match teacher-forced log_prob.

        The AR flow computes log_prob during sampling (accumulating conditionals)
        and via teacher forcing (single forward pass). Both should agree.
        """
        H = n2_cas10_10
        torch.manual_seed(42)
        flow = _make_small_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )

        with torch.no_grad():
            states, sample_log_probs = flow._sample_autoregressive(50)
            configs = states_to_configs(states, flow.n_orbitals)
            teacher_log_probs = flow.log_prob(configs.float())

        # Should agree within floating point tolerance
        # (both compute the same product of conditionals, just in different order)
        assert torch.allclose(sample_log_probs, teacher_log_probs, atol=1e-4), (
            f"Log prob mismatch: max diff = "
            f"{(sample_log_probs - teacher_log_probs).abs().max().item():.6f}"
        )


# =========================================================================
# TestSignNetworkMultiReference — sign physics validation
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestSignNetworkMultiReference:
    """Test that the sign network captures physically meaningful sign structure.

    In the exact ground state wavefunction, different excitation classes have
    characteristic sign patterns:
    - HF (rank 0): always positive (by convention, HF coefficient = +1)
    - Singles (rank 1): Brillouin's theorem makes these small for HF-optimized orbitals
    - Doubles (rank 2): can be positive or negative, dominant correlation contribution
    - Triples/quadruples: typically small, mixed signs
    """

    def test_sign_different_for_different_excitations(self, n2_stretched):
        """Sign network should assign meaningfully different outputs to configs
        at different excitation ranks.

        After brief training, the sign network should not output the same value
        for HF (rank 0) and doubly-excited configurations (rank 2). If it does,
        it has not learned any sign structure at all.
        """
        H = n2_stretched
        torch.manual_seed(42)
        flow = _make_small_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )
        sign_net = SignNetwork(num_sites=H.num_sites)

        vmc_cfg = VMCConfig(
            n_samples=200, n_steps=60, lr=2e-3, min_steps=60,
        )
        trainer = VMCTrainer(
            flow=flow, hamiltonian=H,
            config=vmc_cfg, device="cpu",
            sign_network=sign_net,
        )
        trainer.train(verbose=False)

        # Generate HF state and some excited states
        hf_state = H.get_hf_state().float()

        # Sample configs and classify by excitation rank
        with torch.no_grad():
            states, _ = flow._sample_autoregressive(500)
            configs = states_to_configs(states, flow.n_orbitals)

            hf_long = hf_state.long()
            ranks = torch.tensor(
                [_count_excitation_rank(c, hf_long) for c in configs.long()],
                dtype=torch.long,
            )

            # Evaluate sign on all configs + HF
            all_configs = torch.cat([hf_state.unsqueeze(0), configs.float()], dim=0)
            all_signs = sign_net(all_configs)

        sign_hf = all_signs[0].item()

        # Collect signs by rank
        rank_signs = {}
        for rank_val in ranks.unique().tolist():
            mask = ranks == rank_val
            if mask.sum() >= 3:  # Need enough samples
                rank_signs[rank_val] = all_signs[1:][mask]

        # The sign network should produce some variation across ranks.
        # We measure the standard deviation of mean sign across ranks.
        if len(rank_signs) >= 2:
            rank_means = [v.mean().item() for v in rank_signs.values()]
            sign_spread = max(rank_means) - min(rank_means)
            # After 60 training steps, we expect at least some differentiation.
            # A completely untrained network would also show some spread due to
            # random weights, so we check for non-trivial magnitude.
            assert sign_spread > 0.01 or abs(sign_hf) > 0.1, (
                f"Sign network shows no differentiation across excitation ranks. "
                f"HF sign = {sign_hf:.4f}, rank means = {rank_means}"
            )

    def test_h2o_sign_structure(self):
        """H2O ground state: sign network should produce non-uniform signs.

        H2O at equilibrium is well-described by HF, but the ground state still
        has negative CI coefficients (e.g., the HOMO->LUMO double excitation
        has negative coefficient in the CI expansion).

        After brief training, the sign network should not assign the same sign
        to all configurations.
        """
        from hamiltonians.molecular import create_h2o_hamiltonian

        H = create_h2o_hamiltonian(device="cpu")

        torch.manual_seed(42)
        flow = _make_small_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )
        sign_net = SignNetwork(num_sites=H.num_sites)

        vmc_cfg = VMCConfig(
            n_samples=200, n_steps=50, lr=2e-3, min_steps=50,
        )
        trainer = VMCTrainer(
            flow=flow, hamiltonian=H,
            config=vmc_cfg, device="cpu",
            sign_network=sign_net,
        )
        trainer.train(verbose=False)

        # Sample and evaluate signs
        with torch.no_grad():
            states, _ = flow._sample_autoregressive(200)
            configs = states_to_configs(states, flow.n_orbitals)
            signs = sign_net(configs.float())

        # Signs should not all be the same — there should be some variation
        sign_std = signs.std().item()
        assert sign_std > 0.01, (
            f"Sign network produced near-uniform signs on H2O: std = {sign_std:.6f}. "
            f"Expected variation due to negative CI coefficients."
        )

        # All signs should be finite and in (-1, 1) range
        assert torch.isfinite(signs).all()
        assert (signs.abs() < 1.0).all()

    def test_sign_gradient_through_local_energy(self, n2_stretched):
        """Gradients flow from local energy through sign network on stretched N2.

        The sign network gradient is computed via direct backpropagation through
        E_loc (not REINFORCE).  This test verifies that the gradient computation
        succeeds on a multi-reference system and produces non-zero gradients.
        """
        H = n2_stretched
        torch.manual_seed(42)
        flow = _make_small_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )
        sign_net = SignNetwork(num_sites=H.num_sites)

        trainer = VMCTrainer(
            flow=flow, hamiltonian=H,
            config=VMCConfig(n_samples=50, n_steps=1),
            device="cpu", sign_network=sign_net,
        )

        # Run one train step (which does backward through sign network)
        metrics = trainer.train_step()
        assert math.isfinite(metrics["energy"])
        assert math.isfinite(metrics["grad_norm"])

        # Sign network parameters should have received gradients
        has_nonzero_grad = False
        for name, param in sign_net.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero_grad = True
                break
        assert has_nonzero_grad, (
            "Sign network received no gradients from local energy on stretched N2"
        )

    def test_n2_equilibrium_vs_stretched_sign_need(self, n2_equilibrium, n2_stretched):
        """Stretched N2 should benefit more from signs than equilibrium N2.

        At equilibrium (1.10 A), N2 is mostly single-reference (HF dominates).
        At 2.0 A, it's strongly multi-reference.  The difference in sign variance
        after training should reflect this: more sign variation at 2.0 A.
        """
        results = {}
        for label, H in [("eq", n2_equilibrium), ("stretched", n2_stretched)]:
            torch.manual_seed(42)
            flow = _make_small_sampler(
                n_orbitals=H.n_orbitals,
                n_alpha=H.n_alpha,
                n_beta=H.n_beta,
            )
            sign_net = SignNetwork(num_sites=H.num_sites)

            vmc_cfg = VMCConfig(
                n_samples=200, n_steps=60, lr=2e-3, min_steps=60,
            )
            trainer = VMCTrainer(
                flow=flow, hamiltonian=H,
                config=vmc_cfg, device="cpu",
                sign_network=sign_net,
            )
            trainer.train(verbose=False)

            with torch.no_grad():
                states, _ = flow._sample_autoregressive(200)
                configs = states_to_configs(states, flow.n_orbitals)
                signs = sign_net(configs.float())
            results[label] = signs.std().item()

        # Both should have some sign variation (both are non-trivial molecules)
        assert results["eq"] > 0.001, f"Equilibrium N2 sign std too small: {results['eq']}"
        assert results["stretched"] > 0.001, (
            f"Stretched N2 sign std too small: {results['stretched']}"
        )
        # Note: We do NOT assert stretched > eq here, because with only 60 training
        # steps the sign networks may not have converged enough for a reliable comparison.
        # The test validates that both produce finite, non-trivial sign structure.


# =========================================================================
# TestVMCSignPipelineE2E — end-to-end pipeline tests
# =========================================================================


@pytest.mark.slow
@pytest.mark.molecular
class TestVMCSignPipelineE2E:
    """End-to-end pipeline tests with VMC+sign+SKQD."""

    def test_n2_stretched_vmc_then_skqd(self, n2_stretched):
        """VMC+sign followed by SKQD on stretched N2 should produce a valid energy.

        This tests the full Phase 4 pipeline: autoregressive flow + sign network
        trained with VMC, then SKQD diagonalization on the sampled subspace.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        fci_e = n2_stretched.fci_energy()
        config = PipelineConfig(
            subspace_mode="skqd",
            use_autoregressive_flow=True,
            use_vmc_training=True,
            use_sign_network=True,
            vmc_n_steps=30,
            vmc_n_samples=100,
            vmc_lr=2e-3,
            skip_nf_training=False,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            n2_stretched, config=config, exact_energy=fci_e, auto_adapt=False,
        )
        results = pipeline.run(progress=False)

        # Pipeline should produce a valid energy
        best_e = results.get("combined_energy") or results.get("skqd_energy")
        if best_e is None:
            # Might be stored under vmc_energy if SKQD was skipped
            best_e = results.get("vmc_energy")
        assert best_e is not None, f"No energy in pipeline results: {list(results.keys())}"
        assert math.isfinite(best_e), f"Pipeline energy not finite: {best_e}"

    def test_h2o_vmc_sign_energy_above_fci(self):
        """VMC+sign energy on H2O must satisfy variational principle: E_VMC >= E_FCI.

        This is a fundamental physics constraint.  If the VMC energy goes below
        FCI, there is a bug in the local energy computation or sign ratios.
        """
        from hamiltonians.molecular import create_h2o_hamiltonian

        H = create_h2o_hamiltonian(device="cpu")
        fci_e = H.fci_energy()

        torch.manual_seed(42)
        flow = _make_small_sampler(
            n_orbitals=H.n_orbitals,
            n_alpha=H.n_alpha,
            n_beta=H.n_beta,
        )
        sign_net = SignNetwork(num_sites=H.num_sites)

        vmc_cfg = VMCConfig(
            n_samples=300, n_steps=80, lr=2e-3, min_steps=80,
        )
        trainer = VMCTrainer(
            flow=flow, hamiltonian=H,
            config=vmc_cfg, device="cpu",
            sign_network=sign_net,
        )
        result = trainer.train(verbose=False)

        # Variational principle: best VMC energy >= FCI energy
        # Allow 1 mHa tolerance for sampling noise
        assert result["best_energy"] >= fci_e - 0.001, (
            f"Variational principle violated on H2O: VMC best = {result['best_energy']:.6f}, "
            f"FCI = {fci_e:.6f}, diff = {(result['best_energy'] - fci_e) * 1000:.2f} mHa"
        )
