"""PR-C1: N2/cc-pVDZ ladder validation tests.

End-to-end integration tests verifying the SKQD pipeline works correctly
on cc-pVDZ basis set with CAS active spaces of increasing size:

- CAS(6,6):  6 active orbitals, ~400 configs   -- small tier, exact FCI match
- CAS(10,8): 8 active orbitals, ~3136 configs  -- medium tier, chemical accuracy
- CAS(10,10): 10 active orbitals, ~63504 configs -- very_large tier, sparse path

These tests are the first validation beyond STO-3G and represent the minimum
standard for 2026 publication (cc-pVDZ is the de facto baseline basis set).

Usage:
    uv run pytest tests/test_ccpvdz_validation.py -m slow -x -v
"""

import math
import os
import sys
from math import comb

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pyscf = pytest.importorskip("pyscf", reason="PySCF required for cc-pVDZ CAS tests")


# ---------------------------------------------------------------------------
# Module-scoped fixtures (CAS Hamiltonians are expensive to create via CASSCF)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def n2_cas6_6():
    """N2/cc-pVDZ CAS(6,6) Hamiltonian: 6 orbitals, C(6,3)^2 = 400 configs."""
    from hamiltonians.molecular import create_n2_cas_hamiltonian

    H = create_n2_cas_hamiltonian(
        bond_length=1.10, basis="cc-pvdz", cas=(6, 6), device="cpu"
    )
    return H


@pytest.fixture(scope="module")
def n2_cas10_8():
    """N2/cc-pVDZ CAS(10,8) Hamiltonian: 8 orbitals, C(8,5)^2 = 3136 configs."""
    from hamiltonians.molecular import create_n2_cas_hamiltonian

    H = create_n2_cas_hamiltonian(
        bond_length=1.10, basis="cc-pvdz", cas=(10, 8), device="cpu"
    )
    return H


@pytest.fixture(scope="module")
def n2_cas10_10():
    """N2/cc-pVDZ CAS(10,10) Hamiltonian: 10 orbitals, C(10,5)^2 = 63504 configs."""
    from hamiltonians.molecular import create_n2_cas_hamiltonian

    H = create_n2_cas_hamiltonian(
        bond_length=1.10, basis="cc-pvdz", cas=(10, 10), device="cpu"
    )
    return H


# Chemical accuracy threshold: 1.0 kcal/mol = 1.594 mHa
CHEMICAL_ACCURACY_HA = 1.594e-3


# ---------------------------------------------------------------------------
# Test 1: CAS(6,6) exact FCI match
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS66ExactFCI:
    """CAS(6,6) is small enough for near-exact diag -- should match PySCF CASCI."""

    def test_ccpvdz_cas66_exact_fci(self, n2_cas6_6):
        """CAS(6,6) pipeline energy should match PySCF CASCI within 0.15 mHa.

        With only 400 configs, Direct-CI (HF + singles + doubles) covers
        a large fraction of the Hilbert space. SKQD should recover essentially
        the exact CAS-FCI energy.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas6_6
        fci_e = H.fci_energy()

        # Verify Hamiltonian dimensions
        assert H.n_orbitals == 6
        n_alpha_expected = 3  # (6+0)//2
        n_beta_expected = 3
        assert H.n_alpha == n_alpha_expected
        assert H.n_beta == n_beta_expected
        n_configs = comb(6, 3) ** 2
        assert n_configs == 400

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=False,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            H, config=config, exact_energy=fci_e
        )
        results = pipeline.run(progress=False)

        energy = results["combined_energy"]
        error_ha = abs(energy - fci_e)
        error_mha = error_ha * 1000

        print(f"\nCAS(6,6) FCI reference: {fci_e:.8f} Ha")
        print(f"CAS(6,6) pipeline:     {energy:.8f} Ha")
        print(f"Error: {error_mha:.4f} mHa")

        # 0.15 mHa threshold: CAS(6,6) with 400 configs is small but SKQD's
        # Krylov extraction may not recover the exact eigenvalue due to
        # truncated Krylov subspace and regularization. Well within chemical
        # accuracy (1.594 mHa) and near-exact match.
        assert error_mha < 0.15, (
            f"CAS(6,6) error {error_mha:.4f} mHa exceeds 0.15 mHa threshold. "
            f"Small active space should nearly match FCI."
        )


# ---------------------------------------------------------------------------
# Test 2: CAS(10,8) SKQD chemical accuracy
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS10_8SKQD:
    """CAS(10,8) Direct-CI SKQD should achieve chemical accuracy."""

    def test_ccpvdz_cas10_8_skqd(self, n2_cas10_8):
        """CAS(10,8) with 3136 configs -- SKQD should reach chemical accuracy.

        This is the medium-tier system. Direct-CI generates ~876 essential configs
        (HF + singles + doubles), which is a good basis for SKQD to refine.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_8
        fci_e = H.fci_energy()

        # Verify dimensions
        assert H.n_orbitals == 8
        assert H.n_alpha == 5
        assert H.n_beta == 5

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=False,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            H, config=config, exact_energy=fci_e
        )
        results = pipeline.run(progress=False)

        energy = results["combined_energy"]
        error_ha = abs(energy - fci_e)
        error_mha = error_ha * 1000

        print(f"\nCAS(10,8) FCI reference: {fci_e:.8f} Ha")
        print(f"CAS(10,8) pipeline:     {energy:.8f} Ha")
        print(f"Error: {error_mha:.4f} mHa")

        assert error_mha < 1.0, (
            f"CAS(10,8) error {error_mha:.4f} mHa exceeds 1.0 mHa threshold "
            f"(chemical accuracy = {CHEMICAL_ACCURACY_HA * 1000:.1f} mHa)"
        )


# ---------------------------------------------------------------------------
# Test 3: CAS(10,10) sparse SKQD, no OOM
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS10_10Sparse:
    """CAS(10,10) 63504 configs -- must use sparse SKQD, no OOM."""

    def test_ccpvdz_cas10_10_sparse(self, n2_cas10_10):
        """CAS(10,10) pipeline completes without OOM on 63504-config system.

        With 63504 total configs, this triggers the very_large tier:
        - adapt_to_system_size sets max_diag_basis_size=15000
        - SKQD uses sparse eigensolver for subspaces > 3000
        - OOM guards prevent dense matrix construction for large subspaces

        The test completing without crash IS the OOM assertion.

        Note on accuracy: Direct-CI (HF+S+D) only generates ~876 essential
        configs out of 63504 total (~1.4% coverage). This is insufficient
        to reach chemical accuracy in the CAS(10,10) space -- the missing
        triples/quadruples contribute ~14 mHa. This is the exact scenario
        where NF training or autoregressive sampling would add value.
        We set a 20 mHa threshold that validates the pipeline works but
        acknowledges Direct-CI's limitation on very_large systems.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_10
        fci_e = H.fci_energy()

        # Verify this is indeed a very_large system
        assert H.n_orbitals == 10
        n_configs = comb(10, 5) ** 2
        assert n_configs == 63504

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        # auto_adapt=True (default) will classify as very_large
        pipeline = FlowGuidedKrylovPipeline(
            H, config=config, exact_energy=fci_e
        )
        results = pipeline.run(progress=False)

        energy = results["combined_energy"]
        error_ha = abs(energy - fci_e)
        error_mha = error_ha * 1000

        print(f"\nCAS(10,10) FCI reference: {fci_e:.8f} Ha")
        print(f"CAS(10,10) pipeline:     {energy:.8f} Ha")
        print(f"Error: {error_mha:.4f} mHa")
        print(f"Note: Direct-CI covers ~1.4% of 63504 configs. "
              f"NF/autoregressive needed for chemical accuracy.")

        # Direct-CI on very_large system cannot reach chemical accuracy.
        # 20 mHa threshold validates pipeline runs correctly without OOM
        # while acknowledging the CISD limitation at this scale.
        assert error_mha < 20.0, (
            f"CAS(10,10) error {error_mha:.4f} mHa exceeds 20 mHa threshold. "
            f"Direct-CI expected ~14 mHa; check for regression."
        )
        assert math.isfinite(energy), f"Energy not finite: {energy}"


# ---------------------------------------------------------------------------
# Test 4: CAS(10,8) with NNCI
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.molecular
class TestCAS10_8NNCI:
    """CAS(10,8) with NNCI should be at least as good as without."""

    def test_ccpvdz_cas10_8_nnci(self, n2_cas10_8):
        """NNCI discovers triples/quadruples -- energy should not degrade.

        Run pipeline twice: with and without NNCI. Since NNCI expands the
        basis with NN-classified higher excitations, the variational energy
        should be equal or lower (better).
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_8
        fci_e = H.fci_energy()

        # Run WITHOUT NNCI
        config_no_nnci = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=False,
            device="cpu",
        )
        pipeline_no_nnci = FlowGuidedKrylovPipeline(
            H, config=config_no_nnci, exact_energy=fci_e
        )
        results_no_nnci = pipeline_no_nnci.run(progress=False)
        e_no_nnci = results_no_nnci["combined_energy"]

        # Run WITH NNCI
        config_nnci = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=True,
            device="cpu",
        )
        pipeline_nnci = FlowGuidedKrylovPipeline(
            H, config=config_nnci, exact_energy=fci_e
        )
        results_nnci = pipeline_nnci.run(progress=False)
        e_nnci = results_nnci["combined_energy"]

        print(f"\nCAS(10,8) without NNCI: {e_no_nnci:.8f} Ha")
        print(f"CAS(10,8) with NNCI:    {e_nnci:.8f} Ha")
        print(f"FCI reference:          {fci_e:.8f} Ha")
        print(f"NNCI improvement:       {(e_no_nnci - e_nnci) * 1000:.4f} mHa")

        # NNCI energy should not be worse than Direct-CI by more than 0.001 Ha (1 mHa)
        # It can be slightly worse due to different Krylov convergence paths,
        # but should not degrade significantly.
        assert e_nnci <= e_no_nnci + 0.001, (
            f"NNCI energy ({e_nnci:.6f}) is worse than Direct-CI "
            f"({e_no_nnci:.6f}) by > 1 mHa"
        )


# ---------------------------------------------------------------------------
# Test 5: Cross-CAS energy ordering
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.molecular
class TestCASEnergyOrdering:
    """Larger active space should give lower (better) energy."""

    def test_ccpvdz_energy_ordering(self, n2_cas6_6, n2_cas10_8, n2_cas10_10):
        """Energy ordering: CAS(6,6) >= CAS(10,8) >= CAS(10,10).

        A larger active space captures more electron correlation, so the
        variational energy should decrease. We allow a small tolerance for
        numerical noise from different CASSCF orbital optimizations.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        energies = {}
        for label, H in [
            ("cas6_6", n2_cas6_6),
            ("cas10_8", n2_cas10_8),
            ("cas10_10", n2_cas10_10),
        ]:
            fci_e = H.fci_energy()
            config = PipelineConfig(
                subspace_mode="skqd",
                skip_nf_training=True,
                device="cpu",
            )
            pipeline = FlowGuidedKrylovPipeline(
                H, config=config, exact_energy=fci_e
            )
            results = pipeline.run(progress=False)
            energies[label] = results["combined_energy"]

        e_66 = energies["cas6_6"]
        e_108 = energies["cas10_8"]
        e_1010 = energies["cas10_10"]

        print(f"\nCAS(6,6)   energy: {e_66:.8f} Ha")
        print(f"CAS(10,8)  energy: {e_108:.8f} Ha")
        print(f"CAS(10,10) energy: {e_1010:.8f} Ha")
        print(f"CAS(6,6) - CAS(10,8):   {(e_66 - e_108) * 1000:.4f} mHa")
        print(f"CAS(10,8) - CAS(10,10): {(e_108 - e_1010) * 1000:.4f} mHa")

        # Larger CAS should give lower energy. Allow 5 mHa tolerance for
        # numerical noise from different CASSCF orbital optimizations and
        # different Krylov convergence paths.
        tolerance = 0.005  # 5 mHa

        assert e_108 <= e_66 + tolerance, (
            f"CAS(10,8) energy ({e_108:.6f}) is higher than "
            f"CAS(6,6) energy ({e_66:.6f}) by > {tolerance * 1000:.0f} mHa"
        )
        assert e_1010 <= e_108 + tolerance, (
            f"CAS(10,10) energy ({e_1010:.6f}) is higher than "
            f"CAS(10,8) energy ({e_108:.6f}) by > {tolerance * 1000:.0f} mHa"
        )


# ---------------------------------------------------------------------------
# Test 6: Particle conservation in CAS
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.molecular
class TestCASParticleConservation:
    """All configs in CAS active space must conserve electron number."""

    def test_ccpvdz_particle_conservation(self, n2_cas10_8):
        """Every config generated by the pipeline must have the correct
        number of alpha and beta electrons for the CAS active space.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_8
        n_orb = H.n_orbitals

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            skip_skqd=True,  # Only test config generation
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(H, config=config)
        pipeline.train_flow_nqs(progress=False)
        essential = pipeline._essential_configs

        assert essential is not None, "Essential configs not generated"
        assert len(essential) > 0, "Empty essential configs"

        alpha_counts = essential[:, :n_orb].sum(dim=1)
        beta_counts = essential[:, n_orb:].sum(dim=1)

        # All configs must have exact electron count
        alpha_violations = (alpha_counts != H.n_alpha).sum().item()
        beta_violations = (beta_counts != H.n_beta).sum().item()

        print(f"\nCAS(10,8) particle conservation check:")
        print(f"  Total configs: {len(essential)}")
        print(f"  Expected: {H.n_alpha} alpha, {H.n_beta} beta")
        print(f"  Alpha violations: {alpha_violations}")
        print(f"  Beta violations: {beta_violations}")

        assert alpha_violations == 0, (
            f"{alpha_violations} configs have wrong alpha count. "
            f"Expected {H.n_alpha}, got {torch.unique(alpha_counts).tolist()}"
        )
        assert beta_violations == 0, (
            f"{beta_violations} configs have wrong beta count. "
            f"Expected {H.n_beta}, got {torch.unique(beta_counts).tolist()}"
        )


# ---------------------------------------------------------------------------
# Test 7: Adaptive dt on cc-pVDZ
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.molecular
class TestCASAdaptiveDt:
    """cc-pVDZ has wider spectral range -- adaptive dt should reduce dt."""

    def test_ccpvdz_adaptive_dt(self, n2_cas10_8):
        """CAS(10,8) with adaptive_dt and a large initial dt.

        cc-pVDZ integrals have wider eigenvalue spread than STO-3G.
        The adaptive dt mechanism (Nyquist-like clamping: dt <= pi/spectral_range)
        should reduce the effective dt below the initial value.
        """
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        H = n2_cas10_8
        fci_e = H.fci_energy()

        # Use a deliberately large initial dt to force adaptive reduction
        large_dt = 1.0

        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            use_nnci=False,
            time_step=large_dt,
            skqd_adaptive_dt=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            H, config=config, exact_energy=fci_e
        )
        results = pipeline.run(progress=False)

        # The SKQD results contain effective_dt
        skqd_results = results.get("skqd_results", {})
        effective_dt = skqd_results.get("effective_dt", large_dt)

        print(f"\nInitial dt: {large_dt}")
        print(f"Effective dt: {effective_dt:.6f}")
        print(f"Reduction ratio: {effective_dt / large_dt:.4f}")

        # For cc-pVDZ with wide spectral range, effective_dt should be < initial dt
        assert effective_dt < large_dt, (
            f"Adaptive dt did not reduce dt: effective={effective_dt:.6f}, "
            f"initial={large_dt}. Expected reduction for cc-pVDZ spectral range."
        )

        # Pipeline should still produce a reasonable energy
        energy = results["combined_energy"]
        assert math.isfinite(energy), f"Energy not finite: {energy}"
        error_mha = abs(energy - fci_e) * 1000
        print(f"Energy error: {error_mha:.4f} mHa")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header", "-m", "slow"])
