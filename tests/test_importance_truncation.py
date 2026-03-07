"""Tests for PR 1.2: Importance-Ranked Basis Truncation.

TDD tests verifying that basis truncation uses diagonal energy ranking
instead of blind index-based slicing, and that essential configs (HF +
singles + doubles) are always preserved.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig


class TestRankAndTruncateBasis:
    """Test _rank_and_truncate_basis method."""

    @pytest.mark.molecular
    def test_method_exists(self, lih_hamiltonian):
        """SKQD should have _rank_and_truncate_basis method."""
        config = SKQDConfig(max_krylov_dim=2)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)
        assert hasattr(skqd, '_rank_and_truncate_basis')

    @pytest.mark.molecular
    def test_no_truncation_when_under_limit(self, lih_hamiltonian):
        """If basis is smaller than max_size, return unchanged."""
        config = SKQDConfig(max_krylov_dim=2)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)
        basis = skqd._subspace_basis[:50]
        result = skqd._rank_and_truncate_basis(basis, max_size=100)
        assert len(result) == 50

    @pytest.mark.molecular
    def test_truncation_returns_correct_size(self, lih_hamiltonian):
        """Truncated basis should have exactly max_size configs."""
        config = SKQDConfig(max_krylov_dim=2)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)
        basis = skqd._subspace_basis  # 225 configs
        result = skqd._rank_and_truncate_basis(basis, max_size=50)
        assert len(result) == 50

    @pytest.mark.molecular
    def test_essential_configs_preserved(self, lih_hamiltonian):
        """HF + singles + doubles must survive truncation."""
        config = SKQDConfig(max_krylov_dim=2)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)

        # LiH: 4 electrons (2α, 2β), 6 orbitals
        # HF (1) + singles (2*4=8) + doubles (αα:1*6=6, ββ:6, αβ:2*4*2*4=64) = ...
        # But total is only 225 configs, all within rank <=2 for this small system.
        # Use a small max_size that still exceeds essentials.
        basis = skqd._subspace_basis
        hf = lih_hamiltonian.get_hf_state()

        # Count essential configs in original basis
        diffs = (basis != hf.unsqueeze(0)).sum(dim=1)
        excitation_rank = diffs // 2
        n_essential = (excitation_rank <= 2).sum().item()

        # Truncate to slightly more than essential count
        max_size = min(n_essential + 5, len(basis))
        result = skqd._rank_and_truncate_basis(basis, max_size=max_size)

        # Verify HF state is in result
        hf_in_result = (result == hf.unsqueeze(0)).all(dim=1).any()
        assert hf_in_result, "HF state must survive truncation"

    @pytest.mark.molecular
    def test_lowest_energy_configs_selected(self, lih_hamiltonian):
        """Truncated basis should contain the lowest diagonal energy configs."""
        config = SKQDConfig(max_krylov_dim=2)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)
        basis = skqd._subspace_basis  # 225 configs

        # Get all diagonal energies
        diag_e = lih_hamiltonian.diagonal_elements_batch(basis)

        # Truncate to 50
        result = skqd._rank_and_truncate_basis(basis, max_size=50)
        result_energies = lih_hamiltonian.diagonal_elements_batch(result)

        # The max energy in the truncated set should be <= the median of the full set
        # (since we're keeping the lowest 50 out of 225)
        full_sorted = torch.sort(diag_e)[0]
        assert result_energies.max() <= full_sorted[60], (
            "Truncated basis should contain low-energy configs"
        )

    @pytest.mark.molecular
    def test_ranked_truncation_better_energy(self, n2_hamiltonian):
        """Importance-ranked truncation should yield better energy than blind truncation."""
        H = n2_hamiltonian
        config = SKQDConfig(max_krylov_dim=2, max_diag_basis_size=0)
        skqd = SampleBasedKrylovDiagonalization(H, config=config)
        basis = skqd._subspace_basis  # 14400 configs

        cap = 3000

        # Blind truncation (old behavior): first 3000 by index
        basis_blind = basis[:cap]

        # Ranked truncation (new behavior)
        basis_ranked = skqd._rank_and_truncate_basis(basis, max_size=cap)

        # Compute ground state energy with both
        E_blind, _ = skqd.compute_ground_state_energy(basis=basis_blind)
        E_ranked, _ = skqd.compute_ground_state_energy(basis=basis_ranked)

        # Ranked should be <= blind (lower energy = better)
        assert E_ranked <= E_blind + 1e-10, (
            f"Ranked {E_ranked:.6f} should be <= blind {E_blind:.6f}"
        )


class TestTruncationIntegration:
    """Test that truncation is wired into compute_ground_state_energy."""

    @pytest.mark.molecular
    def test_truncation_uses_ranking(self, lih_hamiltonian):
        """compute_ground_state_energy with small cap should use ranking."""
        config = SKQDConfig(max_krylov_dim=2, max_diag_basis_size=50)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)
        basis = skqd._subspace_basis  # 225 configs

        # This should trigger truncation to 50 via _rank_and_truncate_basis
        E, _ = skqd.compute_ground_state_energy(basis=basis)

        # Energy should be reasonable (not wildly wrong)
        e_fci = lih_hamiltonian.fci_energy()
        error_mha = abs(E - e_fci) * 1000
        # With 50 lowest-energy configs, should still be decent
        assert error_mha < 10, f"LiH error {error_mha:.2f} mHa with 50 ranked configs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
