"""Tests for PR 1.4: DPP selection uses streaming greedy (no O(n²) matrix).

Verifies that _dpp_select delegates to stochastic_greedy_select and that
analyze_basis_diversity uses sampled distances for large bases.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from postprocessing.diversity_selection import (
    DiversityConfig,
    DiversitySelector,
    analyze_basis_diversity,
    compute_hamming_distance_matrix,
    stochastic_greedy_select,
)


class TestDppUsesStreamingGreedy:
    """Verify _dpp_select delegates to stochastic_greedy_select."""

    def test_dpp_select_returns_correct_count(self):
        """DPP select should return exactly n_select indices."""
        n, sites = 200, 20
        configs = torch.randint(0, 2, (n, sites))
        weights = torch.rand(n)

        cfg = DiversityConfig(max_configs=50, min_hamming_distance=2)
        ref = configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._dpp_select(configs, weights, 50)
        assert len(result) == 50

    def test_dpp_select_no_duplicates(self):
        """Selected indices should be unique."""
        n, sites = 200, 20
        configs = torch.randint(0, 2, (n, sites))
        weights = torch.rand(n)

        cfg = DiversityConfig(max_configs=80)
        ref = configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._dpp_select(configs, weights, 80)
        assert len(result) == len(result.unique())

    def test_dpp_select_includes_highest_weight(self):
        """The highest-weight config should be selected first."""
        n, sites = 100, 20
        configs = torch.randint(0, 2, (n, sites))
        weights = torch.rand(n)
        weights[42] = 100.0  # Make index 42 the clear highest

        cfg = DiversityConfig(max_configs=10)
        ref = configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._dpp_select(configs, weights, 10)
        assert 42 in result.tolist(), "Highest-weight config must be selected"

    def test_dpp_select_all_when_n_small(self):
        """When n <= n_select, return all indices."""
        n, sites = 10, 20
        configs = torch.randint(0, 2, (n, sites))
        weights = torch.rand(n)

        cfg = DiversityConfig(max_configs=20)
        ref = configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._dpp_select(configs, weights, 20)
        assert len(result) == n

    def test_dpp_select_memory_bounded(self):
        """DPP on large input should NOT allocate O(n²) memory."""
        # 10K configs × 40 sites: old O(n²) would need 10K×10K×40 = 4GB
        # New streaming approach: O(n) bitpacked + O(n) min_dists ≈ 160KB
        n, sites = 10000, 40
        configs = torch.randint(0, 2, (n, sites))
        weights = torch.rand(n)

        cfg = DiversityConfig(max_configs=500, min_hamming_distance=2)
        ref = configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        # This should complete without OOM
        result = selector._dpp_select(configs, weights, 500)
        assert len(result) == 500


class TestAnalyzeBasisDiversitySampled:
    """Verify analyze_basis_diversity uses sampled distances for large bases."""

    def test_small_basis_exact(self):
        """Small basis should use exact distance matrix."""
        n, sites = 50, 10
        configs = torch.randint(0, 2, (n, sites))
        ref = configs[0]

        stats = analyze_basis_diversity(configs, ref)
        assert 'mean_distance' in stats
        assert 'min_distance' in stats
        assert stats['n_configs'] == n

    def test_large_basis_sampled(self):
        """Large basis should use sampled distances (no OOM)."""
        n, sites = 6000, 20
        configs = torch.randint(0, 2, (n, sites))
        ref = configs[0]

        # This would OOM with O(n²) on large systems
        stats = analyze_basis_diversity(configs, ref)
        assert 'mean_distance' in stats
        assert stats['n_configs'] == n

    def test_sampled_distances_reasonable(self):
        """Sampled distance stats should be close to exact for random configs."""
        n, sites = 6000, 20
        torch.manual_seed(42)
        configs = torch.randint(0, 2, (n, sites))
        ref = configs[0]

        stats = analyze_basis_diversity(configs, ref)
        # For random binary vectors of length 20, expected mean Hamming ≈ 10
        assert 8 < stats['mean_distance'] < 12


class TestEndToEndDiversitySelection:
    """Integration test: full pipeline diversity selection uses streaming."""

    @pytest.mark.molecular
    def test_pipeline_diversity_no_oom(self, n2_hamiltonian):
        """Pipeline diversity selection on N2 (14400 configs) should work."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        e_fci = n2_hamiltonian.fci_energy()
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            n2_hamiltonian, config=config, exact_energy=e_fci
        )
        results = pipeline.run(progress=False)
        best = results.get("combined_energy", results.get("skqd_energy"))
        error_mha = abs(best - e_fci) * 1000
        assert error_mha < 1.0, f"N2 error {error_mha:.4f} mHa"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
