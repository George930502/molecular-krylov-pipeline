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


class TestFindIndicesHashBased:
    """Tests for M1 fix: _find_indices should use hash-based O(n) lookup
    instead of O(n*m) pairwise comparison."""

    def _naive_find_indices(self, all_configs, subset_configs):
        """Reference O(n*m) implementation for correctness checking."""
        indices = []
        for i in range(len(subset_configs)):
            matches = (all_configs == subset_configs[i]).all(dim=1)
            idx = torch.where(matches)[0]
            if len(idx) > 0:
                indices.append(idx[0].item())
        return torch.tensor(indices, device=all_configs.device)

    def test_find_indices_matches_naive(self):
        """Optimized _find_indices must return same results as naive O(n*m) version."""
        torch.manual_seed(123)
        n, sites = 200, 20
        all_configs = torch.randint(0, 2, (n, sites))
        # Select a random subset
        subset_idx = torch.randperm(n)[:50]
        subset_configs = all_configs[subset_idx]

        cfg = DiversityConfig(max_configs=50)
        ref = all_configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._find_indices(all_configs, subset_configs)
        expected = self._naive_find_indices(all_configs, subset_configs)

        assert torch.equal(result, expected), (
            f"Mismatch: result={result.tolist()}, expected={expected.tolist()}"
        )

    def test_find_indices_empty_subset(self):
        """Empty subset should return empty tensor."""
        n, sites = 50, 10
        all_configs = torch.randint(0, 2, (n, sites))
        subset_configs = torch.empty(0, sites, dtype=all_configs.dtype)

        cfg = DiversityConfig()
        ref = all_configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._find_indices(all_configs, subset_configs)
        assert len(result) == 0

    def test_find_indices_single_config(self):
        """Single config subset should find the correct index."""
        n, sites = 100, 12
        torch.manual_seed(42)
        all_configs = torch.randint(0, 2, (n, sites))
        target_idx = 37
        subset_configs = all_configs[target_idx:target_idx + 1]

        cfg = DiversityConfig()
        ref = all_configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._find_indices(all_configs, subset_configs)
        assert len(result) == 1
        assert result[0].item() == target_idx

    def test_find_indices_preserves_order(self):
        """Returned indices must match the order of subset_configs."""
        torch.manual_seed(99)
        n, sites = 100, 14
        all_configs = torch.randint(0, 2, (n, sites))
        # Pick indices in a specific non-sorted order
        subset_idx = torch.tensor([50, 10, 90, 30, 70])
        subset_configs = all_configs[subset_idx]

        cfg = DiversityConfig()
        ref = all_configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._find_indices(all_configs, subset_configs)
        assert result.tolist() == subset_idx.tolist()

    def test_find_indices_missing_config_skipped(self):
        """Configs not found in all_configs should be skipped (not crash)."""
        n, sites = 50, 10
        torch.manual_seed(7)
        all_configs = torch.randint(0, 2, (n, sites))
        # Create a config guaranteed not in all_configs
        missing = torch.ones(1, sites, dtype=all_configs.dtype) * 2  # value=2, not binary
        subset_configs = torch.cat([all_configs[:3], missing], dim=0)

        cfg = DiversityConfig()
        ref = all_configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._find_indices(all_configs, subset_configs)
        # Should find the first 3 but skip the missing one
        assert len(result) == 3

    def test_find_indices_duplicate_configs_returns_first(self):
        """When all_configs has duplicates, return the first occurrence index."""
        sites = 8
        base = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]])
        # Duplicate at index 0 and 5
        all_configs = torch.randint(0, 2, (10, sites))
        all_configs[0] = base
        all_configs[5] = base

        cfg = DiversityConfig()
        ref = all_configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._find_indices(all_configs, base)
        assert len(result) == 1
        # Should return index 0 (first occurrence)
        assert result[0].item() == 0

    def test_find_indices_40_sites(self):
        """Must work correctly for 40-site configs (realistic 40Q system)."""
        torch.manual_seed(55)
        n, sites = 500, 40
        all_configs = torch.randint(0, 2, (n, sites))
        subset_idx = torch.randperm(n)[:100]
        subset_configs = all_configs[subset_idx]

        cfg = DiversityConfig()
        ref = all_configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._find_indices(all_configs, subset_configs)
        expected = self._naive_find_indices(all_configs, subset_configs)
        assert torch.equal(result, expected)

    def test_find_indices_64_sites(self):
        """Must work for 64-site configs (max bitpack width)."""
        torch.manual_seed(77)
        n, sites = 200, 64
        all_configs = torch.randint(0, 2, (n, sites))
        subset_idx = torch.randperm(n)[:30]
        subset_configs = all_configs[subset_idx]

        cfg = DiversityConfig()
        ref = all_configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._find_indices(all_configs, subset_configs)
        expected = self._naive_find_indices(all_configs, subset_configs)
        assert torch.equal(result, expected)

    def test_find_indices_over_64_sites_fallback(self):
        """Systems with >64 sites must use tuple-hash fallback, not crash."""
        torch.manual_seed(88)
        n, sites = 50, 80  # >64 sites
        all_configs = torch.randint(0, 2, (n, sites))
        subset_idx = torch.randperm(n)[:10]
        subset_configs = all_configs[subset_idx]

        cfg = DiversityConfig()
        ref = all_configs[0]
        selector = DiversitySelector(cfg, ref, sites // 2)

        result = selector._find_indices(all_configs, subset_configs)
        expected = self._naive_find_indices(all_configs, subset_configs)
        assert torch.equal(result, expected)

    def test_find_indices_scaling_linear(self):
        """Benchmark: O(n) hash lookup should scale linearly, not quadratically.

        We compare wall-clock time at n=5000 vs n=20000 (4x growth).
        If truly O(n+m), the ratio should be ~4x.
        If O(n*m) with m proportional to n, ratio would be ~16x.
        We use sizes large enough to avoid timer-resolution noise, and
        enough iterations to average out xdist CPU contention.
        """
        import time

        sites = 20

        def time_find_indices(n, m):
            torch.manual_seed(42)
            all_configs = torch.randint(0, 2, (n, sites))
            subset_idx = torch.randperm(n)[:m]
            subset_configs = all_configs[subset_idx]

            cfg = DiversityConfig()
            ref = all_configs[0]
            selector = DiversitySelector(cfg, ref, sites // 2)

            # Warmup (2 rounds to stabilize JIT/alloc)
            selector._find_indices(all_configs, subset_configs)
            selector._find_indices(all_configs, subset_configs)

            n_iter = 5
            start = time.perf_counter()
            for _ in range(n_iter):
                selector._find_indices(all_configs, subset_configs)
            elapsed = time.perf_counter() - start
            return elapsed / n_iter

        t_small = time_find_indices(5000, 1000)
        t_large = time_find_indices(20000, 4000)

        ratio = t_large / max(t_small, 1e-9)
        # O(n+m) hash lookup: ratio should be ~4-6x (4x data growth + overhead)
        # O(n*m) naive: ratio would be ~16x
        # Generous bound of 12x covers CPU contention under xdist
        assert ratio < 12, (
            f"Scaling ratio {ratio:.1f}x suggests non-linear behavior "
            f"(small={t_small*1000:.1f}ms, large={t_large*1000:.1f}ms)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
