"""Tests for PR 1.5: SQD batch parallelization.

The SQD algorithm runs K independent batch diagonalizations per self-consistent
iteration. These are embarrassingly parallel — each batch builds its own projected
Hamiltonian and solves an independent eigenvalue problem.

On DGX Spark with 20 ARM cores, parallelizing 5-10 batches should give 5-8x speedup.
"""

import pytest
import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSQDParallelBatches:
    """Test that parallel batch diag produces correct results."""

    @pytest.mark.molecular
    def test_parallel_matches_sequential(self, lih_hamiltonian):
        """Parallel batch results must match sequential within floating-point tolerance."""
        from krylov.sqd import SQDSolver, SQDConfig

        config = SQDConfig(
            num_batches=5,
            self_consistent_iters=1,
            noise_rate=0.0,
            max_workers=1,  # sequential
        )
        solver_seq = SQDSolver(lih_hamiltonian, config=config)

        config_par = SQDConfig(
            num_batches=5,
            self_consistent_iters=1,
            noise_rate=0.0,
            max_workers=4,  # parallel
        )
        solver_par = SQDSolver(lih_hamiltonian, config=config_par)

        # Generate a basis
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        results_seq = solver_seq.run(basis, progress=False)
        results_par = solver_par.run(basis, progress=False)

        # Energies should match (same batches, same seeds)
        e_seq = results_seq["energy"]
        e_par = results_par["energy"]
        assert abs(e_seq - e_par) < 1e-8, (
            f"Parallel energy {e_par:.10f} != sequential {e_seq:.10f}"
        )

    @pytest.mark.molecular
    def test_parallel_config_exists(self, h2_hamiltonian):
        """SQDConfig should accept max_workers parameter."""
        from krylov.sqd import SQDConfig

        config = SQDConfig(max_workers=4)
        assert config.max_workers == 4

    @pytest.mark.molecular
    def test_parallel_default_is_one(self):
        """Default max_workers should be 1 (sequential, backward compatible)."""
        from krylov.sqd import SQDConfig

        config = SQDConfig()
        assert config.max_workers == 1

    @pytest.mark.molecular
    def test_parallel_energy_accuracy(self, lih_hamiltonian):
        """Parallel SQD must still achieve chemical accuracy on LiH."""
        from krylov.sqd import SQDSolver, SQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        e_fci = lih_hamiltonian.fci_energy()

        config = SQDConfig(
            num_batches=5,
            self_consistent_iters=2,
            noise_rate=0.0,
            max_workers=4,
        )
        solver = SQDSolver(lih_hamiltonian, config=config)

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        results = solver.run(basis, progress=False)
        error_mha = abs(results["energy"] - e_fci) * 1000
        assert error_mha < 5.0, f"Parallel SQD error {error_mha:.4f} mHa too large"


class TestSQDParallelSpeedup:
    """Verify parallel batches provide actual speedup."""

    @pytest.mark.molecular
    @pytest.mark.slow
    def test_parallel_faster_than_sequential(self, n2_hamiltonian):
        """With workers on large distinct batches, parallel should be faster.

        Key requirements for meaningful parallel speedup:
        1. batch_size < n_configs so batches are distinct (not identical copies)
        2. batch_size > 3000 so H build cost justifies thread overhead
        3. Enough batches to utilize multiple workers

        N2 essential configs = ~610 (HF + singles + doubles), but the full
        configuration space is 14400. We generate a large synthetic basis
        to reach the threshold for parallel benefit.
        """
        from krylov.sqd import SQDSolver, SQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(n2_hamiltonian, config=pipe_config)
        essential = pipeline._generate_essential_configs()

        # Expand basis with random valid configs to exceed 3000 threshold
        n_orb = n2_hamiltonian.num_sites // 2
        n_alpha = n2_hamiltonian.n_alpha
        n_beta = n2_hamiltonian.n_beta
        extra_configs = []
        for _ in range(6000):
            alpha = torch.zeros(n_orb)
            alpha[torch.randperm(n_orb)[:n_alpha]] = 1.0
            beta = torch.zeros(n_orb)
            beta[torch.randperm(n_orb)[:n_beta]] = 1.0
            extra_configs.append(torch.cat([alpha, beta]))
        extra = torch.stack(extra_configs)
        basis = torch.unique(torch.cat([essential, extra], dim=0), dim=0)
        print(f"\n  N2 expanded basis size: {len(basis)}")

        # Sequential: batch_size=4000 < basis → distinct batches
        config_seq = SQDConfig(
            num_batches=6, batch_size=4000,
            self_consistent_iters=1, max_workers=1,
        )
        solver_seq = SQDSolver(n2_hamiltonian, config=config_seq)
        t0 = time.perf_counter()
        solver_seq.run(basis, progress=False)
        t_seq = time.perf_counter() - t0

        # Parallel: same config but with workers
        config_par = SQDConfig(
            num_batches=6, batch_size=4000,
            self_consistent_iters=1, max_workers=6,
        )
        solver_par = SQDSolver(n2_hamiltonian, config=config_par)
        t0 = time.perf_counter()
        solver_par.run(basis, progress=False)
        t_par = time.perf_counter() - t0

        speedup = t_seq / t_par
        print(f"  N2 SQD 6 batches: seq={t_seq:.2f}s, par={t_par:.2f}s, speedup={speedup:.1f}x")

        # With distinct 4000-config batches and 6 workers,
        # expect at least modest speedup (> 1.0 means not slower)
        assert speedup > 1.0, f"Parallel speedup only {speedup:.1f}x, should not be slower"

    @pytest.mark.molecular
    def test_small_system_skips_parallel(self, beh2_hamiltonian):
        """Small identical batches should auto-skip parallel (use sequential path)."""
        from krylov.sqd import SQDSolver, SQDConfig

        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(beh2_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        # max_workers=8 but BeH2 (1225 configs) should auto-fallback to sequential
        config = SQDConfig(
            num_batches=8, self_consistent_iters=1, max_workers=8,
        )
        solver = SQDSolver(beh2_hamiltonian, config=config)
        results = solver.run(basis, progress=False)

        # Should still produce correct energy (sequential path)
        e_fci = beh2_hamiltonian.fci_energy()
        error_mha = abs(results["energy"] - e_fci) * 1000
        assert error_mha < 5.0, f"Auto-sequential SQD error {error_mha:.4f} mHa"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
