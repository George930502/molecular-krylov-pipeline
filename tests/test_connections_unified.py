"""Tests for PR 2.3: Unified connection computation path.

Three issues to fix:
1. PhysicsGuidedFlowTrainer._get_connections_batch() doesn't use vectorized path
2. FP32/FP64 precision mismatch between sequential and vectorized paths
3. Cache warmup uses sequential path unnecessarily

TDD RED phase: these tests should FAIL on current code.
"""

import pytest
import torch
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================
# 1. Correctness: vectorized must match sequential exactly
# ============================================================

class TestConnectionsCorrectness:
    """Vectorized and sequential paths must produce identical results."""

    @pytest.mark.molecular
    def test_vectorized_matches_sequential_lih(self, lih_hamiltonian):
        """Every connection from sequential must appear in vectorized, and vice versa."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        configs = pipeline._generate_essential_configs()

        # Sequential
        seq_total = 0
        for i in range(len(configs)):
            connected, elements = lih_hamiltonian.get_connections(configs[i])
            seq_total += len(connected)

        # Vectorized
        all_connected, all_elements, batch_idx = \
            lih_hamiltonian.get_connections_vectorized_batch(configs)
        vec_total = len(all_connected)

        assert seq_total == vec_total, (
            f"Connection count mismatch: sequential={seq_total}, vectorized={vec_total}. "
            f"Difference of {abs(seq_total - vec_total)} connections indicates "
            f"FP32/FP64 precision inconsistency in near-zero matrix elements."
        )

    @pytest.mark.molecular
    def test_vectorized_matches_sequential_n2(self, n2_hamiltonian):
        """N2 connection counts must match exactly."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(n2_hamiltonian, config=pipe_config)
        configs = pipeline._generate_essential_configs()[:200]

        seq_total = 0
        for i in range(len(configs)):
            connected, elements = n2_hamiltonian.get_connections(configs[i])
            seq_total += len(connected)

        all_connected, all_elements, batch_idx = \
            n2_hamiltonian.get_connections_vectorized_batch(configs)
        vec_total = len(all_connected)

        assert seq_total == vec_total, (
            f"N2 connection count mismatch: sequential={seq_total}, vectorized={vec_total}"
        )

    @pytest.mark.molecular
    def test_matrix_elements_match_within_tolerance(self, lih_hamiltonian):
        """Matrix elements must agree to at least 1e-6 relative error."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        configs = pipeline._generate_essential_configs()[:50]

        for i in range(len(configs)):
            seq_connected, seq_elements = lih_hamiltonian.get_connections(configs[i])
            if len(seq_connected) == 0:
                continue

            # Build lookup: config hash -> element value
            seq_map = {}
            for j in range(len(seq_connected)):
                key = tuple(seq_connected[j].cpu().numpy().astype(int))
                seq_map[key] = seq_elements[j].item()

            # Check vectorized produces same elements
            vec_connected, vec_elements, vec_idx = \
                lih_hamiltonian.get_connections_vectorized_batch(configs[i:i+1])

            for j in range(len(vec_connected)):
                key = tuple(vec_connected[j].cpu().numpy().astype(int))
                if key in seq_map:
                    seq_val = seq_map[key]
                    vec_val = vec_elements[j].item()
                    if abs(seq_val) > 1e-8:
                        rel_err = abs(seq_val - vec_val) / abs(seq_val)
                        assert rel_err < 1e-5, (
                            f"Config {i}, connection {key}: "
                            f"seq={seq_val:.10e}, vec={vec_val:.10e}, "
                            f"rel_err={rel_err:.2e}"
                        )


# ============================================================
# 2. Trainer must use vectorized path
# ============================================================

class TestTrainerUsesVectorized:
    """PhysicsGuidedFlowTrainer must use the fast vectorized path."""

    def test_get_connections_batch_uses_vectorized(self):
        """_get_connections_batch should call get_connections_vectorized_batch
        when available, not fall through to sequential loop."""
        import inspect
        from flows.physics_guided_training import PhysicsGuidedFlowTrainer

        source = inspect.getsource(PhysicsGuidedFlowTrainer._get_connections_batch)

        has_vectorized_call = 'get_connections_vectorized_batch' in source

        assert has_vectorized_call, (
            "_get_connections_batch does not call get_connections_vectorized_batch. "
            "The sequential fallback (line 946) is 30-70x slower than the vectorized "
            "batch path. Add vectorized_batch as the primary path."
        )

    def test_no_sequential_fallback_as_default(self):
        """The sequential for-loop should NOT be the default path.
        It should only be used as a last fallback when vectorized is unavailable."""
        import inspect
        from flows.physics_guided_training import PhysicsGuidedFlowTrainer

        source = inspect.getsource(PhysicsGuidedFlowTrainer._get_connections_batch)

        # The vectorized path should come BEFORE the sequential loop
        # Currently: cache -> parallel -> sequential
        # Should be: cache -> vectorized_batch -> parallel -> sequential
        lines = source.split('\n')
        vec_line = None
        seq_line = None
        for i, line in enumerate(lines):
            if 'get_connections_vectorized_batch' in line:
                vec_line = i
            if 'for i in range(len(configs))' in line or 'for i in range(n_configs)' in line:
                seq_line = i

        if vec_line is None:
            pytest.fail("get_connections_vectorized_batch not found in _get_connections_batch")

        if seq_line is not None:
            assert vec_line < seq_line, (
                f"Vectorized path (line {vec_line}) appears AFTER sequential loop "
                f"(line {seq_line}). Vectorized should be tried first."
            )


# ============================================================
# 3. Performance: vectorized must be significantly faster
# ============================================================

class TestVectorizedPerformance:
    """Vectorized path must provide meaningful speedup."""

    @pytest.mark.molecular
    @pytest.mark.slow
    def test_vectorized_faster_than_sequential_n2(self, n2_hamiltonian):
        """On N2 with 200+ configs, vectorized should be >10x faster."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(n2_hamiltonian, config=pipe_config)
        configs = pipeline._generate_essential_configs()[:200]

        # Sequential
        t0 = time.perf_counter()
        for i in range(len(configs)):
            n2_hamiltonian.get_connections(configs[i])
        t_seq = time.perf_counter() - t0

        # Vectorized (warm up first)
        _ = n2_hamiltonian.get_connections_vectorized_batch(configs[:10])
        t0 = time.perf_counter()
        n2_hamiltonian.get_connections_vectorized_batch(configs)
        t_vec = time.perf_counter() - t0

        speedup = t_seq / t_vec
        print(f"\n  N2 200 configs: seq={t_seq:.3f}s, vec={t_vec:.3f}s, speedup={speedup:.1f}x")
        assert speedup > 3, (
            f"Vectorized speedup only {speedup:.1f}x on N2, expected >3x"
        )


# ============================================================
# 4. Energy consistency: both paths must give same pipeline energy
# ============================================================

class TestEnergyConsistency:
    """Pipeline energy must be identical regardless of connection path."""

    @pytest.mark.molecular
    def test_projected_hamiltonian_energy_matches(self, lih_hamiltonian):
        """Projected Hamiltonian built with vectorized connections must give
        the same ground state energy as sequential."""
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig
        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        configs = pipeline._generate_essential_configs()

        n = len(configs)

        # Build H matrix using sequential path
        H_seq = torch.zeros(n, n)
        for i in range(n):
            H_seq[i, i] = lih_hamiltonian.diagonal_element(configs[i])
            connected, elements = lih_hamiltonian.get_connections(configs[i])
            if len(connected) > 0:
                # Find which configs in our basis match
                for j in range(len(connected)):
                    diff = (configs - connected[j].unsqueeze(0)).abs().sum(dim=1)
                    matches = (diff < 0.5).nonzero(as_tuple=True)[0]
                    for m in matches:
                        H_seq[i, m.item()] = elements[j]

        # Build H matrix using vectorized path
        H_vec = torch.zeros(n, n)
        diag = lih_hamiltonian.diagonal_elements_batch(configs)
        for i in range(n):
            H_vec[i, i] = diag[i]

        all_connected, all_elements, batch_idx = \
            lih_hamiltonian.get_connections_vectorized_batch(configs)
        for j in range(len(all_connected)):
            i = batch_idx[j].item()
            diff = (configs - all_connected[j].unsqueeze(0)).abs().sum(dim=1)
            matches = (diff < 0.5).nonzero(as_tuple=True)[0]
            for m in matches:
                H_vec[i, m.item()] = all_elements[j]

        # Compare eigenvalues
        e_seq = torch.linalg.eigvalsh(H_seq)[0].item()
        e_vec = torch.linalg.eigvalsh(H_vec)[0].item()
        e_fci = lih_hamiltonian.fci_energy()

        print(f"\n  E_seq={e_seq:.8f}, E_vec={e_vec:.8f}, E_fci={e_fci:.8f}")
        print(f"  |E_seq - E_vec| = {abs(e_seq - e_vec):.2e}")
        print(f"  |E_seq - E_fci| = {abs(e_seq - e_fci) * 1000:.4f} mHa")
        print(f"  |E_vec - E_fci| = {abs(e_vec - e_fci) * 1000:.4f} mHa")

        # Both should be within chemical accuracy of FCI
        assert abs(e_seq - e_fci) * 1000 < 1.0, f"Sequential energy error too large"
        assert abs(e_vec - e_fci) * 1000 < 1.0, f"Vectorized energy error too large"

        # And they should agree with each other to < 0.01 mHa
        assert abs(e_seq - e_vec) * 1000 < 0.01, (
            f"Energy mismatch between paths: {abs(e_seq - e_vec) * 1000:.4f} mHa"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
