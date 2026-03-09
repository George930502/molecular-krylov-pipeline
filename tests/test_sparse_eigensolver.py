"""Tests for PR 1.1: Sparse eigensolver path.

TDD RED phase — these tests define the target behavior for:
1. MAX_FULL_SUBSPACE_SIZE guard in standard SKQD
2. Sparse matrix construction from get_sparse_matrix_elements()
3. Sparse eigsh matching dense eigh results
4. Krylov expansion using sparse H
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from math import comb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
from hamiltonians.molecular import MolecularHamiltonian


# =============================================================================
# B13: MAX_FULL_SUBSPACE_SIZE guard in standard SKQD
# =============================================================================


class TestMaxFullSubspaceGuard:
    """Standard SKQD must have MAX_FULL_SUBSPACE_SIZE guard like FlowGuidedSKQD."""

    def test_guard_exists(self):
        """SampleBasedKrylovDiagonalization should have MAX_FULL_SUBSPACE_SIZE."""
        assert hasattr(SampleBasedKrylovDiagonalization, 'MAX_FULL_SUBSPACE_SIZE'), \
            "Standard SKQD missing MAX_FULL_SUBSPACE_SIZE guard (B13)"

    def test_guard_value(self):
        """Guard should be 15000 (matching FlowGuidedSKQD). Dense complex128: 15000²×16=3.6GB."""
        assert SampleBasedKrylovDiagonalization.MAX_FULL_SUBSPACE_SIZE == 15000

    @pytest.mark.molecular
    def test_guard_prevents_oom(self, lih_hamiltonian):
        """For systems within limit, subspace should be set up normally."""
        config = SKQDConfig(max_krylov_dim=2)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)
        # LiH has 225 configs — well within guard
        assert hasattr(skqd, '_subspace_basis')
        assert len(skqd._subspace_basis) == 225


# =============================================================================
# Sparse matrix construction
# =============================================================================


class TestSparseMatrixConstruction:
    """Test that get_sparse_matrix_elements produces correct sparse matrices."""

    @pytest.mark.molecular
    def test_sparse_method_exists(self, h2_hamiltonian):
        """MolecularHamiltonian should have get_sparse_matrix_elements()."""
        assert hasattr(h2_hamiltonian, 'get_sparse_matrix_elements'), \
            "MolecularHamiltonian missing get_sparse_matrix_elements()"

    @pytest.mark.molecular
    def test_sparse_matches_dense_h2(self, h2_hamiltonian):
        """Sparse COO from get_sparse_matrix_elements should match dense for H2."""
        from scipy.sparse import coo_matrix

        H = h2_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], dtype=torch.long)
        n = len(configs)

        # Dense reference (includes diagonal)
        H_dense = H.matrix_elements(configs, configs).cpu().numpy().astype(np.float64)
        H_dense = 0.5 * (H_dense + H_dense.T)

        # Sparse COO (off-diagonal only)
        rows, cols, vals = H.get_sparse_matrix_elements(configs)
        rows = rows.cpu().numpy()
        cols = cols.cpu().numpy()
        vals = vals.cpu().numpy().astype(np.float64)

        # Build sparse matrix and add diagonal
        H_sparse = coo_matrix((vals, (rows, cols)), shape=(n, n)).toarray()
        for i in range(n):
            H_sparse[i, i] = H.diagonal_element(configs[i]).item()
        H_sparse = 0.5 * (H_sparse + H_sparse.T)

        np.testing.assert_allclose(
            H_sparse, H_dense, atol=1e-10,
            err_msg="Sparse matrix doesn't match dense matrix for H2"
        )

    @pytest.mark.molecular
    def test_sparse_matches_dense_lih(self, lih_hamiltonian):
        """Sparse COO should match dense for LiH subset."""
        from scipy.sparse import coo_matrix

        H = lih_hamiltonian
        hf = H.get_hf_state()
        configs = [hf]
        n_orb = H.n_orbitals

        # Add single excitations
        for i in range(H.n_alpha):
            for a in range(H.n_alpha, n_orb):
                config = hf.clone()
                config[i] = 0
                config[a] = 1
                configs.append(config)
                if len(configs) >= 20:
                    break
            if len(configs) >= 20:
                break

        configs = torch.stack(configs)
        n = len(configs)

        H_dense = H.matrix_elements(configs, configs).cpu().numpy().astype(np.float64)
        H_dense = 0.5 * (H_dense + H_dense.T)

        rows, cols, vals = H.get_sparse_matrix_elements(configs)
        rows = rows.cpu().numpy()
        cols = cols.cpu().numpy()
        vals = vals.cpu().numpy().astype(np.float64)

        H_sparse = coo_matrix((vals, (rows, cols)), shape=(n, n)).toarray()
        for i in range(n):
            H_sparse[i, i] = H.diagonal_element(configs[i]).item()
        H_sparse = 0.5 * (H_sparse + H_sparse.T)

        # Tolerance relaxed: get_sparse_matrix_elements returns float32,
        # matrix_elements returns float64. ~1e-6 relative error expected.
        np.testing.assert_allclose(
            H_sparse, H_dense, atol=1e-5, rtol=1e-5,
            err_msg="Sparse matrix doesn't match dense for LiH subset"
        )


# =============================================================================
# Sparse eigensolver
# =============================================================================


class TestSparseEigensolver:
    """Test that sparse eigsh produces same ground state as dense eigh."""

    @pytest.mark.molecular
    def test_sparse_eigsh_h2(self, h2_hamiltonian):
        """Sparse eigsh should match dense eigh for H2."""
        from scipy.sparse.linalg import eigsh
        from scipy.linalg import eigh

        H = h2_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], dtype=torch.long)

        H_dense = H.matrix_elements(configs, configs).cpu().numpy().astype(np.float64)
        H_dense = 0.5 * (H_dense + H_dense.T)

        # Dense eigensolver
        evals_dense, _ = eigh(H_dense)

        # Sparse eigensolver
        from scipy.sparse import csr_matrix
        H_csr = csr_matrix(H_dense)
        evals_sparse, _ = eigsh(H_csr, k=1, which='SA')

        np.testing.assert_allclose(
            evals_sparse[0], evals_dense[0], atol=1e-10,
            err_msg="Sparse eigsh doesn't match dense eigh"
        )

    @pytest.mark.molecular
    def test_pipeline_uses_sparse_for_large_basis(self, lih_hamiltonian):
        """Pipeline should use sparse path for basis > dense_threshold."""
        # This test verifies the integration — that the pipeline's eigensolver
        # path can handle sparse matrices when the basis is large enough.
        # For now, we verify the pipeline runs correctly with LiH.
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        e_fci = lih_hamiltonian.fci_energy()
        config = PipelineConfig(
            subspace_mode="skqd",
            skip_nf_training=True,
            device="cpu",
        )
        pipeline = FlowGuidedKrylovPipeline(
            lih_hamiltonian, config=config, exact_energy=e_fci
        )
        results = pipeline.run(progress=False)
        best = results.get("combined_energy", results.get("skqd_energy"))
        error_mha = abs(best - e_fci) * 1000
        assert error_mha < 0.5, f"LiH error {error_mha:.4f} mHa with sparse path"


class TestSparseHamiltonianEigsh:
    """Test the standalone sparse_hamiltonian_eigsh utility function."""

    @pytest.mark.molecular
    def test_sparse_hamiltonian_eigsh_h2(self, h2_hamiltonian):
        """sparse_hamiltonian_eigsh should match FCI for H2 (full space)."""
        from utils.gpu_linalg import sparse_hamiltonian_eigsh

        H = h2_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0], [1, 0, 0, 1],
            [0, 1, 1, 0], [0, 1, 0, 1],
        ], dtype=torch.long)
        evals, evecs = sparse_hamiltonian_eigsh(H, configs, k=2)
        e_fci = H.fci_energy()
        assert abs(float(evals[0]) - e_fci) < 1e-8, (
            f"sparse_hamiltonian_eigsh {float(evals[0]):.10f} != FCI {e_fci:.10f}"
        )

    @pytest.mark.molecular
    def test_sparse_hamiltonian_eigsh_lih(self, lih_hamiltonian):
        """sparse_hamiltonian_eigsh should match FCI for LiH (full space)."""
        from utils.gpu_linalg import sparse_hamiltonian_eigsh
        from itertools import combinations

        H = lih_hamiltonian
        n_orb = H.n_orbitals
        configs = []
        for ac in combinations(range(n_orb), H.n_alpha):
            for bc in combinations(range(n_orb), H.n_beta):
                c = torch.zeros(H.num_sites, dtype=torch.long)
                for i in ac: c[i] = 1
                for i in bc: c[i + n_orb] = 1
                configs.append(c)
        basis = torch.stack(configs)

        evals, evecs = sparse_hamiltonian_eigsh(H, basis, k=2)
        e_fci = H.fci_energy()
        assert abs(float(evals[0]) - e_fci) < 1e-8, (
            f"sparse_hamiltonian_eigsh {float(evals[0]):.10f} != FCI {e_fci:.10f}"
        )

    @pytest.mark.molecular
    def test_evecs_shape(self, h2_hamiltonian):
        """Eigenvectors should have correct shape."""
        from utils.gpu_linalg import sparse_hamiltonian_eigsh

        configs = torch.tensor([
            [1, 0, 1, 0], [1, 0, 0, 1],
            [0, 1, 1, 0], [0, 1, 0, 1],
        ], dtype=torch.long)
        evals, evecs = sparse_hamiltonian_eigsh(h2_hamiltonian, configs, k=2)
        assert evals.shape == (2,)
        assert evecs.shape == (4, 2)


class TestSparseGroundState:
    """Test _sparse_ground_state directly against dense eigensolver."""

    @pytest.mark.molecular
    def test_sparse_ground_state_matches_dense_lih(self, lih_hamiltonian):
        """_sparse_ground_state energy must match dense eigh on LiH."""
        from krylov.skqd import SampleBasedKrylovDiagonalization, SKQDConfig
        from pipeline import FlowGuidedKrylovPipeline, PipelineConfig

        pipe_config = PipelineConfig(skip_nf_training=True, device="cpu")
        pipeline = FlowGuidedKrylovPipeline(lih_hamiltonian, config=pipe_config)
        basis = pipeline._generate_essential_configs()

        config = SKQDConfig(max_diag_basis_size=15000)
        skqd = SampleBasedKrylovDiagonalization(lih_hamiltonian, config=config)

        # Dense path (ground truth)
        E_dense, _ = skqd.compute_ground_state_energy(basis)

        # Sparse path (directly called, bypassing threshold)
        E_sparse, _ = skqd._sparse_ground_state(basis)

        assert abs(E_dense - E_sparse) < 1e-6, (
            f"Sparse {E_sparse:.10f} != dense {E_dense:.10f}, "
            f"diff = {abs(E_dense - E_sparse):.2e}"
        )


class TestScipyExpmMultiply:
    """Test _expm_multiply_step dispatches correctly and produces accurate results."""

    @pytest.mark.molecular
    def test_expm_step_dense_matches_gpu(self, lih_hamiltonian):
        """_expm_multiply_step with dense torch tensor should match gpu_expm_multiply."""
        from krylov.skqd import FlowGuidedSKQD, SKQDConfig
        from utils.gpu_linalg import gpu_expm_multiply

        config = SKQDConfig()
        skqd = FlowGuidedSKQD.__new__(FlowGuidedSKQD)
        skqd.config = config
        skqd.time_step = 0.5

        H = lih_hamiltonian
        # Build a small dense H
        configs = torch.tensor([
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        ], dtype=torch.long)
        H_dense = H.matrix_elements(configs, configs).to(torch.complex128)
        H_dense = 0.5 * (H_dense + H_dense.conj().T)

        psi = torch.ones(4, dtype=torch.complex128) / 2.0

        result_gpu = gpu_expm_multiply(H_dense, psi, t=-1j * 0.5)
        result_step = skqd._expm_multiply_step(H_dense, psi, 0.5)

        np.testing.assert_allclose(
            result_step.cpu().numpy(), result_gpu.cpu().numpy(), atol=1e-12,
            err_msg="_expm_multiply_step dense path doesn't match gpu_expm_multiply"
        )

    @pytest.mark.molecular
    def test_expm_step_sparse_matches_dense(self, lih_hamiltonian):
        """_expm_multiply_step with scipy CSR should match dense path."""
        from krylov.skqd import FlowGuidedSKQD, SKQDConfig
        from scipy.sparse import csr_matrix
        from utils.gpu_linalg import gpu_expm_multiply

        config = SKQDConfig()
        skqd = FlowGuidedSKQD.__new__(FlowGuidedSKQD)
        skqd.config = config
        skqd.time_step = 0.5

        H = lih_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        ], dtype=torch.long)
        H_dense = H.matrix_elements(configs, configs).cpu().numpy().astype(np.float64)
        H_dense = 0.5 * (H_dense + H_dense.T)
        H_csr = csr_matrix(H_dense)
        H_torch = torch.from_numpy(H_dense).to(torch.complex128)

        psi = torch.ones(4, dtype=torch.complex128) / 2.0

        # Dense reference via gpu_expm_multiply
        result_dense = gpu_expm_multiply(H_torch, psi, t=-1j * 0.5)
        # Sparse via scipy expm_multiply
        result_sparse = skqd._expm_multiply_step(H_csr, psi, 0.5)

        np.testing.assert_allclose(
            result_sparse.cpu().numpy(), result_dense.cpu().numpy(), atol=1e-10,
            err_msg="scipy expm_multiply doesn't match dense gpu_expm_multiply"
        )

    @pytest.mark.molecular
    def test_build_hamiltonian_returns_csr_for_large_basis(self, lih_hamiltonian):
        """_build_hamiltonian_in_basis_gpu should return scipy CSR for n >= threshold."""
        from krylov.skqd import FlowGuidedSKQD, SKQDConfig
        from scipy.sparse import issparse
        from itertools import combinations

        H = lih_hamiltonian
        n_orb = H.n_orbitals

        # Build full LiH basis (225 configs)
        configs = []
        for ac in combinations(range(n_orb), H.n_alpha):
            for bc in combinations(range(n_orb), H.n_beta):
                c = torch.zeros(H.num_sites, dtype=torch.long)
                for i in ac:
                    c[i] = 1
                for i in bc:
                    c[i + n_orb] = 1
                configs.append(c)
        basis = torch.stack(configs)

        config = SKQDConfig()
        skqd = FlowGuidedSKQD.__new__(FlowGuidedSKQD)
        skqd.config = config
        skqd.hamiltonian = H

        # Temporarily lower threshold to force sparse path
        old_threshold = FlowGuidedSKQD.KRYLOV_SPARSE_THRESHOLD
        FlowGuidedSKQD.KRYLOV_SPARSE_THRESHOLD = 100
        try:
            H_result = skqd._build_hamiltonian_in_basis_gpu(basis)
            assert issparse(H_result), (
                f"Expected scipy sparse for n={len(basis)} >= threshold=100, "
                f"got {type(H_result)}"
            )
        finally:
            FlowGuidedSKQD.KRYLOV_SPARSE_THRESHOLD = old_threshold

    @pytest.mark.molecular
    def test_build_hamiltonian_returns_dense_for_small_basis(self, lih_hamiltonian):
        """_build_hamiltonian_in_basis_gpu should return dense torch for n < threshold."""
        from krylov.skqd import FlowGuidedSKQD, SKQDConfig

        H = lih_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        ], dtype=torch.long)

        config = SKQDConfig()
        skqd = FlowGuidedSKQD.__new__(FlowGuidedSKQD)
        skqd.config = config
        skqd.hamiltonian = H

        H_result = skqd._build_hamiltonian_in_basis_gpu(configs)
        assert isinstance(H_result, torch.Tensor), (
            f"Expected torch.Tensor for small basis, got {type(H_result)}"
        )
        assert not H_result.is_sparse, "Should be dense torch.Tensor"

    @pytest.mark.molecular
    def test_sparse_expm_preserves_norm(self, lih_hamiltonian):
        """Time evolution via scipy expm_multiply should preserve state norm."""
        from krylov.skqd import FlowGuidedSKQD, SKQDConfig
        from scipy.sparse import csr_matrix

        H = lih_hamiltonian
        configs = torch.tensor([
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        ], dtype=torch.long)
        H_dense = H.matrix_elements(configs, configs).cpu().numpy().astype(np.float64)
        H_dense = 0.5 * (H_dense + H_dense.T)
        H_csr = csr_matrix(H_dense)

        config = SKQDConfig()
        skqd = FlowGuidedSKQD.__new__(FlowGuidedSKQD)
        skqd.config = config
        skqd.time_step = 0.5

        psi = torch.ones(4, dtype=torch.complex128) / 2.0
        for _ in range(10):
            psi = skqd._expm_multiply_step(H_csr, psi, 0.5)

        norm = torch.linalg.norm(psi).item()
        assert abs(norm - 1.0) < 1e-10, (
            f"Norm drifted to {norm} after 10 time steps (should be ~1.0)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
