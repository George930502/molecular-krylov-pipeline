#!/usr/bin/env python
"""N2/cc-pVDZ Benchmark across CAS Active Spaces.

Runs the Flow-Guided Krylov Pipeline on N2 with cc-pVDZ basis set
at increasing active space sizes. This is the minimum acceptable
benchmark for 2026 publication standard.

CAS sizes tested:
- CAS(6,6):   12 qubits,    400 configs  (exact FCI tractable)
- CAS(10,8):  16 qubits,  3,136 configs  (small enough for exact FCI)
- CAS(10,10): 20 qubits, 63,504 configs  (sparse SKQD + NF-guided mode)

Reference energies:
  The Hamiltonian is built from CASSCF-optimized orbitals via
  create_n2_cas_hamiltonian(). The exact ground state of this Hamiltonian
  is obtained by H.fci_energy() (full diagonalization in the active space).
  Pipeline error = |pipeline_energy - H.fci_energy()|.

  PySCF CASCI (non-orbital-optimized) is also computed for cross-reference
  but is NOT used as the primary benchmark target because the Hamiltonian's
  integrals come from CASSCF orbitals, not RHF canonical orbitals.

IBM comparison context:
  IBM's SQD benchmark on N2/cc-pVDZ used 58 circuit qubits (routing overhead).
  Their Ext-SQD (Science, March 2026) improved agreement with HCI.
  Our pipeline uses classical sampling (no quantum hardware) with SKQD.
  The active spaces here are CAS subsets -- not full cc-pVDZ valence space.

Usage:
    python examples/ccpvdz_benchmark.py
    python examples/ccpvdz_benchmark.py --nnci         # Include NNCI expansion
    python examples/ccpvdz_benchmark.py --json out.json # Save detailed results
    python examples/ccpvdz_benchmark.py --cas 6,6       # Run single CAS size
"""

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from math import comb
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

try:
    from pyscf import gto, scf, mcscf

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

from hamiltonians.molecular import create_n2_cas_hamiltonian
from pipeline import FlowGuidedKrylovPipeline, PipelineConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHEMICAL_ACCURACY_MHA = 1.6  # 1 kcal/mol ~ 1.6 mHa
N2_BOND_LENGTH = 1.10  # Angstroms (equilibrium)

# CAS specifications: (nelecas, ncas)
CAS_SPECS = [
    (6, 6),
    (10, 8),
    (10, 10),
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class CASBenchmarkResult:
    """Result of a single CAS benchmark run."""

    cas: str  # e.g. "(6,6)"
    nelecas: int
    ncas: int
    n_qubits: int
    n_configs: int

    # Energies
    fci_energy: float  # Exact FCI of the CASSCF Hamiltonian (Ha)
    pipeline_energy: float  # Pipeline result (Ha)
    error_mha: float  # |pipeline - fci| * 1000

    # Performance
    wall_time_s: float
    peak_memory_mb: float

    # Pipeline details
    basis_size: int
    n_krylov_steps: int
    mode: str  # "Direct-CI SKQD" or "Direct-CI SKQD + NNCI"
    converged: bool

    # Status
    passed: bool  # error < CHEMICAL_ACCURACY_MHA

    # Extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------


def get_peak_memory_mb() -> float:
    """Read VmPeak from /proc/self/status (Linux only). Returns MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmPeak:"):
                    # Format: "VmPeak:   12345 kB"
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def get_current_rss_mb() -> float:
    """Read VmRSS from /proc/self/status (Linux only). Returns MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# CASCI reference energy (PySCF, for cross-reference only)
# ---------------------------------------------------------------------------


def compute_casci_reference(
    nelecas: int, ncas: int, bond_length: float = N2_BOND_LENGTH
) -> float:
    """Compute CASCI energy for N2/cc-pVDZ using PySCF directly.

    CASCI uses RHF canonical orbitals (no orbital optimization).
    This differs from CASSCF, which optimizes active-space orbitals.
    Used for cross-reference only -- not as the primary benchmark target.

    Args:
        nelecas: Number of active electrons.
        ncas: Number of active orbitals.
        bond_length: N-N distance in Angstroms.

    Returns:
        CASCI total energy in Hartree.
    """
    if not PYSCF_AVAILABLE:
        raise RuntimeError("PySCF required for CASCI reference computation")

    mol = gto.Mole()
    mol.atom = [
        ("N", (0.0, 0.0, 0.0)),
        ("N", (0.0, 0.0, bond_length)),
    ]
    mol.basis = "cc-pvdz"
    mol.symmetry = True
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver.conv_tol = 1e-10
    mc.kernel()

    return float(mc.e_tot)


# ---------------------------------------------------------------------------
# Single CAS benchmark
# ---------------------------------------------------------------------------


def run_cas_benchmark(
    nelecas: int,
    ncas: int,
    use_nnci: bool = False,
    verbose: bool = True,
) -> CASBenchmarkResult:
    """Run pipeline benchmark for a single CAS size.

    Args:
        nelecas: Number of active electrons.
        ncas: Number of active orbitals.
        use_nnci: Whether to enable NNCI expansion.
        verbose: Print progress.

    Returns:
        CASBenchmarkResult with all metrics.
    """
    cas_label = f"({nelecas},{ncas})"
    n_qubits = 2 * ncas
    n_alpha = nelecas // 2
    n_beta = nelecas // 2
    n_configs = comb(ncas, n_alpha) * comb(ncas, n_beta)

    print(f"\n{'=' * 70}")
    print(f"CAS{cas_label}  |  {n_qubits} qubits  |  {n_configs:,} configs")
    print(f"{'=' * 70}")

    # --- Step 1: Create Hamiltonian via factory (CASSCF orbitals) ---
    print("Creating CAS Hamiltonian (CASSCF orbital optimization)...")
    t_ham = time.perf_counter()
    H = create_n2_cas_hamiltonian(
        bond_length=N2_BOND_LENGTH,
        basis="cc-pvdz",
        cas=(nelecas, ncas),
        device="cpu",
    )
    t_ham = time.perf_counter() - t_ham
    print(f"  Hamiltonian created: {H.num_sites} sites, "
          f"{H.n_alpha}a+{H.n_beta}b electrons  ({t_ham:.1f}s)")

    # --- Step 2: Compute exact FCI of this Hamiltonian ---
    # This IS the ground truth for the pipeline benchmark:
    # the exact ground state of the CASSCF-orbital Hamiltonian.
    print("Computing exact FCI energy of the CASSCF Hamiltonian...")
    t_fci = time.perf_counter()
    exact_fci = H.fci_energy()
    t_fci = time.perf_counter() - t_fci
    print(f"  FCI energy: {exact_fci:.8f} Ha  ({t_fci:.1f}s)")

    # --- Step 3: PySCF CASCI for cross-reference ---
    casci_energy = None
    try:
        print("Computing PySCF CASCI reference (cross-reference)...")
        t_ref = time.perf_counter()
        casci_energy = compute_casci_reference(nelecas, ncas)
        t_ref = time.perf_counter() - t_ref
        casscf_vs_casci = (exact_fci - casci_energy) * 1000
        print(f"  CASCI energy: {casci_energy:.8f} Ha  ({t_ref:.1f}s)")
        print(f"  CASSCF vs CASCI: {casscf_vs_casci:.2f} mHa "
              "(CASSCF lower due to orbital optimization)")
    except Exception as e:
        print(f"  CASCI reference failed (non-critical): {e}")
        t_ref = 0.0

    # --- Step 4: Configure and run pipeline ---
    mode_label = "Direct-CI SKQD"
    if use_nnci:
        mode_label += " + NNCI"

    print(f"\nRunning pipeline: {mode_label}...")
    config = PipelineConfig(
        subspace_mode="skqd",
        skip_nf_training=True,
        use_nnci=use_nnci,
        device="cpu",
    )

    rss_before = get_current_rss_mb()
    t_start = time.perf_counter()

    pipeline = FlowGuidedKrylovPipeline(
        H, config=config, exact_energy=exact_fci
    )
    results = pipeline.run(progress=verbose)

    wall_time = time.perf_counter() - t_start
    rss_after = get_current_rss_mb()
    peak_mem = get_peak_memory_mb()

    # --- Step 5: Extract results ---
    pipeline_energy = results.get(
        "combined_energy",
        results.get("skqd_energy", results.get("nf_nqs_energy", float("inf"))),
    )
    error_mha = abs(pipeline_energy - exact_fci) * 1000
    passed = error_mha < CHEMICAL_ACCURACY_MHA

    # Extract SKQD-specific details
    skqd_results = results.get("skqd_results", {})
    n_krylov_steps = len(skqd_results.get("krylov_dims", []))
    basis_sizes = skqd_results.get("basis_sizes_combined", [])
    final_basis_size = basis_sizes[-1] if basis_sizes else results.get("nf_basis_size", 0)
    converged = skqd_results.get("converged", False)

    # --- Step 6: Print result ---
    status = "PASS" if passed else "FAIL"
    print(f"\n{'=' * 70}")
    print(f"RESULT: CAS{cas_label}")
    print(f"{'=' * 70}")
    print(f"  FCI Reference:   {exact_fci:.8f} Ha  (exact for CASSCF Hamiltonian)")
    print(f"  Pipeline Energy: {pipeline_energy:.8f} Ha")
    print(f"  Error:           {error_mha:.4f} mHa")
    print(f"  Wall Time:       {wall_time:.1f}s")
    print(f"  Peak RSS:        {rss_after:.0f} MB (delta: {rss_after - rss_before:+.0f} MB)")
    print(f"  Basis Size:      {final_basis_size}")
    print(f"  Krylov Steps:    {n_krylov_steps}")
    print(f"  Converged:       {converged}")
    print(f"  Status:          {status}")
    print(f"{'=' * 70}")

    return CASBenchmarkResult(
        cas=cas_label,
        nelecas=nelecas,
        ncas=ncas,
        n_qubits=n_qubits,
        n_configs=n_configs,
        fci_energy=exact_fci,
        pipeline_energy=pipeline_energy,
        error_mha=error_mha,
        wall_time_s=wall_time,
        peak_memory_mb=peak_mem,
        basis_size=final_basis_size,
        n_krylov_steps=n_krylov_steps,
        mode=mode_label,
        converged=converged,
        passed=passed,
        metadata={
            "bond_length": N2_BOND_LENGTH,
            "basis": "cc-pvdz",
            "casci_energy": casci_energy,
            "hamiltonian_time_s": t_ham,
            "fci_time_s": t_fci,
            "casci_time_s": t_ref,
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "nnci_configs_added": results.get("nnci_configs_added", 0),
            "nf_basis_size": results.get("nf_basis_size", 0),
            "energy_history": skqd_results.get("energy_history", []),
        },
    )


# ---------------------------------------------------------------------------
# Markdown table output
# ---------------------------------------------------------------------------


def print_markdown_table(results: List[CASBenchmarkResult], use_nnci: bool = False):
    """Print results as a markdown table to stdout."""
    mode_str = "Direct-CI SKQD"
    if use_nnci:
        mode_str += " + NNCI"

    print(f"\n## N2/cc-pVDZ Benchmark Results ({mode_str})")
    print()
    print(
        "| CAS | Qubits | Configs | Energy (Ha) "
        "| Error (mHa) | Time (s) | Basis Size | Krylov Steps | Status |"
    )
    print(
        "|-----|--------|---------|-------------"
        "|-------------|----------|------------|--------------|--------|"
    )

    for r in results:
        print(
            f"| {r.cas} "
            f"| {r.n_qubits} "
            f"| {r.n_configs:,} "
            f"| {r.pipeline_energy:.6f} "
            f"| {r.error_mha:.3f} "
            f"| {r.wall_time_s:.1f} "
            f"| {r.basis_size:,} "
            f"| {r.n_krylov_steps} "
            f"| {'PASS' if r.passed else 'FAIL'} |"
        )

    n_pass = sum(1 for r in results if r.passed)
    print()
    print(
        f"Chemical accuracy threshold: {CHEMICAL_ACCURACY_MHA} mHa "
        f"(1 kcal/mol). Result: {n_pass}/{len(results)} PASS."
    )


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def save_json_results(
    results: List[CASBenchmarkResult], path: str, use_nnci: bool = False
):
    """Save detailed results to a JSON file."""
    output = {
        "benchmark": "N2/cc-pVDZ CAS Active Space",
        "molecule": "N2",
        "basis": "cc-pvdz",
        "bond_length_angstrom": N2_BOND_LENGTH,
        "mode": "Direct-CI SKQD" + (" + NNCI" if use_nnci else ""),
        "chemical_accuracy_mha": CHEMICAL_ACCURACY_MHA,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": "cpu",
        "results": [asdict(r) for r in results],
        "summary": {
            "n_pass": sum(1 for r in results if r.passed),
            "n_total": len(results),
            "all_pass": all(r.passed for r in results),
        },
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_cas_arg(cas_str: str) -> List[tuple]:
    """Parse --cas argument like '6,6' or '10,8' into [(nelecas, ncas)]."""
    specs = []
    for part in cas_str.split(";"):
        part = part.strip()
        nelecas, ncas = part.split(",")
        specs.append((int(nelecas.strip()), int(ncas.strip())))
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="N2/cc-pVDZ Benchmark across CAS Active Spaces"
    )
    parser.add_argument(
        "--nnci",
        action="store_true",
        help="Enable NNCI expansion for basis discovery",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Save detailed results to JSON file",
    )
    parser.add_argument(
        "--cas",
        type=str,
        default=None,
        metavar="NELEC,NORB",
        help="Run specific CAS size(s), e.g. '6,6' or '6,6;10,8'",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce pipeline output verbosity",
    )
    args = parser.parse_args()

    if not PYSCF_AVAILABLE:
        print("ERROR: PySCF is required for this benchmark.")
        print("Install with: pip install pyscf")
        sys.exit(1)

    # Determine which CAS sizes to run
    if args.cas:
        cas_list = parse_cas_arg(args.cas)
    else:
        cas_list = CAS_SPECS

    print("=" * 70)
    print("N2/cc-pVDZ Benchmark: CAS Active Space Ladder")
    print("=" * 70)
    print(f"Molecule:   N2 (bond length = {N2_BOND_LENGTH} A)")
    print(f"Basis set:  cc-pVDZ")
    print(f"Mode:       Direct-CI SKQD{' + NNCI' if args.nnci else ''}")
    print(f"Device:     cpu")
    print(f"CAS sizes:  {', '.join(f'({n},{c})' for n, c in cas_list)}")
    print(f"Reference:  H.fci_energy() (exact for CASSCF Hamiltonian)")
    print("=" * 70)

    all_results: List[CASBenchmarkResult] = []

    for nelecas, ncas in cas_list:
        try:
            result = run_cas_benchmark(
                nelecas=nelecas,
                ncas=ncas,
                use_nnci=args.nnci,
                verbose=not args.quiet,
            )
            all_results.append(result)

            # Print intermediate table after each CAS completes
            print_markdown_table(all_results, use_nnci=args.nnci)

        except Exception as e:
            cas_label = f"({nelecas},{ncas})"
            print(f"\nERROR running CAS{cas_label}: {e}")
            traceback.print_exc()
            # Record failure
            n_qubits = 2 * ncas
            n_configs = comb(ncas, nelecas // 2) * comb(ncas, nelecas // 2)
            all_results.append(
                CASBenchmarkResult(
                    cas=cas_label,
                    nelecas=nelecas,
                    ncas=ncas,
                    n_qubits=n_qubits,
                    n_configs=n_configs,
                    fci_energy=0.0,
                    pipeline_energy=float("inf"),
                    error_mha=float("inf"),
                    wall_time_s=0.0,
                    peak_memory_mb=0.0,
                    basis_size=0,
                    n_krylov_steps=0,
                    mode="FAILED",
                    converged=False,
                    passed=False,
                    metadata={"error": str(e)},
                )
            )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print_markdown_table(all_results, use_nnci=args.nnci)

    n_pass = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)
    print(f"\nOverall: {n_pass}/{n_total} PASS")

    if all(r.passed for r in all_results):
        print("All CAS sizes achieved chemical accuracy.")
    else:
        failed = [r.cas for r in all_results if not r.passed]
        print(f"Failed CAS sizes: {', '.join(failed)}")

    # Save JSON if requested
    if args.json:
        save_json_results(all_results, args.json, use_nnci=args.nnci)

    sys.exit(0 if all(r.passed for r in all_results) else 1)


if __name__ == "__main__":
    main()
