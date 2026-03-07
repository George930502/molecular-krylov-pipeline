"""
Persistent disk cache for molecular Hamiltonian data.

Caches expensive computations across pipeline runs:
- Layer 1: MolecularIntegrals (h1e, h2e, nuclear_repulsion, metadata)
- Layer 2: FCI energy

Cache key = hash(geometry + basis + charge + spin).
Storage: ~/.cache/molecular-krylov/<hash>/
Invalidation removes the entire hash directory (both layers together).

Version control:
- cache_version: bumped when cache format changes
- pyscf_version: major.minor mismatch invalidates integrals
- content_sha256: integrity check on load
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

CACHE_DIR = Path.home() / ".cache" / "molecular-krylov"

# Bump this when changing the cache format (field names, npz layout, etc.)
CACHE_VERSION = 2


def _molecule_hash(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str,
    charge: int,
    spin: int,
) -> str:
    """Deterministic hash from molecule specification.

    Geometry coordinates are rounded to 8 decimal places to avoid
    floating-point formatting differences.
    """
    # (order matters for PySCF, so we DON'T sort — we preserve user order)
    geo_str = json.dumps(
        [(atom, tuple(round(c, 8) for c in coords)) for atom, coords in geometry],
        sort_keys=False,
    )
    # Match PySCF's internal _format_basis_name(): lowercase + strip hyphens/underscores/spaces
    basis_normalized = basis.lower().replace('-', '').replace('_', '').replace(' ', '')
    key_str = f"{geo_str}|{basis_normalized}|{charge}|{spin}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _content_sha256(h1e: np.ndarray, h2e: np.ndarray) -> str:
    """SHA-256 of integral arrays for integrity verification."""
    h = hashlib.sha256()
    h.update(h1e.tobytes())
    h.update(h2e.tobytes())
    return h.hexdigest()


def _pyscf_version_str() -> str:
    """Get PySCF major.minor version string, or 'unknown'."""
    try:
        import pyscf
        v = pyscf.__version__
        parts = v.split('.')
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return v
    except Exception:
        return "unknown"


def _cache_dir_for(mol_hash: str) -> Path:
    d = CACHE_DIR / mol_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_integrals(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str,
    charge: int,
    spin: int,
    h1e: np.ndarray,
    h2e: np.ndarray,
    nuclear_repulsion: float,
    n_electrons: int,
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
) -> str:
    """Save molecular integrals to disk cache. Returns the cache hash."""
    mol_hash = _molecule_hash(geometry, basis, charge, spin)
    d = _cache_dir_for(mol_hash)

    # Write to temp files then rename for atomicity.
    # np.savez_compressed auto-appends .npz, so use a name that ends with .npz
    tmp_npz = d / "integrals_tmp.npz"
    np.savez_compressed(tmp_npz, h1e=h1e, h2e=h2e)
    tmp_npz.rename(d / "integrals.npz")

    meta = {
        "cache_version": CACHE_VERSION,
        "pyscf_version": _pyscf_version_str(),
        "content_sha256": _content_sha256(h1e, h2e),
        "nuclear_repulsion": nuclear_repulsion,
        "n_electrons": n_electrons,
        "n_orbitals": n_orbitals,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "basis": basis,
        "charge": charge,
        "spin": spin,
        "geometry": [(a, list(c)) for a, c in geometry],
    }
    tmp_meta = d / "meta.json.tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    tmp_meta.rename(d / "meta.json")

    return mol_hash


def load_integrals(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str,
    charge: int,
    spin: int,
) -> Optional[dict]:
    """Load cached integrals. Returns dict with h1e, h2e, metadata, or None.

    Invalidates cache if:
    - cache_version mismatch (format change)
    - pyscf major.minor version mismatch
    - content SHA-256 integrity check fails
    """
    mol_hash = _molecule_hash(geometry, basis, charge, spin)
    d = CACHE_DIR / mol_hash

    integrals_path = d / "integrals.npz"
    meta_path = d / "meta.json"

    if not integrals_path.exists() or not meta_path.exists():
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)

        # Check cache_version
        cached_version = meta.get("cache_version", 1)
        if cached_version != CACHE_VERSION:
            print(f"[HamiltonianCache] Cache version mismatch "
                  f"(cached={cached_version}, current={CACHE_VERSION}), invalidating")
            _invalidate(d)
            return None

        # Check pyscf version (major.minor)
        cached_pyscf = meta.get("pyscf_version", "unknown")
        current_pyscf = _pyscf_version_str()
        if cached_pyscf != "unknown" and current_pyscf != "unknown":
            if cached_pyscf != current_pyscf:
                print(f"[HamiltonianCache] PySCF version mismatch "
                      f"(cached={cached_pyscf}, current={current_pyscf}), invalidating")
                _invalidate(d)
                return None

        # Load arrays
        with np.load(integrals_path) as data:
            h1e = data["h1e"].copy()
            h2e = data["h2e"].copy()

        # Integrity check
        cached_sha = meta.get("content_sha256")
        if cached_sha is not None:
            actual_sha = _content_sha256(h1e, h2e)
            if cached_sha != actual_sha:
                print(f"[HamiltonianCache] Integrity check failed "
                      f"(expected={cached_sha[:16]}..., got={actual_sha[:16]}...), invalidating")
                _invalidate(d)
                return None

        return {
            "h1e": h1e,
            "h2e": h2e,
            "nuclear_repulsion": meta["nuclear_repulsion"],
            "n_electrons": meta["n_electrons"],
            "n_orbitals": meta["n_orbitals"],
            "n_alpha": meta["n_alpha"],
            "n_beta": meta["n_beta"],
        }
    except Exception as e:
        print(f"[HamiltonianCache] Failed to load integrals: {e}")
        return None


def _invalidate(cache_dir: Path) -> None:
    """Remove a cache directory (integrals + FCI)."""
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def save_fci_energy(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str,
    charge: int,
    spin: int,
    fci_energy: float,
) -> None:
    """Save FCI energy to disk cache."""
    mol_hash = _molecule_hash(geometry, basis, charge, spin)
    d = _cache_dir_for(mol_hash)
    tmp = d / "fci_energy.json.tmp"
    with open(tmp, "w") as f:
        json.dump({"fci_energy": fci_energy, "cache_version": CACHE_VERSION}, f)
    tmp.rename(d / "fci_energy.json")


def load_fci_energy(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str,
    charge: int,
    spin: int,
) -> Optional[float]:
    """Load cached FCI energy. Returns float or None."""
    mol_hash = _molecule_hash(geometry, basis, charge, spin)
    path = CACHE_DIR / mol_hash / "fci_energy.json"

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)
        # Defense-in-depth: check cache_version even though _invalidate
        # removes the entire directory (both integrals + FCI together).
        cached_version = data.get("cache_version", 1)
        if cached_version != CACHE_VERSION:
            return None
        return data["fci_energy"]
    except Exception as e:
        print(f"[HamiltonianCache] Failed to load FCI energy: {e}")
        return None


def clear_cache(
    geometry: Optional[List[Tuple[str, Tuple[float, float, float]]]] = None,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
) -> None:
    """Clear cache for a specific molecule, or all caches if geometry is None."""
    import shutil

    if geometry is None:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print("[HamiltonianCache] Cleared all caches")
    else:
        mol_hash = _molecule_hash(geometry, basis, charge, spin)
        d = CACHE_DIR / mol_hash
        if d.exists():
            shutil.rmtree(d)
            print(f"[HamiltonianCache] Cleared cache for {mol_hash}")


def cache_info() -> dict:
    """Return summary of cached molecules."""
    if not CACHE_DIR.exists():
        return {"entries": 0, "molecules": []}

    molecules = []
    for d in sorted(CACHE_DIR.iterdir()):
        if not d.is_dir():
            continue
        entry = {"hash": d.name, "has_integrals": (d / "integrals.npz").exists()}
        entry["has_fci"] = (d / "fci_energy.json").exists()
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                atoms = [a for a, _ in meta.get("geometry", [])]
                entry["formula"] = "".join(atoms)
                entry["basis"] = meta.get("basis", "?")
                entry["n_orbitals"] = meta.get("n_orbitals", "?")
                entry["pyscf_version"] = meta.get("pyscf_version", "?")
                entry["cache_version"] = meta.get("cache_version", "?")
            except Exception:
                pass
        molecules.append(entry)

    return {"entries": len(molecules), "molecules": molecules}
