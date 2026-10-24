"""
mof_screen/structure.py
=======================
MOF structure helpers: CIF validation, void-site detection, and
K⁺ ion insertion.

All functions accept ASE Atoms objects and return new Atoms objects
or numpy arrays — no CP2K calls, no side effects on disk.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import read

from mof_screen.constants import (
    HUBBARD_U,
    IONIC_RADII,
    UNPAIRED_PER_SITE,
)

log = logging.getLogger("mof_dft.structure")


# ---------------------------------------------------------------------------
# CIF validation
# ---------------------------------------------------------------------------
def validate_cif(cif_path: str) -> Atoms:
    """
    Load and validate a CIF file.

    Checks:
      1. File is readable and parseable by ASE
      2. Non-zero cell volume
      3. No atoms with separation < 0.8 Å (likely unrelaxed structure)
      4. Structure has periodic boundary conditions

    Returns the loaded Atoms object on success.
    Raises ValueError / RuntimeError on failure.
    """
    try:
        atoms = read(cif_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read CIF file '{cif_path}': {exc}") from exc

    # Must have a periodic cell
    if not any(atoms.pbc):
        atoms.set_pbc(True)
        log.warning("  CIF had no PBC — enabling XYZ periodicity.")

    vol = atoms.get_volume()
    if vol <= 0:
        raise ValueError(f"CIF has zero/negative volume ({vol:.3f} Å³) — aborting.")

    # Short-contact warning (may indicate unrelaxed or overlapping atoms)
    if len(atoms) > 1:
        all_dists = atoms.get_all_distances(mic=True)
        np.fill_diagonal(all_dists, np.inf)
        min_d = all_dists.min()
        if min_d < 0.8:
            log.warning(
                f"  Very short bond detected: {min_d:.3f} Å — "
                "CIF may be unrelaxed or contain overlapping atoms!"
            )

    log.info(f"  CIF validated: {atoms.get_chemical_formula()}  "
             f"({len(atoms)} atoms, {vol:.1f} Å³)")
    return atoms


# ---------------------------------------------------------------------------
# Transition-metal detection
# ---------------------------------------------------------------------------
def detect_transition_metals(atoms: Atoms) -> Dict[str, float]:
    """Return {element: U_eff_eV} for TM elements present with U > 0."""
    syms = set(atoms.get_chemical_symbols())
    return {s: HUBBARD_U[s] for s in syms if s in HUBBARD_U and HUBBARD_U[s] > 0.0}


# ---------------------------------------------------------------------------
# Spin multiplicity estimation
# ---------------------------------------------------------------------------
def estimate_multiplicity(atoms: Atoms) -> int:
    """
    Estimate spin multiplicity = 2S + 1.
    Sums unpaired electrons across TM sites, capped at 5 sites per
    element to avoid absurd values for MOFs with many identical nodes.
    """
    counts = Counter(atoms.get_chemical_symbols())
    total  = 0
    for elem, n in counts.items():
        per = UNPAIRED_PER_SITE.get(elem, 0)
        if per > 0:
            total += per * min(n, 5)
    return total + 1


# ---------------------------------------------------------------------------
# Metallic-system heuristic (for smearing decisions)
# ---------------------------------------------------------------------------
_NON_METALS = frozenset({"H", "C", "N", "O", "F", "Cl", "Br", "I", "P", "S", "Se"})

def is_metallic_system(atoms: Atoms) -> bool:
    """
    Return True only for systems that likely need Fermi smearing.
    MOFs (> 80 % non-metal atoms) always return False.
    """
    nm = sum(1 for s in atoms.get_chemical_symbols() if s in _NON_METALS)
    return nm / len(atoms) < 0.2




# ---------------------------------------------------------------------------
# Void-site detection (geometric fallback)
# ---------------------------------------------------------------------------
def find_void_sites(
    atoms: Atoms,
    ion: str    = "K",
    n_sites: int = 1,
    grid_pts: int = 12,
) -> List[np.ndarray]:
    """
    Find the n_sites largest void positions using a fractional-coordinate grid.
    Used as a fallback when chemistry-aware sites cannot be found.
    """
    base_min_dist = 1.8
    scale    = IONIC_RADII.get(ion, 1.0) / IONIC_RADII.get("Li", 0.76)
    min_dist = max(1.5, base_min_dist * scale)

    cell     = atoms.get_cell()
    cell_inv = np.linalg.inv(cell.T)
    pos      = atoms.get_positions()

    lin        = np.linspace(0.05, 0.95, grid_pts)
    grid_frac  = np.array([[i, j, k] for i in lin for j in lin for k in lin])
    grid_cart  = grid_frac @ cell

    min_dists = np.full(len(grid_cart), np.inf)
    for p in pos:
        diff  = grid_cart - p
        df    = diff @ cell_inv.T
        df   -= np.round(df)
        dists = np.linalg.norm(df @ cell, axis=1)
        min_dists = np.minimum(min_dists, dists)

    order  = np.argsort(-min_dists)
    sites: List[np.ndarray] = []

    for idx in order:
        if len(sites) >= n_sites:
            break
        cand      = grid_cart[idx]
        void_size = min_dists[idx]
        if void_size < min_dist:
            continue
        too_close = False
        for placed in sites:
            df  = (cand - placed) @ cell_inv.T
            df -= np.round(df)
            if np.linalg.norm(df @ cell) < min_dist:
                too_close = True
                break
        if not too_close:
            sites.append(cand)
            log.info(
                f"    Void site {len(sites)}: {np.round(cand, 2)} Å  "
                f"(nearest atom {void_size:.2f} Å)"
            )

    if len(sites) < n_sites:
        log.warning(
            f"    Only {len(sites)}/{n_sites} void sites found with "
            f"min_dist > {min_dist:.2f} Å."
        )
    return sites


# ---------------------------------------------------------------------------
# Chemistry-aware binding site detection
# ---------------------------------------------------------------------------
# Covalent radii (Å) for bond-detection heuristics
_COV_RADII: Dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Cl": 1.02, "Zr": 1.75, "Ti": 1.60,
    "Hf": 1.75, "Ce": 2.04,
}


def _min_image_dist(a: np.ndarray, b: np.ndarray,
                    cell: np.ndarray, cell_inv: np.ndarray) -> float:
    """Minimum-image distance between two Cartesian points."""
    df = (a - b) @ cell_inv.T
    df -= np.round(df)
    return float(np.linalg.norm(df @ cell))


def _min_dist_to_framework(
    cand: np.ndarray,
    pos: np.ndarray,
    cell: np.ndarray,
    cell_inv: np.ndarray,
) -> float:
    """Minimum periodic distance from candidate to any framework atom."""
    diff = cand - pos
    df   = diff @ cell_inv.T
    df  -= np.round(df)
    return float(np.min(np.linalg.norm(df @ cell, axis=1)))


def _candidates_too_close(
    cand: np.ndarray,
    placed: List[np.ndarray],
    cell: np.ndarray,
    cell_inv: np.ndarray,
    min_sep: float = 4.0,
) -> bool:
    """Return True if cand is within min_sep of any already-placed candidate."""
    for p in placed:
        if _min_image_dist(cand, p, cell, cell_inv) < min_sep:
            return True
    return False


def _find_k1_sites(
    syms: list, pos: np.ndarray, cell: np.ndarray, cell_inv: np.ndarray,
    min_clearance: float,
) -> List[Tuple[np.ndarray, str, float]]:
    """
    K1 sites: K placed 2.8 Å above aromatic C-ring centroids.
    Identifies 6-membered carbon rings by C–C bond topology.
    """
    sites: List[Tuple[np.ndarray, str, float]] = []
    carbon_idx = [i for i, s in enumerate(syms) if s == "C"]
    used_rings: set = set()

    for i, ci in enumerate(carbon_idx):
        if ci in used_rings:
            continue
        bonded_js = [
            j for j, cj in enumerate(carbon_idx)
            if i != j and _min_image_dist(pos[ci], pos[cj], cell, cell_inv) < 1.7
        ]
        if len(bonded_js) < 2:
            continue
        # BFS to collect ring members
        ring_indices = [ci]
        queue = [carbon_idx[j] for j in bonded_js[:2]]
        visited = {ci}
        while queue and len(ring_indices) < 6:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            ring_indices.append(cur)
            for cj in carbon_idx:
                if cj not in visited and _min_image_dist(pos[cur], pos[cj], cell, cell_inv) < 1.7:
                    queue.append(cj)
        if len(ring_indices) < 5:
            continue
        ring_pos = pos[ring_indices[:6]]
        centroid = ring_pos.mean(axis=0)
        _, _, Vt = np.linalg.svd(ring_pos - centroid)
        normal = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12)
        for sign in (+1.0, -1.0):
            cand  = centroid + sign * 2.8 * normal
            d_min = _min_dist_to_framework(cand, pos, cell, cell_inv)
            if d_min >= min_clearance:
                score = abs(d_min - 2.8)
                sites.append((cand, "K1_aromatic", score))
        used_rings.update(ring_indices)

    return sites


def _find_k2_sites(
    syms: list, pos: np.ndarray, cell: np.ndarray, cell_inv: np.ndarray,
    min_clearance: float,
) -> List[Tuple[np.ndarray, str, float]]:
    """
    K2 sites: K placed 2.0 Å beyond the midpoint of carboxylate O–C–O groups.
    """
    sites: List[Tuple[np.ndarray, str, float]] = []
    oxygen_idx = [i for i, s in enumerate(syms) if s == "O"]
    for ci, cs in enumerate(syms):
        if cs != "C":
            continue
        o_bonded = [
            oi for oi in oxygen_idx
            if _min_image_dist(pos[ci], pos[oi], cell, cell_inv) < 1.6
        ]
        if len(o_bonded) < 2:
            continue
        o1, o2 = o_bonded[0], o_bonded[1]
        midpt     = (pos[o1] + pos[o2]) / 2.0
        direction = midpt - pos[ci]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        cand  = midpt + (direction / norm) * 2.0
        d_min = _min_dist_to_framework(cand, pos, cell, cell_inv)
        if d_min >= min_clearance:
            score = abs(d_min - 2.5)
            sites.append((cand, "K2_carboxylate", score))
    return sites


def _find_k3_sites(
    syms: list, pos: np.ndarray, cell: np.ndarray, cell_inv: np.ndarray,
    min_clearance: float,
) -> List[Tuple[np.ndarray, str, float]]:
    """
    K3 sites: K placed 2.3 Å outward from µ₃-O bridging oxygens on metal nodes.
    """
    sites: List[Tuple[np.ndarray, str, float]] = []
    zr_idx     = [i for i, s in enumerate(syms) if s in ("Zr", "Ti", "Hf", "Ce")]
    oxygen_idx = [i for i, s in enumerate(syms) if s == "O"]
    for oi in oxygen_idx:
        zr_bonded = [zi for zi in zr_idx if _min_image_dist(pos[oi], pos[zi], cell, cell_inv) < 2.5]
        if len(zr_bonded) < 2:
            continue
        zr_centroid = pos[zr_bonded].mean(axis=0)
        direction   = pos[oi] - zr_centroid
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        cand  = pos[oi] + (direction / norm) * 2.3
        d_min = _min_dist_to_framework(cand, pos, cell, cell_inv)
        if d_min >= min_clearance:
            score = abs(d_min - 2.3)
            sites.append((cand, "K3_zr_oxo", score))
    return sites


def find_binding_sites(
    atoms: Atoms,
    ion: str = "K",
    n_sites: int = 1,
) -> List[Tuple[np.ndarray, str, float]]:
    """
    Chemistry-aware K-binding site search for MOF anodes.

    Generates candidate positions at three site types (K1/K2/K3),
    filters by clearance, sorts by proximity score, and returns
    the top n_sites non-overlapping candidates.

    Falls back to geometric void search if no chemistry sites exist.
    Returns: list of (position, site_type, score).
    """
    syms = atoms.get_chemical_symbols()
    pos  = atoms.get_positions()
    cell = atoms.get_cell().array
    cell_inv = np.linalg.inv(cell.T)
    ion_radius    = IONIC_RADII.get(ion, 1.38)
    min_clearance = ion_radius + 1.2

    log.info(f"  Searching for {n_sites} chemically-aware binding site(s) for {ion}…")

    candidates: List[Tuple[np.ndarray, str, float]] = []
    candidates.extend(_find_k1_sites(syms, pos, cell, cell_inv, min_clearance))
    candidates.extend(_find_k2_sites(syms, pos, cell, cell_inv, min_clearance))
    candidates.extend(_find_k3_sites(syms, pos, cell, cell_inv, min_clearance))
    candidates.sort(key=lambda x: x[2])

    if not candidates:
        log.warning("  No chemistry-aware binding sites found — falling back to geometric void search.")
        void_sites = find_void_sites(atoms, ion=ion, n_sites=n_sites)
        return [(s, "void_geometric", 99.0) for s in void_sites]

    selected: List[Tuple[np.ndarray, str, float]] = []
    for cand, stype, score in candidates:
        if len(selected) >= n_sites:
            break
        placed_positions = [p for p, _, _ in selected]
        if _candidates_too_close(cand, placed_positions, cell, cell_inv):
            continue
        selected.append((cand, stype, score))
        log.info(
            f"    Binding site {len(selected)} [{stype}]: "
            f"{np.round(cand, 2)} Å  (clearance={_min_dist_to_framework(cand, pos, cell, cell_inv):.2f} Å)"
        )

    if len(selected) < n_sites:
        # Use higher-resolution grid for high-loading cases
        log.warning(
            f"  Only {len(selected)}/{n_sites} chemistry sites found — "
            f"filling with void search (grid_pts=24 for high loading)."
        )
        grid_res = 24 if n_sites > 10 else 16
        extra = find_void_sites(atoms, ion=ion, n_sites=n_sites - len(selected),
                                grid_pts=grid_res)
        placed_positions = [p for p, _, _ in selected]
        for s in extra:
            if not _candidates_too_close(s, placed_positions, cell, cell_inv):
                selected.append((s, "void_geometric", 99.0))
                placed_positions.append(s)

    return selected


# ---------------------------------------------------------------------------
# Multi-site candidate generator (one per site type, for pre-screening)
# ---------------------------------------------------------------------------
def find_candidate_sites_all_types(
    atoms: Atoms,
    ion: str = "K",
    max_per_type: int = 1,
) -> List[Tuple[np.ndarray, str, float]]:
    """
    Return the best candidate for EACH chemical site type (K1/K2/K3).
    Used for multi-site pre-screening in calc_adsorption.py.
    Returns list of (position, site_type, score).
    """
    syms = atoms.get_chemical_symbols()
    pos  = atoms.get_positions()
    cell = atoms.get_cell().array
    cell_inv = np.linalg.inv(cell.T)
    ion_radius    = IONIC_RADII.get(ion, 1.38)
    min_clearance = ion_radius + 1.2

    by_type: Dict[str, List[Tuple[np.ndarray, str, float]]] = {
        "K1_aromatic":   sorted(_find_k1_sites(syms, pos, cell, cell_inv, min_clearance), key=lambda x: x[2]),
        "K2_carboxylate": sorted(_find_k2_sites(syms, pos, cell, cell_inv, min_clearance), key=lambda x: x[2]),
        "K3_zr_oxo":     sorted(_find_k3_sites(syms, pos, cell, cell_inv, min_clearance), key=lambda x: x[2]),
    }

    result: List[Tuple[np.ndarray, str, float]] = []
    for stype, cands in by_type.items():
        result.extend(cands[:max_per_type])

    if not result:
        log.warning("  No chemistry sites found — using geometric void search.")
        void_sites = find_void_sites(atoms, ion=ion, n_sites=3)
        result = [(s, "void_geometric", 99.0) for s in void_sites]

    return result


# ---------------------------------------------------------------------------
# Ion-loaded structure builder
# ---------------------------------------------------------------------------
def build_loaded_structure(
    atoms: Atoms,
    ion: str = "K",
    n: int   = 1,
) -> Atoms:
    """
    Insert n ions into the MOF at the best chemistry-aware binding sites.
    Returns a new Atoms object (original is not modified).
    Raises RuntimeError if no suitable sites are found.
    """
    sites = find_binding_sites(atoms, ion=ion, n_sites=n)
    if not sites:
        raise RuntimeError(
            f"No binding sites found for {n}× {ion}.  "
            "Ensure the CIF is a fully relaxed structure with open pores."
        )
    loaded = atoms.copy()
    for site_pos, site_type, score in sites:
        loaded += Atoms(
            ion,
            positions=[site_pos],
            cell=atoms.get_cell(),
            pbc=True,
        )
    return loaded

    """
    Find the n_sites largest void positions using a fractional-coordinate grid.

    Algorithm:
      1. Generate an evenly spaced 3-D Cartesian grid inside the unit cell.
      2. For every grid point compute the minimum distance to any framework
         atom (using the minimum-image convention for periodicity).
      3. Pick the n_sites grid points with the largest minimum distance,
         ensuring inserted ions are also ≥ min_dist apart from each other.

    The minimum insertion distance is scaled by the ionic radius of the
    chosen ion relative to Li (the reference).
    """
    # Scale minimum insertion distance by ionic radius
    base_min_dist = 1.8   # Å (reference for Li)
    scale    = IONIC_RADII.get(ion, 1.0) / IONIC_RADII.get("Li", 0.76)
    min_dist = max(1.5, base_min_dist * scale)

    cell     = atoms.get_cell()
    cell_inv = np.linalg.inv(cell.T)
    pos      = atoms.get_positions()

    lin        = np.linspace(0.05, 0.95, grid_pts)
    grid_frac  = np.array([[i, j, k] for i in lin for j in lin for k in lin])
    grid_cart  = grid_frac @ cell   # shape (N, 3)

    # Minimum distance from each grid point to any framework atom (periodic)
    min_dists = np.full(len(grid_cart), np.inf)
    for p in pos:
        diff  = grid_cart - p
        df    = diff @ cell_inv.T
        df   -= np.round(df)
        dists = np.linalg.norm(df @ cell, axis=1)
        min_dists = np.minimum(min_dists, dists)

    order  = np.argsort(-min_dists)
    sites: List[np.ndarray] = []

    for idx in order:
        if len(sites) >= n_sites:
            break
        cand      = grid_cart[idx]
        void_size = min_dists[idx]
        if void_size < min_dist:
            continue
        # Check mutual distance to already placed ions
        too_close = False
        for placed in sites:
            df  = (cand - placed) @ cell_inv.T
            df -= np.round(df)
            if np.linalg.norm(df @ cell) < min_dist:
                too_close = True
                break
        if not too_close:
            sites.append(cand)
            log.info(
                f"    Void site {len(sites)}: {np.round(cand, 2)} Å  "
                f"(nearest atom {void_size:.2f} Å)"
            )

    if len(sites) < n_sites:
        log.warning(
            f"    Only {len(sites)}/{n_sites} void sites found with "
            f"min_dist > {min_dist:.2f} Å. "
            "Try reducing --grid-pts or the ion size."
        )
    return sites


# ---------------------------------------------------------------------------
# Ion-loaded structure builder
# ---------------------------------------------------------------------------
def build_loaded_structure(
    atoms: Atoms,
    ion: str = "K",
    n: int   = 1,
) -> Atoms:
    """
    Insert n ions into the MOF at the largest void sites.
    Returns a new Atoms object (original is not modified).
    Raises RuntimeError if no suitable void sites are found.
    """
    sites = find_void_sites(atoms, ion=ion, n_sites=n)
    if not sites:
        raise RuntimeError(
            f"No void sites found for {n}× {ion}.  "
            "Ensure the CIF is a fully relaxed structure with open pores."
        )
    loaded = atoms.copy()
    for site_pos in sites:
        loaded += Atoms(
            ion,
            positions=[site_pos],
            cell=atoms.get_cell(),
            pbc=True,
        )
    return loaded
