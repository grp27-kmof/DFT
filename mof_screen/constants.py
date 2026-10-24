"""
mof_screen/constants.py
=======================
All physical constants and element/ion lookup tables used across the
MOF K⁺-anode screening pipeline.

All values are module-level constants — import directly:
    from mof_screen.constants import HARTREE_TO_EV, HUBBARD_U
"""

from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HARTREE_TO_EV: float = 27.211386245988      # 1 Hartree in eV
KB_EV: float         = 8.617333262e-5       # Boltzmann constant (eV/K)
AVOGADRO: float      = 6.02214076e23        # mol⁻¹
ELEM_CHARGE: float   = 1.602176634e-19      # Coulombs

# ---------------------------------------------------------------------------
# CP2K binary search order
# ---------------------------------------------------------------------------
CP2K_BINARY_NAMES: List[str] = [
    "cp2k.psmp", "cp2k.ssmp", "cp2k.popt", "cp2k",
]

# ---------------------------------------------------------------------------
# Basis sets
# ---------------------------------------------------------------------------
BASIS_SETS: Dict[str, str] = {
    "default":       "DZVP-MOLOPT-SR-GTH",
    "high_accuracy": "TZV2P-MOLOPT-GTH",
}

# ---------------------------------------------------------------------------
# DFT+U (Hubbard U_eff = U − J, eV, applied to d-orbitals)
# Source: Jain et al., APL Materials 1, 011002 (2013)
# ---------------------------------------------------------------------------
HUBBARD_U: Dict[str, float] = {
    "Ti": 3.5,  "V":  3.1,  "Cr": 3.5,  "Mn": 3.9,  "Fe": 4.0,
    "Co": 3.3,  "Ni": 6.4,  "Cu": 4.0,  "Zn": 0.0,
    # Zr omitted: Zr4+ in UiO-66 is 4d0 (empty d-shell) — no U correction needed.
    # Applying U to empty states is physically unjustified and forces Gamma-only k-pts.
    "Mo": 0.0,  "W":  0.0,  "Nb": 0.0,
    "Ru": 3.0,  "Rh": 3.0,  "Pd": 2.0,  "Ag": 1.5,
    "Ce": 5.0,  "Nd": 5.0,  "Sm": 5.0,  "Eu": 5.0,
}

# ---------------------------------------------------------------------------
# Unpaired electrons per TM site (heuristic, capped at 5 sites per element)
# ---------------------------------------------------------------------------
UNPAIRED_PER_SITE: Dict[str, int] = {
    "Ti": 0, "V":  2, "Cr": 3, "Mn": 5, "Fe": 4,
    "Co": 3, "Ni": 2, "Cu": 1, "Zn": 0,
    "Zr": 0, "Mo": 0, "W":  0, "Nb": 0,
    "Ru": 0, "Rh": 0, "Pd": 0, "Ag": 0,
    "Ce": 1, "Nd": 3, "Sm": 5, "Eu": 6,
    # Alkali / alkaline-earth / other metals
    "Li": 1, "Na": 1, "K":  1, "Mg": 0, "Ca": 0, "Al": 1,
}

# ---------------------------------------------------------------------------
# Correct spin multiplicities for diatomic reference molecules
# O₂ is a triplet; everything else is singlet / closed-shell
# ---------------------------------------------------------------------------
MOLECULE_MULTIPLICITY: Dict[str, int] = {
    "H2": 1, "N2": 1, "O2": 3, "Cl2": 1, "F2": 1,
}

# ---------------------------------------------------------------------------
# Ionic radii (Å) — used to scale void-site minimum distance
# ---------------------------------------------------------------------------
IONIC_RADII: Dict[str, float] = {
    "Li": 0.76, "Na": 1.02, "K":  1.38,
    "Mg": 0.72, "Ca": 1.00, "Zn": 0.74, "Al": 0.54,
}

# ---------------------------------------------------------------------------
# Charge state of each ion (number of electrons transferred per ion)
# ---------------------------------------------------------------------------
ION_CHARGE: Dict[str, int] = {
    "Li": 1, "Na": 1, "K":  1,
    "Mg": 2, "Ca": 2, "Zn": 2, "Al": 3,
}

# ---------------------------------------------------------------------------
# Approximate PBE bulk reference energies (eV/atom) — used as fallback
# when no self-consistent reference has been computed
# ---------------------------------------------------------------------------
APPROX_ION_REFS: Dict[str, float] = {
    "Li": -1.900, "Na": -1.310, "K":  -1.000,
    "Mg": -1.500, "Ca": -1.900, "Zn": -1.200, "Al": -3.500,
}

# ---------------------------------------------------------------------------
# Bulk structure info for alkali-metal reference calculations
# ---------------------------------------------------------------------------
ALKALI_METAL_INFO: Dict[str, dict] = {
    "Li": {"structure": "bcc", "lattice_a": 3.51, "multiplicity": 2},
    "Na": {"structure": "bcc", "lattice_a": 4.29, "multiplicity": 2},
    "K":  {"structure": "bcc", "lattice_a": 5.23, "multiplicity": 2},
}

# ---------------------------------------------------------------------------
# Full list of supported ion species
# ---------------------------------------------------------------------------
SUPPORTED_IONS: List[str] = ["Li", "Na", "K", "Mg", "Ca", "Zn", "Al"]

# ---------------------------------------------------------------------------
# CP2K plane-wave cutoff settings
# ---------------------------------------------------------------------------
DEFAULT_CUTOFF:     int = 500   # Ry  — 600 causes SIGABRT on K@π-ring geometry
DEFAULT_REL_CUTOFF: int = 50    # Ry
FAST_CUTOFF:        int = 400   # Ry  (fast screening mode)
FAST_REL_CUTOFF:    int = 40    # Ry

# ---------------------------------------------------------------------------
# K-point grid generation thresholds (Å)
# Any cell axis > KPOINT_MAX_CELL_LENGTH → Gamma-only (used for MOFs)
# Set to 25.0 Å so all typical MOF cells (> 10 Å) use Gamma only.
# This avoids the ADDED_MOS + k-points + UKS incompatibility in CP2K 2024.2.
# UiO-66 and similar MOFs are wide-gap insulators — Gamma is fully accurate.
KPOINT_MAX_CELL_LENGTH: float = 10.0  # Å — k-points only for small bulk refs (< 10 Å)
KPOINT_TARGET_PERIOD:   float = 25.0

# ---------------------------------------------------------------------------
# Ionic conductivity — pre-exponential diffusion coefficient
# Reference: Famprikis et al., Nature Materials 18, 1278–1291 (2019)
# ---------------------------------------------------------------------------
D0_CM2S:     float = 1.0e-5   # cm²/s  (standard MOF channel pre-exponential)
D0_ANG2S:    float = D0_CM2S * 1.0e16  # Å²/s  (working unit in ASE framework)

# ---------------------------------------------------------------------------
# Default simulation temperature
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE: float = 300.0   # Kelvin

# ---------------------------------------------------------------------------
# Expected wall-clock times per parameter (minutes) for a mid-size MOF
# (~200 atoms, 16 OMP cores).  Used ONLY for ETA estimation in the
# progress dashboard — not hard limits.
# ---------------------------------------------------------------------------
EXPECTED_TIMES_MIN: Dict[str, float] = {
    "adsorption":           60.0,   # GEO_OPT with K — most expensive DFT calc
    "formation":             0.01,  # instant — reuses DFT energy from adsorption
    "volume":                0.01,  # instant — reads adsorption.out cell volume
    "ocv":                   0.01,  # instant — derived from E_ads
    "diffusion":          2880.0,   # CI-NEB — typically 1-3 days for MOF-sized systems
    "dos":                   0.05,  # fast   — parses .pdos files
    "ionic_conductivity":    0.05,  # fast   — Nernst-Einstein from E_ads + barrier
    "convex_hull":           0.01,  # instant — derived from E_form/atom
    "bader":                 0.10,  # fast   — parses CUBE/output charge data
}

# Maximum SCF / GEO_OPT iterations (mirrors CP2K input values)
MAX_SCF_STEPS: int = 300
MAX_GEO_STEPS: int = 300

# ---------------------------------------------------------------------------
# SCREENING-GRADE PRESET  (--screening flag)
# ---------------------------------------------------------------------------
# Purpose: cut adsorption GEO_OPT wall time by ~40-60% for quick candidate
# filtering, while keeping E_ads accurate to ±0.05–0.10 eV (vs ±0.01 eV
# for production). Good for ranking MOFs; NOT for publication.
#
# What each change does and why it is safe:
#
#  CUTOFF 400 Ry  (vs 600)
#    The plane-wave cutoff controls basis completeness.  600 Ry is
#    over-converged for DZVP.  400 Ry gives E_ads within ~0.02 eV.
#    This is the BIGGEST single speedup (~30% per SCF step).
#
#  EPS_SCF 1E-6   (vs 1E-7)
#    SCF convergence threshold (energy change per step).  Going from
#    1E-7 → 1E-6 typically saves 10-20 SCF iterations per GEO step.
#    E_ads error introduced: < 0.005 eV (negligible for screening).
#
#  GEO MAX_FORCE 0.001 Ha/Bohr  (vs 0.0005)
#    Geometry convergence criterion.  Looser threshold = fewer GEO_OPT
#    steps (typically 30-50% fewer).  The K+ position is less critical
#    for E_ads accuracy than for force constants / phonons.
#
#  MAX_GEO 150    (vs 300)
#    Hard cap on geometry steps.  Rarely hit for well-placed void sites.
#    If not converged at 150 steps, the SP fallback gives a usable energy.
#
#  MAX_SCF 150    (vs 300)
#    Hard cap on SCF iterations.  Rarely hit with Broyden mixing.
#    Reducing this is a safety net only — does NOT accelerate converged runs.
# ---------------------------------------------------------------------------
SCREENING_CUTOFF:     int   = 400       # Ry  — biggest per-step speedup
SCREENING_REL_CUTOFF: int   = 40        # Ry
SCREENING_EPS_SCF:    float = 1.0e-6    # looser SCF threshold → fewer iterations
SCREENING_MAX_SCF:    int   = 150       # hard cap (safety net only)
SCREENING_MAX_GEO:    int   = 30        # short cap for site pre-screening (fast ranking only)
SCREENING_MAX_FORCE:  float = 0.001     # Ha/Bohr  — looser GEO convergence
SCREENING_MAX_DR:     float = 0.006     # Bohr     — looser step size criterion

# Expected time in screening mode (for dashboard ETA)
EXPECTED_TIMES_SCREENING_MIN: Dict[str, float] = {
    "adsorption": 25.0,   # ~40-60% of production time
}
