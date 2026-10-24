"""
mof_screen/dataclasses_.py
==========================
All result dataclasses for the MOF K⁺-anode DFT screening pipeline.

Changes from _core.py:
  - BandgapResult removed entirely
  - IonicConductivityResult added (replaces bandgap)
  - ProgressStatus added for live per-parameter progress dashboard
  - ConvexHullResult added  (thermodynamic phase stability / eHull)
  - BaderResult added       (Bader / Hirshfeld / Mulliken charge analysis)
  - AllResults updated accordingly
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enum: progress state for each parameter
# ---------------------------------------------------------------------------
class CalcStatus(str, Enum):
    PENDING  = "PENDING"
    RUNNING  = "RUNNING"
    DONE     = "DONE"
    FAILED   = "FAILED"
    SKIPPED  = "SKIPPED"
    CACHED   = "CACHED"     # loaded from a prior run's JSON cache


# ---------------------------------------------------------------------------
# Per-parameter live progress snapshot
# ---------------------------------------------------------------------------
@dataclass
class ParameterProgress:
    """
    Snapshot of one parameter's execution progress.
    Updated every second by the progress dashboard.
    """
    name:           str             = ""
    status:         CalcStatus      = CalcStatus.PENDING

    # Progress within the current DFT run (0–100 %)
    pct_done:       float           = 0.0

    # Elapsed wall-clock time (seconds)
    elapsed_s:      float           = 0.0

    # Estimated time remaining (seconds); None = unknown
    eta_s:          Optional[float] = None

    # Expected total wall-clock time (seconds) — from EXPECTED_TIMES_MIN
    expected_s:     Optional[float] = None

    # Human-readable phase description, e.g. "GEO step 12/300"
    phase:          str             = ""

    # Current SCF step (for DFT runs)
    scf_step:       int             = 0
    geo_step:       int             = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Result dataclasses — one per parameter
# ---------------------------------------------------------------------------

@dataclass
class IonicConductivityResult:
    """
    Ionic conductivity σ (S/cm) derived via Nernst-Einstein equation.
    Uses the diffusion barrier (eV) and ion adsorption energy (eV).

    Full formula (when barrier available):
        D   = D0 × exp(−Ea / kB T)
        σ   = n × z² × e² × D / (kB × T)

    Fallback (when barrier not available):
        Qualitative rank derived from E_ads alone.
    """
    sigma_s_cm:         Optional[float] = None    # S/cm
    diffusion_coeff:    Optional[float] = None    # cm²/s
    barrier_ev:         Optional[float] = None    # eV  (Ea used)
    temperature_k:      float           = 300.0   # K
    carrier_density:    Optional[float] = None    # ions/cm³
    ion_symbol:         str             = "K"
    z_charge:           int             = 1
    rank:               str             = "UNKNOWN"
    method:             str             = "UNKNOWN"   # "nernst_einstein" | "eads_estimate"
    elapsed_min:        float           = 0.0


@dataclass
class AdsorptionResult:
    """K⁺ ion adsorption energy (eV) from GEO_OPT of MOF + K."""
    e_ads_ev:     Optional[float] = None
    e_mof_ion_ev: Optional[float] = None
    n_ions:       int             = 1
    ion_symbol:   str             = "K"
    relaxed:      bool            = False
    elapsed_min:  float           = 0.0


@dataclass
class FormationResult:
    """MOF formation energy (eV and eV/atom) from elemental references."""
    e_form_ev:          Optional[float] = None
    e_form_per_atom_ev: Optional[float] = None
    refs_complete:      bool            = True
    missing_elements:   List[str]       = field(default_factory=list)


@dataclass
class VolumeResult:
    """Unit-cell volume expansion (%) upon K⁺ insertion."""
    v_before_A3:   Optional[float] = None
    v_after_A3:    Optional[float] = None
    expansion_pct: Optional[float] = None
    cell_relaxed:  bool            = False


@dataclass
class OCVResult:
    """Open-circuit voltage (V) derived from the adsorption energy."""
    ocv_v:      Optional[float] = None
    z_charge:   int             = 1
    n_ions:     int             = 1
    ion_symbol: str             = "K"


@dataclass
class DiffusionBarrierResult:
    """
    K⁺ diffusion barrier (eV) read from a pre-computed NEB/CI-NEB file.
    The file must contain a single float (eV) or a CP2K NEB output.
    """
    barrier_ev:  Optional[float] = None
    source_file: str             = ""
    available:   bool            = False


@dataclass
class DOSResult:
    """Density-of-states info parsed from CP2K PDOS output files."""
    pdos_files:   List[str]       = field(default_factory=list)
    fermi_ev:     Optional[float] = None
    n_pdos_files: int             = 0
    parsed:       bool            = False


@dataclass
class ConvexHullResult:
    """
    Thermodynamic phase stability measured by energy above the convex hull
    (eHull, eV/atom).  A lower eHull means the MOF is closer to the ground
    state of the phase diagram.

        eHull ≤ 0          → on or below hull (thermodynamically stable)
        0 < eHull ≤ 0.05   → metastable (synthesisable at RT)
        0.05 < eHull ≤ 0.1 → marginally metastable
        eHull  > 0.1       → unstable — decomposes spontaneously
    """
    e_hull_ev_per_atom:  Optional[float] = None    # eV/atom above hull
    e_form_per_atom_ev:  Optional[float] = None    # formation energy used
    hull_ref_formula:    str             = ""      # competing-phase label
    n_competing_phases:  int             = 0       # how many competing phases
    stability_label:     str             = "UNKNOWN"
    refs_used:           bool            = False   # True if competing-phase DB used


@dataclass
class BaderResult:
    """
    Charge partitioning result from Bader (Henkelman), Hirshfeld, or
    Mulliken analysis.  Quantifies electron transfer from the inserted ion
    to the MOF framework.

    For a good K⁺ anode:
        ion_charge_e  ≈  +0.8 – +1.0 e  (nearly full electron donation)
    """
    ion_symbol:            str             = "K"
    n_ions:                int             = 1
    ion_charge_e:          Optional[float] = None   # Bader charge on ion (e)
    ion_charge_transfer:   Optional[float] = None   # electrons transferred to MOF
    framework_avg_charge:  Optional[float] = None   # average charge on framework atoms
    method:                str             = "UNKNOWN"   # "bader"|"hirshfeld"|"mulliken"
    parsed:                bool            = False
    all_charges:           List[Dict]      = field(default_factory=list)
    elapsed_min:           float           = 0.0


# ---------------------------------------------------------------------------
# Top-level aggregate result
# ---------------------------------------------------------------------------
@dataclass
class AllResults:
    """
    Complete screening result for one MOF CIF file.
    Serialisable to JSON via dataclasses.asdict().
    """
    formula:              str                      = ""
    n_atoms:              int                      = 0
    cif_path:             str                      = ""
    ion_symbol:           str                      = "K"

    ionic_conductivity:   IonicConductivityResult  = field(
        default_factory=IonicConductivityResult)
    adsorption:           AdsorptionResult         = field(
        default_factory=AdsorptionResult)
    formation:            FormationResult           = field(
        default_factory=FormationResult)
    volume:               VolumeResult             = field(
        default_factory=VolumeResult)
    ocv:                  OCVResult                = field(
        default_factory=OCVResult)
    diffusion_barrier:    DiffusionBarrierResult   = field(
        default_factory=DiffusionBarrierResult)
    dos:                  DOSResult                = field(
        default_factory=DOSResult)
    convex_hull:          ConvexHullResult         = field(
        default_factory=ConvexHullResult)
    bader:                BaderResult              = field(
        default_factory=BaderResult)

    def to_dict(self) -> dict:
        return asdict(self)
