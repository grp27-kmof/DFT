"""
mof_screen/printer.py
=====================
Shared terminal pretty-printer for all parameter result boxes.

Each calc module calls the relevant print_*() function here
immediately after its result is ready (production-grade: never wait
until all calcs finish before showing output).
"""

from __future__ import annotations

import sys
from typing import List

from mof_screen.dataclasses_ import (
    AdsorptionResult,
    BaderResult,
    ConvexHullResult,
    DiffusionBarrierResult,
    DOSResult,
    FormationResult,
    IonicConductivityResult,
    OCVResult,
    VolumeResult,
)

# Box width (characters)
_W = 64


def _box(title: str, lines: List[str]) -> None:
    """Print a neat bordered box to stdout immediately."""
    print("\n" + "┌" + "─" * _W + "┐")
    print(f"│  ✓  {title:<{_W - 4}}│")
    print("├" + "─" * _W + "┤")
    for line in lines:
        # Truncate long lines gracefully
        if len(line) > _W - 2:
            line = line[: _W - 5] + "..."
        print(f"│  {line:<{_W - 2}}│")
    print("└" + "─" * _W + "┘")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Ionic Conductivity
# ---------------------------------------------------------------------------
def print_ionic_conductivity(r: IonicConductivityResult) -> None:
    if r.sigma_s_cm is None and r.rank == "UNKNOWN":
        _box("IONIC CONDUCTIVITY RESULT", ["FAILED — check inputs"])
        return

    sigma_str = (
        f"{r.sigma_s_cm:.3e} S/cm"
        if r.sigma_s_cm is not None
        else "N/A (barrier not provided)"
    )
    diff_str = (
        f"{r.diffusion_coeff:.3e} cm²/s"
        if r.diffusion_coeff is not None
        else "N/A"
    )
    barrier_str = (
        f"{r.barrier_ev:.4f} eV"
        if r.barrier_ev is not None
        else "not provided — rank estimated from E_ads"
    )

    _box(
        "IONIC CONDUCTIVITY RESULT",
        [
            f"σ (Nernst-Einstein) : {sigma_str}",
            f"Diffusion coeff D   : {diff_str}",
            f"Barrier Ea          : {barrier_str}",
            f"Temperature         : {r.temperature_k:.0f} K",
            f"Ion charge z        : {r.z_charge}  ({r.ion_symbol}⁺)",
            f"Method              : {r.method}",
            f"Conductivity rank   : {r.rank}",
            f"Elapsed             : {r.elapsed_min:.1f} min",
            "",
            "σ = n·z²·e²·D / (kB·T)   [Nernst-Einstein]",
            "Higher σ → better rate capability as anode.",
        ],
    )


# ---------------------------------------------------------------------------
# Adsorption Energy
# ---------------------------------------------------------------------------
def print_adsorption(r: AdsorptionResult) -> None:
    if r.e_ads_ev is None:
        _box("ADSORPTION ENERGY RESULT", ["FAILED — check adsorption.out"])
        return

    e = r.e_ads_ev
    if e < -2.0:
        verdict = f"TOO STRONG — {r.ion_symbol} may be trapped (immobile)"
    elif e < -1.0:
        verdict = f"STRONG — excellent {r.ion_symbol} storage site"
    elif e < -0.3:
        verdict = f"MODERATE — {r.ion_symbol} is stable"
    elif e < 0.0:
        verdict = f"WEAK — {r.ion_symbol} barely binds"
    else:
        verdict = f"UNFAVORABLE — {r.ion_symbol} does not bind in MOF"

    _box(
        "ADSORPTION ENERGY RESULT",
        [
            f"E_ads ({r.n_ions}× {r.ion_symbol})   : {e:.4f} eV",
            f"Assessment          : {verdict}",
            f"Structure relaxed   : {r.relaxed}",
            f"Elapsed             : {r.elapsed_min:.1f} min",
            "",
            f"E_ads = E(MOF+{r.ion_symbol}) − E(MOF) − n·E({r.ion_symbol}_bulk)",
            "More negative → stronger binding → better anode.",
        ],
    )


# ---------------------------------------------------------------------------
# Formation Energy
# ---------------------------------------------------------------------------
def print_formation(r: FormationResult) -> None:
    if r.e_form_ev is None:
        _box("FORMATION ENERGY RESULT", ["FAILED — E(MOF) not available"])
        return

    e = r.e_form_per_atom_ev
    if e is None:
        verdict = "UNKNOWN"
    elif e < -1.0:
        verdict = "HIGHLY STABLE"
    elif e < 0.0:
        verdict = "STABLE"
    else:
        verdict = "UNSTABLE (positive E_form)"

    warn = "  ⚠ Refs approximate" if not r.refs_complete else ""
    lines = [
        f"E_form              : {r.e_form_ev:.4f} eV",
        f"E_form/atom         : {r.e_form_per_atom_ev:.4f} eV/atom",
        f"Assessment          : {verdict}",
        f"Refs complete       : {r.refs_complete}{warn}",
    ]
    if r.missing_elements:
        lines.append(f"Missing refs        : {r.missing_elements}")
    _box("FORMATION ENERGY RESULT", lines)


# ---------------------------------------------------------------------------
# Volume Expansion
# ---------------------------------------------------------------------------
def print_volume(r: VolumeResult, ion: str = "K") -> None:
    if r.expansion_pct is None:
        _box("VOLUME EXPANSION RESULT", ["FAILED — check adsorption.out"])
        return

    p = r.expansion_pct
    if abs(p) < 1.0:
        verdict = "EXCELLENT — rigid framework"
    elif p < 5.0:
        verdict = "VERY GOOD — minimal change"
    elif p < 10.0:
        verdict = "GOOD — acceptable for anode"
    elif p < 20.0:
        verdict = "MODERATE — may limit cycle life"
    else:
        verdict = "HIGH — poor cycle stability expected"

    _box(
        "VOLUME EXPANSION RESULT",
        [
            f"V(bare MOF)         : {r.v_before_A3:.3f} Å³",
            f"V(MOF + {ion})      : {r.v_after_A3:.3f} Å³",
            f"Expansion           : {p:.3f} %",
            f"Cell relaxed        : {r.cell_relaxed}",
            f"Assessment          : {verdict}",
        ],
    )


# ---------------------------------------------------------------------------
# Open-Circuit Voltage
# ---------------------------------------------------------------------------
def print_ocv(r: OCVResult) -> None:
    if r.ocv_v is None:
        _box(
            "OPEN-CIRCUIT VOLTAGE RESULT",
            ["NOT COMPUTED — requires adsorption energy"],
        )
        return

    v = r.ocv_v
    if v > 2.0:
        verdict = "HIGH — may cause electrolyte decomposition"
    elif v > 1.0:
        verdict = "MODERATE — check electrolyte window"
    elif v > 0.0:
        verdict = "LOW — suitable for anode (no dendrite risk)"
    else:
        verdict = "NEGATIVE — thermodynamically unfavorable"

    _box(
        "OPEN-CIRCUIT VOLTAGE RESULT",
        [
            f"OCV                 : {v:.4f} V  (vs {r.ion_symbol}⁺/{r.ion_symbol})",
            f"Ion charge (z)      : {r.z_charge}",
            f"n_ions              : {r.n_ions}",
            f"Assessment          : {verdict}",
            "",
            "OCV = −E_ads / (n × z × e)  [eV → V]",
        ],
    )


# ---------------------------------------------------------------------------
# Diffusion Barrier
# ---------------------------------------------------------------------------
def print_diffusion_barrier(r: DiffusionBarrierResult) -> None:
    if not r.available or r.barrier_ev is None:
        _box(
            "DIFFUSION BARRIER RESULT",
            [
                "NOT AVAILABLE — run a separate CI-NEB calculation.",
                "Provide the result file via --barrier-file <path>.",
            ],
        )
        return

    b = r.barrier_ev
    if b < 0.3:
        verdict = "EXCELLENT — fast K⁺ diffusion"
    elif b < 0.5:
        verdict = "GOOD — acceptable kinetics"
    elif b < 0.8:
        verdict = "MODERATE — may limit rate capability"
    else:
        verdict = "HIGH — slow K⁺ transport expected"

    _box(
        "DIFFUSION BARRIER RESULT",
        [
            f"Barrier Ea          : {b:.4f} eV",
            f"Source              : {r.source_file}",
            f"Assessment          : {verdict}",
            "",
            "From pre-computed CI-NEB or NEB barrier file.",
        ],
    )


# ---------------------------------------------------------------------------
# Density of States
# ---------------------------------------------------------------------------
def print_dos(r: DOSResult) -> None:
    if not r.parsed:
        _box(
            "DENSITY OF STATES RESULT",
            ["PDOS not parsed — run with --dos to extract DOS."],
        )
        return

    fermi = (
        f"{r.fermi_ev:.4f} eV" if r.fermi_ev is not None else "N/A"
    )
    lines = [
        f"PDOS files found    : {r.n_pdos_files}",
        f"Fermi energy        : {fermi}",
    ]
    lines += [f"  {p}" for p in r.pdos_files[:8]]
    if len(r.pdos_files) > 8:
        lines.append(f"  ... and {len(r.pdos_files) - 8} more")
    _box("DENSITY OF STATES RESULT", lines)


# ---------------------------------------------------------------------------
# Convex Hull
# ---------------------------------------------------------------------------
def print_convex_hull(r: ConvexHullResult) -> None:
    if r.e_hull_ev_per_atom is None:
        _box(
            "CONVEX HULL RESULT",
            [
                "NOT COMPUTED — formation energy required.",
                "Run formation energy first (or provide competing phases).",
            ],
        )
        return

    eh = r.e_hull_ev_per_atom
    refs_note = (
        f"Using {r.n_competing_phases} competing phase(s)"
        if r.refs_used
        else "Single-compound approx (no competing-phase DB)"
    )

    _box(
        "CONVEX HULL RESULT",
        [
            f"eHull above hull    : {eh:.4f} eV/atom",
            f"E_form/atom used    : {r.e_form_per_atom_ev:.4f} eV/atom",
            f"Hull reference      : {r.hull_ref_formula}",
            f"Stability           : {r.stability_label}",
            f"References          : {refs_note}",
            "",
            "eHull ≤ 0      → stable;  ≤ 0.05 eV/atom → metastable",
            "eHull > 0.10 eV/atom → likely decomposes spontaneously.",
        ],
    )


# ---------------------------------------------------------------------------
# Bader Charge Analysis
# ---------------------------------------------------------------------------
def print_bader(r: BaderResult) -> None:
    if not r.parsed:
        _box(
            "BADER CHARGE ANALYSIS RESULT",
            [
                f"NOT COMPUTED — no charge data found for {r.ion_symbol}.",
                "Add &PRINT → &HIRSHFELD T &END to CP2K input, or",
                "provide CUBE files and ensure 'bader' exe is in PATH.",
            ],
        )
        return

    ion_q = r.ion_charge_e
    if ion_q is None:
        verdict = "UNKNOWN"
    elif ion_q >= 0.8:
        verdict = "EXCELLENT — near-complete electron transfer"
    elif ion_q >= 0.5:
        verdict = "GOOD — substantial charge transfer"
    elif ion_q >= 0.2:
        verdict = "MODERATE — partial charge transfer"
    else:
        verdict = "WEAK — metallic/covalent character"

    ion_str   = f"{ion_q:.4f} e"  if ion_q is not None  else "N/A"
    delta_str = (
        f"{r.ion_charge_transfer:.4f} e"
        if r.ion_charge_transfer is not None else "N/A"
    )
    frame_str = (
        f"{r.framework_avg_charge:.4f} e"
        if r.framework_avg_charge is not None else "N/A"
    )

    _box(
        "BADER CHARGE ANALYSIS RESULT",
        [
            f"Method              : {r.method}",
            f"Ion ({r.ion_symbol}) Bader q    : {ion_str}",
            f"Charge transfer Δq  : {delta_str}  (electrons donated)",
            f"Framework avg charge: {frame_str}",
            f"n_ions              : {r.n_ions}",
            f"Assessment          : {verdict}",
            f"Elapsed             : {r.elapsed_min:.1f} min",
            "",
            "Δq ≈ 1 e → ionic bonding (ideal anode); Δq ≈ 0 → metallic.",
        ],
    )
