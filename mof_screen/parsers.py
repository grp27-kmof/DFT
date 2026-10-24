"""
mof_screen/parsers.py
=====================
CP2K output file parsers.

Each function is self-contained and reads only what it needs from
the CP2K text output.  No side-effects, no global state.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from mof_screen.constants import HARTREE_TO_EV

# ---------------------------------------------------------------------------
# Pre-compiled regexes (module-level for performance)
# ---------------------------------------------------------------------------
_ENERGY_RE = re.compile(
    r"ENERGY\|\s+Total FORCE_EVAL.*?:\s*(-?[\d.]+(?:[Ee][+-]?\d+)?)"
)
# Fallback: CP2K writes this line even on unconverged runs (IGNORE_CONVERGENCE_FAILURE)
_ENERGY_COMPONENT_RE = re.compile(
    r"Total energy:\s*(-?[\d.]+(?:[Ee][+-]?\d+)?)"
)
_VOLUME_RE = re.compile(
    r"CELL\| Volume.*?([\d.]+(?:[Ee][+-]?\d+)?)\s*$"
)
_MO_LINE_RE = re.compile(
    r"^\s*MO\|\s+\d+\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s*$"
)
_EV_SUFFIX_RE = re.compile(
    r"(-?[\d.]+(?:[Ee][+-]?\d+)?)\s+eV\s*$"
)


# ---------------------------------------------------------------------------
# Total DFT energy
# ---------------------------------------------------------------------------
def extract_total_energy(out_file: str) -> Optional[float]:
    """
    Extract the LAST total DFT energy (eV) from a CP2K output file.
    Handles both fixed-point and scientific notation.

    Primary:  'ENERGY| Total FORCE_EVAL' line (only written on converged runs).
    Fallback: 'Total Energy ::' in the energy-components block, which CP2K
              writes even when IGNORE_CONVERGENCE_FAILURE TRUE is set.

    Returns None if the file does not exist or contains no energy line.
    """
    if not os.path.exists(out_file):
        return None
    e_au: Optional[float] = None
    e_au_fallback: Optional[float] = None
    try:
        with open(out_file) as fh:
            for line in fh:
                if "ENERGY| Total FORCE_EVAL" in line:
                    m = _ENERGY_RE.search(line)
                    if m:
                        e_au = float(m.group(1))
                elif "Total energy:" in line:
                    m = _ENERGY_COMPONENT_RE.search(line)
                    if m:
                        e_au_fallback = float(m.group(1))
    except OSError:
        return None
    # Prefer the converged-run summary line; fall back to energy-components line
    best = e_au if e_au is not None else e_au_fallback
    return None if best is None else best * HARTREE_TO_EV


# ---------------------------------------------------------------------------
# SCF convergence check
# ---------------------------------------------------------------------------
def check_scf_converged(out_file: str) -> bool:
    """Return True if the CP2K output contains 'SCF run converged'."""
    if not os.path.exists(out_file):
        return False
    try:
        with open(out_file, errors="replace") as fh:
            return any("SCF run converged" in line for line in fh)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Geometry optimisation completion check
# ---------------------------------------------------------------------------
def check_geo_opt_completed(out_file: str) -> bool:
    """
    Return True only if a GEO_OPT or CELL_OPT run fully completed (converged
    or reached max steps). An incomplete / killed run returns False so that
    the calculation is re-run rather than using partial results.
    """
    if not os.path.exists(out_file):
        return False
    markers = (
        "GEOMETRY OPTIMIZATION COMPLETED",
        "CELL OPTIMIZATION COMPLETED",
        "OPTIMIZATION COMPLETED",   # older CP2K versions
        "Maximum number of optimization steps reached",
    )
    try:
        with open(out_file, errors="replace") as fh:
            content = fh.read()
        return any(m in content for m in markers)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Final unit-cell volume
# ---------------------------------------------------------------------------
def extract_final_volume(out_file: str) -> Optional[float]:
    """
    Extract the LAST 'CELL| Volume' line from a CP2K output (Å³).
    Returns None if not found.
    """
    if not os.path.exists(out_file):
        return None
    vol: Optional[float] = None
    try:
        with open(out_file) as fh:
            for line in fh:
                if "CELL| Volume" in line:
                    m = _VOLUME_RE.search(line.strip())
                    if m:
                        vol = float(m.group(1))
    except OSError:
        return None
    return vol


# ---------------------------------------------------------------------------
# SCF / GEO progress counters (used by the progress dashboard)
# ---------------------------------------------------------------------------
def parse_cp2k_progress(out_file: str) -> dict:
    """
    Read a CP2K output file and return live progress counters:

        {
          "scf_steps": int,      # latest SCF step number
          "outer_scf": int,      # outer SCF cycle counter
          "geo_steps": int,      # GEO_OPT/CELL_OPT step counter
          "converged": bool,
          "energy":   float|None,  # eV
          "last_force": float|None,
        }
    """
    data = {
        "scf_steps": 0,
        "outer_scf": 0,
        "geo_steps": 0,
        "converged": False,
        "energy":    None,
        "last_force": None,
    }
    if not os.path.exists(out_file):
        return data
    try:
        with open(out_file) as fh:
            for line in fh:
                # SCF step counter
                if re.search(r"Step number\s+\d+", line):
                    m = re.search(r"Step number\s+(\d+)", line)
                    if m:
                        data["scf_steps"] = int(m.group(1))

                # Outer SCF cycle
                if "outer SCF" in line and "converged" not in line.lower():
                    m = re.search(r"outer SCF\s+(\d+)", line)
                    if m:
                        data["outer_scf"] = int(m.group(1))

                # GEO_OPT step
                if "STEP|" in line:
                    m = re.search(r"STEP\|\s+(\d+)", line)
                    if m:
                        data["geo_steps"] = int(m.group(1))

                # Convergence flag
                if "SCF run converged" in line:
                    data["converged"] = True

                # Latest energy (Hartree → eV)
                if "ENERGY| Total FORCE_EVAL" in line:
                    m = _ENERGY_RE.search(line)
                    if m:
                        data["energy"] = float(m.group(1)) * HARTREE_TO_EV

                # Max gradient / step (GEO convergence indicator)
                if "Max. gradient" in line or "Max. step" in line:
                    m = re.search(r"([\d.]+(?:[Ee][+-]?\d+)?)\s*$", line.strip())
                    if m:
                        data["last_force"] = float(m.group(1))
    except OSError:
        pass
    return data


# ---------------------------------------------------------------------------
# NEB / BAND run live progress  (used by cp2k_runner for CI-NEB jobs)
# ---------------------------------------------------------------------------
from typing import Any

def parse_neb_progress(out_file: str, ener_file: Optional[str] = None) -> dict:
    """
    Read live NEB BAND progress from a CP2K output file and the .ener log.

    Returns a dict with:
        band_step    : int   – current BAND optimisation step number
        scf_steps    : int   – SCF iterations inside current BAND step (per replica)
        max_force    : float – max NEB force across all replicas (Hartree/Bohr)
        barrier_ev   : float – current barrier estimate (eV), or None
        n_replicas   : int   – number of NEB replicas detected
        elapsed_s    : float – wall time parsed from output, or 0.0
        converged    : bool  – True if BAND convergence reached
    """
    HARTREE_EV = 27.211386245988

    data: dict[str, Any] = {
        "band_step":  0,
        "scf_steps":  0,
        "max_force":  None,
        "barrier_ev": None,
        "n_replicas": 0,
        "elapsed_s":  0.0,
        "converged":  False,
    }

    # ---- 1. Parse the .out file ----------------------------------------
    if os.path.exists(out_file):
        try:
            with open(out_file, errors="replace") as fh:
                content = fh.read()
        except OSError:
            content = ""

        # BAND step counter — CP2K prints "STEP NUMBER = X" in each BAND block
        for m in re.finditer(r"STEP NUMBER\s*=\s*(\d+)", content):
            data["band_step"] = int(m.group(1))

        # SCF step counter inside current replica
        for m in re.finditer(r"Step number\s+(\d+)", content):
            data["scf_steps"] = int(m.group(1))

        # Convergence marker
        if "BAND CONVERGED" in content or "Band calculation converged" in content:
            data["converged"] = True

        # Number of replicas from header
        m = re.search(r"NUMBER OF NEB REPLICA\s*=\s*(\d+)", content)
        if m:
            data["n_replicas"] = int(m.group(1))

        # Current barrier from ENERGIES [au] block (multi-line)
        _energy_pat = (
            r"ENERGIES\s*\[au\]\s*=\s*"
            r"([-\d.]+(?:\s+[-\d.]+)*(?:\n[ \t]+[-\d.]+(?:\s+[-\d.]+)*)*)"
        )
        best_evs = None
        for em in re.finditer(_energy_pat, content, re.IGNORECASE):
            try:
                evs = [float(x) for x in em.group(1).split()]
                if len(evs) >= 3 and all(e < -10.0 for e in evs):
                    best_evs = evs
            except ValueError:
                pass
        if best_evs is not None:
            ref_au = min(best_evs[0], best_evs[-1])
            data["barrier_ev"] = (max(best_evs) - ref_au) * HARTREE_EV
            if data["n_replicas"] == 0:
                data["n_replicas"] = len(best_evs)

    # ---- 2. Supplement with .ener file (more reliable step count) ------
    # The .ener file accumulates across restarts (step index resets to 0 each restart).
    # Total steps = number of valid lines (not max step index).
    # Columns: step  E_1 ... E_R  d_12 d_23 ... d_(R-1,R)
    # We use inter-replica distance spread as convergence proxy since CP2K BAND
    # does NOT write force norms to the .out file.
    _ener = ener_file or (str(Path(out_file).parent / "neb_band-1.ener"))
    if os.path.exists(_ener):
        try:
            lines = [ln.split() for ln in open(_ener).read().splitlines() if ln.strip()]
        except OSError:
            lines = []
        if lines:
            total_cols = len(lines[0])
            n_rep = total_cols // 2
            n_rep = max(n_rep, data["n_replicas"])
            best_barrier = None
            valid_steps = 0
            last_dists: Optional[list] = None
            for cols in lines:
                if len(cols) < 2 * n_rep:
                    continue
                try:
                    int(cols[0])
                    energies = [float(cols[1 + j]) for j in range(n_rep)]
                    # distances between adjacent replicas (columns after energies)
                    dists = [float(cols[1 + n_rep + j]) for j in range(n_rep - 1)]
                except (ValueError, IndexError):
                    continue
                if not all(e < -10.0 for e in energies):
                    continue
                ref_au = min(energies[0], energies[-1])
                best_barrier = (max(energies) - ref_au) * HARTREE_EV
                last_dists = dists
                valid_steps += 1

            # Total steps = number of valid lines across all restarts
            data["band_step"] = valid_steps
            if best_barrier is not None:
                data["barrier_ev"] = best_barrier
            if n_rep > 0:
                data["n_replicas"] = n_rep
            # Distance spread: std-dev of inter-replica distances as geometry proxy
            if last_dists:
                avg = sum(last_dists) / len(last_dists)
                spread = (sum((d - avg) ** 2 for d in last_dists) / len(last_dists)) ** 0.5
                data["max_force"] = spread   # repurpose max_force field for display

    return data
