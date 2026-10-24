"""
mof_screen/calc_neb.py
=======================
PARAMETER: NEB Diffusion Barrier (eV) - runs a full CP2K CI-NEB (BAND) job.

Triggered via --neb-images <dir>.
Cache file: <results_dir>/neb/neb_band.json

Workflow
--------
1. Discover image XYZ files (image00.xyz...imageN.xyz) in the supplied directory
2. Generate CP2K BAND input
3. Run CP2K  (run_cp2k)
4. Parse "Activation energy" from output -> E_a (eV)
5. Cache result -> feeds into Nernst-Einstein ionic conductivity
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from ase import Atoms

from mof_screen.cp2k_runner import run_cp2k
from mof_screen.dataclasses_ import DiffusionBarrierResult
from mof_screen.printer import print_diffusion_barrier
from mof_screen.progress import ProgressDashboard
from mof_screen.state import StateCache
from mof_screen.structure import detect_transition_metals, estimate_multiplicity

log = logging.getLogger("mof_dft.neb")

_PARAM = "diffusion"

_ACT_E_RE = re.compile(
    r"Activation\s+energy[:\s]+(-?[\d.]+(?:[Ee][+-]?\d+)?)\s*eV",
    re.IGNORECASE,
)


def _discover_images(image_dir: str) -> List[Path]:
    d = Path(image_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"NEB image directory not found: {image_dir}")
    candidates = sorted(
        p for p in d.glob("*.xyz")
        if re.search(r"(image|replica|img|frame)[_\-]?\d+", p.stem, re.IGNORECASE)
        or re.fullmatch(r"\d+", p.stem)
    )
    if not candidates:
        candidates = sorted(p for p in d.glob("*.xyz") if "neb" not in p.stem.lower())
    if len(candidates) < 3:
        raise RuntimeError(
            f"Only {len(candidates)} XYZ files found in {image_dir}. Need >= 3."
        )
    log.info(f"  Found {len(candidates)} NEB images in {image_dir}:")
    for p in candidates:
        log.info(f"    {p.name}")
    return candidates


def _sanitise_xyz(src: Path, dst: Path) -> None:
    """
    Write a plain-XYZ version of *src* to *dst*, keeping only the first 4
    columns per atom line (symbol x y z).  Extended-XYZ files produced by
    ASE/MACE/CHGNet carry extra columns (forces, per-atom energies, etc.) that
    cause CP2K's ``COORD_FILE_FORMAT XYZ`` reader to abort with a parse error.
    The comment line (line 2) is preserved as-is; CP2K ignores it.
    """
    lines = src.read_text().splitlines()
    if len(lines) < 2:
        dst.write_text(src.read_text())
        return
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        dst.write_text(src.read_text())
        return
    out_lines = [lines[0], "Plain XYZ (sanitised for CP2K)"]
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            out_lines.append("  {:4s}  {:>18s}  {:>18s}  {:>18s}".format(
                parts[0], parts[1], parts[2], parts[3]))
        else:
            out_lines.append(line)
    dst.write_text("\n".join(out_lines) + "\n")


def _parse_barrier(out_file: str) -> Optional[float]:
    """Parse K+ diffusion barrier from CP2K NEB output.

    Checks, in order:
    1. The given ``out_file`` for an explicit "Activation energy" summary line.
    2. All ``neb_band*.out`` siblings for the same summary line (covers restarts).
    3. ENERGIES [au] blocks across *all* sibling output files (partial runs).
    4. The ``neb_band-1.ener`` energy log (accumulates all steps across restarts).
    """
    out_path = Path(out_file)
    neb_dir  = out_path.parent

    # ---- Collect all output text across restarts ----
    sibling_outs = sorted(neb_dir.glob("neb_band*.out"))
    all_texts = []
    for p in sibling_outs:
        if p.exists() and p.stat().st_size > 0:
            all_texts.append(p.read_text())

    # 1. Explicit activation energy summary (converged run)
    for text in all_texts:
        m = _ACT_E_RE.search(text)
        if m:
            barrier = float(m.group(1))
            log.info(f"  Activation energy (CP2K summary): {barrier:.4f} eV")
            return barrier

    # 2. ENERGIES [au] blocks across all sibling outputs.
    # CP2K prints the replica energies across *multiple continuation lines* like:
    #   ENERGIES [au] =    -1109.765048    -1109.755036    -1109.700836    ...
    #                      -1109.683180    -1109.753546    -1109.740947
    # The continuation lines start with whitespace-only (no uppercase keyword).
    # We collect the leading line AND all subsequent continuation lines
    # (which start with \s+ followed by a digit or minus sign).
    combined_text = "\n".join(all_texts)
    best_barrier_ev: Optional[float] = None

    for m in re.finditer(
        r"ENERGIES\s*\[au\]\s*=\s*([\-\d.]+(?:\s+[\-\d.]+)*(?:\n[ \t]+[\-\d.]+(?:\s+[\-\d.]+)*)*)",
        combined_text,
        re.IGNORECASE,
    ):
        raw = m.group(1)
        try:
            evs = [float(x) for x in raw.split()]
        except ValueError:
            continue
        if len(evs) < 3:
            continue
        # Sanity: all replica energies should be large negatives (< -10 au)
        if not all(e < -10.0 for e in evs):
            continue
        # Physical barrier = max(intermediate energies) - min(endpoint energies)
        # The NEB endpoints are image[0] and image[-1].
        ref_au   = min(evs[0], evs[-1])
        barrier_au = max(evs) - ref_au
        barrier_ev = barrier_au * 27.211386245988
        # Always overwrite to keep the barrier from the latest (most converged) NEB step
        best_barrier_ev = barrier_ev

    if best_barrier_ev is not None:
        log.info(
            f"  Barrier (ENERGIES blocks, most recent step): "
            f"{best_barrier_ev:.4f} eV"
        )
        return best_barrier_ev

    # 3. .ener file — accumulates all steps across interrupted restarts.
    # CP2K writes one line per NEB optimisation step.
    # Line format (n_replicas = R):
    #   step  E_1  E_2 ... E_R  d_12  d_23 ... d_(R-1,R)
    # Column count = 1 + R + (R-1) = 2*R.  We auto-detect R from the first line.
    ener_file = neb_dir / "neb_band-1.ener"
    if ener_file.exists() and ener_file.stat().st_size > 0:
        ener_lines = [ln.split() for ln in ener_file.read_text().splitlines() if ln.strip()]
        if ener_lines:
            # Auto-detect n_replicas from first line: 2*R = len(cols) => R = len/2
            first_cols = ener_lines[0]
            total_cols = len(first_cols)
            # total_cols = 1 + R + (R-1) = 2R  => R = total_cols // 2
            n_rep = total_cols // 2
            if n_rep >= 3:
                for cols in ener_lines:
                    if len(cols) < 1 + n_rep:
                        continue   # incomplete / trailing line
                    try:
                        int(cols[0])  # step index
                        energies = [float(cols[1 + j]) for j in range(n_rep)]
                    except (ValueError, IndexError):
                        continue
                    # Sanity: all energies should be large negatives (< -10 au)
                    if not all(e < -10.0 for e in energies):
                        continue
                    ref_au     = min(energies[0], energies[-1])
                    barrier_au = max(energies) - ref_au
                    best_barrier_ev = barrier_au * 27.211386245988

                if best_barrier_ev is not None:
                    log.info(f"  Barrier (.ener accumulated steps): {best_barrier_ev:.4f} eV")
                    return best_barrier_ev

    log.warning("  Could not parse activation energy from output.")
    return None


# Transition metals that cause charge sloshing in OT solver due to
# near-degenerate d-orbitals. These require Diagonalization + Fermi smearing.
_TM_ELEMENTS = frozenset([
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
])


def _has_transition_metals(atoms: "Atoms") -> bool:
    """Return True if atoms contains any 3d/4d/5d transition metal element."""
    return bool(_TM_ELEMENTS & set(atoms.get_chemical_symbols()))


# Ligand atoms that indicate a mixed-valence/metallic TM environment.
# When C and N are present alongside a TM, it suggests CN⁻ bridges (Prussian
# Blue analogues), pyridyl or imidazolate MOFs — these need Broyden+smearing.
_METALLIC_LIGANDS = frozenset(["N", "C"])

# Chalcogenides and halides form wide-gap semiconductors with TMs even when
# DFT+U is applied (MoS2, WS2, TiO2, FeCl2 etc.). Broyden+DFT+U diverges
# catastrophically for these; OT+CG is the correct and stable choice.
_GAPPED_ANIONS = frozenset(["S", "Se", "Te", "O", "F", "Cl", "Br", "I"])


def _needs_diag_solver(atoms: "Atoms") -> bool:
    """
    Return True if the NEB SCF should use Broyden/Diagonalization + Fermi smearing.

    Rules (in priority order):
    1. No TM elements → False  (pure organic/insulator, OT is always fine)
    2. ONLY closed-shell TMs (Zr4+, Ti4+, Hf4+, Zn2+, etc.) → False
       These are fully oxidised, gapped insulators regardless of C/N ligands.
       Examples: UiO-66 (Zr), ZIF-8 (Zn), MIL-125 (Ti).
    3. TM + only chalcogenide/halide anions, no C/N → False
       (semiconducting TM compounds: MoS2, WS2, TiO2, FeCl2)
    4. TM + C or N ligands present (and open-shell TMs like Fe/Co/Ni/Mn/Cu/Cr/V/Mo)
       → True (mixed-valence MOFs: K2FeNi(CN)6, MOF-74, MIL-53-Fe)
    5. Open-shell TM + no anions (pure metal/alloy) → True
    """
    syms = set(atoms.get_chemical_symbols())
    tms  = _TM_ELEMENTS & syms
    if not tms:
        return False   # Rule 1: no TMs

    # Closed-shell (fully oxidised) TMs that are always gapped insulators.
    # These never require Fermi smearing regardless of what ligands they have.
    CLOSED_SHELL_TM = frozenset(["Zr", "Ti", "Hf", "Zn", "Cd", "Al", "Ga", "In", "Sn"])
    open_shell_tms = tms - CLOSED_SHELL_TM
    if not open_shell_tms:
        log.info(
            "    [NEB SCF] Only closed-shell TMs (Zr/Ti/Hf/Zn ...) "
            "→ OT + CG (fully oxidised, gapped insulator)"
        )
        return False   # Rule 2: closed-shell only

    has_gapped_anion = bool(_GAPPED_ANIONS   & syms)
    has_metallic_lig = bool(_METALLIC_LIGANDS & syms)

    if has_gapped_anion and not has_metallic_lig:
        log.info(
            "    [NEB SCF] TM detected but chalcogenide/halide system "
            "→ OT + CG (gapped semiconductor, Broyden+DFT+U would diverge)"
        )
        return False   # Rule 3: semiconductor

    log.info(
        "    [NEB SCF] Open-shell TM ({}) + C/N ligands → Broyden + Fermi smearing "
        "(mixed-valence MOF)".format(", ".join(sorted(open_shell_tms)))
    )
    return True   # Rule 4 & 5


def _generate_neb_input(
    image_files: List[Path],
    atoms: "Atoms",
    metals: Dict[str, float],
    mult: int,
    label: str,
    nproc_per_replica: int,
    k_spring: float,
    use_high_accuracy: bool,
) -> str:
    from mof_screen.constants import (
        BASIS_SETS, DEFAULT_CUTOFF, DEFAULT_REL_CUTOFF, MAX_SCF_STEPS
    )

    basis      = BASIS_SETS["high_accuracy"] if use_high_accuracy else BASIS_SETS["default"]
    cutoff     = DEFAULT_CUTOFF
    rel_cutoff = DEFAULT_REL_CUTOFF
    use_uks    = mult > 1
    n_images   = len(image_files)
    # 0.05 Ha/Bohr is 10× looser than publication-grade (0.005) but gives barriers
    # accurate to ~0.1-0.2 eV — sufficient for battery-material screening comparisons.
    max_force  = 0.05
    rms_force  = max_force * 0.5

    # Choose SCF solver based on system chemistry (see _needs_diag_solver).
    tm_system = _needs_diag_solver(atoms)
    if tm_system:
        max_scf = 200
        eps_scf = 1.0e-5
    else:
        max_scf = MAX_SCF_STEPS
        eps_scf = 1.0e-5

    # Use the bare-MOF Atoms object for cell — the sanitised XYZ files have no
    # cell information so reading cell from them yields a zero matrix.
    cell = atoms.get_cell()

    cell_lines = "\n".join(
        "      {}  {:.10f}  {:.10f}  {:.10f}".format(name, vec[0], vec[1], vec[2])
        for name, vec in zip("ABC", cell)
    )

    # Read the first NEB image to get all element kinds (includes the guest ion K+
    # which is absent from the bare MOF atoms object).
    from ase.io import read as _ase_read
    try:
        first_img = _ase_read(str(image_files[0]))
        img_symbols = set(first_img.get_chemical_symbols())
    except Exception:
        img_symbols = set()
    symbols = sorted(set(atoms.get_chemical_symbols()) | img_symbols)
    kind_lines = []
    for sym in symbols:
        kind_lines += [
            "    &KIND " + sym,
            "      BASIS_SET " + basis,
            "      POTENTIAL GTH-PBE",
        ]
        if sym in metals:
            kind_lines += [
                "      &DFT_PLUS_U",
                "        L 2",
                "        U_MINUS_J [eV] {:.4f}".format(metals[sym]),
                "      &END DFT_PLUS_U",
            ]
        kind_lines.append("    &END KIND")
    kinds_str = "\n".join(kind_lines)

    replicas = "\n".join(
        "    &REPLICA\n      COORD_FILE_NAME {}\n    &END REPLICA".format(p.name)
        for p in image_files
    )

    uks_str    = "TRUE" if use_uks else "FALSE"
    total_mpi  = n_images * nproc_per_replica

    lines = [
        "! CP2K CI-NEB input generated by mof_screen",
        "! Images: {}  k_spring: {}  DFT: PBE-D3/{}/{}Ry".format(n_images, k_spring, basis, cutoff),
        "! UKS: {}  Mult: {}  DFT+U: {}".format(use_uks, mult, bool(metals)),
        "! Run: mpirun -n {} cp2k.popt -i {}.inp -o {}.out".format(total_mpi, label, label),
        "",
        "&GLOBAL",
        "  RUN_TYPE     BAND",
        "  PROJECT_NAME " + label,
        "  PRINT_LEVEL  MEDIUM",
        "&END GLOBAL",
        "",
        "&MOTION",
        "  &BAND",
        "    BAND_TYPE          CI-NEB",
        "    NUMBER_OF_REPLICA  " + str(n_images),
        "    K_SPRING           " + str(k_spring),
        "    NPROC_REP          " + str(nproc_per_replica),
        "    &CONVERGENCE_CONTROL",
        "      MAX_FORCE {:.4f}".format(max_force),
        "      RMS_FORCE {:.4f}".format(rms_force),
        "    &END CONVERGENCE_CONTROL",
        "    &CI_NEB",
        "      NSTEPS_IT 2",
        "    &END CI_NEB",
        "    &OPTIMIZE_BAND",
        "      OPT_TYPE DIIS",
        "      OPTIMIZE_END_POINTS TRUE",
        "      &DIIS",
        "        MAX_STEPS 100",
        "        NO_LS     TRUE",
        "        N_DIIS    7",
        "      &END DIIS",
        "    &END OPTIMIZE_BAND",
        "    &PROGRAM_RUN_INFO",
        "      INITIAL_CONFIGURATION_INFO TRUE",
        "    &END PROGRAM_RUN_INFO",
        "    &REPLICA_INFO",
        "    &END REPLICA_INFO",
        replicas,
        "  &END BAND",
        "  &PRINT",
        "    &RESTART",
        "      BACKUP_COPIES 2",
        "      &EACH",
        "        BAND 5",
        "      &END EACH",
        "    &END RESTART",
        "    &TRAJECTORY",
        "      FORMAT XYZ",
        "      &EACH",
        "        BAND 1",
        "      &END EACH",
        "    &END TRAJECTORY",
        "  &END PRINT",
        "&END MOTION",
        "",
        "&FORCE_EVAL",
        "  METHOD Quickstep",
        "  &DFT",
        "    BASIS_SET_FILE_NAME BASIS_MOLOPT",
        "    POTENTIAL_FILE_NAME GTH_POTENTIALS",
        "    UKS         " + uks_str,
        "    MULTIPLICITY " + str(mult),
        "    CHARGE      0",
        "    &MGRID",
        "      CUTOFF     " + str(cutoff),
        "      REL_CUTOFF " + str(rel_cutoff),
        "      NGRIDS     5",
        "    &END MGRID",
        "    &QS",
        "      METHOD      GPW",
        "      EPS_DEFAULT 1.0E-12",
        "      EPS_PGF_ORB 1.0E-30",
        "    &END QS",
        "    &POISSON",
        "      PERIODIC XYZ",
        "    &END POISSON",
        "    &SCF",
        f"      MAX_SCF   {max_scf}",
        f"      EPS_SCF   {eps_scf:.1E}",
        "      IGNORE_CONVERGENCE_FAILURE TRUE",
        f"      SCF_GUESS {'RESTART' if list(image_files[0].parent.glob('*-RESTART.wfn')) else 'ATOMIC'}",
        # ---------------------------------------------------------------
        # SCF algorithm: auto-selected based on system type
        #   TM system  → Broyden mixing + Diagonalization + Fermi smearing
        #                Required for Fe, Ni, Co etc. — OT diverges (charge sloshing)
        #   Insulating  → OT + Conjugate Gradient (monotonic, no smearing needed)
        # ---------------------------------------------------------------
        *(
            [
                "      ADDED_MOS 50",
                "      &SMEAR",
                "        METHOD FERMI_DIRAC",
                "        ELECTRONIC_TEMPERATURE [K] 300",
                "      &END SMEAR",
                "      &MIXING",
                "        METHOD  BROYDEN_MIXING",
                "        ALPHA   0.1",
                "        BETA    1.5",
                "        NBUFFER 20",
                "      &END MIXING",
                "      &DIAGONALIZATION",
                "        ALGORITHM STANDARD",
                "      &END DIAGONALIZATION",
                "      &OUTER_SCF ON",
                f"        EPS_SCF {eps_scf:.1E}",
                "        MAX_SCF 50",
                "        OPTIMIZER DIIS",
                "      &END OUTER_SCF",
            ] if tm_system else [
                "      &OT ON",
                "        MINIMIZER    CG",
                "        LINESEARCH   2PNT",
                "        PRECONDITIONER FULL_SINGLE_INVERSE",
                "        ENERGY_GAP   0.001",
                "      &END OT",
            ]
        ),
        "      &PRINT",
        "        &RESTART",
        "          BACKUP_COPIES 2",
        "          &EACH",
        "            QS_SCF 20",
        "          &END EACH",
        "        &END RESTART",
        "      &END PRINT",
        "    &END SCF",
        "    &XC",
        "      &XC_FUNCTIONAL PBE",
        "      &END XC_FUNCTIONAL",
        "      &VDW_POTENTIAL",
        "        POTENTIAL_TYPE PAIR_POTENTIAL",
        "        &PAIR_POTENTIAL",
        "          TYPE                  DFTD3",
        "          REFERENCE_FUNCTIONAL  PBE",
        "          PARAMETER_FILE_NAME   dftd3.dat",
        "          R_CUTOFF              15.0",
        "        &END PAIR_POTENTIAL",
        "      &END VDW_POTENTIAL",
        "      &XC_GRID",
        "        XC_DERIV    SPLINE2",
        "        XC_SMOOTH_RHO NN50",
        "      &END XC_GRID",
        "    &END XC",
        "  &END DFT",
        "  &SUBSYS",
        "    &CELL",
        cell_lines,
        "      PERIODIC XYZ",
        "    &END CELL",
        "    &TOPOLOGY",
        "      COORD_FILE_FORMAT XYZ",
        "      COORD_FILE_NAME   " + image_files[0].name,
        "    &END TOPOLOGY",
        kinds_str,
        "  &END SUBSYS",
        "&END FORCE_EVAL",
        "",
    ]
    return "\n".join(lines)


def calc_neb_barrier(
    image_dir: str,
    out_dir: Path,
    atoms: Atoms,
    num_cores: int,
    mpi_ranks: int                         = 1,
    use_high_accuracy: bool                = False,
    user_mult: Optional[int]               = None,
    nproc_per_replica: int                 = 1,  # Must be 1 for cp2k.ssmp (single MPI process)
    k_spring: float                        = 0.05,
    dashboard: Optional[ProgressDashboard] = None,
    cache: Optional[StateCache]            = None,
) -> DiffusionBarrierResult:
    """Run a CP2K CI-NEB (BAND) job from image XYZ files and return E_a."""
    log.info("")
    log.info("=" * 64)
    log.info("  NEB DIFFUSION BARRIER  (CI-NEB via CP2K BAND)")
    log.info("=" * 64)

    result = DiffusionBarrierResult()

    if cache and cache.is_done(_PARAM):
        log.info("  Loading NEB barrier from cache.")
        data = cache.load(_PARAM) or {}
        result.barrier_ev  = data.get("barrier_ev")
        result.source_file = data.get("source_file", "")
        result.available   = data.get("available", False)
        if dashboard:
            dashboard.mark_cached(_PARAM)
        print_diffusion_barrier(result)
        return result

    if dashboard:
        dashboard.mark_running(_PARAM)

    t0 = time.time()

    try:
        image_files = _discover_images(image_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("  " + str(exc))
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_diffusion_barrier(result)
        return result

    n_images  = len(image_files)
    total_mpi = nproc_per_replica * n_images
    neb_dir   = out_dir / "neb"
    neb_dir.mkdir(parents=True, exist_ok=True)

    local_images = []
    for img in image_files:
        dest = neb_dir / img.name
        # Always write a sanitised (plain 4-col XYZ) version so CP2K can read it.
        # Extended-XYZ files from ASE/MACE/CHGNet carry extra force/energy columns
        # that cause CP2K's COORD_FILE_FORMAT XYZ parser to abort.
        _sanitise_xyz(img, dest)
        log.info("  Sanitised {} -> neb/ (plain XYZ)".format(img.name))
        local_images.append(dest)

    from ase.io import read as _ase_read
    try:
        first_img = _ase_read(str(image_files[0]))
        metals = detect_transition_metals(first_img)
        mult   = user_mult if user_mult else estimate_multiplicity(first_img)
    except Exception:
        metals = detect_transition_metals(atoms)
        mult   = user_mult if user_mult else estimate_multiplicity(atoms)
    label  = "neb_band"
    inp_file = str(neb_dir / (label + ".inp"))
    out_file = str(neb_dir / (label + ".out"))

    inp_text = _generate_neb_input(
        local_images, atoms, metals, mult, label,
        nproc_per_replica, k_spring, use_high_accuracy,
    )
    Path(inp_file).write_text(inp_text)
    log.info("  Input written : " + inp_file)
    log.info("  Run command   : mpirun -n {} cp2k.popt -i {}.inp -o {}.out".format(
        total_mpi, label, label))

    if Path(out_file).exists() and Path(out_file).stat().st_size > 0:
        # Only skip the DFT run if the output is TRULY converged.
        # A partial run will have ENERGIES blocks (from which _parse_barrier returns
        # a value), but it is NOT finished — we must NOT skip it.
        # "Truly converged" means the output contains the explicit activation energy
        # summary OR the BAND CONVERGED marker.
        all_out_texts = "".join(
            p.read_text(errors="replace")
            for p in sorted(neb_dir.glob("neb_band*.out"))
            if p.exists() and p.stat().st_size > 0
        )
        is_converged = (
            _ACT_E_RE.search(all_out_texts) is not None
            or "BAND CONVERGED" in all_out_texts
            or "Band calculation converged" in all_out_texts
        )
        if is_converged:
            existing = _parse_barrier(out_file)
            if existing is not None:
                log.info("  Existing converged NEB output found — skipping DFT run.")
                result.barrier_ev  = existing
                result.available   = True
                result.source_file = out_file
                result.elapsed_min = (time.time() - t0) / 60.0
                if cache:
                    cache.save(_PARAM, result)
                if dashboard:
                    dashboard.mark_done(_PARAM)
                print_diffusion_barrier(result)
                return result
        else:
            existing = _parse_barrier(out_file)
            if existing is not None:
                log.info(
                    f"  Prior partial NEB output found (barrier estimate: {existing:.4f} eV). "
                    f"Run NOT converged — will resume from restart file."
                )

    # Find the latest CP2K geometry restart file.
    # CP2K writes:
    #   neb_band-1.restart          — master BAND restart (coordinates for all replicas)
    #   neb_band-1_0.restart, ..._6.restart  — per-replica WFN restarts (not used here)
    #   neb_band-1.restart.bak-1    — backup copies
    # We want the master file only (no underscore before the digit suffix).
    master_restart_glob = list(neb_dir.glob("neb_band-1.restart"))
    if not master_restart_glob:
        # Fallback: any .restart that is NOT a .bak or per-replica copy
        master_restart_glob = [
            p for p in neb_dir.glob("neb_band-1*.restart")
            if ".bak" not in p.name and re.search(r"neb_band-1\.restart$", p.name)
        ]
    master_restart_glob = sorted(master_restart_glob, key=lambda p: p.stat().st_mtime)
    latest_restart = str(master_restart_glob[-1]) if master_restart_glob else None
    if latest_restart:
        log.info(f"  Found prior NEB restart file: {Path(latest_restart).name}. Resuming geometry!")
        # Show how many NEB optimisation steps have been completed so far
        ener_file = neb_dir / "neb_band-1.ener"
        if ener_file.exists():
            n_done = sum(1 for ln in ener_file.read_text().splitlines() if ln.strip())
            log.info(f"  Prior NEB steps completed: {n_done}")

    log.info("  Launching CP2K CI-NEB ({} replicas, {} MPI ranks)...".format(
        n_images, total_mpi))
    ok = run_cp2k(
        inp_file, out_file,
        num_cores  = num_cores,
        mpi_ranks  = total_mpi,
        restart_coord = latest_restart,
        label      = "CI-NEB({}img)".format(n_images),
        param_name = _PARAM,
        dashboard  = dashboard,
        cwd        = str(neb_dir),
    )

    result.elapsed_min = (time.time() - t0) / 60.0
    result.source_file = out_file

    if not ok:
        log.error("  CP2K BAND failed - check neb/neb_band.out")
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_diffusion_barrier(result)
        return result

    barrier = _parse_barrier(out_file)
    if barrier is None:
        log.error("  Could not parse activation energy from neb_band.out.")
        if dashboard:
            dashboard.mark_failed(_PARAM)
        print_diffusion_barrier(result)
        return result

    result.barrier_ev = barrier
    result.available  = True
    log.info("  K+ diffusion barrier : {:.4f} eV".format(barrier))
    log.info("  Elapsed              : {:.1f} min".format(result.elapsed_min))

    if cache:
        cache.save(_PARAM, result)
    if dashboard:
        dashboard.mark_done(_PARAM)

    print_diffusion_barrier(result)
    return result
