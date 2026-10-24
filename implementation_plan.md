44# MOF DFT Screening — Refactor for K⁺ Anode with Ionic Conductivity

## Background

The current `_core.py` is a monolithic 1860-line file. The mentor requires:

1. **Remove bandgap** → replace with **ionic conductivity**
2. **Each parameter in its own file** — true module separation
3. **CIF-only input** for MOF structures
4. **K⁺ anode focus** (default ion = K)
5. **Each parameter must be runnable independently** — no forced sequential execution
6. **Production-grade logic** — dependency resolution, caching, validation, error recovery

---

## Question 1: Can You Run Any Parameter Individually?

**YES — this is a core design goal.** Every calc module will:

- Accept a **CIF path** + **results directory** and run on its own
- Load cached outputs from disk if a prior calc already ran
- Print its result box immediately when done
- Be callable directly from CLI with `--only <param>`

### Dependency Graph

```
CIF file
    │
    ├─► [adsorption]  ──── needs: CP2K GEO_OPT of MOF+K
    │       │
    │       ├─► [volume]          (instant — reads adsorption.out)
    │       ├─► [ocv]             (instant — E_ads ÷ z)
    │       └─► [ionic_conductivity]  (needs E_ads + optionally diffusion barrier)
    │
    ├─► [formation]   ──── needs: E(MOF) from adsorption.out OR a separate SP run
    │
    ├─► [diffusion]   ──── needs: --barrier-file (pre-computed NEB, fully standalone)
    │
    └─► [dos]         ──── needs: .pdos files from a prior ENERGY run
```

### Standalone Rules per Parameter

| Parameter | Standalone? | What it needs | How it gets it |
|-----------|-------------|---------------|----------------|
| `adsorption` | ✅ Yes | CIF + CP2K | Runs GEO_OPT directly |
| `volume` | ✅ Yes | `adsorption.out` | Loads from results dir |
| `ocv` | ✅ Yes | `adsorption.out` | Loads E_ads from cache |
| `formation` | ✅ Yes | `adsorption.out` or `sp.out` + refs | Loads E(MOF) from cache |
| `diffusion` | ✅ Yes | `--barrier-file` | Reads user-supplied NEB file |
| `dos` | ✅ Yes | `*.pdos` files | Reads from results dir |
| `ionic_conductivity` | ✅ Yes | E_ads + optionally barrier | Loads from cache + barrier |

> [!IMPORTANT]
> **How caching works**: Each calc writes its result to `results/<cif_stem>/<param>.json`.
> Any subsequent calc that needs a prior result reads from that JSON first — it does NOT re-run the DFT.
> This is the production-grade **checkpoint/resume** pattern.

### CLI Examples (Standalone)

```bash
# Run everything (full pipeline)
python -m mof_screen MOF1.cif --ion K --cores 16

# Run ONLY adsorption energy (skips all others)
python -m mof_screen MOF1.cif --only adsorption --ion K --cores 16

# Run ONLY OCV — reads E_ads from cached adsorption.json
python -m mof_screen MOF1.cif --only ocv

# Run ONLY diffusion barrier from an NEB file
python -m mof_screen MOF1.cif --only diffusion --barrier-file neb_result.txt

# Run ONLY ionic conductivity (needs adsorption + barrier cached)
python -m mof_screen MOF1.cif --only ionic_conductivity --barrier-file neb_result.txt

# Run formation + OCV together, skip rest
python -m mof_screen MOF1.cif --only formation ocv

# Batch: rank multiple MOFs
python -m mof_screen MOF1.cif MOF2.cif MOF3.cif --ion K --rank
```

---

## Question 2: Is the Logic Production-Grade?

**The original `_core.py` is research-grade, not production-grade.** Here is what changes:

### Production-Grade Features Added

| Feature | Old `_core.py` | New Design |
|---------|---------------|------------|
| **Module separation** | 1 file | 16 focused files, each independently importable |
| **Dependency caching** | Re-runs all calcs every time | Reads `<param>.json` cache; skips if already done |
| **Selective execution** | Always runs everything | `--only` flag for any subset |
| **Structured state** | JSON only at end | Each calc saves its own JSON immediately on finish |
| **Error isolation** | One failure stops pipeline | Each calc catches its own errors; pipeline continues |
| **Input validation** | Basic checks | Validate CIF, cell volume, bond lengths, ion compatibility |
| **Dry-run mode** | None | `--dry-run` prints plan without running CP2K |
| **Batch ranking** | None | `--rank` compares multiple MOFs with weighted score |
| **Logging** | Single logger | Per-module logger + per-CIF log file |
| **Type safety** | Basic typing | Full `dataclass` + `Optional` + return type annotations |
| **Ionic conductivity** | Missing entirely | Nernst-Einstein formula with fallback estimation |

### Ionic Conductivity Formula (New Calc 1)

Full Nernst-Einstein computation when barrier is available:

$$\sigma = \frac{n \cdot z^2 \cdot e^2 \cdot D_0 \cdot \exp(-E_a / k_B T)}{k_B T}$$

- $E_a$ = diffusion barrier (eV) from NEB file
- $n$ = carrier density = $N_{ions} / V_{cell}$ (ions/Å³)
- $z$ = ion charge (1 for K⁺)
- $D_0$ = 10⁻⁵ cm²/s (standard pre-exponential for MOF channels)
- $T$ = 300 K (configurable via `--temperature`)

When barrier is **not** available → semi-quantitative rank from E_ads:

| E_ads (eV) | Conductivity Rank |
|------------|-------------------|
| < −2.0 | POOR (too strongly bound, ion immobile) |
| −2.0 to −1.0 | MODERATE |
| −1.0 to −0.3 | GOOD |
| −0.3 to  0.0 | EXCELLENT (weak binding = fast hopping) |
| > 0.0 | UNFAVORABLE |

---

## Module Separation (One File Per Parameter)

```
c:\Users\ssanj\OneDrive\Documents\HUB\DFT\
├── mof_screen/
│   ├── __init__.py              ← Public API: run_screening(), __version__
│   ├── constants.py             ← HARTREE_TO_EV, HUBBARD_U, IONIC_RADII, ION_CHARGE, etc.
│   ├── dataclasses_.py          ← All @dataclass result types + AllResults + StateCache
│   ├── cp2k_input.py            ← generate_cp2k_input(), _subsys_block(), _kpoints_block()
│   ├── cp2k_runner.py           ← run_cp2k(), find_cp2k(), ProgressMonitor
│   ├── structure.py             ← find_void_sites(), build_loaded_structure(), validate_cif()
│   ├── parsers.py               ← extract_total_energy(), extract_final_volume(), check_scf_converged()
│   ├── printer.py               ← _box() shared pretty-printer
│   ├── references.py            ← compute_elemental_reference(), load_reference_energies()
│   ├── state.py                 ← StateCache: save/load per-calc JSON, dependency resolution
│   ├── calc_adsorption.py       ← calc_adsorption_energy() — DFT GEO_OPT, standalone ✅
│   ├── calc_ionic_conductivity.py ← calc_ionic_conductivity() — NEW replaces bandgap, standalone ✅
│   ├── calc_formation.py        ← calc_formation_energy() — instant, standalone ✅
│   ├── calc_volume.py           ← calc_volume_expansion() — instant, standalone ✅
│   ├── calc_ocv.py              ← calc_ocv() — instant, standalone ✅
│   ├── calc_diffusion.py        ← calc_diffusion_barrier() — reads NEB file, standalone ✅
│   ├── calc_dos.py              ← calc_dos() — reads .pdos files, standalone ✅
│   ├── workflow.py              ← run_screening() — DAG orchestrator with --only support
│   └── cli.py                   ← argparse CLI, --only, --rank, --dry-run, --temperature
└── _core.py                     ← original backup, untouched
```

---

## Key New Module: `state.py` (Production-Grade Caching)

```python
# state.py — production checkpoint/resume
class StateCache:
    def save(self, param: str, result: dataclass) -> None
        # writes results/<cif_stem>/<param>.json immediately
    def load(self, param: str) -> Optional[dict]
        # reads cached result; returns None if not yet computed
    def is_done(self, param: str) -> bool
        # True if <param>.json exists and is valid
    def clear(self, param: str) -> None
        # deletes cache for one param (to force re-run)
```

CLI flag `--force-rerun adsorption` clears only that param's cache and re-computes it.

---

## Verification Plan

### Automated Tests
```bash
python -c "from mof_screen import run_screening; print('Import OK')"
python -m mof_screen sample.cif --dry-run --ion K
python -m mof_screen sample.cif --only diffusion --barrier-file barrier.txt
python -m mof_screen sample.cif --only ocv    # reads cached adsorption result
```

### Manual Verification
- Full pipeline: `python -m mof_screen mof.cif --ion K --cores 16`
- Results dir has one `.json` per parameter
- `--only ocv` after full run completes instantly from cache
- `summary.json` contains `ionic_conductivity` key, NOT `bandgap`
- Batch rank: `python -m mof_screen *.cif --rank` produces ranked table
