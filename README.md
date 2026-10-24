# MOF K+-Anode DFT Screening — Reference Guide

> **Project**: Predict the best MOF as anode material for a K+ battery
> **Tool**: CP2K (PBE+D3 / DZVP-MOLOPT-SR-GTH)
> **Input**: CIF files of relaxed MOF structures
> **Output**: Per-parameter JSON results + live progress dashboard

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Setup and Requirements](#2-setup-and-requirements)
3. [How to Run — Quick Start](#3-how-to-run--quick-start)
4. [Run Individual Parameters (Standalone)](#4-run-individual-parameters-standalone)
5. [All CLI Flags](#5-all-cli-flags)
6. [Progress Dashboard](#6-progress-dashboard)
7. [Parameters Explained](#7-parameters-explained)
8. [Output Files](#8-output-files)
9. [Caching and Restart System](#9-caching-and-restart-system)
10. [Batch Mode — Rank Multiple MOFs](#10-batch-mode--rank-multiple-mofs)
11. [Common Workflows](#11-common-workflows)

---

## 1. Project Structure

```
DFT/
├── _core.py                        <- original monolithic file (backup, untouched)
├── README.md                       <- this file
└── mof_screen/                     <- NEW modular package
    ├── __init__.py                 # Public API
    ├── __main__.py                 # Enables: python -m mof_screen
    │
    ├── constants.py                # Physical constants, lookup tables
    ├── dataclasses_.py             # All result dataclasses + CalcStatus enum
    ├── state.py                    # JSON checkpoint / cache system
    ├── progress.py                 # Live per-parameter progress dashboard
    ├── printer.py                  # Pretty-print result boxes
    ├── parsers.py                  # CP2K output file parsers
    ├── structure.py                # CIF validation, void detection, ion insertion
    ├── cp2k_input.py               # CP2K input file generator
    ├── cp2k_runner.py              # CP2K launcher (feeds progress dashboard)
    ├── references.py               # Elemental reference energy system
    │
    ├── calc_adsorption.py          # PARAMETER 1: K+ adsorption energy (DFT GEO_OPT)
    ├── calc_ionic_conductivity.py  # PARAMETER 2: Ionic conductivity sigma S/cm
    ├── calc_formation.py           # PARAMETER 3: Formation energy (instant)
    ├── calc_volume.py              # PARAMETER 4: Volume expansion % (instant)
    ├── calc_ocv.py                 # PARAMETER 5: Open-circuit voltage (instant)
    ├── calc_diffusion.py           # PARAMETER 6: Diffusion barrier from NEB file
    ├── calc_dos.py                 # PARAMETER 7: Density of states (parses .pdos)
    │
    ├── workflow.py                 # DAG orchestrator (runs all or --only subset)
    └── cli.py                      # Full argparse CLI
```

---

## 2. Setup and Requirements

### Activate conda environment

```bash
conda activate dft_env
```

### Required Python packages

```bash
pip install ase numpy
```

### CP2K binary

The tool searches PATH automatically in this order:
`cp2k.psmp` → `cp2k.ssmp` → `cp2k.popt` → `cp2k`

Set the data directory (already set in workflow.py):

```bash
export CP2K_DATA_DIR=/home/srsarvanthikha/miniconda/envs/dft_env/share/cp2k/data
```

### Verify setup

```bash
# Check CP2K is found
which cp2k.psmp

# Check the package imports correctly
python -c "from mof_screen import run_screening; print('Import OK')"
```

---

## 3. How to Run — Quick Start

All commands run from the DFT/ directory.

### Full pipeline (all parameters)

```bash
# Standard run — K+ ion, 16 cores
python -m mof_screen MOF1.cif --ion K --cores 16

# With cell relaxation during adsorption
python -m mof_screen MOF1.cif --ion K --cores 16 --cell-opt

# With pre-computed diffusion barrier (enables full Nernst-Einstein sigma)
python -m mof_screen MOF1.cif --ion K --cores 16 --barrier-file neb_barrier.txt

# High-accuracy basis (publication quality, 3-5x slower)
python -m mof_screen MOF1.cif --ion K --cores 32 --high-accuracy

# Fast screening mode (lower cutoffs, quick estimate)
python -m mof_screen MOF1.cif --ion K --cores 16 --fast
```

### Dry run (see the plan without running DFT)

```bash
python -m mof_screen MOF1.cif --dry-run --ion K
```

---

## 4. Run Individual Parameters (Standalone)

Every parameter can be run alone without running the others.
If a prior result (e.g. adsorption.json) exists in the cache it is
loaded automatically — no DFT re-run.

### Dependency table

| Parameter Name | Type | Depends on |
|----------------|------|-----------|
| adsorption | DFT GEO_OPT | Nothing — primary calc |
| formation | Instant | adsorption (E(MOF) from output) |
| volume | Instant | adsorption (reads adsorption.out) |
| ocv | Instant | adsorption (E_ads value) |
| ionic_conductivity | Derived | adsorption + optionally diffusion |
| diffusion | Instant | NEB file (fully standalone) |
| dos | Instant | .pdos files (fully standalone) |

### Examples

```bash
# Run ONLY adsorption energy
python -m mof_screen MOF1.cif --only adsorption --cores 16

# Run ONLY OCV (reads E_ads from adsorption.json cache, instant)
python -m mof_screen MOF1.cif --only ocv

# Run ONLY formation energy
python -m mof_screen MOF1.cif --only formation

# Run ONLY volume expansion
python -m mof_screen MOF1.cif --only volume

# Run ONLY ionic conductivity with a barrier file
python -m mof_screen MOF1.cif --only ionic_conductivity --barrier-file neb.txt

# Run ONLY diffusion barrier (reads NEB file, no DFT needed)
python -m mof_screen MOF1.cif --only diffusion --barrier-file neb_barrier.txt

# Run ONLY DOS (reads .pdos files from results dir)
python -m mof_screen MOF1.cif --only dos --dos

# Run formation + OCV + ionic conductivity together
python -m mof_screen MOF1.cif --only formation ocv ionic_conductivity
```

### Force re-run (clear cache for one param)

```bash
# Re-run only adsorption (ignore cached result)
python -m mof_screen MOF1.cif --force-rerun adsorption --cores 16

# Re-run OCV and formation (keep everything else cached)
python -m mof_screen MOF1.cif --force-rerun ocv formation
```

---

## 5. All CLI Flags

```
python -m mof_screen [CIF_FILES ...] [OPTIONS]
```

### Execution Control

| Flag | Description |
|------|-------------|
| --only PARAM [PARAM ...] | Run only the listed parameters |
| --force-rerun PARAM [PARAM ...] | Clear cache and re-run specific params |
| --dry-run | Print execution plan, no DFT runs |
| --rank | Batch mode: score and rank all input CIFs |

### DFT Settings

| Flag | Default | Description |
|------|---------|-------------|
| --cores N  /  -n N | 16 | OMP_NUM_THREADS per process |
| --mpi-ranks N | 1 | Number of MPI ranks |
| --high-accuracy | off | TZV2P basis (publication quality, 3-5x slower) |
| --screening | off | ALL 5 speed-ups: 40-60% faster, E_ads +-0.05-0.10 eV |
| --fast | off | 400 Ry cutoff only (minor speedup per step) |

### Ion Settings

| Flag | Default | Description |
|------|---------|-------------|
| --ion SYMBOL | K | Ion to insert: Li Na K Mg Ca Zn Al |
| --n-ions N | 1 | Number of ions per unit cell |
| --cell-opt | off | Relax cell vectors during adsorption |
| --multiplicity N | auto | Spin multiplicity override |
| --temperature T | 300 | Temperature K for Nernst-Einstein sigma |

### References

| Flag | Description |
|------|-------------|
| --compute-refs | Run CP2K to compute self-consistent elemental references |
| --ref-energies FILE | JSON file with pre-computed elemental ref energies |
| --ion-ref-ev EV | Manual bulk K reference energy eV/atom |

### Optional Inputs

| Flag | Description |
|------|-------------|
| --barrier-file FILE | Path to NEB barrier file (plain eV or CP2K NEB output) |
| --dos | Parse CP2K PDOS files for density of states |
| --cp2k-data-dir PATH | Override CP2K_DATA_DIR |

---

## 6. Progress Dashboard

When running, the terminal shows a live dashboard updated every second:

```
Parameters      Status     Progress          Elapsed   ETA
adsorption      RUNNING    [████████░░] 52%  45m12s    41m00s
formation       PENDING    [░░░░░░░░░░]  0%    --        --
volume          PENDING    [░░░░░░░░░░]  0%    --        --
ocv             PENDING    [░░░░░░░░░░]  0%    --        --
ionic_cond      PENDING    [░░░░░░░░░░]  0%    --        --
diffusion       SKIPPED    [──────────]  --    --        --
dos             SKIPPED    [──────────]  --    --        --
Overall         [████░░░░░░░░░░░░░░░░] 16%   45m12s   ~236m
```

### Status meanings

| Status | Meaning |
|--------|---------|
| PENDING | Not yet started |
| RUNNING | Currently executing |
| DONE | Completed successfully |
| CACHED | Loaded from prior run JSON |
| SKIPPED | Not requested |
| FAILED | Error occurred — check run.log |

### How % is estimated

- GEO_OPT: percent = geo_step / max_geo_steps * 100
- SCF within each GEO step adds a fractional sub-increment
- Instant calcs (formation, OCV, volume): jump to 100% immediately
- ETA calculated once > 2% complete

---

## 7. Parameters Explained

### adsorption — K+ Adsorption Energy (eV)

```
E_ads = E(MOF+K) - E(MOF) - n_ions * E(K_bulk per atom)
```

- Run type: GEO_OPT (DFT, most expensive step)
- Negative = K+ binds to MOF (good anode)
- -0.3 to -2.0 eV is the ideal range

### ionic_conductivity — sigma (S/cm) — REPLACES BANDGAP

Full Nernst-Einstein (when --barrier-file is provided):

```
D   = D0 * exp(-Ea / kB * T)       Arrhenius diffusion
n   = N_ions / V_cell              carrier density cm^-3
sigma = n * z^2 * e^2 * D / (kB * T)   S/cm
```

Estimate from E_ads (when no barrier file):

| E_ads range | Conductivity rank |
|-------------|-------------------|
| < -2.0 eV | POOR (ion trapped) |
| -2.0 to -1.0 eV | MODERATE |
| -1.0 to -0.3 eV | GOOD |
| -0.3 to 0.0 eV | EXCELLENT |
| > 0.0 eV | UNFAVORABLE |

### formation — Formation Energy (eV/atom)

```
E_form = E(MOF) - sum_i [ n_i * E(element_i per atom) ]
```

- Instant — reuses E(MOF) from adsorption.out
- < 0 eV/atom = thermodynamically stable

### volume — Volume Expansion (%)

```
dV% = (V_after - V_before) / V_before * 100
```

- < 5% = excellent for cycle stability

### ocv — Open-Circuit Voltage (V)

```
OCV = -E_ads / (n_ions * z * e)
```

- 0–1 V = suitable for anode

### diffusion — Diffusion Barrier (eV)

- Read from pre-computed CI-NEB or NEB output file
- < 0.3 eV = excellent K+ mobility

### dos — Density of States

- Parses .pdos files produced automatically during GEO_OPT
- Extracts Fermi energy

---

## 8. Output Files

After running, a results/MOF_name/ directory is created:

```
results/MOF1/
├── adsorption.inp          # CP2K input for GEO_OPT
├── adsorption.out          # CP2K output (raw DFT output)
├── adsorption.out.stderr   # CP2K stderr (check for errors)
├── mof_with_k.cif          # MOF + inserted K+ structure
├── mof_k-pos-1.xyz         # GEO_OPT trajectory
│
├── adsorption.json         <- per-param cache
├── formation.json          <- per-param cache
├── volume.json             <- per-param cache
├── ocv.json                <- per-param cache
├── ionic_conductivity.json <- per-param cache
├── diffusion.json          <- per-param cache (if --barrier-file given)
├── dos.json                <- per-param cache (if --dos given)
│
├── summary.json            <- complete results all params
└── run.log                 <- full timestamped log
```

### Reading results in Python

```python
import json

with open("results/MOF1/summary.json") as f:
    data = json.load(f)

print(data["ionic_conductivity"]["sigma_s_cm"])    # sigma in S/cm
print(data["ionic_conductivity"]["rank"])          # EXCELLENT/GOOD/etc
print(data["adsorption"]["e_ads_ev"])              # E_ads in eV
print(data["ocv"]["ocv_v"])                        # OCV in V
print(data["formation"]["e_form_per_atom_ev"])     # eV/atom
print(data["volume"]["expansion_pct"])             # percent
print(data["diffusion_barrier"]["barrier_ev"])     # eV
```

---

## 9. Caching and Restart System

Each parameter writes its result to its own JSON file immediately when done.

```
results/MOF1/adsorption.json   <- written the moment adsorption finishes
results/MOF1/ocv.json          <- written the moment OCV finishes
```

### Benefits

- Kill the job at any time — re-run and it picks up where it stopped
- Run OCV alone after a full run — reads adsorption.json, returns in < 1 second
- Change just one param — use --force-rerun to clear only that JSON

### Cache logic flow

```
python -m mof_screen MOF1.cif --only ocv
                              |
                 state.py: does ocv.json exist?
                   YES  -> load and print instantly (< 1 sec)
                   NO   -> check dependencies: adsorption.json exists?
                             YES -> compute OCV from E_ads (instant)
                             NO  -> report missing dependency
```

---

## 10. Batch Mode — Rank Multiple MOFs

```bash
python -m mof_screen MOF1.cif MOF2.cif MOF3.cif --ion K --cores 16 --rank
```

Prints a ranked comparison table after all MOFs are screened:

```
Rank  MOF           Score   E_ads      E_form/at   OCV     dV%   Conductivity
   1  HKUST-1       0.734  -1.253 eV  -0.421 eV   0.521V  2.1%  GOOD
   2  MIL-101       0.612  -0.876 eV  -0.318 eV   0.743V  4.3%  GOOD
   3  ZIF-8         0.391  -0.412 eV  +0.021 eV   0.350V 12.7%  MODERATE

BEST K+ ANODE CANDIDATE: HKUST-1
```

### Ranking weights

| Metric | Weight | Ideal value |
|--------|--------|-------------|
| E_ads | 30% | -0.3 to -2.0 eV |
| E_form/atom | 20% | Most negative |
| OCV | 20% | 0–1 V |
| Volume expansion | 15% | < 5% |
| Ionic conductivity rank | 15% | EXCELLENT |

---

## 11. Common Workflows

### Workflow A: Standard first screening

```bash
# Fast mode for quick check
python -m mof_screen MOF1.cif --fast --ion K --cores 16

# If promising, run full accuracy
python -m mof_screen MOF1.cif --ion K --cores 16
```

### Workflow B: You have a NEB barrier already

```bash
python -m mof_screen MOF1.cif --ion K --cores 16 --barrier-file neb_K.txt
```

### Workflow C: Check one result from a completed run

```bash
# OCV in < 1 second (from cache)
python -m mof_screen MOF1.cif --only ocv

# Formation energy (from cache)
python -m mof_screen MOF1.cif --only formation
```

### Workflow D: Re-run a failed step

```bash
# GEO_OPT crashed — clear adsorption cache and retry
python -m mof_screen MOF1.cif --force-rerun adsorption --cores 16
```

### Workflow E: Screen 5 MOFs and pick the best

```bash
python -m mof_screen MOF1.cif MOF2.cif MOF3.cif MOF4.cif MOF5.cif \
    --ion K --n-ions 2 --cores 16 --barrier-file neb.txt --rank
```

### Workflow F: Compute elemental references once for all MOFs

```bash
# Compute references (run once per project)
python -m mof_screen MOF1.cif --compute-refs --ion K --cores 16

# All subsequent MOFs reuse cached references
python -m mof_screen MOF2.cif --ion K --cores 16
python -m mof_screen MOF3.cif --ion K --cores 16
```

---

## Quick Reference Card

```
ACTION                                   COMMAND
----------------------------------------------------------------------
Full run                                 python -m mof_screen MOF.cif
Full run + barrier                       python -m mof_screen MOF.cif --barrier-file neb.txt
See plan only (no DFT)                   python -m mof_screen MOF.cif --dry-run
Run ONLY adsorption                      python -m mof_screen MOF.cif --only adsorption
Run ONLY OCV                             python -m mof_screen MOF.cif --only ocv
Run ONLY ionic conductivity              python -m mof_screen MOF.cif --only ionic_conductivity
Run ONLY diffusion barrier               python -m mof_screen MOF.cif --only diffusion --barrier-file neb.txt
Re-run one param (clear cache)           python -m mof_screen MOF.cif --force-rerun adsorption
Rank multiple MOFs                       python -m mof_screen *.cif --rank
Screening mode (40-60% faster)           python -m mof_screen MOF.cif --screening
High accuracy (publication quality)      python -m mof_screen MOF.cif --high-accuracy
Fast screening (quick estimate)          python -m mof_screen MOF.cif --fast
Different temperature                    python -m mof_screen MOF.cif --temperature 350
Multiple ions per cell                   python -m mof_screen MOF.cif --n-ions 4
Read all results in Python               import json; json.load(open("results/MOF/summary.json"))
----------------------------------------------------------------------
```

