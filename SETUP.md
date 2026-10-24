# Setup Guide: MOF DFT Screening Pipeline

This guide explains how to set up the environment and run the MOF K⁺-anode DFT screening pipeline (`mof_screen`) on a new system.

## 1. Prerequisites

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed on your new system.

## 2. Environment Setup

We recommend using `conda` to manage the environment because it handles both Python packages and the `cp2k` binary gracefully.

1. **Create the environment and install dependencies**  
   Run the following command to create a new environment named `dft_env`, installing Python 3.11, CP2K (from conda-forge), and necessary Python packages:

   ```bash
   conda create -n dft_env -c conda-forge python=3.11 "cp2k=*=mpi_openmpi_*" ase matplotlib scipy numpy
   ```

2. **Activate the environment**

   ```bash
   conda activate dft_env
   ```

*(Note: The pipeline uses the `cp2k.popt` command. The conda-forge `cp2k` package provides this executable automatically.)*

## 3. Running the Pipeline

Ensure that you have your MOF `.cif` file (e.g., `UiO-66.cif`) in your working directory along with the `mof_screen` code.

### Running the Pipeline sequentially or by individual parameters

The pipeline calculates several parameters. Because many parameters depend on the results of earlier calculations, they should be run in a specific **sequential order** if you are running them individually.

**Sequential Dependency Order:**
1. **Adsorption (`adsorption`)**: The most expensive DFT step. It performs a geometry optimization of the MOF with the inserted K⁺ ion. **This must be run first.**
2. **Volume (`volume`)**: Instant calculation, reads the optimized cell volume from the `adsorption` output.
3. **OCV (`ocv`)**: Instant calculation, derives the Open Circuit Voltage using the `adsorption` energy.
4. **Formation Energy (`formation`)**: Instant calculation, reuses the DFT energy from `adsorption`.
5. **Convex Hull (`convex_hull`)**: Instant calculation, requires the `formation` energy per atom.
6. **Bader Charges (`bader`)**: Parses the charge data from the `adsorption` output.
7. **Density of States (`dos`)**: Parses the `.pdos` files generated during the `adsorption` run.
8. **Diffusion (`diffusion`)**: Runs the CI-NEB calculation to find the diffusion barrier. Can take days depending on the system.
9. **Ionic Conductivity (`ionic_conductivity`)**: Requires both the `adsorption` energy and the `diffusion` barrier to calculate via the Nernst-Einstein equation.

---

#### Commands for Individual Parameters

If you wish to run these steps one by one, follow this sequence:

**1. Adsorption (Run this first)**
```bash
python3 -m mof_screen UiO-66.cif --only adsorption --ion K --cores 28
```

**2. Volume, OCV, Formation, Bader, and DOS (Run after Adsorption)**
```bash
python3 -m mof_screen UiO-66.cif --only volume
python3 -m mof_screen UiO-66.cif --only ocv
python3 -m mof_screen UiO-66.cif --only formation
python3 -m mof_screen UiO-66.cif --only bader
python3 -m mof_screen UiO-66.cif --only dos
```

**3. Convex Hull (Run after Formation)**
```bash
python3 -m mof_screen UiO-66.cif --only convex_hull
```

**4. Diffusion (NEB Barrier)**
Requires a directory of NEB images (e.g., `neb/`).
```bash
python3 -m mof_screen UiO-66.cif --only diffusion --neb-images neb --ion K --cores 28 --nproc-rep 1
```

**5. Ionic Conductivity (Run after Adsorption and Diffusion)**
```bash
python3 -m mof_screen UiO-66.cif --only ionic_conductivity
```

### Live Monitoring of the NEB calculation

If you run the NEB calculation in the background, or if you want a dedicated terminal window to monitor the CI-NEB progress in real-time, you can use the standalone watch script:

```bash
conda activate dft_env
python3 watch_neb.py results/UiO-66/neb
```

This will display a live-updating dashboard showing the current BAND step, SCF iterations, barrier estimate, and convergence status.
