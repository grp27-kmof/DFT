# Linux Setup & Run Guide — MOF K+ Anode DFT Screening
# ======================================================
# For: Ubuntu 20.04 / 22.04 / 24.04 (fresh PC or HPC node)
# ======================================================
# Save this file and follow step by step.
# Run all commands in bash terminal unless stated otherwise.

## SECTION 1 — INSTALL MINICONDA (skip if already installed)
##------------------------------------------------------------

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Install silently to ~/miniconda3
bash miniconda.sh -b -p $HOME/miniconda3

# Activate conda in current shell
source $HOME/miniconda3/etc/profile.d/conda.sh

# Make conda available in all future shells
echo 'source $HOME/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc

# Confirm conda works
conda --version
# Expected output: conda 24.x.x  (or similar)


## SECTION 2 — CREATE THE DFT CONDA ENVIRONMENT
##------------------------------------------------

# Create environment with Python 3.11
conda create -n dft_env python=3.11 -y

# Activate it
conda activate dft_env

# Install Python packages needed by mof_screen
pip install ase numpy

# Verify imports work
python -c "import ase, numpy; print('ASE', ase.__version__, '  NumPy', numpy.__version__)"


## SECTION 3 — INSTALL CP2K
##--------------------------
# Option A: from conda-forge (easiest, recommended)
conda activate dft_env
conda install -c conda-forge cp2k -y

# Check it installed correctly
which cp2k.psmp
cp2k.psmp --version
# Expected: CP2K version 2024.x  (or similar)

# Set the data directory (CP2K needs this to find basis sets)
# Find the actual path:
find $HOME/miniconda3 -name "BASIS_MOLOPT" 2>/dev/null | head -1
# Copy the directory from the output above (everything up to /BASIS_MOLOPT)
# Then add it to your bashrc:
export CP2K_DATA_DIR=$(dirname $(find $HOME/miniconda3 -name "BASIS_MOLOPT" 2>/dev/null | head -1))
echo "export CP2K_DATA_DIR=$CP2K_DATA_DIR" >> ~/.bashrc
source ~/.bashrc

# Verify CP2K_DATA_DIR is set
echo $CP2K_DATA_DIR
# Should print something like: /home/user/miniconda3/envs/dft_env/share/cp2k/data

# Option B: from Ubuntu apt (older version, may work)
# sudo apt update && sudo apt install cp2k -y

# Option C: load module on HPC cluster
# module load cp2k/2024.1


## SECTION 4 — COPY THE mof_screen PACKAGE
##------------------------------------------
# Transfer the entire DFT/ folder from your Windows machine to the Linux PC.

# METHOD A: USB / shared drive — copy the DFT folder, then on Linux:
cd ~
mkdir -p projects
# Place DFT/ inside ~/projects/DFT

# METHOD B: SCP from Windows (run in Windows PowerShell)
# scp -r "C:\Users\ssanj\OneDrive\Documents\HUB\DFT" user@linux-ip:~/projects/

# METHOD C: Git (if you push to GitHub first)
# git clone https://github.com/yourname/DFT.git ~/projects/DFT

# Go to the project directory
cd ~/projects/DFT

# Confirm the package is there
ls mof_screen/
# Should list: __init__.py  cli.py  constants.py  workflow.py  etc.


## SECTION 5 — FINAL VERIFICATION (run this once before any DFT job)
##--------------------------------------------------------------------

conda activate dft_env
cd ~/projects/DFT

# 1. Package imports
python -c "from mof_screen import run_screening; print('Import OK')"

# 2. CP2K binary found by the tool
python -c "
from mof_screen.cp2k_runner import find_cp2k
cp2k = find_cp2k()
print('CP2K found at:', cp2k)
"

# 3. CIF validation (put your .cif files in ~/projects/DFT/)
python -c "
from mof_screen.structure import validate_cif
atoms = validate_cif('MOF1.cif')
print('CIF OK:', atoms.get_chemical_formula(), len(atoms), 'atoms')
"

# 4. Dry-run to see the execution plan
python -m mof_screen MOF1.cif --dry-run --ion K


## SECTION 6 — RUN COMMANDS (no NEB/diffusion input needed)
##-----------------------------------------------------------
# NOTE: diffusion and dos are automatically OMITTED in all commands below.
# All other 5 parameters run fine without any extra files.

# Parameters that WILL run:
#   adsorption          (DFT GEO_OPT — the main heavy calc)
#   formation           (instant — derived from adsorption result)
#   volume              (instant — reads adsorption output)
#   ocv                 (instant — derived from adsorption result)
#   ionic_conductivity  (derived — uses E_ads estimate, no barrier needed)

# Parameters OMITTED (no input file needed = just leave them out):
#   diffusion           (needs NEB file — skipped)
#   dos                 (needs --dos flag — not passed = skipped)


## 6A — RUN ALL 5 PARAMETERS IN ONE GO (recommended first run)
##-------------------------------------------------------------

# Standard production run (all 5 params, 16 cores)
python -m mof_screen MOF1.cif \
    --ion K \
    --cores 16

# With more ions per unit cell (e.g. 2 K+ per cell)
python -m mof_screen MOF1.cif \
    --ion K \
    --n-ions 2 \
    --cores 16

# Screening mode — 40-60% faster, good for quickly filtering candidates
# Use this first, then production on the best MOF
python -m mof_screen MOF1.cif \
    --ion K \
    --cores 16 \
    --screening

# See what would run WITHOUT actually starting DFT (dry run)
python -m mof_screen MOF1.cif \
    --ion K \
    --dry-run

# Multiple MOFs in one command + print ranked table at the end
python -m mof_screen MOF1.cif MOF2.cif MOF3.cif \
    --ion K \
    --cores 16 \
    --rank

# All CIF files in the folder at once
python -m mof_screen *.cif \
    --ion K \
    --cores 16 \
    --rank


## 6B — RUN ONLY ADSORPTION (the DFT step, most time consuming)
##--------------------------------------------------------------

python -m mof_screen MOF1.cif \
    --only adsorption \
    --ion K \
    --cores 16

# With cell relaxation (more accurate, ~2x slower)
python -m mof_screen MOF1.cif \
    --only adsorption \
    --ion K \
    --cores 16 \
    --cell-opt

# Screening mode for quick adsorption estimate
python -m mof_screen MOF1.cif \
    --only adsorption \
    --ion K \
    --cores 16 \
    --screening


## 6C — RUN ONLY IONIC CONDUCTIVITY
##------------------------------------
# (reads E_ads from cache, ranks by adsorption strength, instant)

python -m mof_screen MOF1.cif \
    --only ionic_conductivity \
    --ion K

# With a specific temperature (default is 300 K)
python -m mof_screen MOF1.cif \
    --only ionic_conductivity \
    --ion K \
    --temperature 350


## 6D — RUN ONLY OCV (Open-Circuit Voltage)
##-------------------------------------------
# (instant, reads E_ads from adsorption cache)

python -m mof_screen MOF1.cif \
    --only ocv \
    --ion K


## 6E — RUN ONLY FORMATION ENERGY
##----------------------------------
# (instant, reads E(MOF) from adsorption output)

python -m mof_screen MOF1.cif \
    --only formation \
    --ion K


## 6F — RUN ONLY VOLUME EXPANSION
##----------------------------------
# (instant, reads cell volume from adsorption output)

python -m mof_screen MOF1.cif \
    --only volume \
    --ion K


## 6G — RUN MULTIPLE SPECIFIC PARAMETERS TOGETHER
##---------------------------------------------------
# Mix and match any subset

# OCV + ionic conductivity together (both instant after adsorption done)
python -m mof_screen MOF1.cif \
    --only ocv ionic_conductivity \
    --ion K

# formation + volume + ocv (all instant, need adsorption done first)
python -m mof_screen MOF1.cif \
    --only formation volume ocv \
    --ion K

# Everything except adsorption (use when adsorption already finished)
python -m mof_screen MOF1.cif \
    --only formation volume ocv ionic_conductivity \
    --ion K


## 6H — RE-RUN A SPECIFIC PARAMETER (clear its cache and redo)
##--------------------------------------------------------------
# Use this if a step failed or you changed a setting

# Re-run only adsorption
python -m mof_screen MOF1.cif \
    --force-rerun adsorption \
    --ion K \
    --cores 16

# Re-run OCV with different n-ions
python -m mof_screen MOF1.cif \
    --force-rerun ocv \
    --ion K \
    --n-ions 2


## SECTION 7 — RECOMMENDED WORKFLOW ORDER
##-----------------------------------------

# STEP 1: Quick screening of all candidates (fast mode)
python -m mof_screen MOF1.cif MOF2.cif MOF3.cif \
    --screening --ion K --cores 16 --rank

# STEP 2: Pick the best 1-2 MOFs, run full production
python -m mof_screen MOF1.cif \
    --ion K --cores 16

# STEP 3: Check individual results instantly (no DFT re-run)
python -m mof_screen MOF1.cif --only ocv --ion K
python -m mof_screen MOF1.cif --only ionic_conductivity --ion K
python -m mof_screen MOF1.cif --only formation --ion K
python -m mof_screen MOF1.cif --only volume --ion K


## SECTION 8 — RUN IN BACKGROUND (for long adsorption jobs)
##-----------------------------------------------------------

# Run in background and write all output to a log file
nohup python -m mof_screen MOF1.cif \
    --ion K \
    --cores 16 \
    > mof1_run.log 2>&1 &

# Check job status
tail -f mof1_run.log          # live log tail (Ctrl+C to stop watching)
jobs                          # list background jobs
ps aux | grep mof_screen      # check if still running

# On HPC with SLURM scheduler, create job.sh:
cat > job.sh << 'SLURM'
#!/bin/bash
#SBATCH --job-name=mof_dft
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=mof_%j.log

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate dft_env
cd ~/projects/DFT

python -m mof_screen MOF1.cif \
    --ion K \
    --cores 16
SLURM

# Submit the job
sbatch job.sh


## SECTION 9 — READ RESULTS
##--------------------------

# After any run, results are in: results/MOF1/summary.json
cat results/MOF1/summary.json

# Or read in Python
python3 << 'EOF'
import json
with open("results/MOF1/summary.json") as f:
    d = json.load(f)

print("Formula          :", d["formula"])
print("E_ads (eV)       :", d["adsorption"]["e_ads_ev"])
print("OCV (V)          :", d["ocv"]["ocv_v"])
print("E_form/atom (eV) :", d["formation"]["e_form_per_atom_ev"])
print("Volume exp (%)   :", d["volume"]["expansion_pct"])
print("Ionic cond. rank :", d["ionic_conductivity"]["rank"])
print("Sigma (S/cm)     :", d["ionic_conductivity"]["sigma_s_cm"])
EOF

# View raw CP2K output log (for debugging)
less results/MOF1/adsorption.out

# Full run log with timestamps
less results/MOF1/run.log


## SECTION 10 — TROUBLESHOOTING
##--------------------------------

# Problem: "CP2K binary not found"
# Solution: make sure cp2k is in PATH
conda activate dft_env
which cp2k.psmp          # should print a path
# If not found, reinstall:
conda install -c conda-forge cp2k -y

# Problem: "Import error: No module named 'ase'"
# Solution:
conda activate dft_env
pip install ase numpy

# Problem: "No void sites found"
# Solution: your MOF may be too dense for 1 ion — try fewer ions
python -m mof_screen MOF1.cif --only adsorption --ion K --n-ions 1 --cores 16

# Problem: SCF not converging (adsorption fails)
# Solution: use screening mode which has looser settings
python -m mof_screen MOF1.cif \
    --force-rerun adsorption \
    --ion K \
    --cores 16 \
    --screening

# Problem: want to start fresh for a MOF (delete all cached results)
rm -rf results/MOF1/
python -m mof_screen MOF1.cif --ion K --cores 16

# Problem: CP2K_DATA_DIR not found error
echo $CP2K_DATA_DIR
# If empty, set it:
export CP2K_DATA_DIR=$(dirname $(find $HOME/miniconda3 -name "BASIS_MOLOPT" 2>/dev/null | head -1))
echo "export CP2K_DATA_DIR=$CP2K_DATA_DIR" >> ~/.bashrc
source ~/.bashrc


## QUICK REFERENCE CARD
##-----------------------
#
# ACTION                                   COMMAND
# -------------------------------------------------------------------------
# All 5 params (no diffusion/dos)          python -m mof_screen MOF.cif --ion K --cores 16
# Screening mode (40-60% faster)           python -m mof_screen MOF.cif --ion K --cores 16 --screening
# See plan, no DFT run                     python -m mof_screen MOF.cif --dry-run --ion K
#
# Only adsorption (DFT, slow)              python -m mof_screen MOF.cif --only adsorption --ion K --cores 16
# Only OCV (instant)                       python -m mof_screen MOF.cif --only ocv --ion K
# Only formation energy (instant)          python -m mof_screen MOF.cif --only formation --ion K
# Only volume expansion (instant)          python -m mof_screen MOF.cif --only volume --ion K
# Only ionic conductivity (instant)        python -m mof_screen MOF.cif --only ionic_conductivity --ion K
#
# OCV + ionic conductivity together        python -m mof_screen MOF.cif --only ocv ionic_conductivity --ion K
# All instant params (need ads done)       python -m mof_screen MOF.cif --only formation volume ocv ionic_conductivity --ion K
# Re-run one failed param                  python -m mof_screen MOF.cif --force-rerun adsorption --ion K --cores 16
# Rank multiple MOFs                       python -m mof_screen *.cif --ion K --cores 16 --rank
# Run in background                        nohup python -m mof_screen MOF.cif --ion K --cores 16 > run.log 2>&1 &
# Read results                             python3 -c "import json; print(json.load(open('results/MOF1/summary.json')))"
# -------------------------------------------------------------------------
