# CP2K Thresholds for K⁺-Anode MOF Screening

Below are all the convergence threshold parameters used in the DFT pipeline for evaluating MOFs as K⁺ battery anodes. 

## 1. Adsorption, Volume, Formation, and OCV
These parameters rely on the main geometry optimization (`GEO_OPT`) calculation. The thresholds change depending on whether you are running in **Production/High-Accuracy** mode or **Screening** mode.

### Production / High-Accuracy Mode (Default)
Used for highly accurate, publication-quality results.
- **`EPS_SCF`**: `1.0E-7` (Very tight convergence for the Self-Consistent Field wavefunction).
- **`MAX_SCF`**: `150` (Hard cap on inner SCF iterations).
- **Outer SCF MAX_SCF**: `50` (Outer loop iterations for OT minimizer).
- **`MAX_FORCE`**: `0.0005 Ha/Bohr` (Strict geometry optimization convergence).
- **`MAX_DR`**: `0.002 Bohr` (Strict step-size constraint).
- **`MAX_GEO`**: `300` (Maximum number of geometry optimization steps).
- **Plane-Wave Cutoff**: Varies dynamically based on the basis set (usually >600 Ry for DZVP/TZV2P).

### Fast Screening Mode (`--screening`)
Used for rapidly scanning many MOFs to find the top candidates (40-60% faster).
- **`EPS_SCF`**: `1.0E-6` (Looser SCF convergence).
- **`MAX_SCF`**: `150` (Hard cap).
- **`MAX_FORCE`**: `0.001 Ha/Bohr` (Looser geometry optimization convergence).
- **`MAX_GEO`**: `150` (Hard cap on geometry steps to prevent endless struggling).
- **Plane-Wave Cutoff**: `400 Ry` (Hardcoded limit for speed).

---

## 2. Diffusion Barrier (CI-NEB)
The CI-NEB (Nudged Elastic Band) calculation has its own specific convergence thresholds for the elastic band optimization.
- **`MAX_FORCE`**: Usually `0.002 Ha/Bohr` (Determines when the forces perpendicular to the diffusion path have converged).
- **`RMS_FORCE`**: Usually `0.005 Ha/Bohr` (Root mean square force across the band).
- **`BAND_TYPE`**: `CI-NEB` (Climbing-Image NEB ensures the highest-energy image precisely locates the transition state).
- **`EPS_SCF`**: Inherits the `1.0E-7` tight threshold from the production defaults since accurate forces are critical for NEB.

---

## 3. Ionic Conductivity
- **Temperature**: `300 K` (Default parameter used in the Nernst-Einstein equation to calculate ionic conductivity from the NEB barrier, adjustable via `--temperature`).

---

## 4. Density of States (DOS) & Bader Charges
- These properties do not run separate optimizations. They are parsed from the wavefunctions and electron density cubes generated during the successfully converged **Adsorption** step, so they implicitly rely on the Production/Screening thresholds listed in Section 1.
