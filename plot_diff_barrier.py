import matplotlib.pyplot as plt
import numpy as np

# Load the final NEB energies (adjust filename if necessary)
# CP2K writes the distances in column 2 and energies in column 6 in the .ener file
data = np.loadtxt("neb_band-1.ener")

# Extract the final step data (the last 3 rows for your 3 replicas)
# Assuming pristine state is 0.0 eV, we plot the relative energies
distances = data[-3:, 1] 
energies = data[-3:, 5] * 27.2114  # Convert Hartree to eV if necessary

plt.figure(figsize=(8, 5))
plt.plot(distances, energies, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.title("Potassium Ion Diffusion Barrier on MoS2 (4x4x1)", fontsize=14, fontweight='bold')
plt.xlabel("Reaction Coordinate (Bohr)", fontsize=12)
plt.ylabel("Energy (eV)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Highlight the peak
peak_idx = np.argmax(energies)
plt.annotate(f"{energies[peak_idx]:.4f} eV", 
             (distances[peak_idx], energies[peak_idx]), 
             textcoords="offset points", 
             xytext=(0,10), ha='center', fontweight='bold', color='red')

plt.tight_layout()
plt.savefig("diffusion_barrier.png", dpi=300)
print("Plot saved as diffusion_barrier.png!")