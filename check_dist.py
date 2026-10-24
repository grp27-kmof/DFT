import numpy as np
from ase.io import read
from pathlib import Path

neb_dir = Path("/home/ubuntu/Documents/dft-d/results/UiO-66/neb")
if not neb_dir.exists():
    neb_dir = Path("/home/ubuntu/Documents/dft-d/neb")

if not neb_dir.exists():
    print("Could not find neb dir")
else:
    images = sorted(neb_dir.glob("image0*.xyz"))
    if not images:
        images = sorted(neb_dir.glob("*.xyz"))
        
    print(f"Found {len(images)} images")
    if len(images) > 1:
        initial = read(images[0])
        final = read(images[-1])
        
        pos_i = initial.positions[-1]
        pos_f = final.positions[-1]
        print(f"Initial K pos: {pos_i}")
        print(f"Final K pos:   {pos_f}")
        
        dist = np.linalg.norm(pos_f - pos_i)
        print(f"Raw Cartesian distance: {dist:.2f} Angstroms")
        
        cell = initial.get_cell()
        if np.linalg.norm(cell) > 0:
            diff = pos_f - pos_i
            diff_wrapped = diff - np.round(diff @ np.linalg.inv(cell)) @ cell
            print(f"PBC Wrapped distance: {np.linalg.norm(diff_wrapped):.2f} Angstroms")
        else:
            print("No cell found in XYZ to do PBC wrapping check.")
