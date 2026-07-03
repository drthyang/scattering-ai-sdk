# 2D slices

Single 2D maps: reciprocal-space slices (e.g. HK0 planes), detector images,
or real-space maps (e.g. 3D-ΔPDF slices).

Ideal starter set:

- one slice containing the features you mentioned: **Bragg peaks, diffuse
  signal, an Al (sample-environment) ring, realistic noise/background**
- one slice where you'd want **line cuts** taken (note where the interesting
  cut directions are)

Formats: `.npy`/`.npz`, HDF5/NeXus, ASCII grid, or image formats — anything.
In `INVENTORY.md`, note the axes (e.g. "H along x, K along y, L = 0 ± 0.1"),
units, and any known artifacts (masked regions, beamstop, gaps).
