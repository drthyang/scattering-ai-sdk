# 3D volumes

Full reciprocal-space volumes (or 3D-ΔPDF volumes), typically HDF5/NeXus.

One volume is enough to start. In `INVENTORY.md`, note:

- axes and their ranges (e.g. H, K, L extents and step sizes)
- which slices you routinely take from it (plane, position, thickness,
  orientation) — these define the slicing tool's requirements
- the file's internal layout if non-obvious (dataset paths inside the HDF5)
