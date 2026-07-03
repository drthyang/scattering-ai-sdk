# Inventory

| File(s) | Axes & units | What it is | What to extract / known answer |
|---------|--------------|------------|--------------------------------|
| `TbTi3Bi4_22K_mmm_(0,k,l)_[h,0,0]_...401x401x301_mmm_cc.nxs` | Mantid MDHistoWorkspace NeXus; signal stored (D2,D1,D0) = (301, 401, 401); bin edges: D0 [h00] −12…12 (401 bins), D1 −30…30 (401 bins), D2 −5…5 (301 bins); HKL units; oriented lattice + unit cell included | Neutron diffuse scattering volume, CORELLI (SNS), TbTi3Bi4 at 22 K, mmm-symmetrized, cross-correlation elastic | Slice at given axis/position/thickness → 2D maps (e.g. (0,k,l) plane); mask/NaN handling; later: Bragg/diffuse feature extraction on slices |

Notes: 2D slices are generated from this volume (user workflow), so `data/2d/`
stays empty for now — the slicing tool produces its own test slices.
