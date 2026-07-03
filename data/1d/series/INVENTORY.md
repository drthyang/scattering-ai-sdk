# Inventory

| File(s) | Axes & units | What it is | What to extract / known answer |
|---------|--------------|------------|--------------------------------|
| `GaNb4Se8_*_T_base_*K_..._tth.dat_bgsub_...dat` (20 files, 5.0–99.1 K in ~5 K steps) | col 1 = 2θ-like scattering angle (0.009–16.3, 4096 pts), col 2 = background-subtracted intensity; **masked points = −3.0 sentinel** | Synchrotron total-scattering patterns of GaNb4Se8 (lacunar spinel) vs temperature; DetZ 4461 mm; used for PDF generation | Track Bragg peaks vs T; structural (Jahn-Teller-type) transition expected in the 30–50 K range. SDK finding (2026-07-03): peak-center changepoints at 36.5–51.7 K on all four strong peaks (@3.666, 4.494, 4.764, 5.186), improvement ratios 3.4–21.7 — pinned in `tests/test_series.py` |
