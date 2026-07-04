# Inventory

| File(s) | Axes & units | What it is | What to extract / known answer |
|---------|--------------|------------|--------------------------------|
| `GaNb4Se8_*_T_base_*K_..._tth.dat_bgsub_...dat` (20 files, 5.0–99.1 K in ~5 K steps) | col 1 = 2θ-like scattering angle (0.009–16.3, 4096 pts), col 2 = background-subtracted intensity; **masked points = −3.0 sentinel** | Synchrotron total-scattering patterns of GaNb4Se8 (lacunar spinel) vs temperature; DetZ 4461 mm; used for PDF generation | Track Bragg peaks vs T; structural (Jahn-Teller-type) transition expected in the 30–50 K range. SDK finding (2026-07-03): peak-center changepoints at 36.5–51.7 K on all four strong peaks (@3.666, 4.494, 4.764, 5.186), improvement ratios 3.4–21.7 — pinned in `tests/test_series.py` |
| `GaNb4Se8_*_T_base_*K_..._q.dat_bgsub_scale_0.28_shift_110.gr` (21 files, 5.8–104.2 K in ~5 K steps) | pdfgetx format: INI header + `#### start data`; col 1 = r (Å, 0–100, step 0.01, 10001 pts), col 2 = G (1/Å²) | PDF G(r) companions to the Bragg series above — real-space local structure vs temperature. diffpy.pdfgetx 2.1.1, `mode = xray`, composition Ga Nb4 Se8, qmin 0.8, qmax 27, rpoly 0.8; DetZ 3674 mm (from the `q.dat` reduction, not the `tth.dat` set) | Track first-shell peak positions/amplitudes vs T; the local-structure signature of the same 30–50 K transition seen in the Bragg data. Same-run cross-check: does the PDF changepoint coincide with the Bragg one? |

Notes: the `.gr` series (21 pts, 5.8–104.2 K) and the `tth.dat` Bragg series (20
pts, 5.0–99.1 K) are separate reductions of the same experiment (`q.dat` at DetZ
3674 mm vs `tth.dat` at DetZ 4461 mm) — the temperature grids differ slightly,
so pair by nearest T rather than by index. One label quirk: the `base_5.8K` file
records `inputfile = ...base_2.8K...` in its header (a 3 K mismatch); all other
20 files' filename and header temperatures agree exactly.
