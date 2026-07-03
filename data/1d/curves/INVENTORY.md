# Inventory

| File(s) | Axes & units | What it is | What to extract / known answer |
|---------|--------------|------------|--------------------------------|
| `Neutron_NOM_9999_GaTa4Se8_at_5K_50_clean_SQ.dat` | col 1 = Q (1/Å, 0–45, step 0.02), col 2 = S(Q); `#` comment header | Neutron total scattering S(Q), NOMAD (SNS), GaTa4Se8 at 5 K | Transform to G(r) and match the pdfgetx `.gr` below (same run) |
| `Neutron_NOM_9999_GaTa4Se8_at_5K_50_clean_SQ.gr` | pdfgetx format: INI header + `#### start data`; col 1 = r (Å), col 2 = G (1/Å²) | G(r) produced by diffpy.pdfgetx 2.1.1 from the S(Q) above | Reference answer for the S(Q)→G(r) transform tool |
| `XRAY_FeCoSn_100K_converted.fq` | pdfgetx format; col 1 = Q (1/Å), col 2 = F(Q) (1/Å) | X-ray total scattering F(Q), NSLS-II 28-ID (PDF beamline), FeCoSn at 100 K, Kapton background subtracted | Peak finding/fitting on a real x-ray pattern |
| `XRAY_FeCoSn_100K_converted.gr` | pdfgetx format; col 1 = r (Å, step 0.01), col 2 = G (1/Å²) | G(r) from pdfgetx 2.2.1 (rpoly = 0.6) for the F(Q) above | Fit low-r peaks (bond lengths) with uncertainties |

Notes: pdfgetx INI headers carry reduction provenance (qmin/qmax, background
files, corrections) — the reader preserves them as metadata.

**Open question (2026-07-03):** the NOMAD `.dat` stores S(Q)−1 (high-Q tail →
0), and the SDK transform of it produces a physically standard G(r): negative
−4πρ₀r low-r slope, first peak at 2.52 Å, amplitudes ±3–4 Å⁻². The companion
`.gr` file matches in shape (|corr| = 0.9985) but is **inverted and ~11×
smaller** (ref ≈ −0.09 × SDK result), with a positive low-r region — not a
standard G(r). The x-ray pair shows no such discrepancy (corr +0.998, scale
0.92). Worth checking how that neutron `.gr` was produced.
