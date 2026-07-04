# Case studies (roadmap D5)

Reproducible **known answers on real data** — the evidence base for the
publication and end-to-end integration checks of the packs. Each case pins a
required claim on the report text (never exact wording).

Run them:

```bash
scattering-ai case-studies           # human-readable
scattering-ai case-studies --json    # machine-readable
```

Two RMC cases run on committed demo data (`examples/rmc_monitor_demo/`), so they
execute in CI. The rest point at datasets under `data/` that are gitignored
(facility data): they **skip cleanly** when the data is absent and run when a
researcher has it locally. A skipped case is a case whose data has not landed
yet — cf. the E8 watch queue — not a failure. A case never *fails* unless the
known answer regresses.

| Case | Data | Known answer (must contain) | Publication angle |
|------|------|------------------------------|-------------------|
| `rmc_local_vs_average_conflict` | committed `stalled_run.json` | `conflict` (Bragg ↑ vs PDF ↓) | Local-vs-average conflict detection |
| `rmc_healthy_convergence` | committed `healthy_run.json` | `decreasing`, no `conflict` | Convergence monitoring baseline |
| `pdf_inverted_neutron_gr` | `data/1d/curves/*GaTa4Se8*SQ.gr` | `invert` | PDF QC — inverted-`.gr` sign gotcha |
| `phase_transition_gaNb4Se8` | `data/1d/series/*tth.dat*.dat` | `transition` | Phase-transition detection from a T-series |
| `diffuse_anisotropy_corelli` | `data/3d/volumes/*.nxs` | `anisotrop` | 3D diffuse feature explanation |

New cases go in `src/scattering_ai/evaluation/case_studies.py`. When a
correction pins a new known answer, add it here and as a regression case
(`tests/regressions/`) via the growth loop, so the publication's evidence and
the test suite grow from the same source.
