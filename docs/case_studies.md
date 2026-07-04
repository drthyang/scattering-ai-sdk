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

## Data required for the full test suite

CI runs green with **no external data** — every data-dependent test skips
cleanly. To exercise the full suite (the three data-gated case studies) locally,
drop these gitignored facility datasets at these paths:

| Case | Path (glob) | File | Known answer to reproduce |
|------|-------------|------|----------------------------|
| Inverted `.gr` | `data/1d/curves/*GaTa4Se8*SQ.gr` | `Neutron_NOM_9999_GaTa4Se8_at_5K_50_clean_SQ.gr` | `pdf` pack flags the non-standard/inverted G(r) |
| Phase transition | `data/1d/series/*tth.dat*.dat` | GaNb4Se8 2θ T-series (20 files, 5–99 K) | `data` pack reports a transition (30–50 K) |
| Diffuse anisotropy | `data/3d/volumes/*.nxs` | CORELLI TbTi3Bi4 volume | `diffuse` pack flags ~4.5× sampling anisotropy |

Everything else — 285 tests including the whole growth loop, all packs on
synthetic fixtures, and the two committed RMC case studies — needs no external
data. See each `data/**/INVENTORY.md` for the exact axes/units/known-answer of
files already dropped locally.

### Future builds (E8 watch queue — data not yet arrived)

The self-improvement watch queue (`scattering-ai learn watch`) flags these the
moment matching data lands in `data/`:

| Build | Data needed | Filename hint (globs matched) |
|-------|-------------|-------------------------------|
| `ins` domain pack (S(Q,ω), phonons) | inelastic-neutron S(Q,ω) map(s) | `*sqw*`, `*inelastic*`, `*s_q_e*`, `ins/**` |
| T3 Spinvert-style spin refinement | measured **magnetic** diffuse I(Q) reference | `*magnetic_diffuse*`, `*spinvert*`, `*_mag_iq*` |
| Symmetry rep-theory (irreps, MAXMAGN) | representation-theory tables (not a `data/` drop) | — external tables |
