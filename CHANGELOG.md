# Changelog

## Unreleased

### Structure lookup, magnetic PDF, spin correlations, punch-and-fill
- **`lookup_structures`** (OPTIMADE): query the open crystal-structure
  databases (default COD) by elements/formula to identify candidate phases —
  the SDK's one network-using tool (only the query leaves the machine).
  Live check: Ga+Nb+Se returns GaNb4Se8, F-43m, a = 10.42 Å.
- **`simulate_mpdf_from_mcif`**: the ideal magnetic PDF of an ordered spin
  structure (Frandsen/Billinge form) — negative peaks mark AFM-correlated
  distances. Builds on the mCIF moment parsing.
- **`spin_correlations_from_mcif`**: ⟨Ŝ·Ŝ⟩ per neighbour shell (+1 FM / −1
  AFM / 0 uncorrelated) — the spinvert-style real-space fingerprint.
- **Punch-and-fill for the 3D-ΔPDF**: punched Bragg holes are now backfilled
  by iterative local averaging before the FFT (default on; `fill=false` to
  skip), suppressing the punch-lattice imprint.
- ROADMAP gains a **magnetic diffuse plan** (SpinHarmony-informed): powder
  magnetic I(Q) (Scatty-like), spinvert-style RMC spin refinement, and
  spinteract-style interaction refinement as staged future work, plus the
  `skill_magnetic_diffuse` / `skill_frustration_check` skill candidates.

### PDF model fitting from a structure (community priority #1)
- `simulate_gr_from_cif`: the model G(r) of a crystal structure — supercell
  pair sums with **neutron scattering lengths** (or x-ray Z weighting) and
  Gaussian broadening. Physics check: Ti's negative b makes the Ti–O shell a
  negative dip with neutrons and a positive peak with x-rays.
- `fit_gr_model`: fit a structure model (CIF) to a measured G(r) — scale,
  peak width σ, and a lattice-scale factor — with the **Rw** metric and a
  data/model/difference overlay figure. Self-consistency: exact parameter
  recovery at Rw ≈ 1e-4; a wrong model gives Rw ≈ 1. A comparison fit —
  full refinement (ADPs, occupancies) remains PDFgui/diffpy territory.
- **CIF + G(r) together auto-route to `pdf`** (model-comparison intent; a CIF
  alone still routes to `symmetry`), and the pdf pack reports the fit with the
  overlay figure automatically.

### Multiple phase transitions (user-reported correction)
- GaNb4Se8's true transitions are ~50 K and ~29 K; the old single-changepoint
  detector reported a fictitious 39 K (an average forced between two real
  kinks). `detect_transitions` now fits up to two changepoints (3-segment
  model), and detections from many peaks/observables are **clustered** into
  distinct candidate transitions instead of collapsed into one median. The
  real series now yields ~54.3 K (bracketing 50 K at the 5 K step size) and
  ~39 K as separate candidates.

### Fixes + performance (full live verification pass)
- **Fix:** file entries containing glob characters (`[h,0,0]`, `(0,k,l)` — real
  facility filenames) were mangled by glob expansion, mis-routing e.g. the
  CORELLI volume to the generic domain. An existing literal path now always wins
  over glob interpretation.
- **Performance:** 3D-ΔPDF FFT now uses scipy's multithreaded pocketfft
  (~4× faster end-to-end on the real CORELLI volume) and `punch_bragg`
  estimates its robust threshold from a fixed-seed subsample instead of sorting
  tens of millions of voxels.
- **Verified live on Ollama (gemma4:26b), 5/5 domains** with real data: phase
  transition at 39 K (figures), inverted-G(r) flagged, diffuse volume figure +
  anisotropy, Pm-3m + 5 transition pathways (structure + tree figures), and an
  `.rmc6f` configuration summary — all high confidence, full provenance.

### MCP server — resources + prompts
- The MCP server now exposes, beyond the tools + skills + `analyze`:
  - **Resources**: the curated knowledge base (`knowledge://…`, path-traversal
    guarded) and a `scattering-ai://domains` overview.
  - **Prompts**: `domain_guidance` (a technique's grounding rules) and
    `analyze_files` (a ready-to-send analysis request).
- `analyze` tool's `domain` enum now includes `symmetry`. README gained a
  "Connect from Claude Code / an IDE" MCP section.

### Two new composite skills
- `skill_symmetry_overview` (structure): the full symmetry picture of a CIF in
  one call — space group + Wyckoff, maximal subgroups (transition pathways, with
  a tree figure), pseudosymmetric parent, and reflection conditions.
- `skill_delta_pdf` (new `3D volumes` category): 3D-ΔPDF of a diffuse volume
  (punch → apodize → centred FFT) plus its strongest real-space correlations and
  a central-plane plot.

### New technique tools (symmetry + ported from sibling repos)
- **Symmetry:** `systematic_absences` (symmetry-allowed reflections / extinction
  conditions — a symmetry-filtered Bragg checklist) and `standardize_cell`
  (conventional/primitive setting + transformation matrix). Remaining assessed
  items (klassengleiche subgroups, irrep/symmetry-mode decomposition, k-vector →
  maximal magnetic groups) need external representation-theory tables and are
  documented as future work.
- **3D-ΔPDF** (`delta_pdf` tool, `tools/delta_pdf.py`): the difference-PDF of a
  diffuse volume via the standard punch → apodize → centred-FFT recipe — an
  independent MIT implementation (nebula3d is AGPL, so its source was not
  copied). `Volume3D.load_data()` reads the full array. Validated on the real
  CORELLI volume.
- **RMCProfile files** (`tools/rmc_files.py`, adapted from the MIT rmc-toolkits):
  `read_rmc6f` (average unit cell, supercell, composition, folded positions),
  `read_rmc_csv` (R-value/χ² logs), and `rmc_density_map` (KDE slab of a config,
  revealing split sites / disorder). `.rmc6f` files auto-route to the `rmc`
  domain, which reports the configuration.

### Structure visualization + organized skills
- **CIF/mCIF visualization** (`plot_structure` tool, `skill_visualize_structure`):
  a matplotlib 3D render of the unit cell — element-coloured atoms, bonds, and
  magnetic moment arrows for an mCIF. The CIF reader now parses `_atom_site_moment`
  (mCIF) and transforms moments as axial vectors on symmetry expansion. The
  `symmetry` domain report now includes a structure figure alongside the subgroup
  tree.
- **Skills reorganized by category** into per-module files (`patterns`, `series`,
  `slices`, `structure`) with a `category` tag; `register_skills` aggregates them
  and `ToolRegistry.skills_by_category()` gives a readable overview. Domain packs
  and third parties add skill modules the same way.

### Symmetry domain pack (B4 — third technique pack)
- New `symmetry` domain (spglib-backed, `[symmetry]` extra) for crystal
  structures (CIF), auto-routed from `.cif`/`.mcif`:
  - **`find_symmetry`** — FINDSYM-like: space group (number/symbol/Hall), point
    group, crystal system, and Wyckoff sites at a chosen tolerance.
  - **`subgroup_tree`** — the **maximal subgroups** of the space group: the
    group-subgroup pathways a structural phase transition can take, with index
    and number of domain variants, drawn as a tree figure. Verified against the
    International Tables (Pm-3m → P4/mmm i3×3, R-3m i4×4, P432/P-43m/Pm-3 i2).
  - **`pseudosymmetry_scan`** — relax the tolerance to find a higher-symmetry
    parent phase a distorted structure sits under.
  - **`magnetic_symmetry`** — the magnetic (Shubnikov) space group of an ordered
    magnetic structure from its moments.
  - CIF reader gained `read_structure` (atom-site loop + symmetry-operation
    expansion of the asymmetric unit), a versioned prompt `symmetry_interpret/v1`,
    and a group-subgroup / phase-transition knowledge file.
- Scope: translationengleiche subgroups (cell-multiplying klassengleiche and
  k-vector→maximal-magnetic-group representation analysis are future work).
- The agent now ensures the report workspace exists before packs write figures.

### Docs + entry-point consistency
- New [QUICKSTART.md](QUICKSTART.md): setup, the simplest agent-skills example,
  and advanced usage (LLM, chat, other interfaces, writing your own skill).
  README trimmed with a docs header.
- `analyze(data={"files": [...]})` now expands **globs and directories** just like
  the CLI (shared `core.files.expand_files`), so a folder or `"scans/*.dat"` runs
  a whole series through the Python API.
- Robustness: `auto_mask_value` no longer treats a value that dominates the data
  (flat/degenerate curves) as a mask sentinel, and the series transition scan is
  guarded so an odd curve can never crash `analyze()`.

### Summarizing figures + phase-transition detection
- Every report can now carry **summarizing figures** (`report.figures`, rendered
  in the Markdown report). Domain diagnostics generate a plot that supports the
  conclusion, the agent attaches the paths, and the LLM is told to reference
  them in its interpretation:
  - **data**: detects a temperature/field **series**, tracks the strongest peaks
    across the whole range, reports any **phase transition** (T_c + which peaks
    move), and emits a waterfall + per-peak tracking figure. This is the
    "observe a phase transition" path for a plain scan series (validated on
    GaNb4Se8 → T_c ≈ 39 K).
  - **pdf**: G(r)/S(Q) overview plot with peaks marked.
  - **diffuse**: log-scale intensity map (Bragg peaks marked); for a volume, a
    representative fine-resolution plane is cut automatically.
- Pack prompts bumped (`pdf_interpret/v2`, `diffuse_interpret/v2`,
  `data_analysis/v3`) to reference the figures and, for a series, to read the
  precomputed transition rather than re-deriving it.

### Domain auto-routing — one entry point
- `analyze()` now defaults to `domain="auto"`: the SDK picks the technique pack
  from the input (`domains/router.py`) — RMC run state → `rmc`, a
  reciprocal-space volume/slice → `diffuse`, a G(r)/S(Q)/F(Q) curve → `pdf`,
  anything else → `data`. The resolved pack is on `report.domain`, and an
  offline run explains the routing in its first observation.
- `scattering-ai analyze --file X` needs no `--domain` (and `--question` now
  defaults); the MCP `analyze` tool's `domain` is optional (`auto`). Explicit
  `domain=...` still overrides.
- **LLM loop hardened** (the fix for local models emitting non-JSON): the parser
  strips `<think>` reasoning blocks and code fences and extracts the last
  balanced JSON object; on failure the agent does one corrective retry asking
  for JSON only before falling back. No more silent "interpretation unavailable"
  on the first prose reply.

### Provenance enforcement (D4)
- Reports with an incomplete provenance block are now **rejected by output
  validation** (`ReportValidationError`), not silently returned. `sdk_version`,
  `schema_version`, `input_hash` (sha256), and `timestamp` are always required;
  `model` and `prompt_version` become required once an LLM contributes
  interpretation. `Provenance.missing_fields()` / `AnalysisReport.assert_valid()`
  expose the check. When an LLM runs without an explicit `model_id`, the client
  class name is recorded so the report stays attributable (never fabricated).

### Diffuse-scattering domain pack (B4 — second technique pack)
- New `diffuse` domain: `analyze(domain="diffuse", ...)`. Deterministic,
  cheap diagnostics that route by file type:
  - **Volume (.nxs):** axis ranges / lattice and **anisotropic-sampling**
    detection (broadening along a coarsely-sampled axis is resolution, not the
    sample) — from metadata only, no full-array read.
  - **2D slice (.npz):** Bragg-punch / mask **coverage**, **contaminant powder
    rings** (Al/Cu/steel/V) as sample-environment artifacts, and Bragg-vs-diffuse
    character (sharp-peak count).
  - Versioned prompt `diffuse_interpret/v1` + a diffuse/3D-ΔPDF knowledge file.
    Validated on the real CORELLI TbTi3Bi4 volume (4.5× anisotropy, Cu ring).

### PDF / total-scattering domain pack (B4 — first technique pack)
- New `pdf` domain: `analyze(domain="pdf", ...)` / `scattering-ai analyze
  --domain pdf`. Deterministic diagnostics on G(r)/S(Q)/F(Q) files, encoding
  real-data gotchas:
  - **G(r) baseline slope** — a non-negative low-r slope flags a non-standard
    G(r) (RDF g(r), differential PDF, or sign inversion). Catches the known
    inverted neutron `.gr`.
  - **Low-r artifact** — significant |G(r)| below the first bond (~1 Å) =
    termination ripple or normalization error, not coordination.
  - **First-peak position** — nearest-neighbour distance candidate.
  - **S(Q)-vs-S(Q)−1 convention mismatch** — a file named S(Q) whose high-Q
    tail → 0 stores S(Q)−1 (the NOMAD gotcha), with the right `input_kind`.
  - **Qmax / range** — reports the termination-ripple period.
  - Versioned prompt `pdf_interpret/v1` and a curated PDF-conventions knowledge
    file; validated on the real FeCoSn and GaTa4Se8 data.
- **Architecture (D10):** deterministic next-check rules moved out of
  `core/agent.py` onto the `DomainPack` (`next_check_rules`); the agent merges
  them generically, so a new technique pack needs no core reasoning changes.

### Transition tracing — robustness
- `skill_scan_series_transitions` now *monitors* the strongest peaks across the
  whole scan and reports each peak's center/FWHM/height change plus a plain
  human-readable `summary`, not just a bare verdict.
- Tracking candidates are picked from the **mean over the series**
  (`stack_series`), so a peak present in only one curve (noise) is no longer
  chosen for tracking.
- The masked-region sentinel (e.g. −3.0) is **auto-detected** (`auto_mask_value`,
  `load_series(mask_value="auto")`) from its repetition in the low tail, so the
  transition is found even when the caller forgets to pass it. `inspect_series`
  reports both the detected and the used value.
- Chat prompt `chat/v3`: transition / "what changes with temperature" questions
  are steered to the one skill on the whole file glob, with an explicit
  instruction to report which peaks moved and the changepoint.

## 0.1.0 — 2026-07-03

First working release: the tools-first foundation, built and validated
against real data (NOMAD & NSLS-II 28-ID total scattering, CORELLI diffuse
volume, GaNb4Se8 temperature series).

### Core
- Versioned request/report schemas (Pydantic) with full provenance: input
  hash, model, prompt version, retrieved knowledge, tool-call trace.
- Single-agent loop: deterministic diagnostics → cited knowledge retrieval →
  multi-round LLM tool dispatch → validated report; graceful offline
  (deterministic-only) mode.
- Interactive chat sessions (Milestone 2): persistent history + artifact
  workspace across turns, transcript with tool trace.
- Provider-agnostic LLM layer; OpenAI-compatible client covers LM Studio,
  Ollama, vLLM, and OpenAI via one `base_url`.

### Data tools (18) and skills (3)
- 1D: format-sniffing IO (pdfgetx, NOMAD ASCII, NeXus NXdata), crop,
  error-propagating rebin, unbiased iterative background, peak finding,
  bounded pseudo-Voigt fitting with uncertainties and quality flags,
  S(Q)↔G(r) transforms with convention auto-detection.
- 2D: peak detection, robust azimuthal profiles with annulus completeness,
  powder-ring candidates matched to contaminant d-spacings, line cuts.
- 3D: MDHistoWorkspace loader (hyperslab reads), axis-aligned and oblique
  (HKL basis) slicing; Busing-Levy B matrix for non-orthogonal cells.
- Series: parameter parsing from filenames, peak tracking with per-point
  fits, weighted changepoint transition detection.
- CIF reading (cell/symmetry/formula) with geometric d-spacing prediction.
- Plotting: curves, fits with residual panels, slices, waterfalls, tracking,
  ring profiles — from Python, CLI, agent tools, HTTP, and MCP.
- Skills (composite validated workflows): `skill_characterize_slice`,
  `skill_fit_pattern_peaks`, `skill_scan_series_transitions`.

### Integration surfaces
- Python API (`analyze`, `Agent`, `ChatSession`), CLI (`analyze`, `chat`,
  `plot`, `serve`, `mcp`), FastAPI service, MCP server (any agent host),
  RMC Monitor connector.

### Domain content
- RMC domain pack (run-health diagnostics, curated RMCProfile knowledge),
  scattering knowledge base, evaluation harness pinning required claims,
  grounding (no invented file references), and tool-use honesty (fit claims
  require fit calls) — including live-model behavioral evals.
