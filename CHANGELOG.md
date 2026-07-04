# Changelog

## Unreleased

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
