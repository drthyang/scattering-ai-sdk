# Changelog

## Unreleased

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
