# Changelog

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
