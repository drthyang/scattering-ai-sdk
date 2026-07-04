# scattering-ai-sdk — Roadmap

## Project Vision

Build a reusable **AI SDK for scattering science**: a domain-grounded reasoning layer for AI-assisted RMC, PDF/total scattering, diffraction, phonon, and diffuse-scattering workflows.

The goal is not to replace scientific computation. The goal is to add a disciplined reasoning layer that can understand structured scientific outputs, retrieve domain knowledge, call analysis tools, detect workflow problems, and generate trustworthy scientific interpretations — while making clear that final scientific judgment remains with the researcher.

This SDK should become the shared AI layer behind tools such as:

- RMC Monitor
- RMC Phonon Dynamics
- Neutron Diffuse Toolkit
- future PDF, symmetry, phonon, and refinement tools

Core principle:

```text
Scientific codes compute.
Domain tools evaluate.
The LLM reasons, explains, and guides.
```

---

## Design Decisions

Decisions are recorded here so future contributors see *why*, and so any of them can be revisited explicitly rather than drifting. When a decision is superseded, mark it superseded and add the new one — do not delete history.

| # | Decision | Rationale | Status |
|---|----------|-----------|--------|
| D1 | Name: repo `scattering-ai-sdk`, package `scattering_ai`, CLI `scattering-ai` | Focused identity matching the actual domain (scattering techniques); less crowded than "scientific-ai"; better framing for a future publication | Accepted (2026-07) |
| D2 | LLM backend: OpenAI-compatible chat API first | LM Studio, Ollama, vLLM, and OpenAI cloud all speak it — one client covers local and cloud from day one. Anthropic and other native clients come later behind the same protocol | Accepted (2026-07) |
| D3 | Schemas: Pydantic v2 models as single source of truth | Validation, JSON Schema export, versioning, and serialization in one place | Accepted (2026-07) |
| D4 | License: MIT | Maximize adoption in academic and facility settings | Accepted (2026-07) |
| D5 | Python ≥ 3.10 | Modern typing (`X \| Y`, `ParamSpec`) without excluding HPC/facility environments | Accepted (2026-07) |
| D6 | Local-first by design | Scattering data is often unpublished or facility-restricted; the SDK must be fully functional with local models and must never require sending data to a cloud service | Accepted (2026-07) |
| D7 | Vertical slice before framework | Build one useful end-to-end path (RMC run health) before generalizing | Accepted (2026-07) |
| D8 | Tools-first reprioritization | Experience with the interpretation-only RMC prototype showed limited value in commentary without action. Deterministic data-operation tools (1D fit/rebin/transform, 2D feature extraction, 3D slicing) become the center of Track B; the agent's power comes from orchestrating tools, with RAG in a supporting role. Tools are organized by data dimensionality; temperature/field dependence is a series axis over 1D/2D data, not a separate toolkit | Accepted (2026-07) |
| D9 | numpy/scipy in core; h5py optional | Data tools are now the SDK's center, so numeric deps are core; HDF5/NeXus volume support stays an extra (`[volumes]`) for lightweight installs | Accepted (2026-07) |
| D10 | Domain-owned next-check rules | The deterministic "what to check next" rules for a technique are domain content, not core logic; they live on the `DomainPack`, not in `core/agent.py`. Surfaced while building the first technique pack (B4): keeps the "no core reasoning changes per new pack" invariant honest and testable | Accepted (2026-07) |
| D11 | First technique pack = PDF / total scattering | Chosen over phonons/diffuse for the first B4 pack because the tools (S(Q)→G(r), peak fitting, series) and validated real data (FeCoSn, GaTa4Se8) already exist, and it directly feeds the RMC workflow. Encodes the user's known real-data gotchas (inverted-.gr sign, NOMAD S(Q)−1 naming) as deterministic diagnostics | Accepted (2026-07) |
| D12 | Phonons belong to an inelastic-neutron-scattering (INS) domain | Phonon analysis is one capability of INS (S(Q,ω), dispersions, DOS, dynamic structure factor), not a standalone technique. The future pack is `ins` (energy-resolved scattering), with phonon skills inside it — not a `phonons` pack. Keeps the domain axis = measurement technique, consistent with `pdf`/`diffuse` | Accepted (2026-07) |
| D13 | Symmetry pack built on spglib (`[symmetry]` extra) | Crystallography is correctness-critical; spglib is the field-standard, well-tested engine (space groups, Wyckoff, magnetic). Subgroup trees are computed from spglib's operations (general maximal-subgroup enumeration in the primitive setting, typed via `get_spacegroup_type_from_symmetry`), verified against International Tables. Avoids re-implementing crystallographic databases | Accepted (2026-07) |

---

## Target Architecture

```text
Scientific Application
(RMC Monitor / Phonon App / Diffuse Toolkit / any agent host)
        |
        v
Integration Surface
(Python API | CLI | FastAPI service | MCP server)
        |
        v
scattering_ai SDK
        |
        |-- Core Agent Runtime        (orchestrates the loop below)
        |-- Domain Packs              (RMC, PDF, phonons, diffuse, symmetry, ...)
        |     schemas + skills + diagnostics + knowledge + eval cases
        |-- Tool Calling Layer        (read-only by default)
        |-- Knowledge / RAG           (curated, cited)
        |-- Rule-Based Diagnostics    (deterministic, runs before the LLM)
        |-- Report Generator          (Markdown + JSON, with provenance)
        |-- Evaluation Suite          (domain behavior regression tests)
        |
        v
LLM Backend Layer (provider-agnostic)
(OpenAI-compatible: LM Studio / Ollama / vLLM / OpenAI · later: Anthropic, MLX)
```

The SDK accepts structured scientific state from external applications and returns structured, provenance-carrying reasoning outputs.

### Package layout

```text
src/scattering_ai/
├── core/         # agent runtime, schemas, config, context — domain-agnostic
├── llm/          # LLMClient protocol + provider clients
├── domains/      # domain packs (rmc/, pdf/, phonons/, diffuse/, symmetry/)
├── rag/          # indexing, retrieval, citations
├── tools/        # generic tool layer (files, plots, statistics)
├── reports/      # report templates and renderers
└── evaluation/   # eval harness, rubrics, regression cases
```

Placement rule: **if it is specific to one scattering technique, it lives in that technique's domain pack; if two or more domains need it, it moves to core/tools/rag.** New capabilities should start inside a domain pack and be promoted to core only when a second domain actually needs them.

---

## Extensibility Architecture

Flexibility for future implementations is a design requirement, not an afterthought. Four mechanisms carry it:

### 1. Domain packs as plugins

A domain pack is a self-contained bundle:

```text
domain pack = input schema extensions
            + skills
            + deterministic diagnostics
            + knowledge files
            + evaluation cases
```

Built-in packs live under `scattering_ai/domains/`. External packs register via the Python entry point group `scattering_ai.domains`, so third parties can ship e.g. `scattering-ai-qens` or a facility-specific pack without touching core. The core runtime discovers packs at startup and routes requests by the `domain` field.

### 2. Provider-agnostic LLM layer

All model access goes through one protocol:

```python
class LLMClient(Protocol):
    def complete(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse: ...
    @property
    def capabilities(self) -> ModelCapabilities: ...  # tool_use, vision, structured_output, context_window
```

Skills query capability flags and degrade gracefully (e.g., skip plot-image analysis on a text-only local model instead of failing). Switching from a local model to a cloud model is a config change, never a code change.

### 3. Versioned schemas

Every request and report carries a `schema_version`. Within a major version, changes are additive only (new optional fields). Pydantic models are the single source of truth; JSON Schema is exported from them for non-Python consumers.

### 4. Stable public API vs. internal

Only symbols exported from the top-level `scattering_ai` namespace are covered by semver. Everything under submodules may change between minor versions until 1.0. This keeps early iteration fast without breaking integrators.

---

## Design Principles

1. **The LLM does not invent data.** Every numerical claim must come from structured input, a tool result, or a retrieved document — and be traceable to it.
2. **Computation stays outside the LLM.** RMCProfile, Phonopy, DFT, PDF, and diffuse-scattering codes remain the source of numerical truth.
3. **Rule-based diagnostics run before LLM reasoning.** Convergence problems, constraint violations, dataset conflicts, and missing files are detected deterministically; the LLM interprets, it does not detect.
4. **RAG provides domain knowledge.** Small, curated, cited. High-quality notes beat large unfiltered documents.
5. **Skills are modular.** Each scientific capability is a small, testable unit inside a domain pack.
6. **Tools are explicit and safe.** Read-only by default. Anything that modifies files, reruns jobs, or consumes significant compute requires explicit user approval.
7. **Outputs are structured and separated.** Reports distinguish *Observation* (what the data shows), *Interpretation* (what it may mean), and *Recommendation* (what to do next), in both Markdown and JSON.
8. **Every report carries provenance.** See below.
9. **Evaluation is mandatory.** Domain behavior tests ship with every domain pack to prevent confident-but-wrong reasoning.
10. **Local-first.** Fully functional with local models; no data leaves the machine unless the user configures a cloud backend.

---

## Provenance & Reproducibility

This is the scientific credibility story and the evidence base for a future publication. Every report embeds a provenance block:

```json
{
  "provenance": {
    "sdk_version": "0.1.0",
    "schema_version": "1",
    "input_hash": "sha256:...",
    "model": "backend id + model name",
    "prompt_version": "rmc_health/v3",
    "retrieved_chunks": ["knowledge/rmcprofile/convergence_interpretation.md#slope"],
    "tool_calls": [{"tool": "extract_r_values", "args_hash": "..."}],
    "timestamp": "ISO-8601"
  }
}
```

Consequences:

- Prompts are versioned assets in the repo, not inline strings.
- The same input + same model + same knowledge state should be re-runnable and auditable.
- Regression tests pin against provenance-carrying reports, not raw prose.

---

## Core Data Flow

```text
Application exports scientific state
        → SDK validates input schema
        → Rule-based diagnostics run (deterministic)
        → Relevant knowledge retrieved (cited)
        → Agent selects skills/tools if needed
        → LLM generates interpretation
        → SDK validates output schema + grounding
        → Application receives report (Markdown + JSON + provenance)
```

---

## Schemas

### Input (v1 minimum)

```json
{
  "schema_version": "1",
  "project": "example_project",
  "domain": "rmc",
  "question": "Is this RMC run healthy?",
  "data": {
    "run_summary": {},
    "r_values": [],
    "files": [],
    "plots": [],
    "metadata": {}
  },
  "options": {
    "use_rag": true,
    "use_tools": true,
    "report_format": "markdown"
  }
}
```

### Output (v1 minimum)

```json
{
  "schema_version": "1",
  "status": "ok",
  "summary": "short conclusion",
  "observations": [],
  "interpretation": [],
  "warnings": [],
  "recommended_next_checks": [],
  "citations": [],
  "used_tools": [],
  "confidence": "low | medium | high",
  "provenance": {}
}
```

Domain packs may extend `data` with typed sub-schemas; the envelope stays stable.

---

## Development Tracks

Work is organized into four tracks. Phases within a track are sequential; tracks run in parallel where noted. This replaces a rigid single ladder: after the first vertical slice (A0 → A1 → B1), tracks B, C, and D can advance independently.

```text
Track A — Core Runtime        A0 bootstrap → A1 schemas+LLM → A2 agent loop → A3 orchestration
Track B — Domain Capability   B1 RMC slice → B2 knowledge/RAG → B3 tools/skills → B4 more domains
Track C — Integration         C1 Python API → C2 CLI → C3 FastAPI → C4 MCP server → C5 app connectors → C6 data adapters
Track D — Trust & Quality     D1 diagnostics → D2 reports → D3 evaluation harness → D4 provenance → D5 publication
```

---

### Track A — Core Runtime

#### A0 — Project Bootstrap

- **Goal:** Clean, installable Python package with minimal infrastructure.
- **Deliverables:** `pyproject.toml`, `src/scattering_ai/` layout, README, MIT license, `.gitignore`, pytest setup.
- **Definition of Done:**
  ```bash
  pip install -e ".[dev]"
  python -c "import scattering_ai; print(scattering_ai.__version__)"
  pytest
  ```
- **Non-goals:** CI, docs site, CLI, any agent logic.
- **Risks:** none meaningful; keep it under a day.

#### A1 — Schemas + LLM Layer

- **Goal:** Typed request/report envelope and one working LLM backend.
- **Deliverables:** Pydantic schemas (input/output above), `LLMClient` protocol, OpenAI-compatible client (LM Studio / Ollama / vLLM / OpenAI via `base_url`), config with env-based backend selection.
- **Definition of Done:** the same one-shot completion runs against LM Studio and Ollama changing only config; schema round-trip tests pass.
- **Non-goals:** streaming, Anthropic client, tool-calling loop, retries/rate limiting.

#### A2 — Agent Loop

- **Goal:** Single-agent loop: diagnostics → retrieval → optional tool calls → grounded answer → validated report.
- **Deliverables:** agent runtime in `core/`, tool-call dispatch, output validation (schema + "no uncited numbers" check).
- **Definition of Done:** Track B1's RMC health analysis runs end-to-end through this loop.
- **Non-goals:** multi-agent coordination, planning, memory.

#### A3 — Advanced Orchestration (much later)

- **Goal:** Coordinated multi-domain reasoning (coordinator + domain agents) for questions like *"Why does this sample show local distortion below T_N?"* spanning PDF, RMC, symmetry, and phonon evidence.
- **Definition of Done:** a cross-domain question is decomposed, routed to ≥2 domain packs, and answered with an integrated, cited interpretation.
- **Warning:** Do not start here. Only after the single-agent SDK is stable, evaluated, and in real use.

---

### Track B — Domain Capability

#### B1 — RMC Run Health (the first vertical slice)

- **Goal:** The first useful thing: an **RMC Run Health Agent**.
- **Input:** RMC Monitor summary JSON, R-values over time, plot/file metadata, log snippets.
- **Deliverables — deterministic diagnostics first:**
  - Rwp trend classification (decreasing / flat / oscillating / increasing; last-N-step slope)
  - Bragg vs PDF disagreement; neutron vs x-ray disagreement
  - missing expected files; warnings/errors in logs
  - suspiciously good fit with limited constraints
- **Initial skills:** `analyze_rmc_convergence`, `check_dataset_conflicts`, `check_constraint_warnings`, `summarize_rmc_run`.
- **Definition of Done:** given a static RMC Monitor JSON, the SDK answers: what is happening, why it may matter scientifically, what to check next, and what evidence supports each conclusion.
- **Non-goals:** live monitoring, rerunning RMC, any other domain.

#### B2 — Knowledge Base + RAG

- **Goal:** Domain knowledge beyond the input JSON.
- **Deliverables:** curated Markdown knowledge under `knowledge/rmcprofile/` (basics, R-values and fit quality, Bragg vs total scattering, constraints and cutoffs, common failure modes, convergence interpretation, partial PDFs, magnetic RMC) and `knowledge/scattering/` (reciprocal space, total scattering, PDF, neutron vs x-ray contrast). Simple keyword retrieval first; embeddings second (local-first: `sqlite-vec` or Chroma). Top 3–5 chunks, citations as file path + section title.
- **Definition of Done:** *"Why can the PDF improve while the Bragg fit gets worse?"* retrieves local-vs-average-structure knowledge and produces a grounded, cited answer.
- **Non-goals:** large document ingestion, PDF-of-papers parsing, web retrieval.
- **Risk:** knowledge sprawl — keep it small and curated; every file needs an owner-reviewed pass.

#### B3 — Data Operations Toolkit (the center of the SDK, per D8)

- **Goal:** Make the assistant active instead of passive. An agent is only as
  capable as its tools: deterministic, typed, testable operations on real
  scattering data. The LLM chooses tools and interprets results; every number
  comes from a tool.
- **Organizing principle:** dimensionality, with reduction downward
  (3D → 2D slices → 1D cuts), and a **series axis** (temperature, field, time)
  layered over 1D/2D data rather than a separate toolkit.
- **Deliverables by dimension:**
  - **IO:** readers for the formats in `data/` — NOMAD-style ASCII, diffpy
    pdfgetx `.fq`/`.gr` (metadata header preserved as provenance), generic
    columns, Mantid MDHistoWorkspace NeXus volumes; more formats as they land
    in `data/`.
  - **1D:** crop, rebin (error-propagating), iterative background estimation,
    peak finding, robust peak fitting (bounded, uncertainties, fit-quality
    flags the agent can reason about), Fourier transforms (S(Q) ↔ G(r)).
  - **2D:** peak/blob detection, background and noise estimation, masking,
    powder-ring detection via azimuthal integration with contaminant
    identification against known d-spacings (Al, Cu, steel — knowledge-backed),
    line cuts and ROI integration → 1D.
  - **3D:** volume loader with lattice metadata; slicing at position/thickness/
    orientation → 2D (wrap common operations; do not rebuild Mantid).
  - **Series:** track fitted features across a parameter axis (position, width,
    intensity vs T/H); transition detection.
  - Tool registry with safety tiers, wired into the agent loop (A2 dispatch).
- **Acceptance = known answers on real data:** every toolkit component must
  reproduce a result the user already trusts on the datasets in `data/`
  (e.g. transform NOMAD S(Q) → G(r) and match the pdfgetx `.gr` for the same
  run; recover known peak positions; slice the CORELLI volume to a familiar
  map).
- **Safety rule:** read-only by default; anything mutating or compute-heavy
  requires explicit user approval.
- **Definition of Done:** for *"fit the peaks in this pattern"* or *"cut this
  slice along [h,0,0] and fit the profile"*, the agent executes the tool chain
  and reports fitted values with uncertainties — no numbers from the LLM.

#### B4 — Multi-Domain Expansion

- **Goal:** Second and third domain packs, proving the plugin architecture.
- **First pack — PDF / total scattering (active, per D11):** a `pdf` domain pack
  built entirely on the existing tools, adding deterministic diagnostics on
  G(r)/S(Q) files:
  - **Low-r artifact:** significant |G(r)| below the first physical bond
    distance = termination ripple or normalization error.
  - **G(r) baseline slope:** near the origin G(r) must fall as −4πρr; a
    non-negative slope flags a non-standard convention or sign inversion
    (encodes the user's inverted neutron `.gr` gotcha).
  - **First-peak position:** nearest-neighbour candidate, as grounding.
  - **S(Q) convention mismatch:** a file named `S(Q)` whose high-Q tail → 0
    actually stores S(Q)−1 (the NOMAD gotcha).
  - **Qmax / range:** report the termination-ripple driver.
  - Versioned prompt `pdf_interpret/v1`, curated PDF knowledge, eval cases
    pinned to the real FeCoSn / GaTa4Se8 data.
- **Second pack — diffuse / 3D-ΔPDF (done):** the `diffuse` domain diagnoses
  reciprocal-space volumes and 2D slices — Bragg-punch/mask coverage,
  contaminant powder rings (Al/Cu/steel/V), anisotropic sampling, and
  Bragg-vs-diffuse character; knowledge on diffuse scattering and 3D-ΔPDF.
  Validated on the real CORELLI TbTi3Bi4 volume (flags its 4.5× anisotropic
  sampling and a Cu ring candidate).
- **Third pack — symmetry (done, spglib per D13):** the `symmetry` domain
  analyses a crystal structure (CIF). Tools: `find_symmetry` (FINDSYM-like space
  group + Wyckoff), `subgroup_tree` (maximal subgroups = phase-transition
  pathways, with domain-variant counts and a tree figure), `pseudosymmetry_scan`
  (parent-phase search), `magnetic_symmetry` (Shubnikov group from moments).
  Verified against the International Tables (Pm-3m subgroups).
  - *Additional symmetry tools worth adding later:* systematic-absence /
    reflection-condition prediction (ties symmetry to the diffraction/pdf packs),
    cell standardization, klassengleiche (cell-multiplying) subgroups for
    ordering transitions, symmetry-mode / irrep decomposition (AMPLIMODES-style),
    and k-vector → maximal magnetic space group representation analysis (MAXMAGN).
    These need extra tables or representation machinery beyond spglib.
- **Later candidate packs (order by user need):**
  - **INS (inelastic neutron scattering, incl. phonons — per D12):** S(Q,ω),
    dispersions, DOS features, flat/soft branches, acoustic-mode checks, dynamic
    structure factor; phonon analysis is a capability *inside* this pack, not a
    standalone `phonons` pack.
- **Definition of Done:** a user question routes to the correct domain pack and knowledge base; adding the pack required **no changes to core reasoning** (agent loop, schemas, tool layer, report generator) — only a new pack module, its registration, and its knowledge/eval assets. If core needs a change (as D10 did), make it a general one and fix it before the next pack.

---

### Track C — Integration Surface

#### C1 — Python API (first)

```python
from scattering_ai import analyze

report = analyze(domain="rmc", question="Is this run healthy?", data=rmc_monitor_json)
print(report.markdown)
```

- **Definition of Done:** RMC Monitor can `pip install` the SDK and get a report object with `.markdown`, `.json`, `.provenance`.

#### C2 — CLI

```bash
scattering-ai analyze path/to/rmc_monitor_summary.json --domain rmc --out report.md
```

#### C3 — FastAPI Service (optional deployment mode)

`POST /analyze`, `POST /chat`, `POST /index-knowledge`, `GET /health` — same core, for apps that prefer a local service over an import.

#### C4 — MCP Server

Expose the SDK as MCP tools so any agent host (Claude Code, IDEs, other
assistants) can drive scattering analysis. This turns the SDK from a library
into infrastructure other AI systems can use — the key step toward the
universal-foundation goal: capability decoupled from interpretation, with the
same tested core behind every door (Python API, CLI, MCP, app connectors).

Plan:

1. **Thin adapter, zero new logic**: `server/mcp.py` maps each `AgentTool`
   in the registry to an MCP tool 1:1 (the registry's JSON schemas are the
   single source of truth). Official `mcp` Python package, stdio transport,
   launched via `scattering-ai mcp`.
2. **Two tiers**: the individual data tools (a capable host model does its
   own orchestration), plus one high-level `analyze` tool running the full
   loop (diagnostics → RAG → local-LLM interpretation) for hosts that want
   packaged behavior.
3. **Same guarantees**: workspace artifacts, structured summaries, and
   provenance identical to the built-in agent loop.
- **Definition of Done:** Claude Code, connected via `claude mcp add`, can
  slice the CORELLI volume, cut a profile, and fit peaks — chaining the
  SDK's tools itself — and every number in its answer traces to a tool
  result.

#### C5 — App Connectors

Thin adapters per application (`connectors/rmc_monitor.py` first): validate app-specific JSON, map to SDK scientific state, return reports. Apps never embed AI logic; they call the SDK. RMC Monitor gets an **AI Analysis** panel (Analyze current run / Explain warning / Compare with previous run / Generate report).

#### C6 — Data Format Adapters (later)

Read standard formats directly — NeXus, CIF, RMCProfile file conventions — so the SDK can ingest facility data without app-specific export steps.

- **Track C rule:** each surface wraps the same core `analyze()`; no surface gets its own reasoning logic.

---

### Track D — Trust & Quality

#### D1 — Deterministic Diagnostics

Built with B1; listed here because the diagnostics layer is a trust feature: everything detectable without an LLM must be detected without an LLM.

#### D2 — Report Generation

- **Report types:** short summary, run health, multi-run comparison, experiment daily summary, publication-style interpretation, troubleshooting.
- **Structure:** Conclusion → Evidence → Scientific interpretation → Warnings → Recommended next checks → Files used → Knowledge used → Confidence → Provenance.
- **Definition of Done:** reports are clear enough to paste into a lab notebook or project log with minimal editing.

#### D3 — Evaluation Harness

- **Case types:** known-healthy run, known-stalled run, bad-Bragg/good-PDF, good-Bragg/bad-PDF, constraint violation, missing files, deliberately ambiguous case.
- **Rubric:** factual grounding, correct use of evidence, no invented numbers, correct uncertainty handling, useful next checks, domain correctness, clarity.
- **Method:** test for required claims, warnings, and evidence references — never exact wording. Every domain pack ships its own eval cases.
- **Definition of Done:** `pytest tests/evaluation/` gates releases; a model or prompt change that breaks domain behavior fails CI.

#### D4 — Provenance Enforcement

Provenance block (above) becomes mandatory in output validation; reports without full provenance are rejected by the SDK itself.

#### D5 — Publication

- **Angle:** *A domain-grounded agent framework for AI-assisted scattering and atomistic modeling workflows.*
- **Claim:** improved workflow efficiency, reproducibility, and interpretability from combining structured scientific state, deterministic diagnostics, curated knowledge, tool-augmented LLM reasoning, and testable reports.
- **Required evidence:** case studies (RMC convergence monitoring; local-vs-average conflict detection; phonon mode interpretation from RMC ensembles; 3D-ΔPDF feature explanation; multi-run comparison), benchmark tasks, before/after workflow comparison, error analysis, domain-expert evaluation, reproducible examples, open-source repo.

---

## Milestones

### Milestone 1 — RMC Run Health Report *(A0 + A1 + A2 + B1 + C1 + C2 + D2)* — ✅ done (2026-07-03)

```text
Input:  RMC Monitor JSON
Output: Markdown + JSON report — convergence status, fit quality, suspicious
        behavior, possible scientific meaning, recommended next checks,
        evidence from input data, knowledge used, provenance.
```

```bash
scattering-ai analyze path/to/rmc_monitor_summary.json --domain rmc --out report.md
```

### Milestone 2 — Interactive RMC Assistant *(+ B2 + B3)* — ✅ done (2026-07-03; `chat/v3`)

Chat-style interaction grounded in tools and knowledge:

> **User:** Why is this run not improving?
> **Assistant:** The Rwp trend is nearly flat over the last N steps. The PDF component improved slightly, but the Bragg component worsened, suggesting a conflict between local and average structure constraints. Check the following…

### Milestone 3 — RMC Monitor Integration *(+ C5)*

RMC Monitor shows an AI Analysis panel; the monitor owns zero AI logic.

### Milestone 4 — Cross-App Scattering Assistant *(+ B4, C4)*

The same SDK powers RMC Monitor, RMC Phonon Dynamics, and Neutron Diffuse Toolkit, and is reachable via MCP. At this point the project is a platform.

---

## Versioning & Release Policy

- **Semver.** `0.x` until Milestone 2 ships; breaking changes allowed in `0.x` minors with CHANGELOG notes.
- **`1.0`** when: schemas are stable, two domain packs exist, the evaluation harness gates releases, and one external app integrates in production.
- **CHANGELOG.md** from the first tagged release.
- **CI:** GitHub Actions once there are meaningful tests (post-A1), not before.
- **Deprecations:** one minor version of warning before removal, post-1.0.

---

## Current Status (2026-07-03)

The tools-first foundation is **built and validated on real data**; `v0.1.0` is
tagged and the repo is pushed. What exists now:

| Track | Done | Notes |
|-------|------|-------|
| A — Core Runtime | A0, A1, **A2 (full tool dispatch)** | A3 multi-agent still deferred |
| B — Domain Capability | B1 (RMC health), B2 (RAG), **B3 (1D/2D/3D + series tools, skills)**, **B4 (`pdf` + `diffuse` + `symmetry` technique packs)** | plugin architecture proven three times with no core reasoning changes; next pack `ins` |
| C — Integration | C1 (Python API), C2 (CLI), C3 (FastAPI), C4 (MCP), C5 (RMC connector), C6 (NeXus / CIF / mCIF / RMCProfile `.rmc6f` readers) | all surfaces wrap the same core |
| D — Trust & Quality | D1 (diagnostics), D2 (**reports + summarizing figures**), D3 (eval harness), **D4 (provenance enforcement)** | reports carry figures the LLM reasons over; incomplete-provenance reports are rejected; D5 later |

Built beyond the original slice: a broad agent-tool registry + composite skills
organized by category (patterns / series / slices / structure), CIF/mCIF
structure visualization, interactive chat (Milestone 2), plotting toolkit, MCP server, a
**robust transition-tracing workflow** (stacked peak selection, auto-detected
mask sentinel, per-peak monitoring summary — validated on the GaNb4Se8 39 K
structural transition), and **domain auto-routing** (one entry point picks the
pack from the input; `analyze(data={"files":[…]})` with no domain). CI, CHANGELOG.

Also built: **3D-ΔPDF** of a diffuse volume (punch → apodize → centred FFT,
validated on the real CORELLI volume), **symmetry extras** (systematic
absences, cell standardization) on top of the space-group/subgroup/pseudosymmetry
tools, and **RMCProfile file reading** (`.rmc6f` configurations + R-value logs +
a KDE disorder map). The five domains are `data`, `pdf`, `diffuse`, `symmetry`,
`rmc`; 196 tests.

Every report now carries **summarizing figures** the LLM reasons over: the
`data` pack detects a T/field series and reports the **phase transition**
(T_c + which peaks move) with waterfall + tracking plots; `pdf` and `diffuse`
emit an overview/map. All four packs verified live on Ollama (gemma4:26b):
GaNb4Se8 series → transition at 39 K (high confidence, figures); diffuse volume
→ 4.5× anisotropy + figure the model cites by path; PDF flags the inverted
neutron `.gr`. Auto-routing + a JSON-retry loop fixed the earlier local-model
"interpretation unavailable" failure.

## Immediate Next Actions

The old bottom-up tool ladder is **done**. The next frontier is proving the
**plugin architecture with a real technique pack** (Track B4) and hardening
trust (D4), not more one-off tools.

Done recently: B4 `pdf` + `diffuse` + `symmetry` packs; D10 next-check-rules
move; D4 provenance enforcement; domain **auto-routing** (one entry point);
LLM-loop hardening (reasoning-block/prose-tolerant JSON + retry); **summarizing
figures** in reports with first-class **phase-transition** detection (`data`
pack) and **maximal-subgroup trees** (`symmetry` pack).

Ported from sibling repos (2026-07): 3D-ΔPDF (independent MIT implementation of
the standard windowed-FFT method; nebula3d is AGPL so its source was not copied),
RMCProfile readers + KDE density map (adapted from the MIT rmc-toolkits). Symmetry
gained systematic absences + cell standardization.

```text
Next:
1.  B4 — `ins` pack (inelastic neutron scattering, incl. phonons per D12):
    diagnostics on S(Q,ω) / dispersion / DOS data once such data lands in data/
    (acceptance = known answer). Deferred until example data exists — this is
    also the home for rmc-phonon's reciprocal-space / k-path utilities.
2.  Symmetry, when representation-theory tables are available: klassengleiche
    subgroups, irrep / symmetry-mode decomposition, k-vector → maximal magnetic
    space groups (MAXMAGN).
3.  D5 groundwork — assemble reproducible case-study runs the packs now enable
    (phase transition, inverted-.gr, diffuse contaminant/anisotropy, RMC
    convergence) for the publication evidence base.
4.  Chat polish: surface figures inline; optional streaming.
5.  A3 (much later): cross-domain coordinator once ≥3 packs are in real use.
```

Rule still holds: every deterministic check must reproduce a known answer on
real data in `data/` before the agent is allowed to rely on it.

---

## Long-Term Vision

**Horizon 1 (~1 year) — a trusted run-health copilot.** The SDK is the AI layer inside the author's own tools, giving disciplined, cited, provenance-carrying answers about RMC and PDF runs. Scientists trust it because everything deterministic is deterministic, and everything interpretive is cited and hedged honestly.

**Horizon 2 (~2–3 years) — an extensible platform.** Multiple domain packs (phonons, diffuse, symmetry, QENS), community-contributed packs via the plugin interface, MCP access from any agent host, data-format adapters for facility data, and a publication establishing the framework. Other groups build on it without forking it.

**Horizon 3 (aspirational) — a disciplined scientific collaborator across the scattering workflow.** From experiment planning and data-reduction QA through modeling, interpretation, and publication-ready reporting:

```text
It reads results.
It retrieves domain knowledge.
It checks deterministic diagnostics.
It calls tools when needed.
It explains uncertainty.
It recommends next steps.
It produces reproducible reports.
```

**What it will never do:**

- Replace the researcher's scientific judgment or sign off on conclusions.
- Invent, extrapolate, or "smooth over" numbers not present in inputs, tool results, or cited knowledge.
- Silently modify data, rerun computations, or consume compute without explicit approval.
- Require cloud access: local-first operation is permanent, not transitional.

The standard is higher than a generic chatbot: trustworthy enough for real scientific workflows, transparent enough to audit, and honest enough to say "the evidence is insufficient."
