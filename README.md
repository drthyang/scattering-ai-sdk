# scattering-ai-sdk

**An AI reasoning layer for scattering science** — RMC, total scattering / PDF,
diffuse scattering, and diffraction workflows.

```text
Scientific codes compute.  Domain tools evaluate.  The LLM reasons, explains, and guides.
```

Point it at a data file and it detects the technique, runs deterministic
diagnostics, retrieves cited domain knowledge, optionally calls an LLM to
interpret, and returns a structured, **provenance-carrying** report **with
summarizing figures**. It never invents numbers, never mutates data, and works
fully offline with local models.

> Status: **early, but useful.** Five domains (`data`, `pdf`, `diffuse`,
> `symmetry`, `rmc`), auto-routing, figure-backed reports, an evaluation harness,
> and a human-reviewed **self-improvement loop** are in place and validated on
> real data.

**Docs:** [Quickstart](QUICKSTART.md) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md) · [Self-improvement](docs/self_improvement.md) · [Case studies](docs/case_studies.md)

## What it can tell you

| You give it | It reports (deterministically, then the LLM interprets) | Figure |
|-------------|--------------------------------------------------------|--------|
| A **T/field scan** of patterns | Whether there's a **phase transition**, its T_c, and which peaks move | waterfall + peak tracking |
| A **G(r) / S(Q)** curve | Non-standard/inverted G(r), low-r artifacts, first-neighbour distance, S(Q) vs S(Q)−1 | overview plot |
| A **G(r) + a CIF** | **Does this structure explain the PDF?** — model fit with Rw, peak width, lattice scale | data/model/difference |
| A **diffuse volume / slice** | Bragg-vs-diffuse character, contaminant rings, Bragg-punch coverage, sampling anisotropy; **3D-ΔPDF** (punch → apodize → FFT) | log-scale map / ΔPDF slice |
| A **crystal structure (CIF/mCIF)** | Space group + Wyckoff sites, **maximal subgroups** (phase-transition pathways), pseudosymmetric parent, magnetic space group | structure view + subgroup tree |
| **RMC monitor state** | Convergence trend, Bragg/PDF & neutron/x-ray conflicts, missing files | — |

## Install

```bash
pip install -e ".[dev]"    # development
pip install -e ".[all]"    # + LLM client, figures, volumes (h5py), symmetry (spglib), API, MCP
```

Python ≥ 3.10. Figures need the `plots` extra (matplotlib); volumes need `volumes`
(h5py); symmetry needs `symmetry` (spglib).

## Quick start

```python
from scattering_ai import analyze

# Domain auto-detected from the input; pass domain=... to override.
report = analyze(data={"files": ["FeCoSn_100K.gr"]})
print(report.domain)      # -> "pdf"
print(report.figures)     # -> [".../pdf_overview.png"]
print(report.markdown)    # full report (or report.model_dump_json())
```

Add a local LLM for interpretation (one client covers LM Studio / Ollama / vLLM / OpenAI):

```python
from scattering_ai import Agent, AnalysisRequest
from scattering_ai.core.config import SDKConfig
from scattering_ai.llm.openai_compatible import OpenAICompatibleClient

cfg = SDKConfig.ollama(model="qwen3:32b")       # or .lm_studio() / .openai() / .from_env()
agent = Agent(llm=OpenAICompatibleClient(cfg), model_id=f"ollama:{cfg.model}")
report = agent.analyze(AnalysisRequest(
    question="Is there a phase transition across temperature?",
    data={"files": ["scans/"]},                 # a folder or glob of the T-series
))
```

CLI — just point it at a file (domain auto-detected):

```bash
scattering-ai analyze --file my_pattern.gr                     # offline diagnostics + figure
scattering-ai analyze --file 'scan_dir/*.dat' --out report.md  # a T-series -> phase transition
scattering-ai analyze --file scan_dir --backend ollama --model qwen3:32b   # + LLM interpretation
```

## Domains

Auto-detected from the input, or set explicitly (`--domain` / `domain=`):

- **`data`** — generic tool-driven analysis; detects a parametric series and hunts phase transitions.
- **`pdf`** — total scattering: G(r), S(Q), F(Q).
- **`diffuse`** — single-crystal diffuse scattering / 3D-ΔPDF (volumes and slices).
- **`symmetry`** — crystallographic symmetry from a CIF: space group, Wyckoff sites, maximal subgroups, pseudosymmetry, magnetic groups ([`symmetry`] extra, spglib).
- **`rmc`** — RMCProfile run health; reads `.rmc6f` configurations (cell, supercell, composition) and R-value logs.

Each pack is self-contained (diagnostics + knowledge + prompt + next-check rules
+ figures). Third parties add packs via the `scattering_ai.domains` entry point
without touching core.

## Other interfaces

```bash
scattering-ai chat --backend ollama --model qwen3:32b --file 'data/1d/series/*.dat'
scattering-ai plot my_pattern.gr --fit "2.64,3.73"     # quick-look plots (peaks/fits/slices/series)
scattering-ai serve --port 8551                        # HTTP API ([api] extra)
claude mcp add scattering-ai -- scattering-ai mcp      # expose tools to any MCP host
```

```python
from scattering_ai.connectors.rmc_monitor import analyze_monitor   # apps own zero AI logic
report = analyze_monitor(monitor_json)
```

### MCP server

Expose the SDK to any MCP host (Claude Code, IDEs, other assistants):

```bash
pip install -e ".[mcp]"
claude mcp add scattering-ai -- scattering-ai mcp     # Claude Code
```

For a generic host, register the stdio command `scattering-ai mcp` (JSON form):

```json
{ "mcpServers": { "scattering-ai": { "command": "scattering-ai", "args": ["mcp"] } } }
```

The host then gets, all from the same tested core:
- **Tools** — every SDK tool + skill 1:1, plus a high-level `analyze` (auto-routes and returns a provenance-carrying report). Set `SCATTERING_AI_MODEL` to have `analyze` use a local LLM inside the host.
- **Resources** — the curated knowledge base (`knowledge://…`) and a `scattering-ai://domains` overview.
- **Prompts** — `domain_guidance` (a technique's grounding rules) and `analyze_files` (a ready-to-send analysis request).

**Skills** — validated multi-step workflows the agent invokes as one call,
organized by category (`series & transitions`, `1D patterns`, `2D slices`,
`structure`, `3D volumes`, `magnetic`); each returns a summary, `figures`, and
an audited step chain. See the [Quickstart](QUICKSTART.md#agent-skills).

## How it works

```text
input → auto-route to a domain pack → deterministic diagnostics (+ figures)
      → cited knowledge retrieval → optional LLM interpretation (tool-calling)
      → schema + provenance validation → report (Markdown + JSON)
```

- **Diagnostics run before the LLM** — everything detectable without a model is.
- **Every number is traceable** to input data, a tool result, or a cited document.
- **Reports are attributable** — the SDK rejects reports with incomplete provenance.
- **Local-first** — no data leaves your machine unless you configure a cloud backend.

## Self-improvement (the growth loop)

Because capability here is **data-gated** — the correctness rule is *reproduce a
known answer on real data*, and datasets, corrections, and failure modes arrive
over months of research — the SDK is built to improve *during* use. An opt-in,
**redacted**, local loop captures what real use reveals and turns it into
**reviewable** improvements. It never edits scientific logic, prompts, schemas,
or code on its own: it observes and proposes; a human approves; changes land as
eval-gated, diff-only commits (design: [docs/self_improvement.md](docs/self_improvement.md)).

```bash
export SCATTERING_AI_JOURNAL=~/.scattering_ai/journal   # opt-in; default off
scattering-ai analyze --file my_data.gr                 # each run/chat/MCP call may log a redacted episode
scattering-ai learn status                              # aggregate view of the journal
scattering-ai learn correct --target transition_temperature \
    --statement "GaNb4Se8 transitions are 50 K and 29 K, not 39 K" --value 50,29
scattering-ai learn review                              # signals -> tiered, reviewable proposals
scattering-ai learn apply  --id <id> --approve          # Tier-0 data: eval-gated diff (human commits)
scattering-ai learn brief  --id <id>                    # Tier-1/2: an agent work order (never a patch)
scattering-ai learn watch  --brief                      # data-gated builds unblocked by new data/
scattering-ai case-studies                              # known-answer runs on real data
```

Everything is category labels + identifiers — never data values — and every
apply is gated by the test suite and audited. A human always makes the commit.

## Development

```bash
pip install -e ".[dev]" && pytest && ruff check .
```

## Acknowledgements

The RMCProfile file readers and the KDE density map are adapted from the
MIT-licensed [rmc-toolkits](https://github.com/drthyang/rmc-toolkits). The
3D-ΔPDF is an independent implementation of the standard windowed-FFT method
(as in [nebula3d](https://github.com/drthyang/nebula3d)).

## License

MIT
