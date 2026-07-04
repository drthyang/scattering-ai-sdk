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

> Status: **early, but useful.** Four domain packs, auto-routing, figure-backed
> reports, and an evaluation harness are in place and validated on real data.

**Docs:** [Quickstart](QUICKSTART.md) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md)

## What it can tell you

| You give it | It reports (deterministically, then the LLM interprets) | Figure |
|-------------|--------------------------------------------------------|--------|
| A **T/field scan** of patterns | Whether there's a **phase transition**, its T_c, and which peaks move | waterfall + peak tracking |
| A **G(r) / S(Q)** curve | Non-standard/inverted G(r), low-r artifacts, first-neighbour distance, S(Q) vs S(Q)−1 | overview plot |
| A **diffuse volume / slice** | Bragg-vs-diffuse character, contaminant rings, Bragg-punch coverage, sampling anisotropy; **3D-ΔPDF** (punch → apodize → FFT) | log-scale map / ΔPDF slice |
| A **crystal structure (CIF/mCIF)** | Space group + Wyckoff sites, **maximal subgroups** (phase-transition pathways), pseudosymmetric parent, magnetic space group | structure view + subgroup tree |
| **RMC monitor state** | Convergence trend, Bragg/PDF & neutron/x-ray conflicts, missing files | — |

## Install

```bash
pip install -e ".[dev]"    # development
pip install -e ".[all]"    # + LLM client, figures (matplotlib), volumes (h5py), API, MCP
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

**Skills** — validated multi-step workflows the agent invokes as one call,
organized by category (`series & transitions`, `1D patterns`, `2D slices`,
`structure`); each returns a summary, `figures`, and an audited step chain.
See the [Quickstart](QUICKSTART.md#agent-skills).

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
