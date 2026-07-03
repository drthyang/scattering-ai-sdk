# scattering-ai-sdk

**AI SDK for scattering science** — a domain-grounded reasoning layer for AI-assisted RMC, PDF/total scattering, diffraction, phonon, and diffuse-scattering workflows.

> Status: **pre-alpha**. The first vertical slice works end-to-end: RMC run health analysis with deterministic diagnostics, cited knowledge retrieval, optional LLM interpretation, and provenance-carrying reports. See [ROADMAP.md](ROADMAP.md).

## What it is

Scientific codes (RMCProfile, Phonopy, PDF and diffuse-scattering tools) compute. This SDK adds a disciplined reasoning layer on top:

```text
Scientific codes compute.
Domain tools evaluate.
The LLM reasons, explains, and guides.
```

It accepts structured scientific state from applications, runs deterministic diagnostics, retrieves curated domain knowledge, optionally calls read-only analysis tools, and returns structured, cited, provenance-carrying reports. It never invents numbers, never mutates data, and works fully offline with local models.

## Install

```bash
pip install -e ".[dev]"        # development
pip install -e ".[llm]"        # + OpenAI-compatible LLM client (LM Studio / Ollama / vLLM / OpenAI)
```

Requires Python ≥ 3.10.

## Quick start

Python API — works fully offline (deterministic diagnostics + cited knowledge retrieval); add an LLM client for scientific interpretation:

```python
from scattering_ai import analyze

report = analyze(domain="rmc", question="Is this run healthy?", data=rmc_monitor_json)
print(report.markdown)      # or report.model_dump_json()
```

```python
from scattering_ai import Agent, AnalysisRequest
from scattering_ai.core.config import SDKConfig
from scattering_ai.llm.openai_compatible import OpenAICompatibleClient

config = SDKConfig.lm_studio(model="your-model")   # or .ollama() / .openai() / .from_env()
agent = Agent(llm=OpenAICompatibleClient(config), model_id=config.model)
```

CLI:

```bash
scattering-ai analyze examples/rmc_monitor_demo/stalled_run.json               # offline
scattering-ai analyze run.json --backend lmstudio --model m --out report.md   # with local LLM

# tool-driven analysis of a data file: the agent inspects, cuts, and fits
scattering-ai analyze --file my_pattern.gr \
    --question "Fit the main peaks below 6 A" --backend ollama --model qwen3:32b
```

Quick-look plots for judging results (peaks, fits with residuals, slices, series):

```bash
scattering-ai plot my_pattern.gr                          # curve + detected peaks
scattering-ai plot my_pattern.gr --fit "2.64,3.73"        # fit + residual panel
scattering-ai plot slice.npz --log                        # 2D slice
scattering-ai plot series_*K.dat --mask-value=-3.0        # waterfall vs T
```

HTTP API (`pip install ".[api]"`):

```bash
scattering-ai serve --port 8551    # GET /health /tools, POST /tools/{name} /analyze
```

MCP server — expose the tools to any agent host (Claude Code, IDEs, ...):

```bash
claude mcp add scattering-ai -- scattering-ai mcp
```

The host model then chains the SDK's 12 data tools itself (volume slicing,
line cuts, peak fitting, ring detection, series tracking), plus a high-level
`analyze` tool running the full diagnostics → knowledge → report loop.

Application connectors — apps own zero AI logic:

```python
from scattering_ai.connectors.rmc_monitor import analyze_monitor

report = analyze_monitor(monitor_json)   # dict or path; returns AnalysisReport
panel.show(report.markdown)
```

## Design principles

- Rule-based diagnostics run **before** LLM reasoning — everything detectable without an LLM is detected without an LLM.
- Every numerical claim traces to input data, a tool result, or a cited knowledge document.
- Reports separate **Observation** / **Interpretation** / **Recommendation** and carry full provenance.
- Local-first: no data leaves your machine unless you configure a cloud backend.
- Domains (RMC, PDF, phonons, diffuse, symmetry, …) are plugins; core stays technique-agnostic.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT
