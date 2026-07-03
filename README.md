# scattering-ai-sdk

**AI SDK for scattering science** — a domain-grounded reasoning layer for AI-assisted RMC, PDF/total scattering, diffraction, phonon, and diffuse-scattering workflows.

> Status: **pre-alpha**. The package skeleton and schemas exist; the first vertical slice (RMC run health analysis) is under construction. See [ROADMAP.md](ROADMAP.md).

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

## Quick look (target API)

```python
from scattering_ai import analyze  # coming in Milestone 1

report = analyze(domain="rmc", question="Is this run healthy?", data=rmc_monitor_json)
print(report.markdown)
```

Today the package provides the typed request/report schemas and the provider-agnostic LLM client layer:

```python
from scattering_ai import AnalysisRequest, AnalysisReport
from scattering_ai.core.config import SDKConfig

config = SDKConfig.lm_studio()   # or .ollama() / .openai() / .from_env()
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
