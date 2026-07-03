# RMC Monitor Demo

Two synthetic RMC Monitor exports for trying the SDK:

- `healthy_run.json` — both R-value series converging, files present, clean log.
- `stalled_run.json` — Bragg worsening while the PDF improves (dataset
  conflict), a missing partials file, low move acceptance, and a constraint
  warning in the log.

## Deterministic diagnostics only (no LLM needed)

```bash
scattering-ai analyze examples/rmc_monitor_demo/stalled_run.json
```

## With a local LLM

Start LM Studio (or Ollama) with a model loaded, then:

```bash
scattering-ai analyze examples/rmc_monitor_demo/stalled_run.json \
    --backend lmstudio --model your-model-name --out report.md
```
