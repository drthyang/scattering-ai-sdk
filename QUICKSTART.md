# Quickstart

Get from install to a figure-backed scientific answer in a few minutes.

- [Setup](#setup)
- [Simplest example](#simplest-example) — one call, auto-routed
- [Agent skills](#agent-skills) — validated multi-step workflows
- [Advanced usage](#advanced-usage) — LLM interpretation, chat, other interfaces, extending

---

## Setup

```bash
git clone https://github.com/drthyang/scattering-ai-sdk.git
cd scattering-ai-sdk
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -e ".[all]"      # everything: figures, LLM, volumes, symmetry, API, MCP
```

Minimal installs if you don't need everything:

| Extra | Adds | Needed for |
|-------|------|------------|
| `plots` | matplotlib | summarizing figures |
| `llm` | openai client | LLM interpretation (LM Studio / Ollama / vLLM / OpenAI) |
| `volumes` | h5py | 3D NeXus volumes (diffuse, 3D-ΔPDF) |
| `symmetry` | spglib | crystallographic symmetry (CIF/mCIF) |
| `api` / `mcp` | FastAPI / MCP | HTTP service / MCP server |

Verify:

```bash
python -c "import scattering_ai; print(scattering_ai.__version__)"
pytest -q
```

---

## Simplest example

Point `analyze` at a file — the domain is detected from the input, diagnostics
run, and a summarizing figure is generated. No LLM required.

```python
from scattering_ai import analyze

report = analyze(data={"files": ["FeCoSn_100K.gr"]})

print(report.domain)      # -> "pdf"  (auto-detected)
print(report.summary)     # deterministic conclusion
print(report.figures)     # -> [".../pdf_overview.png"]
print(report.markdown)    # full report (Observation / Interpretation / Figures / Provenance)
```

Or from the command line:

```bash
scattering-ai analyze --file FeCoSn_100K.gr --out report.md
```

---

## Agent skills

A **skill** is a validated multi-step workflow the agent runs as one call — it
chains the right tools in the right order, returns the evidence, a plain-English
summary, **figures**, and an audited list of every step it took.

The built-in skills, grouped by category (see `registry.skills_by_category()`):

| Category | Skill | Answers | Figures |
|----------|-------|---------|---------|
| series & transitions | `skill_scan_series_transitions` | Is there a phase transition? Where? | waterfall + peak tracking |
| 1D patterns | `skill_fit_pattern_peaks` | Fit the peaks (with uncertainties) | fit + residual |
| 2D slices | `skill_characterize_slice` | What's in this 2D slice? | slice map + ring profile |
| structure | `skill_visualize_structure` | Show this crystal structure (CIF/mCIF) | unit cell + moments |
| structure | `skill_symmetry_overview` | Space group + subgroups + transitions | subgroup tree |
| 3D volumes | `skill_delta_pdf` | 3D-ΔPDF + real-space correlations | central-plane ΔPDF |

Individual tools are called the same way — e.g. crystallographic symmetry from a
CIF (needs the `symmetry` extra):

```python
registry.execute("find_symmetry", {"path": "structure.cif"})      # -> Pm-3m (#221), Wyckoff sites
tree = registry.execute("subgroup_tree", {"path": "structure.cif"})  # phase-transition pathways
print([s["international"] for s in tree["subgroups"]], tree["plot"])  # + a subgroup-tree figure
```

### Simplest skill call

Skills run over a tool **registry** bound to a workspace folder (where plots and
intermediate artifacts are saved):

```python
from scattering_ai.tools.registry import default_toolkit

registry = default_toolkit("workspace/")     # folder for figures + artifacts

result = registry.execute("skill_scan_series_transitions", {
    "paths": ["scans/"],                      # a directory, a glob, or a list of files
})

print(result["summary"])
# Tracked 3 peak(s) across 20 curves (T (K) 5–99.1). 2 trend(s) show a changepoint
# near 39.02 (range 36.5–41.55); likely a transition. Confirm against the plots.

print(result["verdict"]["transition_estimate"])   # 39.02  (K)
print(result["figures"])                           # ['.../series.png', '.../tracking.png', ...]
print([s["tool"] for s in result["steps"]])        # audited chain of tool calls
```

Each entry in `paths` may be a directory or a glob (`"scans/*.dat"`) so you never
retype long filenames. The masked-region sentinel (e.g. `-3.0`) is auto-detected.

### The other two skills

```python
# Fit the strongest peaks in a 1D pattern, with uncertainties and a residual plot
fit = registry.execute("skill_fit_pattern_peaks", {"path": "pattern.gr", "xmax": 6.0})
for p in fit["peak_table"]:
    print(p["center"], "±", p["center_err"])
print(fit["assessment"]["fit_trustworthy"], fit["figures"])

# First look at a 2D slice cut from a volume: Bragg peaks, contaminant rings, plot
look = registry.execute("skill_characterize_slice", {
    "path": "volume.nxs", "axis": 0, "center": 0.0, "thickness": 0.1,
})
print(look["n_bragg_peaks"], look["contaminant_matches"], look["figures"])
```

Every skill result includes `figures` (all plots it produced) and `steps` (the
audited tool chain), so the output is both visual and reproducible.

---

## Advanced usage

### Add an LLM for interpretation

One client covers LM Studio, Ollama, vLLM, and OpenAI — switching is a config
change. The model reasons over the diagnostics and the generated figures; every
number stays traceable and the report is rejected if provenance is incomplete.

```python
from scattering_ai import Agent, AnalysisRequest
from scattering_ai.core.config import SDKConfig
from scattering_ai.llm.openai_compatible import OpenAICompatibleClient

cfg = SDKConfig.ollama(model="qwen3:32b")          # or .lm_studio() / .openai() / .from_env()
agent = Agent(llm=OpenAICompatibleClient(cfg), model_id=f"ollama:{cfg.model}")

report = agent.analyze(AnalysisRequest(
    question="Is there a phase transition across temperature?",
    data={"files": ["scans/"]},                    # auto-routes to the data pack
))
print(report.summary, report.confidence)
print(report.figures)                              # waterfall + tracking, cited in the report
```

### Interactive chat

Chat keeps history and the artifact workspace across turns and invokes skills
for you:

```bash
scattering-ai chat --backend ollama --model qwen3:32b --file 'scans/*.dat'
you> is there a phase transition in this series?
```

### Other interfaces

```bash
scattering-ai serve --port 8551                    # HTTP API ([api] extra)
claude mcp add scattering-ai -- scattering-ai mcp  # tools + skills + knowledge resources + prompts
```

```python
from scattering_ai.connectors.rmc_monitor import analyze_monitor   # app connector
report = analyze_monitor(monitor_json)             # apps own zero AI logic
```

### Register your own skill

A skill is a function over the registry plus an `AgentTool` entry:

```python
from scattering_ai.tools.registry import AgentTool, default_toolkit, _params
from scattering_ai.skills.base import SkillRun

def my_workflow(registry, path):
    run = SkillRun(registry)
    found = run.call("find_peaks_1d", path=path)
    fit = run.call("plot_fit_1d", path=path, centers=[p["x"] for p in found["peaks"][:3]])
    return run.finish(rwp=fit.get("rwp"))          # `figures` + `steps` added automatically

registry = default_toolkit("workspace/")
registry.add(AgentTool(
    "skill_my_workflow", "SKILL: find and fit the top-3 peaks.",
    _params({"path": {"type": "string"}}, ["path"]),
    lambda **kw: my_workflow(registry, **kw),
))
print(registry.execute("skill_my_workflow", {"path": "pattern.gr"})["figures"])
```

Third parties can ship a whole **domain pack** (diagnostics + knowledge + prompt
+ skills) via the `scattering_ai.domains` entry point without touching core — see
[ROADMAP.md](ROADMAP.md), "Extensibility Architecture".

---

See the [README](README.md) for the full feature list and [ROADMAP.md](ROADMAP.md)
for design decisions.
