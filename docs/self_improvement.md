# Human-reviewed self-improvement

A controlled loop that captures workflow failures, warnings, validation results,
and **human corrections**, then turns recurring patterns into **reviewable
proposals**. The SDK never modifies scientific logic, prompts, schemas, or code
automatically.

## The non-negotiable principle

> The system may **observe** and **propose**. Every change to logic, prompts,
> schemas, or code passes through an explicit **human approval** step and lands
> as a reviewable git diff, gated on the evaluation suite — never a silent edit.

This is enforced structurally (tiers, denylists, eval gates), not by policy.

## Decisions (locked)

- **Apply autonomy: diff-only, human commits.** Even an approved, eval-passing
  Tier-0 change is written to the working tree for review; a human makes the
  commit. Nothing lands without a human hand on the commit.
- **LLM role: prose only.** Signal extraction and evidence are deterministic;
  an LLM may rewrite a proposal's title/rationale into readable English. It can
  never introduce or alter evidence or suggested values.
- **Local-first.** Journal and proposals are plain files in a directory you
  choose. Nothing is transmitted anywhere.

## Architecture

```
episodes (append-only JSONL journal)   ← passive, deterministic, redacted
      │
 signals (typed observations)          ← deterministic extraction, no LLM
      │
proposals (reviewable improvements)    ← templated; LLM prose only
      │
 review + guarded apply (human)        ← approval → git diff → eval-gated
```

It reuses existing capture points (no parallel machinery): `ChatSession`
`tool_trace` / `on_tool_call`, `SkillRun.steps`, the report's `Finding` list and
`Confidence`, `Provenance` + `ReportValidationError` (D4), and fit-quality
outputs (`rw`).

## What is captured (redacted)

An **episode** stores category labels and identifiers only — never data arrays,
evidence values, cell parameters, or message text: `domain, model, input_hash,
file basenames, tool names, finding {diagnostic, severity}, confidence,
provenance_complete, n_figures, outcome, duration`.

**Signals** derived deterministically: tool error, unhandled warning,
interpretation-unavailable, JSON-retry used, provenance gap, validation failure,
empty result, low confidence, poor fit, mis-route.

**Corrections** — the highest-value signal, from a **human only**, never
inferred: `record_correction(episode, target, statement, corrected_value)` (API
+ `learn correct` CLI + a chat affordance). *Example:* "GaNb4Se8 transitions are
50 K and 29 K, not 39 K" → becomes a pinned regression eval.

## Proposals — tiered by risk

| Tier | Change class | On approval |
|------|-------------|-------------|
| **0 — data** | eval/regression case, knowledge snippet, `next_check_rule` entry, numeric config threshold | write diff → run eval + tests → **land only if green** (human commits) |
| **1 — draft-only** | prompt *version bump* (drafted, never activated), tool description, router rule | write a draft; a human edits & commits |
| **2 — never touched** | scientific algorithms, schemas, core code | open a task with evidence; the system does **not** write these files |

## Safety invariants (to be tested)

1. Capture and proposal generation never write to `src/`.
2. No proposal auto-applies; every apply needs explicit approval.
3. Tier-0 apply is eval-gated and reversible (git); Tier-1 is draft-only;
   Tier-2 files are denylisted — attempting to auto-edit one **raises**.
4. Full audit: proposal → evidence episodes → applied commit.
5. Local-first; nothing transmitted; the LLM only phrases already-local data.

## Phases

- **P1 — Capture (DONE):** `learning/journal.py` — append-only, redacted,
  opt-in episode log wired into `Agent.analyze`; `scattering-ai learn status`.
  Read-only; a journaling failure never affects analysis.
- **P2 — Signals + corrections:** deterministic signal extraction;
  `record_correction`; `learn signals` clustering report.
- **P3 — Proposals + review:** templated proposals (LLM prose only);
  `learn review` CLI; proposals markdown.
- **P4 — Guarded apply:** Tier-0 appliers (diff + eval gate + tier enforcement);
  Tier-1 drafts; Tier-2 task hand-off.
- **P5 — Close the loop:** demonstrate correction → eval case → verified fix
  end-to-end, replaying the GaNb4Se8 case as the acceptance test.

## Using P1 today

```bash
export SCATTERING_AI_JOURNAL=~/.scattering_ai/journal   # opt-in; default is off
scattering-ai analyze --file my_data.gr                 # each run appends an episode
scattering-ai learn status                              # aggregate view of the journal
```

Or in Python: `Agent(journal="path/").analyze(request)`.
