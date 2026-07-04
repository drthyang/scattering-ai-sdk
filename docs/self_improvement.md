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

## Safety invariants (enforced + tested in `tests/test_learning.py`)

1. Capture and proposal generation never write to `src/` — capture/signals/
   proposals only read; the denylist blocks `src/` writes at apply.
2. No proposal auto-applies; every apply needs explicit approval
   (`test_nothing_applies_without_approval`).
3. Tier-0 apply is eval-gated and reversible; Tier-1 is draft-only; Tier-2
   files are denylisted — attempting to write one **raises** `TierViolation`
   (`test_denylist_blocks_source_and_schemas`, `test_tier0_reverts_*`).
4. Full audit: proposal → evidence episodes → outcome, in `applied.jsonl`
   (`test_tier0_lands_only_when_gate_green`). A human makes the git commit.
5. Local-first; nothing transmitted; the LLM only phrases already-local prose
   (`test_prose_polish_touches_only_title_and_rationale`).

## Phases

- **P1 — Capture (DONE):** `learning/journal.py` — append-only, redacted,
  opt-in episode log wired into `Agent.analyze`; `scattering-ai learn status`.
  Read-only; a journaling failure never affects analysis.
- **P2 — Signals + corrections (DONE):** `learning/signals.py` — deterministic
  signal extraction (`error_outcome`, `unhandled_warning`, `provenance_gap`,
  `low_confidence`, `interpretation_unavailable`, `empty_result`) with no LLM and
  no data values; human-only `Agent.record_correction` + `learn correct` CLI
  (corrections stored verbatim in `corrections.jsonl`, referencing the episode
  they correct); `learn signals` clusters signals by (type, domain, key) ranked
  by severity × recurrence. `ChatSession.record_correction` is the in-chat
  affordance — an explicit human act (never model-inferred), attributed to the
  chat session and landing in the same journal as `analyze` runs.
- **P3 — Proposals + review (DONE):** `learning/proposals.py` — each recurring
  signal cluster becomes a tier-classified `Proposal` from a **deterministic**
  template (tier, change class, evidence, and any corrected value come from the
  signals, never a model). Corrections always propose (Tier-0 regression eval,
  or Tier-1 router rule for a mis-route); other signals propose only past a
  recurrence threshold (default 3). An optional LLM `polish_prose` rewrites
  **only** `title`/`rationale` — `_apply_polish` re-applies just those two
  fields, so even a misbehaving model can't touch evidence or suggested values.
  `learn review` renders proposals as markdown grouped by tier (`--json`,
  `--write proposals.md`).
- **P4 — Guarded apply (DONE):** `learning/apply.py` — `apply_proposal` under
  the tier guarantees. Nothing applies without `approve=True` (CLI `--approve`;
  otherwise a dry run). Tier-0 writes a diff into an **allowlisted** data
  location (`tests/regressions/`, `knowledge/`), runs the eval gate
  (`pytest_gate`, a subprocess), and keeps it only if **green** — else reverts
  byte-for-byte (prior content restored, new files + empty dirs removed). Tier-1
  writes an inert draft; Tier-2 writes a task with evidence; neither touches
  source. A `_DENYLIST` (`src/`, `schemas.py`) is enforced on **every** write and
  **raises** `TierViolation` even if a proposal is mislabeled. Every apply is
  audited to `applied.jsonl` (proposal → episodes → files → gate → outcome).
  New `tests/test_regressions.py` is the permanent gate: applied cases live in
  `tests/regressions/*.json`, each carrying a `check` the gate runs — a
  `series_transitions` case synthesizes a temperature series and asserts the
  **real** `detect_transitions` recovers the corrected temperatures, so a
  regression in the scientific code path fails the gate (no gitignored data
  needed); other cases fall back to a well-formedness check.
- **P5 — Close the loop (DONE):** `test_real_pytest_gate_runs_the_science_end_to_end`
  and the `learn apply` CLI test drive correction → proposal → gate → pinned case
  end-to-end; the GaNb4Se8 correction (`transition_temperature = 50,29`) lands as
  a Tier-0 regression case whose data-dependent check runs `detect_transitions`
  through the real subprocess gate.

## Using P1 today

```bash
export SCATTERING_AI_JOURNAL=~/.scattering_ai/journal   # opt-in; default is off
scattering-ai analyze --file my_data.gr                 # each run appends an episode
scattering-ai learn status                              # aggregate view of the journal
scattering-ai learn signals                             # cluster recurring signals + corrections
scattering-ai learn correct --episode <id> \            # record a human correction
    --target transition_temperature \
    --statement "GaNb4Se8 transitions are 50 K and 29 K, not 39 K" --value 50,29
scattering-ai learn review                              # turn signals into reviewable proposals
scattering-ai learn apply --id <proposal-id>            # dry run: show what would change
scattering-ai learn apply --id <proposal-id> --approve  # Tier-0: eval-gated diff (human commits)
```

Or in Python: `Agent(journal="path/").analyze(request)` and
`agent.record_correction(episode, target=..., statement=..., corrected_value=...)`.
