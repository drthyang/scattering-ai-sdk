"""Human-reviewed self-improvement — Phase 4: guarded apply.

Turns an approved :class:`~scattering_ai.learning.proposals.Proposal` into a
change, under the safety invariants that make this loop trustworthy:

1. **Nothing auto-applies.** ``apply_proposal`` refuses unless ``approve=True``
   is passed explicitly (the CLI requires ``--approve``).
2. **Tier enforcement is structural, not advisory.**
   - *Tier 0 (data)* writes a diff into an **allowlisted** repo location
     (``tests/regressions/``, ``knowledge/``), runs the eval gate, and keeps the
     change only if the gate is **green** — otherwise it is reverted byte-for-
     byte. A human makes the git commit.
   - *Tier 1 (draft-only)* writes an inert draft into the journal; it is never
     activated. A human edits and commits it.
   - *Tier 2 (never touched)* writes only a task note with evidence; it never
     writes source.
3. **A denylist is enforced on every write.** Any attempt to write under
   ``src/`` (scientific logic, schemas, core code) or a ``schemas.py`` **raises**
   ``TierViolation`` — even if a proposal is mislabeled Tier 0.
4. **Every apply is audited** to ``applied.jsonl``: proposal → evidence episodes
   → files → gate result → outcome.
5. **Local-first.** Everything is local files; nothing is transmitted.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from scattering_ai.learning.proposals import Proposal

# Targets whose corrected value is a set of transition temperatures — a
# regression case for one of these gets a data-dependent check that exercises
# the real changepoint detector (see tests/test_regressions.py).
_SERIES_TRANSITION_TARGETS = {
    "transition_temperature", "transition_temperatures", "transitions",
}


def _parse_floats(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text or "")]


def synthesize_check(domain: str, target: str, corrected_value: str) -> dict:
    """Derive a machine-checkable assertion from a correction.

    For a ``series``/transition-temperature correction this pins a
    *data-dependent* check the eval gate can actually run against the real
    scientific code path; otherwise it falls back to a well-formedness check.
    Deterministic — the corrected numbers come straight from the human's value.
    """
    numbers = _parse_floats(corrected_value)
    if domain == "series" and target in _SERIES_TRANSITION_TARGETS and numbers:
        return {"kind": "series_transitions", "temperatures": numbers[:2], "tol": 8.0}
    return {"kind": "value_present"}

_AUDIT_FILE = "applied.jsonl"

# Paths automated apply must NEVER write, at any tier. This is the structural
# guard behind "Tier-2 files are never touched": scientific algorithms, schemas,
# core pipeline, prompts, and skills all live under src/.
_DENYLIST = ("src/",)
_DENY_SUFFIX = ("schemas.py",)

# Tier-0 may only write data into these locations (checked in addition to the
# denylist). Regression fixtures and knowledge snippets are data, not code.
_TIER0_ALLOWLIST = ("tests/regressions/", "knowledge/")


class TierViolation(RuntimeError):
    """Raised when a write would touch a denylisted (Tier-2) path, or a Tier-0
    write escapes its data allowlist."""


class GateResult(BaseModel):
    passed: bool
    detail: str = ""


class FileWrite(BaseModel):
    path: str  # relative to the write base (repo root for Tier 0; journal otherwise)
    content: str


class AppliedRecord(BaseModel):
    """Audit entry: the full chain from proposal to outcome."""

    proposal_id: str
    tier: int
    change_class: str
    outcome: str  # applied | reverted | draft | task | refused
    approved: bool = False
    gate_passed: bool | None = None
    gate_detail: str = ""
    files: list[str] = Field(default_factory=list)
    evidence_episodes: list[str] = Field(default_factory=list)
    timestamp: str = ""


# --- path safety --------------------------------------------------------------

def _normalize(path: str) -> str:
    p = path.replace("\\", "/")
    if p.startswith("/") or ".." in Path(p).parts:
        raise TierViolation(f"unsafe path outside the write base: {path!r}")
    return p


def assert_not_denylisted(path: str) -> str:
    """Raise ``TierViolation`` if the path is source/schema/core (any tier)."""
    p = _normalize(path)
    if any(p.startswith(pfx) for pfx in _DENYLIST) or p.endswith(_DENY_SUFFIX):
        raise TierViolation(
            f"refusing to write {path!r}: source, schemas, and core code are "
            "Tier-2 (never touched by automated apply)")
    return p


def assert_tier0_target(path: str) -> str:
    """A Tier-0 write must be denylist-clean *and* inside the data allowlist."""
    p = assert_not_denylisted(path)
    if not any(p.startswith(pfx) for pfx in _TIER0_ALLOWLIST):
        raise TierViolation(
            f"Tier-0 apply may only write data ({', '.join(_TIER0_ALLOWLIST)}); "
            f"{path!r} is out of scope")
    return p


# --- writers: change_class -> the files it wants to write ---------------------

def _writes_regression_eval(p: Proposal) -> list[FileWrite]:
    payload = {
        "proposal_id": p.id,
        "source": "human_correction",
        "domain": p.domain,
        "target": p.key,
        "corrected_value": p.corrected_value,
        "statement": p.rationale,
        "evidence_episodes": p.evidence_episodes,
        "check": synthesize_check(p.domain, p.key, p.corrected_value),
    }
    return [FileWrite(path=f"tests/regressions/{p.id}.json",
                      content=json.dumps(payload, indent=2) + "\n")]


def _writes_next_check_rule(p: Proposal) -> list[FileWrite]:
    slug = f"{p.domain or 'general'}__{p.key or 'rule'}"
    md = (
        f"# Proposed next-check rule ({p.domain or 'general'})\n\n"
        f"- **trigger:** `{p.key}`\n"
        f"- **evidence:** {p.evidence_count} occurrence(s)\n\n"
        f"{p.suggested_action}\n\n"
        f"> Drafted by the self-improvement loop from proposal `{p.id}`. "
        "Wire into the pack's `next_check_rules` after review.\n"
    )
    return [FileWrite(path=f"knowledge/next_checks/{slug}.md", content=md)]


def _writes_draft(p: Proposal) -> list[FileWrite]:
    md = (
        f"# DRAFT (not activated): {p.title}\n\n"
        f"- **proposal:** `{p.id}`  ·  **tier 1 · {p.change_class}**\n"
        f"- **domain:** {p.domain or '—'}  ·  **evidence:** {p.evidence_count}\n\n"
        f"## Rationale\n\n{p.rationale}\n\n"
        f"## Suggested action\n\n{p.suggested_action}\n\n"
        "> This is a draft only. A human edits and commits it; the loop never "
        "activates prompts, router rules, or tool descriptions on its own.\n"
    )
    return [FileWrite(path=f"drafts/{p.id}.md", content=md)]


def _writes_task(p: Proposal) -> list[FileWrite]:
    md = (
        f"# TASK: {p.title}\n\n"
        f"- **proposal:** `{p.id}`  ·  **tier 2 · {p.change_class}**\n"
        f"- **domain:** {p.domain or '—'}  ·  **evidence:** {p.evidence_count} "
        f"occurrence(s)\n"
        f"- **episodes:** {', '.join(p.evidence_episodes) or '—'}\n\n"
        f"## Rationale\n\n{p.rationale}\n\n"
        f"## Suggested action\n\n{p.suggested_action}\n\n"
        "> Tier 2: scientific algorithms, schemas, and core code are never "
        "written by the loop. This task hands the evidence to a human.\n"
    )
    return [FileWrite(path=f"tasks/{p.id}.md", content=md)]


_WRITERS: dict[str, Callable[[Proposal], list[FileWrite]]] = {
    "regression_eval": _writes_regression_eval,
    "next_check_rule": _writes_next_check_rule,
    "prompt_version_bump": _writes_draft,
    "router_rule": _writes_draft,
    "task": _writes_task,
}


# --- the eval gate ------------------------------------------------------------

def pytest_gate(repo_root: Path,
                targets: tuple[str, ...] = ("tests/test_regressions.py",)) -> GateResult:
    """Default Tier-0 gate: the change lands only if this test target is green.

    Runs in a subprocess so it can't disturb the running interpreter. The
    default target is the regression suite (fast, and the applied case must
    itself pass); widen ``targets`` to gate on more of the suite.
    """
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *targets],
        cwd=str(repo_root), capture_output=True, text=True)
    tail = (proc.stdout or "")[-800:] + (proc.stderr or "")[-400:]
    return GateResult(passed=proc.returncode == 0, detail=tail.strip())


# --- filesystem helpers (byte-exact, reversible) ------------------------------

def _snapshot(base: Path, writes: list[FileWrite]) -> dict[str, bytes | None]:
    """Record prior contents so a failed gate can be reverted byte-for-byte."""
    snap: dict[str, bytes | None] = {}
    for w in writes:
        target = base / w.path
        snap[w.path] = target.read_bytes() if target.exists() else None
    return snap


def _write_all(base: Path, writes: list[FileWrite]) -> list[str]:
    for w in writes:
        target = base / w.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(w.content, encoding="utf-8")
    return [w.path for w in writes]


def _revert(base: Path, snapshot: dict[str, bytes | None]) -> None:
    for rel, prior in snapshot.items():
        target = base / rel
        if prior is None:
            target.unlink(missing_ok=True)
            # best-effort: drop dirs we newly created and left empty
            parent = target.parent
            while parent != base and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
        else:
            target.write_bytes(prior)


# --- repo discovery + audit ---------------------------------------------------

def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / ".git").exists():
            return d
    return here


def record_applied(journal, record: AppliedRecord) -> None:
    with (journal.dir / _AUDIT_FILE).open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


def applied_records(journal) -> list[AppliedRecord]:
    path = journal.dir / _AUDIT_FILE
    if not path.exists():
        return []
    return [AppliedRecord.model_validate_json(ln)
            for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --- the guarded applier ------------------------------------------------------

def apply_proposal(proposal: Proposal, *, journal, repo_root: Path | str | None = None,
                   approve: bool = False,
                   gate: Callable[[Path], GateResult] | None = None) -> AppliedRecord:
    """Apply one proposal under the tier's guarantees, and audit the outcome.

    ``approve`` must be ``True`` — nothing applies implicitly. ``gate`` overrides
    the default :func:`pytest_gate` (used by tests to inject a deterministic
    result). Returns the :class:`AppliedRecord`; a human makes the git commit.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def audit(outcome: str, files=(), gate_passed=None, gate_detail="") -> AppliedRecord:
        rec = AppliedRecord(
            proposal_id=proposal.id, tier=proposal.tier,
            change_class=proposal.change_class, outcome=outcome, approved=approve,
            gate_passed=gate_passed, gate_detail=gate_detail, files=list(files),
            evidence_episodes=list(proposal.evidence_episodes), timestamp=now)
        record_applied(journal, rec)
        return rec

    if not approve:  # invariant 1: nothing auto-applies
        return audit("refused")

    writer = _WRITERS.get(proposal.change_class)
    if writer is None:
        return audit("refused", gate_detail=f"no applier for {proposal.change_class}")
    writes = writer(proposal)

    if proposal.tier == 0:
        base = Path(repo_root) if repo_root else find_repo_root()
        for w in writes:
            assert_tier0_target(w.path)  # allowlist + denylist, may raise
        snapshot = _snapshot(base, writes)
        files = _write_all(base, writes)
        result = (gate or pytest_gate)(base)
        if result.passed:
            return audit("applied", files, True, result.detail)
        _revert(base, snapshot)  # reversible: gate red -> nothing kept
        return audit("reverted", [], False, result.detail)

    # Tier 1 (draft) and Tier 2 (task): inert local files, no gate, no source.
    base = journal.dir
    for w in writes:
        assert_not_denylisted(w.path)  # defense in depth
    files = _write_all(base, writes)
    return audit("draft" if proposal.tier == 1 else "task", files)
