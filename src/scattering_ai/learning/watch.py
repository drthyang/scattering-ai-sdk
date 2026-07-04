"""Human-reviewed self-improvement — Phase E8: data-gated capability queue.

Capability the roadmap wants but lacks data for (INS / S(Q,ω), Spinvert
reference sets, ...) sits in a **watch queue**. When matching data lands in
``data/``, the loop flags it as *ready to build* and turns it into a Tier-2 task
proposal — which E7 packages into an agent brief. This makes the roadmap's
data-gating explicit and self-clearing: the SDK notices when the thing blocking
a build has arrived, instead of the researcher having to remember.

No data is read or transmitted — only filenames are matched against globs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from scattering_ai.learning.proposals import Proposal, _slug


class WatchItem(BaseModel):
    """A capability that is blocked until matching data appears in ``data/``."""

    name: str
    title: str
    globs: list[str] = Field(default_factory=list)  # recursive globs under data/
    build_hint: str = ""


# The standing queue of data-gated builds (roadmap: `ins` pack, T3 refinement).
WATCH_QUEUE: list[WatchItem] = [
    WatchItem(
        name="ins_pack",
        title="INS domain pack (inelastic neutron scattering, incl. phonons)",
        globs=["**/*sqw*", "**/*inelastic*", "**/*s_q_e*", "**/ins/**/*"],
        build_hint=("Scaffold the `ins` domain pack (S(Q,ω), dispersions, DOS, "
                    "phonons per D12) with a known-answer eval on the new data; "
                    "home for rmc-phonon's k-path utilities."),
    ),
    WatchItem(
        name="spinvert_refine",
        title="Spinvert-style magnetic RMC refinement (T3)",
        globs=["**/*magnetic_diffuse*", "**/*spinvert*", "**/*_mag_iq*"],
        build_hint=("Build the T3 spin-configuration refinement loop, validated "
                    "against this measured magnetic-diffuse reference data."),
    ),
]


def scan_watch_queue(data_root: str | Path,
                     queue: list[WatchItem] | None = None) -> list[tuple[WatchItem, list[str]]]:
    """Return ``(item, matched basenames)`` for every watch item whose data has
    arrived under ``data_root``."""
    root = Path(data_root)
    out: list[tuple[WatchItem, list[str]]] = []
    if not root.exists():
        return out
    for item in queue if queue is not None else WATCH_QUEUE:
        matched: set[str] = set()
        for pattern in item.globs:
            matched.update(p.name for p in root.glob(pattern) if p.is_file())
        if matched:
            out.append((item, sorted(matched)))
    return out


def watch_proposals(data_root: str | Path,
                    queue: list[WatchItem] | None = None) -> list[Proposal]:
    """A Tier-2 task proposal for each watch item now unblocked by data."""
    proposals: list[Proposal] = []
    for item, files in scan_watch_queue(data_root, queue):
        shown = ", ".join(files[:5]) + (", ..." if len(files) > 5 else "")
        proposals.append(Proposal(
            id=_slug("p", "2", "watch", item.name),
            tier=2,
            change_class="task",
            signal_type="data_arrival",
            domain="",
            key=item.name,
            title=f"Data landed — ready to build: {item.title}",
            rationale=(f"{len(files)} matching file(s) in data/ ({shown}). "
                       f"{item.build_hint}"),
            suggested_action=item.build_hint,
            evidence_episodes=[],
            evidence_count=len(files),
        ))
    return proposals


def watch_status(data_root: str | Path,
                 queue: list[WatchItem] | None = None) -> list[dict]:
    """Human-readable queue: each item with whether its data has arrived."""
    ready = {item.name: files for item, files in scan_watch_queue(data_root, queue)}
    return [
        {"name": item.name, "title": item.title,
         "ready": item.name in ready, "files": ready.get(item.name, [])}
        for item in (queue if queue is not None else WATCH_QUEUE)
    ]
