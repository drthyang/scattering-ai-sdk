"""RMC Monitor connector (roadmap C5).

The application owns zero AI logic: it exports JSON, calls
``analyze_monitor``, and displays the returned report. This module owns the
translation from monitor payload shapes to the SDK's AnalysisRequest.

Accepted payload shapes (checked in this order):

1. **Full request** — already AnalysisRequest-shaped
   (has ``domain`` and ``question``): validated as-is.
2. **RMC state** — the typed run state at top level
   (has ``series``; optionally ``expected_files``/``present_files``/
   ``log_tail``): wrapped into ``data.metadata.rmc``.
3. **Legacy flat** — just ``r_values`` (single overall series) plus optional
   ``files``: mapped onto ``AnalysisData`` directly.

The recommended export for new RMC Monitor versions is shape 2 — see
``RMCRunState`` in ``scattering_ai.domains.rmc.schemas`` for the fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scattering_ai.core.agent import Agent
from scattering_ai.core.schemas import AnalysisReport, AnalysisRequest

DEFAULT_QUESTION = "Is this RMC run healthy?"


def from_monitor_json(
    payload: dict[str, Any] | str | Path, question: str = ""
) -> AnalysisRequest:
    """Convert an RMC Monitor payload (dict or JSON file path) to a request."""
    if not isinstance(payload, dict):
        payload = json.loads(Path(payload).read_text())

    if "domain" in payload and "question" in payload:
        request = AnalysisRequest.model_validate(payload)
        if question:
            request.question = question
        return request

    if "series" in payload:
        from scattering_ai.domains.rmc.schemas import RMCRunState

        state = RMCRunState.model_validate(payload)  # validate before wrapping
        return AnalysisRequest(
            domain="rmc",
            question=question or DEFAULT_QUESTION,
            data={"metadata": {"rmc": state.model_dump()}},
        )

    if "r_values" in payload:
        return AnalysisRequest(
            domain="rmc",
            question=question or DEFAULT_QUESTION,
            data={
                "r_values": payload["r_values"],
                "files": payload.get("files", []),
                "metadata": {k: v for k, v in payload.items()
                             if k not in ("r_values", "files")},
            },
        )

    raise ValueError(
        "Unrecognized RMC Monitor payload: expected an AnalysisRequest "
        "(domain+question), an RMC state (series), or a legacy payload "
        f"(r_values). Got keys: {sorted(payload)}"
    )


def analyze_monitor(
    payload: dict[str, Any] | str | Path,
    question: str = "",
    agent: Agent | None = None,
) -> AnalysisReport:
    """One call for applications: payload in, provenance-carrying report out.

    Pass an ``Agent`` configured with an LLM for interpretation; the default
    runs deterministic diagnostics + knowledge retrieval only (offline).
    """
    request = from_monitor_json(payload, question=question)
    return (agent or Agent()).analyze(request)
