"""Pinned regression cases applied by the self-improvement loop (P4/P5).

Each JSON file in ``tests/regressions/`` is a machine-checkable fact captured
from a **human correction** and applied by ``scattering-ai learn apply``. This
module is the *eval gate*: a Tier-0 apply writes its case here, runs this test,
and lands only if it stays green.

Each case carries a ``check`` describing how to verify it:

- ``value_present`` — the case is well-formed and carries a target + corrected
  value (the universal minimum, no data needed).
- ``series_transitions`` — the corrected value is a set of transition
  temperatures; we synthesize a temperature series with slope changes at those
  temperatures and assert the **real** changepoint detector recovers them. This
  runs the actual scientific code path deterministically, without the (gitignored)
  raw data — so a regression in ``detect_transitions`` would fail the gate.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scattering_ai.tools.series import detect_transitions

_DIR = Path(__file__).parent / "regressions"
_FILES = sorted(_DIR.glob("*.json")) if _DIR.exists() else []


def _synthetic_series(kinks: list[float]) -> tuple[list[float], list[float]]:
    """A temperature grid and a continuous piecewise-linear quantity whose slope
    changes at each kink temperature — the signature a transition leaves in a
    tracked peak parameter."""
    temps = list(np.arange(5.0, 106.0, 5.0))
    slopes = [0.0008, -0.0020, 0.0015][: len(kinks) + 1]
    values, v = [], 0.10
    for i, t in enumerate(temps):
        if i > 0:
            seg = min(sum(1 for k in kinks if k <= temps[i - 1] + 1e-9), len(slopes) - 1)
            v += slopes[seg] * (t - temps[i - 1])
        values.append(v)
    return temps, values


def _check_series_transitions(case: dict) -> None:
    expected = sorted(float(t) for t in case["check"]["temperatures"])
    tol = float(case["check"].get("tol", 8.0))
    temps, values = _synthetic_series(expected)
    result = detect_transitions(temps, values)
    found = [t["param"] for t in result["transitions"]]
    assert result["detected"], f"{case['proposal_id']}: no transition detected"
    for temp in expected:
        assert any(abs(temp - f) <= tol for f in found), (
            f"{case['proposal_id']}: no changepoint within {tol} K of {temp} "
            f"(found {found})")


@pytest.mark.parametrize("path", _FILES, ids=[p.stem for p in _FILES])
def test_regression_case(path):
    case = json.loads(path.read_text(encoding="utf-8"))
    assert case.get("proposal_id"), f"{path.name}: missing proposal_id"
    assert case.get("target"), f"{path.name}: missing target"
    assert str(case.get("corrected_value", "")).strip(), \
        f"{path.name}: missing corrected_value"

    kind = case.get("check", {}).get("kind", "value_present")
    if kind == "series_transitions":
        _check_series_transitions(case)


def test_regressions_loader_valid_when_empty():
    # A Tier-0 gate must be runnable before any case exists.
    assert _DIR.exists() or not _FILES
