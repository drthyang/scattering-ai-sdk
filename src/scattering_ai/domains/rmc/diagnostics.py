"""Deterministic diagnostics for RMC runs.

Everything here must be computable without an LLM. Each diagnostic returns
findings whose ``evidence`` carries the numbers behind the conclusion, so
downstream reasoning can cite instead of recompute.
"""

from __future__ import annotations

from enum import Enum

from scattering_ai.core.findings import Finding, Severity
from scattering_ai.domains.rmc.schemas import RMCRunState, RMCSeries

TREND_WINDOW = 20
# Fractional change over the window below which a series counts as flat.
FLAT_THRESHOLD = 0.02
# Fraction of sign changes in successive diffs above which a series oscillates.
OSCILLATION_RATIO = 0.5
# Relative amplitude around the trend line below which jitter is ignored:
# a series alternating within 0.5% of its mean is flat, not oscillating.
OSCILLATION_AMPLITUDE = 0.005
MIN_POINTS = 4


# Deterministic next-check rules for RMC findings (decision D10: technique
# content lives on the domain pack, not in core). Keyed by finding rule-key
# (``diagnostic`` or ``diagnostic:trend``).
NEXT_CHECK_RULES: dict[str, str] = {
    "rwp_trend:flat": "Check the move acceptance rate; near-zero acceptance suggests "
    "over-tight constraints, very high acceptance suggests loose dataset weights.",
    "rwp_trend:oscillating": "Inspect dataset weights for competing datasets or "
    "constraints pulling the configuration in opposite directions.",
    "rwp_trend:increasing": "Check for mid-run changes to weights or constraints and "
    "verify the restart configuration file.",
    "dataset_kind_conflict": "Compare partial PDFs and rebalance dataset weights; a "
    "robust local-structure signal should survive moderate weight changes.",
    "radiation_conflict": "Identify which partial correlations dominate each probe "
    "(neutron b vs x-ray Z weighting) and check element-specific misfit.",
    "missing_files": "Locate or regenerate the missing files before trusting the analysis.",
    "log_error": "Inspect the run log around the reported error line.",
    "log_warning": "Review the warning in the run log and confirm it is benign.",
}


class Trend(str, Enum):
    DECREASING = "decreasing"
    FLAT = "flat"
    OSCILLATING = "oscillating"
    INCREASING = "increasing"
    INSUFFICIENT = "insufficient_data"


def classify_trend(values: list[float], window: int = TREND_WINDOW) -> tuple[Trend, dict]:
    """Classify the recent trend of an R-value series.

    Uses a least-squares slope over the last ``window`` points, normalized by
    the window mean, so thresholds are scale-free (Rwp in % or absolute chi^2
    behave the same).
    """
    if len(values) < MIN_POINTS:
        return Trend.INSUFFICIENT, {"n_points": len(values)}

    tail = values[-window:]
    n = len(tail)
    mean = sum(tail) / n
    if mean == 0:
        return Trend.INSUFFICIENT, {"n_points": n, "mean": 0.0}

    xs = range(n)
    x_mean = (n - 1) / 2
    slope = sum((x - x_mean) * (y - mean) for x, y in zip(xs, tail, strict=True)) / sum(
        (x - x_mean) ** 2 for x in xs
    )
    rel_change = slope * (n - 1) / mean

    diffs = [b - a for a, b in zip(tail, tail[1:], strict=False)]
    nonzero = [d for d in diffs if d != 0]
    sign_changes = sum(1 for a, b in zip(nonzero, nonzero[1:], strict=False) if (a > 0) != (b > 0))
    oscillation = sign_changes / max(len(nonzero) - 1, 1)

    residuals = [y - (mean + slope * (x - x_mean)) for x, y in zip(xs, tail, strict=True)]
    rel_amplitude = (sum(r * r for r in residuals) / n) ** 0.5 / abs(mean)

    evidence = {
        "window": n,
        "first": tail[0],
        "last": tail[-1],
        "relative_change_over_window": round(rel_change, 4),
        "oscillation_ratio": round(oscillation, 3),
        "relative_amplitude": round(rel_amplitude, 4),
    }

    if (
        abs(rel_change) < FLAT_THRESHOLD
        and oscillation > OSCILLATION_RATIO
        and rel_amplitude > OSCILLATION_AMPLITUDE
    ):
        return Trend.OSCILLATING, evidence
    if rel_change <= -FLAT_THRESHOLD:
        return Trend.DECREASING, evidence
    if rel_change >= FLAT_THRESHOLD:
        return Trend.INCREASING, evidence
    return Trend.FLAT, evidence


def check_convergence(state: RMCRunState) -> list[Finding]:
    findings: list[Finding] = []
    for series in state.series:
        trend, evidence = classify_trend(series.values)
        severity = {
            Trend.DECREASING: Severity.INFO,
            Trend.FLAT: Severity.WARNING,
            Trend.OSCILLATING: Severity.WARNING,
            Trend.INCREASING: Severity.WARNING,
            Trend.INSUFFICIENT: Severity.INFO,
        }[trend]
        findings.append(
            Finding(
                diagnostic="rwp_trend",
                severity=severity,
                message=f"R-value series '{series.label}' is {trend.value} "
                f"over the last {evidence.get('window', 0)} points.",
                evidence={"series": series.label, "trend": trend.value, **evidence},
            )
        )
    return findings


def _trend_of(series: RMCSeries) -> Trend:
    trend, _ = classify_trend(series.values)
    return trend


def check_dataset_conflicts(state: RMCRunState) -> list[Finding]:
    """Flag pairs of datasets pulling the fit in opposite directions."""
    findings: list[Finding] = []
    improving = {Trend.DECREASING}
    worsening = {Trend.INCREASING}

    def conflict_pairs(axis: str, key):
        groups: dict[str, list[RMCSeries]] = {}
        for s in state.series:
            groups.setdefault(key(s), []).append(s)
        labels = [k for k in groups if k != "other"]
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                for sa in groups[a]:
                    for sb in groups[b]:
                        ta, tb = _trend_of(sa), _trend_of(sb)
                        if (ta in improving and tb in worsening) or (
                            ta in worsening and tb in improving
                        ):
                            findings.append(
                                Finding(
                                    diagnostic=f"{axis}_conflict",
                                    severity=Severity.WARNING,
                                    message=f"Dataset conflict: '{sa.label}' is {ta.value} "
                                    f"while '{sb.label}' is {tb.value}.",
                                    evidence={
                                        "axis": axis,
                                        "series_a": sa.label,
                                        "trend_a": ta.value,
                                        "series_b": sb.label,
                                        "trend_b": tb.value,
                                    },
                                )
                            )

    conflict_pairs("dataset_kind", lambda s: s.kind)
    conflict_pairs("radiation", lambda s: s.radiation)
    return findings


def check_missing_files(state: RMCRunState) -> list[Finding]:
    present = set(state.present_files)
    missing = [f for f in state.expected_files if f not in present]
    if not missing:
        return []
    return [
        Finding(
            diagnostic="missing_files",
            severity=Severity.ERROR,
            message=f"{len(missing)} expected file(s) missing: {', '.join(missing)}.",
            evidence={"missing": missing},
        )
    ]


def check_log_warnings(state: RMCRunState) -> list[Finding]:
    findings: list[Finding] = []
    for line in state.log_tail:
        lowered = line.lower()
        if "error" in lowered:
            findings.append(
                Finding(
                    diagnostic="log_error",
                    severity=Severity.ERROR,
                    message="Error found in log tail.",
                    evidence={"line": line.strip()},
                )
            )
        elif "warning" in lowered:
            findings.append(
                Finding(
                    diagnostic="log_warning",
                    severity=Severity.WARNING,
                    message="Warning found in log tail.",
                    evidence={"line": line.strip()},
                )
            )
    return findings


def run_all(state: RMCRunState) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(check_convergence(state))
    findings.extend(check_dataset_conflicts(state))
    findings.extend(check_missing_files(state))
    findings.extend(check_log_warnings(state))
    return findings
