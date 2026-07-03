from scattering_ai.core.findings import Severity
from scattering_ai.domains.rmc.diagnostics import (
    Trend,
    check_dataset_conflicts,
    classify_trend,
    run_all,
)
from scattering_ai.domains.rmc.schemas import RMCRunState, RMCSeries


def decreasing(n=30, start=20.0, step=0.3):
    return [start - i * step for i in range(n)]


def increasing(n=30, start=10.0, step=0.2):
    return [start + i * step for i in range(n)]


def flat(n=30, value=8.0):
    return [value + 0.001 * (i % 2) for i in range(n)]


def oscillating(n=30, value=10.0, amplitude=0.15):
    return [value + amplitude * (-1) ** i for i in range(n)]


def test_classify_trends():
    assert classify_trend(decreasing())[0] == Trend.DECREASING
    assert classify_trend(increasing())[0] == Trend.INCREASING
    assert classify_trend(flat())[0] == Trend.FLAT
    assert classify_trend(oscillating())[0] == Trend.OSCILLATING
    assert classify_trend([1.0, 2.0])[0] == Trend.INSUFFICIENT


def test_trend_evidence_carries_numbers():
    trend, evidence = classify_trend(decreasing())
    assert trend == Trend.DECREASING
    assert evidence["relative_change_over_window"] < 0
    assert evidence["first"] > evidence["last"]


def test_healthy_run_has_no_warnings():
    state = RMCRunState(
        series=[
            RMCSeries(label="bragg_neutron", kind="bragg", radiation="neutron",
                      values=decreasing()),
            RMCSeries(label="pdf_neutron", kind="pdf", radiation="neutron",
                      values=decreasing(start=15.0)),
        ],
        expected_files=["run.log"],
        present_files=["run.log"],
    )
    findings = run_all(state)
    assert all(f.severity == Severity.INFO for f in findings)


def test_stalled_run_flags_flat_trend():
    state = RMCRunState(series=[RMCSeries(label="overall", values=flat())])
    findings = run_all(state)
    flags = [f for f in findings if f.diagnostic == "rwp_trend"]
    assert flags[0].severity == Severity.WARNING
    assert flags[0].evidence["trend"] == "flat"


def test_bragg_pdf_conflict_detected():
    state = RMCRunState(
        series=[
            RMCSeries(label="bragg_neutron", kind="bragg", radiation="neutron",
                      values=increasing()),
            RMCSeries(label="pdf_neutron", kind="pdf", radiation="neutron",
                      values=decreasing()),
        ]
    )
    conflicts = check_dataset_conflicts(state)
    assert any(f.diagnostic == "dataset_kind_conflict" for f in conflicts)


def test_no_conflict_when_both_improve():
    state = RMCRunState(
        series=[
            RMCSeries(label="bragg", kind="bragg", values=decreasing()),
            RMCSeries(label="pdf", kind="pdf", values=decreasing(start=12.0)),
        ]
    )
    assert check_dataset_conflicts(state) == []


def test_missing_files_and_log_scan():
    state = RMCRunState(
        expected_files=["run.log", "run.rmc6f"],
        present_files=["run.log"],
        log_tail=["step 1000 ok", "WARNING: bond length constraint violated"],
    )
    findings = run_all(state)
    diagnostics = {f.diagnostic for f in findings}
    assert "missing_files" in diagnostics
    assert "log_warning" in diagnostics
    missing = next(f for f in findings if f.diagnostic == "missing_files")
    assert missing.evidence["missing"] == ["run.rmc6f"]
    assert missing.severity == Severity.ERROR
