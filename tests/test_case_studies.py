"""Roadmap D5: reproducible case-study runs.

The two RMC cases run on committed demo data; the data-gated cases skip cleanly.
"""

from scattering_ai.evaluation.case_studies import (
    CASE_STUDIES,
    run_all,
    run_case_study,
)


def _by_name(results):
    return {r.name: r for r in results}


def test_committed_rmc_cases_pass_and_data_gated_never_fail():
    results = _by_name(run_all())

    # committed demo data -> always runs. local-vs-average conflict on the
    # stalled run; decreasing/no-conflict on the healthy run.
    assert results["rmc_local_vs_average_conflict"].status == "passed"
    assert results["rmc_healthy_convergence"].status == "passed"

    # data-gated cases: pass when the (gitignored) data is present locally,
    # skip when absent (e.g. CI) — but never fail.
    for name in ("pdf_inverted_neutron_gr", "phase_transition_gaNb4Se8",
                 "diffuse_anisotropy_corelli"):
        assert results[name].status in ("passed", "skipped"), results[name]


def test_every_case_declares_a_publication_angle():
    for cs in CASE_STUDIES:
        assert cs.publication_angle and cs.must_contain
        assert cs.kind in ("monitor", "files")


def test_failed_check_is_reported(tmp_path):
    from scattering_ai.evaluation.case_studies import CaseStudy

    bogus = CaseStudy(
        name="bogus", description="Is this RMC run healthy?",
        publication_angle="x", kind="monitor",
        data=["examples/rmc_monitor_demo/healthy_run.json"],
        must_contain=["this phrase will never appear in the report"])
    result = run_case_study(bogus)
    assert result.status == "failed" and result.unmet
