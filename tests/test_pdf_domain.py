"""B4 first technique pack: PDF / total-scattering domain.

Unit tests pin the deterministic diagnostics on synthetic curves (run in CI
without data); the real-data tests reproduce the known gotchas on the user's
files (skipped when absent). The architecture tests assert the pack plugs into
the same DomainPack interface with no core reasoning changes.
"""

from pathlib import Path

import numpy as np
import pytest

from scattering_ai import analyze
from scattering_ai.core.findings import Severity
from scattering_ai.domains.pdf import diagnostics as d
from scattering_ai.domains.registry import get_domain
from scattering_ai.tools.models import Curve1D

DATA = Path(__file__).parents[1] / "data" / "1d" / "curves"
XRAY_GR = DATA / "XRAY_FeCoSn_100K_converted.gr"
NEUTRON_GR = DATA / "Neutron_NOM_9999_GaTa4Se8_at_5K_50_clean_SQ.gr"
NEUTRON_SQ = DATA / "Neutron_NOM_9999_GaTa4Se8_at_5K_50_clean_SQ.dat"

RNG = np.random.default_rng(7)


def _gr(baseline_slope: float, low_r_spike: float = 0.0) -> Curve1D:
    """Synthetic G(r): a low-r baseline of the given sign plus real peaks."""
    x = np.linspace(0.01, 20.0, 3000)
    y = baseline_slope * x * np.exp(-x / 8.0)
    for height, center in [(5.0, 2.5), (3.0, 4.0), (2.0, 5.6)]:
        y += height * np.exp(-4 * np.log(2) * (x - center) ** 2 / 0.25**2)
    y += RNG.normal(0, 0.01, x.size)
    if low_r_spike:
        y[x < 0.8] += low_r_spike
    return Curve1D(x=x, y=y, xlabel="r (Å)", ylabel="G(r)")


def _sq(tail_level: float) -> Curve1D:
    """Synthetic S(Q) oscillating around ``tail_level`` at high Q."""
    q = np.linspace(0.5, 30.0, 2000)
    y = tail_level + 0.6 * np.exp(-q / 6.0) * np.sin(2 * q)
    return Curve1D(x=q, y=y, xlabel="Q (1/Å)", ylabel="S(Q)")


# ---------------------------------------------------------------- unit tests


def test_is_r_space_routing():
    assert d.is_r_space(_gr(-0.25))
    assert not d.is_r_space(_sq(1.0))
    assert not d.is_r_space(_sq(0.0))


def test_healthy_gr_baseline_ok_and_first_peak():
    curve = _gr(baseline_slope=-0.25)
    kinds = {f.diagnostic: f for f in d.check_baseline_slope(curve) + d.check_first_peak(curve)}
    assert "pdf_baseline_ok" in kinds
    assert kinds["pdf_baseline_ok"].severity == Severity.INFO
    assert kinds["pdf_baseline_ok"].evidence["slope"] < 0
    assert abs(kinds["pdf_first_peak"].evidence["r"] - 2.5) < 0.1


def test_inverted_gr_flags_baseline_slope():
    """Positive low-r slope = not a standard G(r) (the inverted-.gr gotcha)."""
    curve = _gr(baseline_slope=+0.25)
    findings = d.check_baseline_slope(curve)
    assert findings[0].diagnostic == "pdf_baseline_slope"
    assert findings[0].severity == Severity.WARNING
    assert findings[0].evidence["slope"] > 0


def test_low_r_artifact_flagged():
    clean = d.check_low_r_artifact(_gr(-0.25))
    assert clean == []  # no spurious low-r structure
    spiked = d.check_low_r_artifact(_gr(-0.25, low_r_spike=4.0))
    assert spiked and spiked[0].diagnostic == "pdf_low_r_artifact"
    assert spiked[0].evidence["ratio"] > d.LOW_R_ARTIFACT_FRACTION


def test_sq_convention_mismatch_by_name_and_tail():
    # named like S(Q), tail -> 0  => stores S(Q)-1
    mism = d.check_sq_convention(_sq(tail_level=0.0), "Neutron_9999_clean_SQ.dat")
    assert mism[0].diagnostic == "sq_convention_mismatch"
    assert mism[0].severity == Severity.WARNING
    # a true S(Q) (tail -> 1) is not a mismatch
    ok = d.check_sq_convention(_sq(tail_level=1.0), "sample_SofQ.dat")
    assert ok[0].diagnostic == "sq_convention"
    assert ok[0].severity == Severity.INFO


def test_q_range_reports_ripple_period():
    findings = d.check_q_range(_sq(1.0))
    ev = findings[0].evidence
    assert ev["qmax"] > ev["qmin"]
    assert abs(ev["ripple_period"] - 2 * np.pi / ev["qmax"]) < 1e-3


def test_missing_and_unreadable_files():
    assert d.diagnose_file("/no/such/file.gr")[0].diagnostic == "missing_files"


# ------------------------------------------------------- architecture (B4)


def test_pdf_pack_registered_with_same_interface():
    pack = get_domain("pdf")
    assert pack.name == "pdf"
    assert pack.prompt_version == "pdf_interpret/v2"
    # next-check rules live on the pack (decision D10), not in core
    assert "pdf_baseline_slope" in pack.next_check_rules
    # rmc rules stayed with their pack through the same move
    assert "rwp_trend:flat" in get_domain("rmc").next_check_rules


def test_pack_next_check_rules_surface_in_offline_report(tmp_path):
    """The generic agent floor pulls next-checks from pack.next_check_rules —
    no PDF-specific code in core."""
    x = np.linspace(0.01, 20.0, 3000)
    y = 0.25 * x * np.exp(-x / 8.0)  # positive low-r slope -> baseline warning
    for h, c in [(5.0, 2.5), (3.0, 4.0)]:
        y += h * np.exp(-4 * np.log(2) * (x - c) ** 2 / 0.25**2)
    path = tmp_path / "inverted.gr"
    # pdfgetx marker + #L label (2-space separated) so the loader tags r-space
    rows = "\n".join(f"{a} {b}" for a, b in zip(x, y, strict=True))
    path.write_text("outputtype = gr\n#### start data\n#L r  G(r)\n" + rows)
    report = analyze(domain="pdf", question="Is this standard G(r)?",
                     data={"files": [str(path)]})
    assert any("does not fall" in w for w in report.warnings)
    assert any("standard G(r)" in c for c in report.recommended_next_checks)


# ------------------------------------------------------------- real data


@pytest.mark.skipif(not XRAY_GR.exists(), reason="real data not present")
def test_real_xray_gr_is_healthy():
    report = analyze(domain="pdf", question="Is this G(r) healthy?",
                     data={"files": [str(XRAY_GR)]})
    assert report.warnings == []
    text = " ".join(report.observations).lower()
    assert "negative slope" in text and "first pdf peak" in text


@pytest.mark.skipif(not NEUTRON_GR.exists(), reason="real data not present")
def test_real_neutron_gr_flagged_inverted():
    """Known open question: this neutron .gr is inverted/rescaled vs standard."""
    report = analyze(domain="pdf", question="Is this standard G(r)?",
                     data={"files": [str(NEUTRON_GR)]})
    assert any("does not fall" in w for w in report.warnings)


@pytest.mark.skipif(not NEUTRON_SQ.exists(), reason="real data not present")
def test_real_nomad_sq_convention_mismatch():
    report = analyze(domain="pdf", question="What convention is this S(Q)?",
                     data={"files": [str(NEUTRON_SQ)]})
    assert any("S(Q)-1" in w for w in report.warnings)
