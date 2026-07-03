"""Tests against real datasets in data/ (skipped when files are absent —
data/ is gitignored, so CI machines won't have them).

These pin known answers: peak positions, transform agreement with
diffpy.pdfgetx, and physical sanity of G(r). See data/*/INVENTORY.md.
"""

from pathlib import Path

import numpy as np
import pytest

from scattering_ai.tools.curves import detect_sq_convention, find_peaks, sq_to_gr
from scattering_ai.tools.io import load_curve

CURVES = Path(__file__).parents[1] / "data" / "1d" / "curves"
NOMAD_SQ = CURVES / "Neutron_NOM_9999_GaTa4Se8_at_5K_50_clean_SQ.dat"
NOMAD_GR = CURVES / "Neutron_NOM_9999_GaTa4Se8_at_5K_50_clean_SQ.gr"
XRAY_FQ = CURVES / "XRAY_FeCoSn_100K_converted.fq"
XRAY_GR = CURVES / "XRAY_FeCoSn_100K_converted.gr"

needs = pytest.mark.skipif(not NOMAD_SQ.exists(), reason="real data not present")


@needs
def test_load_nomad_ascii():
    curve = load_curve(NOMAD_SQ)
    assert curve.x.size == 2251
    assert curve.x[0] == 0.0 and abs(curve.x[-1] - 45.0) < 1e-6
    # File is named SQ but stores S(Q)-1 (high-Q tail -> 0): detection must
    # catch this — transforming the wrong convention inverts the physics.
    assert detect_sq_convention(curve) == "sq_minus_1"


@needs
def test_load_pdfgetx_with_metadata():
    curve = load_curve(NOMAD_GR)
    assert curve.xlabel.startswith("r")
    assert curve.meta["format"] == "pdfgetx"
    assert curve.meta["pdfgetx"]["qmax"] == "35"
    assert curve.meta["pdfgetx"]["mode"] == "neutron"


@needs
def test_xray_fq_to_gr_matches_pdfgetx():
    """Transform the x-ray F(Q) and compare with pdfgetx's own G(r)."""
    fq = load_curve(XRAY_FQ)
    ref = load_curve(XRAY_GR)
    gr = sq_to_gr(fq, qmin=0.5, qmax=28.0, rmax=20.0, dr=0.01, input_kind="fq")

    mask = (ref.x >= 1.0) & (ref.x <= 15.0)
    mine = np.interp(ref.x[mask], gr.x, gr.y)
    theirs = ref.y[mask]
    corr = float(np.corrcoef(mine, theirs)[0, 1])
    scale = float(np.polyfit(mine, theirs, 1)[0])
    assert corr > 0.99
    assert abs(scale - 1.0) < 0.15  # small scale offset from termination handling


@needs
def test_neutron_sq_to_gr_is_physical():
    """The neutron transform must produce a physically valid G(r).

    Note: the companion .gr file appears inverted and rescaled relative to a
    standard G(r) (positive low-r region, amplitude ~10x small); shape
    correlation is checked, but sign/scale agreement is intentionally NOT
    asserted against it. See data/1d/curves/INVENTORY.md.
    """
    sq = load_curve(NOMAD_SQ)
    ref = load_curve(NOMAD_GR)
    gr = sq_to_gr(sq, qmin=0.5, qmax=35.0, rmax=20.0, dr=0.01)

    # Low-r slope ~ -4*pi*rho0*r must be negative
    low = (gr.x >= 0.3) & (gr.x <= 1.2)
    assert np.polyfit(gr.x[low], gr.y[low], 1)[0] < -0.1

    # First coordination peak at the known Ga/Ta-Se bond distance (~2.5 A)
    window = (gr.x >= 1.5) & (gr.x <= 4.0)
    first_peak = gr.x[window][np.argmax(gr.y[window])]
    assert 2.4 < first_peak < 2.6

    # Shape agreement with the pdfgetx output (sign-agnostic)
    mask = (ref.x >= 1.0) & (ref.x <= 15.0)
    mine = np.interp(ref.x[mask], gr.x, gr.y)
    assert abs(float(np.corrcoef(mine, ref.y[mask])[0, 1])) > 0.99


@needs
def test_find_real_xray_pdf_peaks():
    ref = load_curve(XRAY_GR)
    peaks = find_peaks(ref, min_prominence=0.5)
    positions = sorted(p["x"] for p in peaks)[:3]
    # FeCoSn: nearest-neighbor metal-metal/metal-Sn distances ~2.6 A
    assert 2.4 < positions[0] < 2.8
