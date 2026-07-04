"""3D difference pair distribution function (3D-ΔPDF) from a diffuse volume.

The ΔPDF is the Fourier transform of the diffuse scattering (total minus Bragg):
its peaks are inter-site pair correlations — positive where a pair distance is
more common than in the average structure, negative where less. The standard
recipe (Weber & Simonov, Z. Kristallogr. 2012) is: punch out the Bragg peaks,
apodize the diffuse volume in Q-space to suppress truncation ripple, subtract
the mean (kills the r=0 pileup), and take the centred Fourier transform.

This is an independent implementation of that well-established method (not a
port of any one package), so it stays deterministic, dependency-light, and MIT.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _window(n: int, kind: str) -> np.ndarray:
    if kind == "none":
        return np.ones(n)
    x = np.linspace(-1.0, 1.0, n)
    if kind == "gaussian":
        return np.exp(-0.5 * (x / 0.5) ** 2)
    return 0.5 * (1.0 + np.cos(np.pi * x))  # Hann (default)


def punch_bragg(data: np.ndarray, n_sigma: float = 6.0, radius: int = 2) -> np.ndarray:
    """Zero out sharp Bragg voxels (intensity > median + n_sigma·MAD) and a
    small neighbourhood, leaving the smooth diffuse signal."""
    finite = np.isfinite(data)
    med = np.median(data[finite])
    mad = np.median(np.abs(data[finite] - med)) or float(np.std(data[finite]))
    thresh = med + n_sigma * 1.4826 * mad
    mask = np.isfinite(data) & (data > thresh)
    if radius > 0 and mask.any():
        try:
            from scipy import ndimage

            mask = ndimage.binary_dilation(mask, iterations=radius)
        except ImportError:
            pass
    out = data.copy()
    out[mask] = np.nan
    return out


def compute_delta_pdf(data: np.ndarray, apodization: str = "hann",
                      subtract_mean: bool = True) -> np.ndarray:
    """Centred Fourier transform of an (already Bragg-punched) diffuse volume.

    ``data`` has its Q=0 origin at the array centre; masked voxels are NaN.
    Returns the real 3D-ΔPDF with r=0 at the centre."""
    vol = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0)
    win = [_window(n, apodization) for n in vol.shape]
    vol = vol * win[0][:, None, None] * win[1][None, :, None] * win[2][None, None, :]
    if subtract_mean:
        vol = vol - vol.mean()
    # Q=0 sits at the array centre; the correct centred transform is
    # fftshift(fftn(ifftshift(·))) — without ifftshift a linear phase ramp
    # flips real-space features by pixel parity.
    ft = np.fft.fftn(np.fft.ifftshift(vol))
    return np.fft.fftshift(ft.real)


def central_slices(dpdf: np.ndarray) -> dict[str, np.ndarray]:
    """The three principal central planes of a ΔPDF volume."""
    cx, cy, cz = (s // 2 for s in dpdf.shape)
    return {"xy": dpdf[:, :, cz], "xz": dpdf[:, cy, :], "yz": dpdf[cx, :, :]}


def delta_pdf_summary(dpdf: np.ndarray) -> dict[str, Any]:
    finite = dpdf[np.isfinite(dpdf)]
    return {
        "shape": list(dpdf.shape),
        "max_positive": round(float(finite.max()), 5),
        "max_negative": round(float(finite.min()), 5),
        "note": "positive peaks = pair distances more common than average; "
        "negative = less common; low-r features can be Bragg-subtraction residue",
    }
