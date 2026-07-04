# Diffuse scattering and 3D-ΔPDF

## What diffuse scattering is

Sharp Bragg peaks encode the average, periodic structure. **Diffuse scattering**
is the broad, structured intensity *between* the Bragg peaks: it comes from
deviations from the average — short-range order, local displacements,
correlated disorder, static or dynamic. Reading it means characterizing that
broad intensity, not the sharp maxima.

- **Sharp local maxima** in a reciprocal-space slice are Bragg peaks (or the
  residue of imperfectly removed ones), not diffuse features.
- Diffuse intensity is weak (often 10²–10⁴× below Bragg), so dynamic range,
  background, and contamination matter a lot.

## Contaminant powder rings

A single crystal gives spots; **rings** come from powder in the beam — the
aluminium sample can, copper/steel mounts, or a vanadium can. Ring |Q| matches
2π/d of the contaminant. These are **artifacts**, not sample diffuse scattering:
exclude the annuli before analysis.

- A ring spanning a full annulus (**high completeness**) is convincing.
- A short arc (**low completeness**) is often just Bragg spots crossing the
  annulus — weak evidence. Confirm in the most isotropic plane.

## Bragg-punch and mask artifacts

To study diffuse scattering the Bragg peaks are often "punched out" (removed in
a small sphere around each). This leaves **holes**; detector gaps and masks add
more. Azimuthal averages and diffuse integrals over poorly covered annuli are
biased low and jagged. Always check coverage before trusting an average.

## Anisotropic sampling

Instruments sample reciprocal space unevenly — typically fine in-plane and
coarse along the rotation/out-of-plane axis. A diffuse feature can look broader
or streakier simply because that direction is sampled coarsely. Compare planes
at **matched bin widths** before attributing anisotropy to the sample.

## 3D-ΔPDF (real-space view)

The **3D difference pair distribution function** is the Fourier transform of the
diffuse intensity (total minus Bragg). Its peaks are **inter-site pair
correlations**: positive = a pair distance more common than in the average
structure, negative = less common. They are correlations between sites, not
bonds, and their sign and decay length describe how disorder is organized.

- Truncation and the Bragg subtraction quality both leave ripples/artifacts in
  ΔPDF; low-r features can be subtraction residue rather than real correlations.
