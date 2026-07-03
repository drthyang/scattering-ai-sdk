# Total Scattering and the PDF

## What total scattering measures

Total scattering is the full coherent scattering — Bragg peaks plus diffuse
scattering — measured over a wide Q range with careful corrections and
normalization. The Bragg component encodes the average periodic structure; the
diffuse component encodes deviations from it: static disorder, correlated
displacements, and short-range order.

## The pair distribution function

The PDF, G(r), is the Fourier transform of the normalized total scattering
structure factor. It gives the real-space distribution of interatomic
distances, weighted by the scattering power of each atom pair. Peaks at low r
correspond to well-defined coordination shells; the decay and broadening of
peaks with r reflects how quickly local order decorrelates toward the average
structure.

Reading a PDF:

- Peak positions: bond lengths and coordination-shell distances.
- Peak widths: static and thermal disorder, and correlated motion (nearest
  neighbor peaks are often sharper than the average structure predicts because
  bonded atoms move together).
- Low-r region below the first bond length: should contain only termination
  ripples; real intensity there indicates normalization or reduction problems.

## Practical limits

- Qmax sets real-space resolution; finite Qmax produces termination ripples of
  period about 2*pi/Qmax around sharp peaks.
- Statistical noise grows with r in G(r); interpretation of weak high-r
  features requires caution.
- Instrument resolution damps the PDF at high r; model this before
  interpreting a loss of high-r peaks as reduced structural coherence.
