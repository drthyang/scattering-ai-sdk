# Bragg vs Total Scattering in RMC

## Local versus average structure

Bragg peaks encode the long-range **average** periodic structure: only the
component of the atomic configuration that repeats coherently across the
crystal contributes to them. Total scattering (Bragg plus diffuse) and its
Fourier transform, the pair distribution function (PDF), additionally encode
**local** correlations: short-range distortions, correlated displacements, and
disorder that do not propagate coherently.

A single RMC configuration can therefore fit both quantities at once, but they
constrain different aspects of the model. Improving one at the expense of the
other usually means the local structure and the average structure are genuinely
different — or that the fit is trading physical signal in one dataset for noise
in another.

## Why the PDF can improve while the Bragg fit worsens

Common, physically meaningful reasons:

- Real local distortions (e.g. off-center cation displacements, Jahn-Teller
  distortions, correlated tilts) that average out over long range. Fitting them
  improves G(r) at low r while pushing atoms away from ideal average positions,
  which broadens or weakens calculated Bragg intensities.
- Short-range order in a solid solution or a frustrated magnet: the diffuse
  signal (captured in the PDF) demands correlations the average structure
  forbids.

Common pathological reasons:

- Relative weights (or sigma values) of the datasets favor the PDF too
  strongly, so the algorithm sacrifices Bragg agreement for marginal PDF gains.
- The PDF is being over-fit at high r where statistical noise and Fourier
  truncation ripples dominate.
- The supercell is too small to represent both the local motif and the correct
  average, so the two constraints compete artificially.

## What to check

- Compare partial PDFs before and after: are the changes chemically sensible?
- Check the magnitude of atomic displacements from average positions against
  the displacement parameters of the average-structure refinement.
- Re-run with rebalanced dataset weights; a robust local-structure signal
  survives moderate weight changes.
- Verify Qmax and truncation ripple positions before interpreting new low-r
  features as distortions.

## Related

See also: local vs average structure, PDF interpretation, dataset weighting.
