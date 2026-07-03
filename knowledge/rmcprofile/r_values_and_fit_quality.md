# R-values and Fit Quality in RMC

## What RMC agreement factors mean

RMC refinements (e.g. RMCProfile) minimize a chi-squared-like sum over all
fitted datasets: total scattering structure factor S(Q) or F(Q), the pair
distribution function G(r) or D(r), and optionally a Bragg profile. Each
dataset carries its own agreement factor and weight (often expressed as a
sigma); the reported "Rwp"-style values are per-dataset measures of misfit.

A lower agreement factor is not automatically a better model. RMC has enormous
configurational freedom, so it can fit noise as easily as signal — fit quality
must always be judged together with the physical plausibility of the resulting
configuration.

## Healthy values and comparisons

- Comparisons of agreement factors are meaningful only within the same dataset,
  normalization, and weighting scheme. Never compare raw values across
  different data reductions.
- A fit dramatically better than the estimated noise level of the data is a
  warning sign of over-fitting, especially when few constraints are applied.
- Persistent misfit concentrated in specific r-ranges or Q-ranges is more
  informative than the global number: low-r misfit often points to wrong
  closest-approach constraints; high-Q misfit to inadequate displacement
  modeling or normalization.

## Suspiciously good fits

If the fit is excellent but constraints are weak (no bond-length constraints,
generous closest-approach distances, no coordination constraints), inspect the
configuration: unphysical bond lengths, collapsed coordination polyhedra, and
atoms fitting noise are common. Tighten constraints and confirm the fit
degrades only modestly; a physical model should not depend on unphysical
freedom.
