# Common RMC Failure Modes

## Over-fitting noise

RMC will happily use its many degrees of freedom to fit statistical noise and
Fourier truncation ripples, producing atomic displacements with no physical
meaning. Warning signs: fit visibly better than the noise level, features in
partial PDFs at chemically impossible distances, and results that change
qualitatively between repeat runs with different random seeds.

## Supercell too small

A small configuration box cannot host long-wavelength correlations, limits the
maximum r of the calculated PDF (r < half the shortest box edge), and makes
local and average structure compete artificially. Symptoms: misfit at high r,
Bragg/PDF conflicts that disappear with a larger box, and finite-size ripples.

## Data reduction problems

Normalization errors, inadequate background subtraction, and absorption or
multiple-scattering corrections show up as smooth misfits the configuration
cannot fix. RMC then absorbs them into unphysical structure. Persistent smooth
residuals in S(Q), a G(r) that does not oscillate around the correct baseline,
or misfit that no constraint change affects all point to reduction problems
upstream of the refinement.

## Qmax truncation ripples

Terminating the Fourier transform at finite Qmax produces ripples with period
about 2*pi/Qmax around sharp PDF peaks. These ripples are commonly mistaken
for split peaks or local distortions. Before interpreting a new low-r feature,
check whether its spacing matches the truncation period and whether it moves
when the transform is redone with a different Qmax.

## Restart and bookkeeping errors

Runs restarted from the wrong configuration file, mismatched data files, or
mid-run changes to weights and constraints produce discontinuities in the
agreement-factor history. A sudden jump in R-values at a restart boundary is a
bookkeeping problem until proven otherwise.
