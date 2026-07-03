# Constraints and Cutoffs in RMC

## Why constraints matter

RMC has far more degrees of freedom than the data can determine. Constraints —
closest-approach distances, bond-length and bond-angle restraints, coordination
constraints, and (for molecular systems) rigid units — inject chemical
knowledge that the scattering data alone cannot provide. The refined
configuration is only as physical as its constraints.

## Closest-approach cutoffs

Cutoffs set the minimum allowed distance between pairs of atom types. Set them
from known ionic/covalent radii and the first physical peak of each partial
PDF:

- Too small: atoms overlap unphysically to fit low-r noise or termination
  ripples; partial PDFs show intensity below chemically possible distances.
- Too large: the first real coordination peak cannot be reproduced, misfit
  concentrates at low r, and acceptance rates collapse.

## Bond and coordination constraints

Distance-window and coordination-number constraints keep polyhedra intact
during the fit. Symptoms of over-constraint: acceptance rate near zero, flat
agreement factors from the start, and misfit that cannot decrease. Symptoms of
under-constraint: excellent fits with broken coordination environments and
implausible bond-length distributions.

## Balancing datasets and constraints

Constraint strength and dataset weights compete. When a constraint conflicts
with a dataset (e.g. a distance window fights an apparent short bond in G(r)),
decide which encodes better knowledge before loosening either. A conflict is
often the interesting scientific result, not an inconvenience.
