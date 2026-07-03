# 1D series (temperature / field / time dependent)

Stacks of 1D curves measured against a control parameter: temperature scans,
field scans, time series.

Drop either:

- one file per point, with the parameter in the filename
  (e.g. `scan_005K.dat`, `scan_010K.dat`, ...), or
- a single multi-column / HDF5 file containing the whole stack.

In `INVENTORY.md`, note the parameter values and units, and what should be
tracked across the series (peak position? width? intensity? a transition
temperature you already know the answer to — known answers make the best
tests).
