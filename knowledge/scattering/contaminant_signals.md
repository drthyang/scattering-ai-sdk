# Sample-Environment and Contaminant Signals

## Powder rings from the sample environment

Single-crystal diffuse scattering measurements almost always include
polycrystalline material in the beam: the aluminum of cryostat shields and
sample holders, copper from cold fingers, stainless steel, or a vanadium can.
These produce Debye-Scherrer rings — constant-|Q| circles — superimposed on
the single-crystal pattern. In reciprocal-space maps plotted in r.l.u. the
rings appear as ellipses (axes scaled by the lattice constants), which is a
quick visual signature distinguishing them from crystal diffuse scattering.

## Characteristic d-spacings (Å)

- **Aluminum** (fcc, a = 4.0495): 2.338 (111), 2.025 (200), 1.432 (220),
  1.221 (311), 1.169 (222), 1.012 (400), 0.929 (331), 0.905 (420)
- **Copper** (fcc, a = 3.6149): 2.087 (111), 1.808 (200), 1.278 (220),
  1.090 (311), 1.044 (222)
- **bcc iron / steel** (a = 2.8665): 2.027 (110), 1.433 (200), 1.170 (211),
  1.013 (220)
- **Vanadium can** (bcc, a = 3.027): 2.141 (110), 1.514 (200), 1.236 (211)
  — vanadium is nearly incoherent for neutrons, so its rings are weak but
  its incoherent background is significant.

Ring |Q| = 2π/d. Slight mismatches (~1–2%) from thermal contraction at low
temperature and alloying are normal; cryostat aluminum at 5 K has a smaller
lattice constant than the room-temperature value.

## Distinguishing rings from crystal features

- A ring fills its annulus at all azimuths; single-crystal Bragg peaks and
  diffuse features are localized in azimuth. Azimuthal profiles using a
  robust statistic (median or lower quartile) suppress crystal peaks while
  keeping rings.
- In anisotropic slices (one reciprocal axis covering a much smaller |Q|
  range), annuli are short arcs and this discrimination weakens — ring
  candidates found there deserve suspicion, and should be confirmed in the
  most isotropic available plane.
- Rings persist unchanged across temperature series while sample features
  evolve; a temperature-independent "diffuse" feature at a contaminant |Q|
  is almost certainly the sample environment.

## Other common artifacts

- **Bragg-peak punching residue**: removing Bragg peaks before 3D-ΔPDF
  leaves regular holes; incomplete punching leaves sharp positive residue at
  reciprocal-lattice points.
- **Detector gaps and beamstop**: appear as NaN/masked regions; any feature
  aligned with a gap boundary is suspect.
- **Symmetrization artifacts**: applying point-group symmetry duplicates any
  artifact into all equivalent positions, making instrumental features look
  like physics.
