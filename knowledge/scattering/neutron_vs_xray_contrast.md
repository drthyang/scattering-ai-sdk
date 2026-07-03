# Neutron vs X-ray Contrast

## Different probes, different sensitivity

X-rays scatter from electrons: the atomic form factor scales roughly with
atomic number Z and falls off with Q. Neutrons scatter from nuclei: the
coherent scattering length b varies non-monotonically across the periodic
table and between isotopes, and is independent of Q.

Consequences for joint refinements:

- Neutrons often see light elements (H/D, Li, O) far better than x-rays,
  especially next to heavy elements.
- Neighboring elements in Z can be nearly indistinguishable to x-rays but have
  very different neutron scattering lengths (and vice versa), so the two
  probes weight partial correlations differently.
- Neutrons couple to magnetic moments; magnetic diffuse scattering appears in
  neutron data but never in x-ray total scattering at comparable magnitude.

## Interpreting neutron/x-ray disagreement in a fit

When a configuration fits one probe and not the other, the misfit is
concentrated in the partials where the two probes' weightings differ most.
Useful checks:

- Identify which partial pair correlations dominate each dataset (weighting by
  b_i*b_j versus Z_i*Z_j) and see whether the disagreement localizes there.
- For hydrogen-containing samples, check inelasticity/Placzek corrections in
  the neutron data before blaming the model.
- Consider whether a magnetic diffuse contribution is present in the neutron
  data and absent from the model.

Genuine element-specific disorder (e.g. one sublattice disordered) also
produces probe-dependent misfit and can be the scientific result.
