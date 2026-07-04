# PDF conventions and low-r artifacts

## What G(r) is

The pair distribution function from total scattering is the sine Fourier
transform of the reduced structure function F(Q) = Q[S(Q) − 1]:

```
G(r) = (2/π) ∫ Q[S(Q) − 1] sin(Qr) dQ,  from Qmin to Qmax
```

`G(r)` (the **reduced** PDF, PDFgui/pdfgetx convention) oscillates around zero
and, below the first interatomic distance, follows the average-density limit

```
G(r) → −4πρr    as r → 0
```

so its **baseline has a negative slope near the origin**. Peaks in G(r) sit at
interatomic distances; peak areas scale with coordination number weighted by
scattering power.

## Conventions that get confused

- **g(r)** (the atomic pair-density / RDF-like function) tends to 1 at large r
  and is **non-negative**; it is not G(r). Its low-r slope is not −4πρr.
- **R(r) = 4πρr²g(r)** is the radial distribution function (area = coordination
  number). Different baseline again.
- A **sign-inverted or mis-scaled export** can flip G(r) upside down. Tell-tale:
  the low-r slope is positive instead of negative, and the first "peak" points
  the wrong way. If the low-r slope is not negative, the file is probably not a
  standard G(r) — check the reduction before interpreting peaks.

## S(Q) vs S(Q) − 1 (the naming trap)

Facility exports are inconsistent. A file **named** `S(Q)`/`SofQ` whose high-Q
limit tends to **0** actually stores **S(Q) − 1** (common for NOMAD `SofQ`
outputs); a true S(Q) tends to **1** at high Q. Feeding S(Q) − 1 to a transform
that expects S(Q) (or vice-versa) rescales/insverts the PDF. Always transform
with the convention that matches the high-Q tail.

## Low-r artifacts

Structure **below the shortest physical bond (~1 Å)** is never a bond. Sources:

- **Termination ripple**: truncating the transform at a finite Qmax convolves
  G(r) with a sinc, adding ripples of period ≈ 2π/Qmax around every real peak
  and below the first one. Higher Qmax → smaller-period, lower-amplitude ripple.
- **Normalization / background error**: an over- or under-subtracted background,
  a wrong number density ρ, or a bad Qmin introduces a slowly varying low-r
  offset or slope. This shows up as large |G| at small r that is not ripple.

When a low-r feature is large relative to the real G(r) amplitude, suspect
normalization first, then Qmax. It is not coordination.

## Qmax and peak shape

Qmax limits real-space resolution: broader Qmax → sharper, better-separated
PDF peaks. Sharp features needing high Qmax (e.g. resolving two close bonds) are
unreliable if Qmax is low. Report Qmax alongside any peak-width claim.
