# Symmetry and structural phase transitions

## Space group, Wyckoff sites, point group

The **space group** is the full set of symmetry operations of a crystal
(rotations, screw axes, glides, translations) — 230 types in 3D. Detected from a
structure (atoms + cell) at a chosen position **tolerance**: too tight and noise
lowers the apparent symmetry, too loose and a real distortion is averaged away.

Each atom sits on a **Wyckoff position** with a **site symmetry** (the point
group that leaves it fixed). Atoms on special positions (high site symmetry)
have their coordinates fixed by symmetry; general positions are free.

## Group-subgroup relations and transitions

A structural phase transition that lowers symmetry takes the crystal from a
high-symmetry **parent** space group G to a **subgroup** H ⊂ G. Landau theory
says a continuous (second-order) transition follows a single irreducible
representation, and the low-symmetry group is a **maximal subgroup** of the
parent — you cannot skip steps.

Two kinds of maximal subgroup:

- **translationengleiche (t):** same lattice (translations), fewer point-group
  operations — a **ferroic** distortion (ferroelastic/ferroelectric,
  octahedral tilt). Index = order ratio (2, 3, 4, ...).
- **klassengleiche (k):** same point group, a larger cell — an **ordering** or
  superstructure transition (cell doubling, cation ordering). These need a
  supercell and are not enumerated from a single cell.

When a t-subgroup has several symmetry-equivalent orientations in the parent
(e.g. the three tetragonal axes of a cubic parent), each becomes a **domain
variant** on cooling. The number of variants equals the index for that step.

**Reading the maximal-subgroup list:** it enumerates the *allowed* pathways. The
transition that actually occurs is the one whose distortion the data support —
a matching cell metric, split or newly-forbidden reflections, and a plausible
order parameter. The list narrows the candidates; it does not choose among them.

## Pseudosymmetry (finding the parent)

A distorted structure often retains **pseudosymmetry**: at a looser tolerance it
snaps to a higher-symmetry space group. That higher group is a candidate
**parent phase**, and the tight (real) structure is one of its distorted
subgroups. Pseudosymmetry is how you find the parent when you only have the
low-symmetry phase.

## Magnetic (Shubnikov) space groups

Adding time reversal (moment flip) to the 230 space groups gives the 1651
**magnetic space groups**. Types: I (no time reversal), II (grey / paramagnetic,
time reversal alone is a symmetry), III and IV (black-white, time reversal
coupled to a rotation or a translation). The magnetic space group of an ordered
structure is fixed by the moment arrangement. Enumerating the *maximal* magnetic
groups allowed by a propagation vector **k** is representation analysis (a
separate step) and is not the same as reading off the group of a given ordering.
