# Data Drop Zone

Put real datasets here for developing and testing the SDK's data tools.
Each subfolder README says what belongs in it.

```text
data/
├── 1d/
│   ├── curves/     single 1D patterns (diffraction, S(Q), G(r), line cuts)
│   └── series/     temperature/field-dependent stacks of 1D curves
├── 2d/
│   ├── slices/     single 2D maps (reciprocal or real space)
│   └── series/     temperature/field-dependent stacks of 2D slices
└── 3d/
    └── volumes/    3D reciprocal-space volumes (HDF5 / NeXus)
```

## Ground rules

- **Git ignores everything under `data/` except READMEs**, so nothing large
  or unpublished is committed by accident. If a small file is safe to share
  as a permanent test fixture, say so and it can be force-added.
- Any format is fine to start: ASCII (`.xye`, `.dat`, `.csv`, `.gr`, `.sq`),
  `.npy`/`.npz`, HDF5/NeXus, instrument-specific — the IO layer will be built
  around whatever lands here. Exotic formats are useful too; include one.
- Redacting/truncating is fine; structure matters more than the science.

## Per-dataset notes (important)

For each file (or group of files), add a short entry to the `INVENTORY.md`
in its folder — a few words is enough, but these three things matter:

1. **Axes and units** — e.g. "col 1 = Q in 1/Å, col 2 = intensity, col 3 = sigma".
2. **What it is** — technique, instrument if relevant, sample (can be "oxide A").
3. **What you'd want extracted** — e.g. "fit the three peaks near Q=1.5–2.5",
   "there is an Al ring here", "transition around 40 K". This becomes the
   acceptance test for the tools.
