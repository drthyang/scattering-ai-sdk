"""Deterministic symmetry diagnostics for a crystal structure (CIF).

Detects the space group and Wyckoff sites, enumerates the maximal subgroups
(the group-subgroup steps a structural phase transition can take, drawn as a
tree), and searches for a higher-symmetry parent by relaxing the tolerance.
All computed by spglib; needs the ``symmetry`` extra.
"""

from __future__ import annotations

from pathlib import Path

from scattering_ai.core.findings import Finding, Severity

NEXT_CHECK_RULES: dict[str, str] = {
    "symmetry_subgroups": "Match each maximal subgroup against the observed "
    "low-symmetry phase (cell metric, split reflections, new extinctions); the "
    "transition path is the subgroup whose distortion the data support.",
    "pseudosymmetry": "Treat the looser-tolerance space group as a candidate "
    "parent phase and check whether the distortion is the order parameter of a "
    "parent -> subgroup transition.",
    "symmetry_unavailable": "Install the symmetry extra "
    "(pip install scattering-ai-sdk[symmetry]) to enable spglib analysis.",
    "cif_unreadable": "Confirm the CIF has a unit cell and an _atom_site loop; "
    "provide all atoms (P1) or the symmetry-operation loop for expansion.",
    "missing_files": "Locate or regenerate the missing files before analysis.",
}


def _structure(path: str):
    from scattering_ai.tools.cif import read_structure

    s = read_structure(path)
    return s["lattice"], s["positions"], s["species"], s


def diagnose_file(path: str, workspace=None) -> list[Finding]:
    if not Path(path).exists():
        return [Finding(diagnostic="missing_files", severity=Severity.ERROR,
                        message=f"Input file not found: {path}", evidence={"path": path})]
    try:
        from scattering_ai.tools import symmetry as sym
    except ImportError:
        return [Finding(diagnostic="symmetry_unavailable", severity=Severity.ERROR,
                        message="Symmetry analysis needs spglib "
                        "(pip install scattering-ai-sdk[symmetry]).", evidence={})]
    try:
        lattice, positions, species, meta = _structure(path)
    except Exception as exc:
        return [Finding(diagnostic="cif_unreadable", severity=Severity.ERROR,
                        message=f"Could not read structure from {Path(path).name}: "
                        f"{type(exc).__name__}: {exc}", evidence={"path": path})]

    try:
        fs = sym.find_symmetry(lattice, positions, species)
    except ImportError:
        return [Finding(diagnostic="symmetry_unavailable", severity=Severity.ERROR,
                        message="Symmetry analysis needs spglib.", evidence={})]
    if "error" in fs:
        return [Finding(diagnostic="symmetry_undetermined", severity=Severity.WARNING,
                        message=fs["error"], evidence=fs)]

    findings = [Finding(
        diagnostic="space_group", severity=Severity.INFO,
        message=f"Space group {fs['international']} (#{fs['number']}, "
        f"{fs['crystal_system']}, point group {fs['point_group']}); "
        f"{len(fs['wyckoff_sites'])} Wyckoff site(s) from {meta['n_atoms']} atoms.",
        evidence={**fs, "n_atoms": meta["n_atoms"], "n_asymmetric": meta["n_asymmetric"]},
    )]

    if workspace is not None:
        try:
            from scattering_ai.tools.structure_viz import plot_structure

            img = plot_structure(lattice, positions, species,
                                 Path(workspace) / "structure.png",
                                 moments=meta.get("moments"), title=Path(path).stem)
            findings.append(Finding(
                diagnostic="structure_figure", severity=Severity.INFO,
                message="Rendered the unit cell with atoms"
                + (" and magnetic moments" if meta.get("moments") else "") + ".",
                evidence={"figures": [img]}))
        except Exception:
            pass

    figures: list[str] = []
    try:
        tree = sym.maximal_subgroups(lattice, positions, species)
        if workspace is not None and tree.get("subgroups"):
            try:
                figures.append(sym.subgroup_tree_figure(
                    tree, Path(workspace) / "subgroup_tree.png"))
            except ImportError:
                pass
            except Exception:
                pass
        labels = ", ".join(
            f"{s['international']} (#{s['number']}, i{s['index']}"
            + (f"×{s['n_variants']}" if s["n_variants"] > 1 else "") + ")"
            for s in tree.get("subgroups", []))
        findings.append(Finding(
            diagnostic="symmetry_subgroups", severity=Severity.INFO,
            message=f"{tree.get('n_maximal_subgroups', 0)} maximal subgroup(s) — "
            f"structural-transition pathways: {labels or 'none'}.",
            evidence={**tree, "figures": figures},
        ))
    except Exception as exc:
        findings.append(Finding(
            diagnostic="symmetry_subgroups", severity=Severity.WARNING,
            message=f"Subgroup enumeration failed: {type(exc).__name__}: {exc}",
            evidence={}))

    try:
        ps = sym.pseudosymmetry_scan(lattice, positions, species)
        if ps.get("pseudosymmetric"):
            hi = ps["highest_symmetry"]
            findings.append(Finding(
                diagnostic="pseudosymmetry", severity=Severity.WARNING,
                message=f"Pseudosymmetry: at a looser tolerance the structure has "
                f"higher symmetry {hi['international']} (#{hi['number']}) — a likely "
                "parent phase of a symmetry-lowering transition.",
                evidence=ps))
    except Exception:
        pass
    return findings


def run_all(files: list[str], workspace=None) -> list[Finding]:
    findings: list[Finding] = []
    for f in files:
        findings.extend(diagnose_file(f, workspace))
    return findings
