"""Structure lookup via OPTIMADE — identify candidate phases from a database.

OPTIMADE is the federated REST API spoken by the open crystal-structure
databases (COD, Materials Project, OQMD, Alexandria, ...). Given elements
and/or a formula, this returns candidate structures (formula, cell, source
link) so the agent can match an observed pattern against known phases.

Note: this tool calls an **external web service** (default: the Crystallography
Open Database). It is the one deliberately network-using tool in the SDK — the
local-first rule still holds for your data, which never leaves the machine;
only the element/formula query is sent.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

# COD: open, no API key, solid OPTIMADE endpoint.
DEFAULT_BASE_URL = "https://www.crystallography.net/cod/optimade"
TIMEOUT = 20.0


def build_filter(elements: list[str] | None = None,
                 formula: str | None = None,
                 exclusive: bool = False) -> str:
    """An OPTIMADE filter string from elements and/or a chemical formula."""
    clauses = []
    if elements:
        quoted = ", ".join(f'"{e}"' for e in elements)
        if exclusive:
            clauses.append(f'elements HAS ONLY {quoted}')
        else:
            clauses.append(f'elements HAS ALL {quoted}')
    if formula:
        clauses.append(f'chemical_formula_reduced="{formula}"')
    if not clauses:
        raise ValueError("provide elements and/or formula")
    return " AND ".join(clauses)


def parse_structures(payload: dict[str, Any], max_results: int = 10) -> list[dict[str, Any]]:
    """Normalize an OPTIMADE /structures response into compact candidates."""
    out = []
    for entry in payload.get("data", [])[:max_results]:
        attrs = entry.get("attributes", {}) or {}
        cell = attrs.get("lattice_vectors")
        cell_params = None
        if cell:
            import numpy as np

            m = np.asarray(cell, dtype=float)
            a, b, c = (float(np.linalg.norm(v)) for v in m)
            cell_params = [round(x, 4) for x in (a, b, c)]
        out.append({
            "id": entry.get("id"),
            "formula": attrs.get("chemical_formula_reduced")
            or attrs.get("chemical_formula_descriptive"),
            "elements": attrs.get("elements"),
            "n_elements": attrs.get("nelements"),
            "space_group": attrs.get("_cod_sg"),  # provider-specific when present
            "cell_lengths": cell_params,
        })
    return out


def query_structures(elements: list[str] | None = None,
                     formula: str | None = None,
                     exclusive: bool = False,
                     max_results: int = 10,
                     base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    """Query an OPTIMADE provider for candidate structures."""
    def fetch(filt: str) -> dict[str, Any]:
        url = (f"{base_url}/v1/structures?filter={urllib.parse.quote(filt)}"
               f"&page_limit={max_results}")
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:  # noqa: S310
            return json.loads(resp.read().decode())

    filt = build_filter(elements, formula, exclusive)
    try:
        payload = fetch(filt)
    except Exception as exc:
        if exclusive:  # some providers (e.g. COD) don't implement HAS ONLY
            try:
                filt = build_filter(elements, formula, exclusive=False)
                payload = fetch(filt)
            except Exception as exc2:
                return {"error": f"OPTIMADE query failed ({type(exc2).__name__}: "
                        f"{exc2}); the tool needs network access"}
        else:
            return {"error": f"OPTIMADE query failed ({type(exc).__name__}: {exc}); "
                    "the tool needs network access"}
    candidates = parse_structures(payload, max_results)
    meta = payload.get("meta", {}) or {}
    return {
        "filter": filt,
        "provider": base_url,
        "n_returned": len(candidates),
        "n_available": meta.get("data_returned"),
        "candidates": candidates,
        "note": "candidate phases from an open database; verify against the "
        "observed cell and reflections (e.g. inspect_cif / systematic_absences)",
    }
