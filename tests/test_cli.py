import json
from pathlib import Path

from scattering_ai.cli import main

EXAMPLES = Path(__file__).parents[1] / "examples" / "rmc_monitor_demo"


def test_cli_stalled_run_end_to_end(tmp_path, capsys):
    out_md = tmp_path / "report.md"
    out_json = tmp_path / "report.json"
    rc = main(
        [
            "analyze",
            str(EXAMPLES / "stalled_run.json"),
            "--out",
            str(out_md),
            "--json-out",
            str(out_json),
        ]
    )
    assert rc == 0

    md = out_md.read_text()
    assert "## Conclusion" in md
    assert "increasing" in md  # bragg series worsening
    assert "missing" in md.lower()  # missing partials file

    payload = json.loads(out_json.read_text())
    assert payload["schema_version"] == "1"
    assert payload["warnings"]
    assert payload["provenance"]["input_hash"].startswith("sha256:")
    assert payload["citations"]


def test_cli_healthy_run_prints_to_stdout(capsys):
    rc = main(["analyze", str(EXAMPLES / "healthy_run.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Conclusion" in out
    assert "healthy" in out.lower()


def test_cli_bare_payload_auto_routes(tmp_path, capsys):
    """A bare payload now auto-routes (no --domain needed): r_values -> rmc."""
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"r_values": [10.0, 9.0, 8.0, 7.0, 6.0]}))
    rc = main(["analyze", str(bare)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "decreasing" in out  # rmc diagnostics ran


def test_cli_bare_payload_with_flags(tmp_path, capsys):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"r_values": [20.0 - 0.4 * i for i in range(30)]}))
    rc = main(["analyze", str(bare), "--domain", "rmc", "--question", "Converging?"])
    assert rc == 0
    assert "decreasing" in capsys.readouterr().out
