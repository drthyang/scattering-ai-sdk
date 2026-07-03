import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from scattering_ai.server.api import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(workspace=tmp_path / "ws"))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_tools_listing(client):
    tools = client.get("/tools").json()
    names = [t["name"] for t in tools]
    assert "analyze" in names and "fit_peaks_1d" in names and "plot_1d" in names


def test_call_tool_endpoint(client, tmp_path):
    x = np.linspace(0, 10, 800)
    y = 1 + 5 * np.exp(-((x - 4) ** 2) / 0.02)
    path = tmp_path / "c.dat"
    np.savetxt(path, np.column_stack([x, y]))

    response = client.post("/tools/find_peaks_1d", json={"path": str(path)})
    assert response.status_code == 200
    assert response.json()["n_peaks"] == 1

    error = client.post("/tools/no_such_tool", json={}).json()
    assert "error" in error


def test_analyze_endpoint(client):
    body = {
        "domain": "rmc",
        "question": "Is this run healthy?",
        "data": {"r_values": [20.0 - 0.4 * i for i in range(30)]},
    }
    response = client.post("/analyze", json=body)
    assert response.status_code == 200
    report = response.json()
    assert report["provenance"]["input_hash"].startswith("sha256:")
    assert any("decreasing" in o for o in report["observations"])
