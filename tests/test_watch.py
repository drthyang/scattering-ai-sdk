"""Self-improvement E8: the data-gated capability watch queue."""

import json

from scattering_ai.learning.briefs import brief_from_proposal
from scattering_ai.learning.watch import (
    WatchItem,
    scan_watch_queue,
    watch_proposals,
    watch_status,
)

_QUEUE = [WatchItem(name="ins_pack", title="INS pack",
                    globs=["**/*sqw*", "**/ins/**/*"], build_hint="build ins")]


def test_no_matching_data_means_not_ready(tmp_path):
    (tmp_path / "1d").mkdir()
    (tmp_path / "1d" / "curve.gr").write_text("x")
    assert scan_watch_queue(tmp_path, _QUEUE) == []
    assert watch_proposals(tmp_path, _QUEUE) == []
    status = watch_status(tmp_path, _QUEUE)
    assert status[0]["ready"] is False


def test_matching_data_unblocks_a_tier2_task_proposal(tmp_path):
    d = tmp_path / "2d" / "ins"
    d.mkdir(parents=True)
    (d / "sample_sqw_5K.dat").write_text("x")  # matches **/*sqw* and **/ins/**/*
    matched = scan_watch_queue(tmp_path, _QUEUE)
    assert len(matched) == 1 and matched[0][0].name == "ins_pack"
    assert "sample_sqw_5K.dat" in matched[0][1]

    props = watch_proposals(tmp_path, _QUEUE)
    assert len(props) == 1
    p = props[0]
    assert p.tier == 2 and p.change_class == "task" and p.signal_type == "data_arrival"
    assert p.evidence_count == 1

    # E8 composes with E7: a ready build becomes an agent brief
    brief = brief_from_proposal(p)
    assert brief.tier == 2 and "build ins" in brief.intent


def test_missing_data_root_is_safe(tmp_path):
    assert scan_watch_queue(tmp_path / "nope", _QUEUE) == []


def test_cli_learn_watch(tmp_path, capsys, monkeypatch):
    from scattering_ai.cli import main

    d = tmp_path / "3d"
    d.mkdir()
    (d / "run_inelastic.nxs").write_text("x")  # matches the real queue's ins globs
    rc = main(["learn", "watch", "--data", str(tmp_path)])
    assert rc == 0
    status = json.loads(capsys.readouterr().out)
    ins = next(s for s in status if s["name"] == "ins_pack")
    assert ins["ready"] is True

    rc = main(["learn", "watch", "--data", str(tmp_path), "--brief"])
    assert rc == 0
    assert "Improvement brief" in capsys.readouterr().out
