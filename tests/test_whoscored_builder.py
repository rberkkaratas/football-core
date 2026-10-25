"""The ``whoscored`` snapshot builder: private manifest, verify, drop log, reconciliation, inputs,
the CLI and the helpers shared with StatsBomb (spec ``runs/MS4/whoscored/spec.md`` §1.1, §1.4,
§1.8, §1.9, §3, §5)."""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import snapshot_cases as snc
import statsbomb_cases as sbc
import whoscored_cases as wsc

from football_core import cli, schema
from football_core.io import snapshot, statsbomb, whoscored

NAME = "ws-fixture"
SORT_BY = {
    "shots": ("match_date", "competition_id", "season_id", "match_id", "period",
              "seconds_in_period", "shot_id"),
    "freeze_frames": ("shot_id",),
    "matches": ("match_date", "match_id"),
}  # fmt: skip
LOG_LINES = [
    "whoscored: dropped 2 match(es): wrong_competition",
    "whoscored: dropped 1 match(es): pending",
    "whoscored: dropped 3 match(es): unvalidated",
    "whoscored: dropped 4 match(es): no_event_file",
    "whoscored: kept 3 match(es) without a score: unscored",
    "whoscored: kept 1 match(es) without a score: extra_time",
    "whoscored: kept 1 match(es) ending before minute 85",
]


def _archive(tmp_path: Path, fx: wsc.Fixture | None = None) -> Path:
    return wsc.write_archive(tmp_path / "archive", wsc.standard() if fx is None else fx)


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    rc = cli.main(list(argv))
    out, err = capsys.readouterr()
    return rc, out, err


def _forbidden() -> set[str]:
    """Strings that identify fixture rows: match ids, file names, team ids and names."""
    fx = wsc.standard()
    out = {str(m.mid) for m in fx.matches} | {m.stem for m in fx.matches}
    out |= set(wsc.TEAM_NAMES.values()) | {f"ws:{t}" for t in wsc.TEAM_NAMES}
    return out | {str(wsc.ORPHAN)}


@snc.needs_git
def test_manifest_private_and_verify(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    repo = snc.git_repo(tmp_path / "repo")
    result = snapshot.snapshot_from_builder(NAME, "whoscored", inputs=[archive], root=repo)
    assert result.status == "created"
    text = snc.manifest_path(repo, NAME).read_text(encoding="utf-8")
    m = json.loads(text)
    assert m["builder"] == "whoscored"
    source = m["source"]
    assert (source["kind"], source["uri"], source["files"]) == ("files", whoscored.ARCHIVE_URI, [])
    assert whoscored.ARCHIVE_URI == "whoscored-csv-archive"
    # The revision digests exactly the files the loader opened (spec §1.9).
    opened = sorted(archive / p for p in wsc.expected_files(archive))
    expected = snapshot.source_from_files(opened, root=archive, list_files=False)
    assert source["revision"] == expected.revision
    leaks = sorted(s for s in _forbidden() if s in text)
    assert leaks == [], leaks
    assert {t["name"]: tuple(t["sort_by"]) for t in m["tables"]} == SORT_BY
    rows = {t["name"]: t["rows"] for t in m["tables"]}
    assert rows == {"shots": len(wsc.ALL_SHOTS), "freeze_frames": 0, "matches": len(wsc.KEPT)}
    assert snapshot.verify_snapshot(snc.manifest_path(repo, NAME)).ok
    again = snapshot.snapshot_from_builder(NAME, "whoscored", inputs=[archive], root=repo)
    assert again.status == "unchanged"


def test_logs_counts(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    archive = _archive(tmp_path)
    caplog.set_level(logging.DEBUG, logger="football_core.io.whoscored")
    result = whoscored.build_snapshot(snapshot.BuildRequest((archive,), {}))
    records = [r for r in caplog.records if r.name == "football_core.io.whoscored"]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.WARNING, line) for line in LOG_LINES
    ]
    assert sorted(result.tables) == ["freeze_frames", "matches", "shots"]
    assert result.tables["freeze_frames"].height == 0
    assert result.tables["shots"].height == len(wsc.ALL_SHOTS)


@snc.needs_git
def test_refuses_unreconciled(tmp_path: Path) -> None:
    fx = wsc.standard()
    fx.match("main").scores = (2, 2)  # the shots give 1-2
    archive = _archive(tmp_path, fx)
    whoscored.load(archive)  # loads: reconciliation is the builder's
    repo = snc.git_repo(tmp_path / "repo")
    before = snc.tree(repo)
    with pytest.raises(ValueError, match=f"ws:{wsc.MAIN}"):
        snapshot.snapshot_from_builder(NAME, "whoscored", inputs=[archive], root=repo)
    snc.assert_nothing_written(repo, NAME, before)


@pytest.mark.parametrize(
    ("inputs", "params"),
    [
        ((), {}),
        (("a", "b"), {}),
        (("missing-archive",), {"commit": "x"}),
    ],
    ids=["no_input", "two_inputs", "a_param"],
)
def test_inputs_and_params(tmp_path: Path, inputs: tuple[str, ...], params: dict[str, str]) -> None:
    # The inputs do not exist: a builder that read before checking would raise
    # FileNotFoundError instead.
    request = snapshot.BuildRequest(tuple(tmp_path / p for p in inputs), params)
    with pytest.raises(ValueError) as info:
        whoscored.build_snapshot(request)
    assert not isinstance(info.value, OSError)
    if not params:
        assert "input" in str(info.value)


@snc.needs_git
def test_cli(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    repo = snc.git_repo(tmp_path / "repo")
    args = ["snapshot", NAME, "--builder", "whoscored", "--root", str(repo)]
    rc, out, err = _run(capsys, *args, "--input", str(archive))
    assert rc == 0, err
    rows = len(wsc.ALL_SHOTS) + len(wsc.KEPT)
    assert out.startswith(f"created snapshot '{NAME}': 3 table(s), {rows} row(s); ")
    rc, out, err = _run(capsys, "verify", str(snc.manifest_path(repo, NAME)))
    assert rc == 0, err
    assert out.startswith("ok: ")
    for bad in (["--input", str(tmp_path / "nowhere")], []):
        rc, out, err = _run(capsys, *args, *bad)
        assert rc == 1, (bad, err)
        assert err.startswith("fc: error:") and "Traceback" not in err
    rc, out, _ = _run(capsys, "builders")
    assert rc == 0 and "whoscored" in out.splitlines()
    assert snapshot.BUILTIN_BUILDERS["whoscored"] == "football_core.io.whoscored:build_snapshot"
    assert snapshot.get_builder("whoscored") is whoscored.build_snapshot


def test_shared_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert statsbomb.reconcile_goals is schema.reconcile_goals
    assert {k: tuple(v) for k, v in snapshot.C3_SORT_BY.items()} == SORT_BY
    ws = whoscored.build_snapshot(snapshot.BuildRequest((_archive(tmp_path),), {}))
    assert ws.sort_by is snapshot.C3_SORT_BY

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"network access attempted: {args!r}")

    monkeypatch.setattr(statsbomb, "_http_get", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    cache = tmp_path / "sb-cache"
    sbc.write_cache(cache, sbc.repo_files(sbc.standard_matches()))
    sb = statsbomb.build_snapshot(snapshot.BuildRequest((cache,), {"commit": sbc.COMMIT}))
    assert sb.sort_by is snapshot.C3_SORT_BY
