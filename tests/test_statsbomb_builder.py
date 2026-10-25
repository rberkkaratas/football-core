"""The ``statsbomb`` snapshot builder and ``fc snapshot --builder statsbomb`` (spec
``runs/MS2/c4/spec.md`` §1, §5). The fixture cache is complete, so no network is touched;
``_http_get`` and ``urlopen`` are patched to fail if anything tries."""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import snapshot_cases as snc
import statsbomb_cases as sbc

from football_core import cli
from football_core.io import snapshot, statsbomb

SORT_BY = {
    "shots": (
        "match_date",
        "competition_id",
        "season_id",
        "match_id",
        "period",
        "seconds_in_period",
        "shot_id",
    ),
    "freeze_frames": ("shot_id",),
    "matches": ("match_date", "match_id"),
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"network access attempted: {args!r}")

    monkeypatch.setattr(statsbomb, "_http_get", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def _cache(tmp_path: Path, matches: list[sbc.Match] | None = None) -> Path:
    files = sbc.repo_files(sbc.standard_matches() if matches is None else matches)
    cache = tmp_path / "cache"
    sbc.write_cache(cache, files)
    return cache


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    rc = cli.main(list(argv))
    out, err = capsys.readouterr()
    return rc, out, err


def _snapshot_args(cache: Path | None, root: Path, *extra: str) -> list[str]:
    args = ["snapshot", "sb-fixture", "--builder", "statsbomb", "--root", str(root)]
    if cache is not None:
        args += ["--input", str(cache)]
    return [*args, *extra]


def test_builtin_statsbomb_registered() -> None:
    assert snapshot.BUILTIN_BUILDERS["statsbomb"] == "football_core.io.statsbomb:build_snapshot"
    assert "statsbomb" in snapshot.builder_names()
    assert snapshot.get_builder("statsbomb") is statsbomb.build_snapshot


def test_fc_snapshot_statsbomb_fixture(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cache, root = _cache(tmp_path), tmp_path / "out"
    args = _snapshot_args(cache, root, "--param", f"commit={sbc.COMMIT}")
    rc, out, err = _run(capsys, *args)
    assert rc == 0, err
    assert err == ""
    rows = sbc.SHOT_ROWS + sbc.FREEZE_FRAME_ROWS + 3
    assert out.startswith(f"created snapshot 'sb-fixture': 3 table(s), {rows} row(s); ")
    manifest = snc.manifest_path(root, "sb-fixture")
    tables = snapshot.load_snapshot(manifest)
    assert sorted(tables) == ["freeze_frames", "matches", "shots"]
    assert tables["shots"].height == sbc.SHOT_ROWS
    assert tables["freeze_frames"].height == sbc.FREEZE_FRAME_ROWS
    matches = tables["matches"]
    assert dict(matches.schema) == sbc.MATCH_DTYPES
    assert matches.height == 3 and matches.null_count().sum_horizontal().item() == 0
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["builder"] == "statsbomb"
    assert (m["source"]["kind"], m["source"]["uri"], m["source"]["revision"]) == (
        "git",
        statsbomb.OPEN_DATA_URL,
        sbc.COMMIT,
    )
    assert statsbomb.OPEN_DATA_URL == "https://github.com/statsbomb/open-data"
    assert {t["name"]: tuple(t["sort_by"]) for t in m["tables"]} == SORT_BY
    rc, out, err = _run(capsys, *args)
    assert rc == 0, err
    assert out.startswith("unchanged snapshot 'sb-fixture'")
    assert _run(capsys, "verify", str(manifest))[0] == 0


def _unreconciled(tmp_path: Path) -> Path:
    a = sbc.match_a()
    a.home_score = 4
    return _cache(tmp_path, [a, sbc.match_b()])


@pytest.mark.parametrize(
    ("case", "needle"),
    [
        ("no_input", "exactly one --input"),
        ("two_inputs", "exactly one --input"),
        ("unknown_param", "season"),
        ("bad_commit", "commit"),
        ("unreconciled", str(sbc.A)),
    ],
)
def test_builder_errors_exit_1(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, case: str, needle: str
) -> None:
    root = tmp_path / "out"
    commit = ["--param", f"commit={sbc.COMMIT}"]
    if case == "no_input":
        args = _snapshot_args(None, root, *commit)
    elif case == "two_inputs":
        args = _snapshot_args(_cache(tmp_path), root, "--input", str(tmp_path), *commit)
    elif case == "unknown_param":
        args = _snapshot_args(_cache(tmp_path), root, *commit, "--param", "season=1")
    elif case == "bad_commit":
        args = _snapshot_args(_cache(tmp_path), root, "--param", "commit=abc")
    else:
        args = _snapshot_args(_unreconciled(tmp_path), root, *commit)
    rc, out, err = _run(capsys, *args)
    assert rc == 1
    assert err.startswith("fc: error:"), err
    assert "Traceback" not in err
    assert needle in err
    assert not snc.manifest_path(root, "sb-fixture").exists()


def test_builder_logs_skipped_360(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    broken = sbc.one_match(
        sbc.shot("fake-shot-d01"),
        match_id=900004,
        three_sixty=[sbc.visible_area("fake-shot-d01", sbc.AREA_A01)],
        extra_files={"data/three-sixty/900004.json": b"[\x00\x00]"},
    )
    cache = _cache(tmp_path, [*sbc.standard_matches(), broken])
    caplog.set_level(logging.WARNING, logger="football_core.io.statsbomb")
    result = statsbomb.build_snapshot(snapshot.BuildRequest((cache,), {"commit": sbc.COMMIT}))
    records = [r for r in caplog.records if r.name == "football_core.io.statsbomb"]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.WARNING, "statsbomb: 1 match(es) without usable 360 data: sb:900004")
    ]
    assert sorted(result.tables) == ["freeze_frames", "matches", "shots"]
    assert {k: tuple(v) for k, v in result.sort_by.items()} == SORT_BY
    assert (result.source.kind, result.source.uri, result.source.revision) == (
        "git",
        statsbomb.OPEN_DATA_URL,
        sbc.COMMIT,
    )
    has_360 = result.tables["matches"].filter(pl.col("match_id") == "sb:900004")["has_360"]
    assert has_360.to_list() == [False]
