"""``whoscored.load``: the keep rule, the duplicate match, home/away, scores and the competition
table (spec ``runs/MS4/whoscored/spec.md`` §1.2-§1.5, §2.1, §2.3, §3, §6)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import MappingProxyType
from typing import Any

import polars as pl
import pytest
import whoscored_cases as wsc

from football_core import schema
from football_core.io import whoscored


def _match(loaded: Any, mid: int) -> dict[str, Any]:
    rows = loaded.archive.matches.filter(pl.col("match_id") == f"ws:{mid}").to_dicts()
    assert len(rows) == 1, f"ws:{mid}: {len(rows)} row(s)"
    return rows[0]


def test_status_rule(ws_standard: Any, tmp_path: Path) -> None:
    archive = ws_standard.archive
    assert isinstance(archive.counts, MappingProxyType)
    assert list(archive.counts.items()) == list(wsc.COUNTS.items())
    assert whoscored.DROP_REASONS == (
        "wrong_competition",
        "pending",
        "unvalidated",
        "no_event_file",
    )
    assert whoscored.SCORE_NOTES == ("unscored", "extra_time")
    assert whoscored.SHORT_NOTE == "ended_before_85"
    assert sorted(archive.matches["match_id"].to_list()) == sorted(f"ws:{m}" for m in wsc.KEPT)
    drops = sum(archive.counts[k] for k in whoscored.DROP_REASONS)
    assert archive.matches.height + drops == wsc.MATCH_IDS_ROWS + 1  # one orphan event file
    # Non-ok rows count under their status whether or not they have a file.
    fx = wsc.standard()
    assert not fx.match("null_nofile").has_file and fx.match("null_file").has_file
    # An archive whose rows are all dropped gives valid empty tables.
    only = wsc.Fixture([fx.match("pending"), fx.match("ok_nofile1")])
    empty = wsc.load_fixture(tmp_path, only).archive
    assert empty.shots.height == empty.matches.height == empty.freeze_frames.height == 0
    assert empty.shots.schema == schema.SHOT_SCHEMA
    assert empty.matches.schema == schema.MATCH_SCHEMA
    assert dict(empty.counts) == {k: 0 for k in wsc.COUNTS} | {"pending": 1, "no_event_file": 1}


def test_dropped_files_never_read(ws_standard: Any) -> None:
    """The pending, wrong_competition, null-status and orphan files are garbage: opening any
    of them would raise. ``Archive.files`` lists exactly what was opened, sorted."""
    root, archive = ws_standard
    files = archive.files
    assert isinstance(files, tuple) and list(files) == sorted(files)
    rel = {Path(p).resolve().relative_to(root.resolve()).as_posix() for p in files}
    assert len(rel) == len(files)
    assert rel == wsc.expected_files(root)
    fx = wsc.standard()
    for key in ("pending", "wrong2", "null_file", "dup_wrong"):
        assert fx.match(key).event_path(root).relative_to(root).as_posix() not in rel, key
    assert not any(str(wsc.ORPHAN) in p for p in rel)


@pytest.mark.parametrize("case", ["wrong_copy_first", "wrong_copy_last"])
def test_duplicate_match(case: str, tmp_path: Path) -> None:
    fx = wsc.standard()
    ok, wrong = fx.match("dup_ok"), fx.match("dup_wrong")
    assert ok.events == wrong.events  # same global ids in both copies, both parseable
    assert wrong.folder < ok.folder  # the wrong copy's folder is listed and read first
    if case == "wrong_copy_last":  # the Championship copy is now the wrong one, read last
        ok.status, wrong.status = "wrong_competition", "ok"
        ok, wrong = wrong, ok
    loaded = wsc.load_fixture(tmp_path, fx)
    archive = loaded.archive
    ids = archive.matches["match_id"].to_list()
    assert f"ws:{ok.mid}" in ids and f"ws:{wrong.mid}" not in ids
    assert archive.counts["wrong_competition"] == 2
    kept = _match(loaded, ok.mid)
    assert kept["competition_id"] == f"ws:{ok.folder}"
    dup = archive.shots.filter(
        pl.col("shot_id").is_in([f"ws:{wsc.DUP_SHOT1}", f"ws:{wsc.DUP_SHOT2}"])
    )
    assert dup.height == 2
    assert dup["match_id"].unique().to_list() == [f"ws:{ok.mid}"]
    assert dup["competition_id"].unique().to_list() == [f"ws:{ok.folder}"]


@pytest.mark.parametrize("route", ["metadata", "names"])
def test_home_away(ws_standard: Any, route: str) -> None:
    if route == "metadata":
        m, home, away, shooter = wsc.main_match(), wsc.B, wsc.A, wsc.SAVED
        assert m.scores is not None
    else:
        m, home, away, shooter = wsc.names_match(), wsc.MM, wsc.L, wsc.N_SHOT1
        assert m.scores is None and m.validated
    prematch = [e for e in m.events if e.period == "PreMatch"]
    assert prematch[0].team == away  # "first team in the file = home" would be wrong
    row = _match(ws_standard, m.mid)
    assert (row["home_team_id"], row["away_team_id"]) == (f"ws:{home}", f"ws:{away}")
    shots = ws_standard.archive.shots.filter(pl.col("match_id") == f"ws:{m.mid}")
    by_team = dict(
        zip(shots["shot_id"], zip(shots["team_id"], shots["is_home"], strict=True), strict=True)
    )
    assert m.event(shooter).team == home
    assert by_team[f"ws:{shooter}"] == (f"ws:{home}", True)
    assert all(is_home == (team == f"ws:{home}") for team, is_home in by_team.values())


def test_scores(ws_standard: Any) -> None:
    archive = ws_standard.archive
    scores = {
        r["match_id"]: (r["home_score"], r["away_score"])
        for r in archive.matches.select("match_id", "home_score", "away_score").to_dicts()
    }
    assert scores[f"ws:{wsc.MAIN}"] == wsc.MAIN_SCORE
    assert scores[f"ws:{wsc.OG}"] == wsc.OG_SCORE  # own goal for the away side, shoot-out out
    # Extra time: the metadata holds the 90-minute score (1-1, checked), so it is nulled.
    assert wsc.et_match().scores == wsc.ET_SCORE
    assert scores[f"ws:{wsc.ET}"] == (None, None)
    for mid in (wsc.NAMES, wsc.SHORT, wsc.DUP_OK):  # no metadata
        assert scores[f"ws:{mid}"] == (None, None), mid
    assert archive.counts["unscored"] == 3 and archive.counts["extra_time"] == 1
    # The short match is kept as recorded and counted; its PostGame row (minute 90) is ignored.
    assert f"ws:{wsc.SHORT}" in scores and archive.counts["ended_before_85"] == 1
    # Goals reconcile in every scored match.
    scored = archive.matches.filter(pl.col("home_score").is_not_null())
    assert scored.height == 2
    assert schema.reconcile_goals(archive.shots, scored).height == 0


def test_competition_table(ws_standard: Any) -> None:
    table = whoscored.COMPETITIONS
    assert len(table) == 11
    assert all(isinstance(c, whoscored.Competition) for c in table.values())
    assert {c.gender for c in table.values()} == {"male"}
    assert [f for f, c in table.items() if c.calendar_year] == ["World_Cup_2026"]
    names = {f: c.name for f, c in table.items()}
    for folder, name in [
        ("Bundesliga", "1. Bundesliga"),
        ("Belgium_Pro_League", "Belgian Pro League"),
        ("Championship", "Championship"),
        ("La_Liga", "La Liga"),
        ("Ligue_1", "Ligue 1"),
        ("Premier_League", "Premier League"),
        ("Serie_A", "Serie A"),
        ("World_Cup_2026", "FIFA World Cup"),
    ]:
        assert names[folder] == name, folder
    assert {"Eredivisie", "Primeira_Liga", "Super_Lig"} <= set(table)
    expected = {  # match -> (competition_id, season_id, competition_name, season_name, date)
        wsc.MAIN: ("ws:Championship", "ws:2025-2026", "Championship", "2025/2026",
                   dt.date(2025, 8, 16)),
        wsc.OG: ("ws:World_Cup_2026", "ws:2025-2026", "FIFA World Cup", "2026",
                 dt.date(2026, 6, 14)),
        wsc.NAMES: ("ws:Belgium_Pro_League", "ws:2025-2026", "Belgian Pro League", "2025/2026",
                    dt.date(2025, 8, 23)),
        wsc.SHORT: ("ws:Championship", "ws:2026-2027", "Championship", "2026/2027",
                    dt.date(2026, 8, 15)),
    }  # fmt: skip
    for mid, values in expected.items():
        row = _match(ws_standard, mid)
        got = tuple(row[c] for c in ("competition_id", "season_id", "competition_name",
                                     "season_name", "match_date"))  # fmt: skip
        assert got == values, mid
        assert (row["gender"], row["provider"], row["has_360"]) == ("male", "whoscored", None)
        assert row["schema_version"] == schema.SCHEMA_VERSION
