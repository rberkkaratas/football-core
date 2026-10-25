"""``whoscored.load``: one test per build ratify item that states behaviour (build log
``runs/MS4/whoscored/build.md`` "Architect: ratify" 1-11).

Each test starts from a small archive (``whoscored_cases.mini`` or one named match), mutates
one cell or file, and asserts the precondition its reverted rule would need, so it cannot pass
by coincidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import polars as pl
import pytest
import whoscored_cases as wsc

from football_core.io import snapshot, whoscored

STEM = f"16082025_{wsc.MAIN}"


def _row(archive: Any, gid: int) -> dict[str, Any]:
    (row,) = archive.shots.filter(pl.col("shot_id") == f"ws:{gid}").to_dicts()
    return row


def _main_row(archive: Any) -> dict[str, Any]:
    (row,) = archive.matches.filter(pl.col("match_id") == f"ws:{wsc.MAIN}").to_dicts()
    return row


def _main_meta_without(fx: wsc.Fixture, key: str) -> None:
    m = fx.match("main")
    meta = m.metadata()
    del meta[key]
    m.meta_text = json.dumps(meta)


# 1. Metadata present with both scores null -> "unscored".
def test_metadata_both_scores_null_is_unscored(tmp_path: Path) -> None:
    fx = wsc.mini()
    fx.match("main").meta_overrides.update(home_score=None, away_score=None)
    wsc.write_archive(tmp_path, fx)
    assert fx.match("main").meta_path(tmp_path).is_file()  # metadata exists: not the no-file path
    archive = whoscored.load(tmp_path)
    row = _main_row(archive)
    assert row["home_score"] is None and row["away_score"] is None
    assert archive.counts["unscored"] == 1
    assert any(p.name.endswith("_metadata.json") for p in archive.files)  # home/away from it


# 2. One match id in two event files -> ValueError (never a silent replacement).
def test_match_id_in_two_event_files(tmp_path: Path) -> None:
    names = wsc.names_match()  # no metadata: a second copy with another date would load
    wsc.write_archive(tmp_path, wsc.Fixture([names]))
    path = names.event_path(tmp_path)
    copy = path.with_name(f"24082025_{wsc.NAMES}.csv")
    assert copy.name > path.name  # read second: replacing would keep the copy
    copy.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="two event files") as info:
        whoscored.load(tmp_path)
    assert str(wsc.NAMES) in str(info.value)


# 3. Metadata competition_key null or missing is accepted; a non-str start_time is a ValueError.
@pytest.mark.parametrize("case", ["key_null", "key_missing"])
def test_metadata_competition_key_absent(tmp_path: Path, case: str) -> None:
    fx = wsc.mini()
    if case == "key_null":
        fx.match("main").meta_overrides["competition_key"] = None
    else:
        _main_meta_without(fx, "competition_key")
    archive = wsc.load_fixture(tmp_path, fx).archive
    assert _main_row(archive)["home_score"] == wsc.MAIN_SCORE[0]  # metadata was read


@pytest.mark.parametrize("value", [20250816, None, ["2025-08-16T15:00:00"]])
def test_metadata_start_time_not_str(tmp_path: Path, value: object) -> None:
    fx = wsc.mini()
    if value is None:
        _main_meta_without(fx, "start_time")
    else:
        fx.match("main").meta_overrides["start_time"] = value
    wsc.write_archive(tmp_path, fx)
    with pytest.raises(ValueError, match="start_time") as info:
        whoscored.load(tmp_path)
    assert str(wsc.MAIN) in str(info.value)


# 4. A half-present point maps per element.
def test_half_present_point_per_element(tmp_path: Path) -> None:
    full = wsc.load_fixture(tmp_path / "full", wsc.mini()).archive
    fx = wsc.mini()
    saved = fx.match("main").event(wsc.SAVED)
    assert saved.type == "SavedShot" and saved.blocked and saved.mouth
    saved.blocked = (saved.blocked[0], "")
    saved.mouth = (saved.mouth[0], "")
    half = _row(wsc.load_fixture(tmp_path / "half", fx).archive, wsc.SAVED)
    ref = _row(full, wsc.SAVED)
    assert ref["end_y"] is not None and ref["goal_z"] is not None
    assert half["end_x"] == pytest.approx(ref["end_x"]) and half["end_y"] is None
    assert half["goal_y"] == pytest.approx(ref["goal_y"]) and half["goal_z"] is None


# 5. teamId is integer text on every row; a null period on any row is an unknown period.
def test_team_id_checked_on_non_shot_rows(tmp_path: Path) -> None:
    fx = wsc.mini()
    ev = fx.match("main").event(wsc.DUPKEY_A50_1)
    assert not ev.is_shot
    ev.raw["teamId"] = f"{wsc.A}.5"
    wsc.write_archive(tmp_path, fx)
    with pytest.raises(ValueError) as info:
        whoscored.load(tmp_path)
    # Reported as a malformed teamId, not as a third team.
    assert f"teamId '{wsc.A}.5'" in str(info.value) and STEM in str(info.value)


def test_null_period_on_non_shot_row(tmp_path: Path) -> None:
    fx = wsc.mini()
    ev = fx.match("main").event(wsc.DUPKEY_A50_1)
    assert not ev.is_shot
    ev.raw["period"] = ""
    wsc.write_archive(tmp_path, fx)
    with pytest.raises(ValueError, match="unknown period None") as info:
        whoscored.load(tmp_path)
    assert STEM in str(info.value)


# 6. The q55 value may be a str or a non-bool int (its text must be integer text).
@pytest.mark.parametrize(("value", "ok"), [(24, True), ("24.0", True), (24.0, False)])
def test_q55_value_type(tmp_path: Path, value: object, ok: bool) -> None:
    fx = wsc.mini()
    goal = fx.match("main").event(wsc.GOAL)
    assert goal.related == 24 and 55 in goal.qids
    for item in goal.quals:
        if item["type"]["value"] == 55:
            item["value"] = value
    wsc.write_archive(tmp_path, fx)
    if not ok:
        with pytest.raises(ValueError, match="q55") as info:
            whoscored.load(tmp_path)
        assert STEM in str(info.value)
        return
    row = _row(whoscored.load(tmp_path), wsc.GOAL)
    assert (row["assist_type"], row["assist_event_id"]) == ("pass", f"ws:{wsc.PASS_B24}")


# 7. A kept match with no event in PERIODS counts ended_before_85.
def test_no_in_play_event_counts_short(tmp_path: Path) -> None:
    names = wsc.names_match()  # names route: team names come from the PreMatch rows
    names.events = [e for e in names.events if e.period in whoscored.NON_PLAY_PERIODS]
    assert max(e.minute for e in names.events) >= whoscored.SHORT_MATCH_MINUTE  # PostGame row
    archive = wsc.load_fixture(tmp_path, wsc.Fixture([names])).archive
    assert archive.matches.height == 1 and archive.shots.height == 0
    assert archive.counts["ended_before_85"] == 1


# 8. Read errors on an opened file become a ValueError naming the file.
@pytest.mark.parametrize("case", ["header_not_utf8", "minute_overflow", "match_ids_parse"])
def test_read_errors_name_the_file(tmp_path: Path, case: str) -> None:
    fx = wsc.mini()
    main = fx.match("main")
    if case == "minute_overflow":  # passes INT_TEXT; too large for Int64 (a polars cast error)
        main.event(wsc.DUPKEY_A50_1).raw["minute"] = "9" * 30
    wsc.write_archive(tmp_path, fx)
    path = main.event_path(tmp_path)
    if case == "header_not_utf8":  # the csv module reads the header
        path.write_bytes(b"\xff\xfe" + path.read_bytes())
    elif case == "match_ids_parse":  # header fine, polars cannot parse the body
        path = tmp_path / "match_ids" / f"{main.ids_stem}.csv"
        path.write_text(path.read_text(encoding="utf-8") + '"1,ok\n', encoding="utf-8")
    with pytest.raises(ValueError) as info:
        whoscored.load(tmp_path)
    assert path.name in str(info.value)


# 9. Names route: a null validated name fails like an unmatched one; home must differ from away.
@pytest.mark.parametrize("case", ["home_null", "away_null", "same_name"])
def test_names_route_pair(tmp_path: Path, case: str) -> None:
    names = wsc.names_match()
    assert names.scores is None  # the names route
    wsc.write_archive(tmp_path, wsc.Fixture([names]))
    ids = tmp_path / "match_ids" / f"{names.ids_stem}.csv"
    home, away = wsc.TEAM_NAMES[names.home], wsc.TEAM_NAMES[names.away]
    text = ids.read_text(encoding="utf-8")
    assert text.count(f",{home},{away}\n") == 1
    pair = {"home_null": f",,{away}", "away_null": f",{home},", "same_name": f",{away},{away}"}
    ids.write_text(text.replace(f",{home},{away}\n", pair[case] + "\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="validated teams") as info:
        whoscored.load(tmp_path)
    assert names.stem in str(info.value)


# 10. New public names; the §2.1 tables and C3_SORT_BY are read-only.
def test_public_names_and_read_only_tables() -> None:
    assert whoscored.EVENT_COLUMNS == wsc.REQUIRED
    assert whoscored.MATCH_ID_COLUMNS == (
        "match_id", "validation_status", "competition_key", "validated_home_team",
        "validated_away_team",
    )  # fmt: skip
    tables = {
        "PERIODS": whoscored.PERIODS,
        "SHOT_TYPES": whoscored.SHOT_TYPES,
        "BODY_PARTS": whoscored.BODY_PARTS,
        "PATTERNS": whoscored.PATTERNS,
        "STATUSES": whoscored.STATUSES,
        "COMPETITIONS": whoscored.COMPETITIONS,
        "C3_SORT_BY": snapshot.C3_SORT_BY,
    }
    for name, table in tables.items():
        assert isinstance(table, MappingProxyType), name
        with pytest.raises(TypeError):
            table["new"] = None  # type: ignore[index]


# 11. A missing root names <root>/events.
def test_missing_root_names_events(tmp_path: Path) -> None:
    root = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError) as info:
        whoscored.load(root)
    assert str(root / "events") in str(info.value)
