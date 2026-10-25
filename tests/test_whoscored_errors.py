"""``whoscored.load`` totality: one ``ValueError`` per spec §3.1 item, missing paths, and the row
scope of the checks (spec ``runs/MS4/whoscored/spec.md`` §1.2, §1.3, §1.4, §2.2, §3, §3.1).

Each case mutates one cell, file or row of a fresh standard archive, so the error can only come
from the rule the case names. Messages must name the file (or match) and the bad value.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import whoscored_cases as wsc

from football_core import schema
from football_core.io import whoscored

Mutation = Callable[[wsc.Fixture], None]
STEM = f"16082025_{wsc.MAIN}"  # the MAIN event file
SAVED_IDS = (str(wsc.SAVED), "20")  # either "event id" (global id or eventId) may be named


def _raw(gid: int, column: str, text: str) -> Mutation:
    def mutate(fx: wsc.Fixture) -> None:
        fx.match("main").event(gid).raw[column] = text

    return mutate


def _quals(text_of: Callable[[str], str], gid: int = wsc.SAVED) -> Mutation:
    """Replace a row's qualifier text, given the valid text."""

    def mutate(fx: wsc.Fixture) -> None:
        ev = fx.match("main").event(gid)
        ev.raw["qualifiers"] = text_of(repr(ev.quals))

    return mutate


def _append(bad: str) -> Callable[[str], str]:
    return lambda valid: valid[:-1] + ", " + bad + "]"


def _add_qual(qid: int) -> Mutation:
    def mutate(fx: wsc.Fixture) -> None:
        fx.match("main").event(wsc.SAVED).quals.append(wsc.q(qid))

    return mutate


def _set(key: str, attr: str, value: object) -> Mutation:
    def mutate(fx: wsc.Fixture) -> None:
        setattr(fx.match(key), attr, value)

    return mutate


def _meta(**overrides: object) -> Mutation:
    def mutate(fx: wsc.Fixture) -> None:
        fx.match("main").meta_overrides.update(overrides)

    return mutate


def _meta_text(text: str) -> Mutation:
    return _set("main", "meta_text", text)


def _layout(columns: tuple[str, ...]) -> Mutation:
    return _set("main", "layout", columns)


def _no_pattern(fx: wsc.Fixture) -> None:
    ev = fx.match("main").event(wsc.SAVED)
    ev.quals = [e for e in ev.quals if e["type"]["value"] != 22]


def _related_without_q55(fx: wsc.Fixture) -> None:
    fx.match("main").event(wsc.SAVED).related = 24


def _referenced_duplicate_key(fx: wsc.Fixture) -> None:
    ev = fx.match("main").event(wsc.UNRESOLVED)  # team A; A has two global ids for eventId 50
    ev.quals = [wsc.q(22), wsc.q(20), wsc.q(55, "50")]
    ev.related = 50


def _third_team(fx: wsc.Fixture) -> None:
    fx.match("main").event(wsc.DUPKEY_A50_1).team = 9999


def _stem_date(fx: wsc.Fixture) -> None:
    m = fx.match("main")
    m.date, m.scores = "31022026", None  # no metadata: only the stem carries the date


def _second_metadata(fx: wsc.Fixture) -> None:
    m = fx.match("main")
    path = f"events/{m.folder}/{m.season}/metadata/17082025_{wsc.MAIN}_metadata.json"
    fx.extra_files[path] = json.dumps(m.metadata())


def _nan_metadata(fx: wsc.Fixture) -> None:
    m = fx.match("main")
    m.meta_text = json.dumps(m.metadata()).replace('"home_score": 1', '"home_score": NaN')
    assert "NaN" in m.meta_text


def _stray_csv(fx: wsc.Fixture) -> None:
    fx.extra_files["events/Championship/2025-2026/notes.csv"] = "a,b\n1,2\n"


def _match_id_twice(fx: wsc.Fixture) -> None:
    again = wsc.standard().match("ok_nofile1")
    again.key, again.folder, again.season = "again", "World_Cup_2026", "2025-2026"
    fx.matches.append(again)


def _two_ok_copies(fx: wsc.Fixture) -> None:
    fx.match("dup_wrong").status = "ok"


CASES: dict[str, tuple[Mutation, tuple[str | tuple[str, ...], ...]]] = {
    # Periods (every row) and the row scope's period rule on shot rows.
    "period_unknown_on_non_shot": (_raw(wsc.DUPKEY_A50_1, "period", "ExtraFirstHalf"),
                                   (STEM, "ExtraFirstHalf")),
    "period_prematch_on_shot": (_raw(wsc.SAVED, "period", "PreMatch"), (STEM, "PreMatch")),
    "period_postgame_on_shot": (_raw(wsc.SAVED, "period", "PostGame"), (STEM, "PostGame")),
    "minute_fraction_on_non_shot": (_raw(wsc.DUPKEY_A50_1, "minute", "40.5"), (STEM, "40.5")),
    # Header.
    "header_duplicate_column": (_layout((*wsc.LAYOUT_A, "x")), (STEM,)),
    "header_missing_column": (_layout(tuple(c for c in wsc.LAYOUT_A if c != "blockedY")),
                              (STEM, "blockedY")),
    # Ids.
    "id_underscore": (_raw(wsc.SAVED, "id", "2901000_101.0"), (STEM, "2901000_101")),
    "id_fraction": (_raw(wsc.SAVED, "id", "2901000101.5"), (STEM, "2901000101.5")),
    "event_id_arabic_indic_digits": (_raw(wsc.SAVED, "eventId", "٢٠"), (STEM, "٢٠")),
    "player_id_spaces": (_raw(wsc.SAVED, "playerId", " 7 "), (STEM, " 7 ")),
    "player_id_empty_on_shot": (_raw(wsc.SAVED, "playerId", ""), (STEM,)),
    "team_id_empty": (_raw(wsc.SAVED, "teamId", ""), (STEM,)),
    # Numbers.
    "x_underscore": (_raw(wsc.SAVED, "x", "8_3"), (STEM, "8_3")),
    "y_spaces": (_raw(wsc.SAVED, "y", " 7 "), (STEM, " 7 ")),
    "x_empty": (_raw(wsc.SAVED, "x", ""), (STEM,)),
    "x_above_100": (_raw(wsc.SAVED, "x", "100.5"), (STEM, "100.5")),
    "y_negative": (_raw(wsc.SAVED, "y", "-0.5"), (STEM, "-0.5")),
    "goal_mouth_exponent": (_raw(wsc.SAVED, "goalMouthY", "1e1"), (STEM, "1e1")),
    "blocked_nan": (_raw(wsc.SAVED, "blockedX", "nan"), (STEM, "nan")),
    "blocked_above_100": (_raw(wsc.SAVED, "blockedY", "101.0"), (STEM, "101.0")),
    "second_fraction": (_raw(wsc.SAVED, "second", "12.5"), (STEM, "12.5")),
    "second_empty_on_shot": (_raw(wsc.SAVED, "second", ""), (STEM,)),
    # Qualifiers (literal text, value walk, top-level shape).
    "qualifiers_tuple": (_quals(lambda v: "(" + v[1:-1] + ",)"), (STEM, SAVED_IDS)),
    "qualifiers_unclosed": (_quals(lambda v: v[:-1]), (STEM, SAVED_IDS)),
    "qualifiers_not_a_list": (_quals(lambda v: repr(wsc.q(22))), (STEM, SAVED_IDS)),
    "qualifier_bool_id": (_quals(_append("{'type': {'value': True, 'displayName': 'X'}}")),
                          (STEM, SAVED_IDS)),
    "qualifier_negative_id": (_quals(_append("{'type': {'value': -1, 'displayName': 'X'}}")),
                              (STEM, SAVED_IDS)),
    "qualifier_no_display_name": (_quals(_append("{'type': {'value': 212}}")),
                                  (STEM, SAVED_IDS)),
    "qualifier_int_key": (_quals(_append("{'type': {'value': 212, 'displayName': 'L'}, 1: 'x'}")),
                          (STEM, SAVED_IDS)),
    "qualifier_bool_value": (_quals(_append("{'type': {'value': 212, 'displayName': 'L'}, "
                                            "'value': False}")), (STEM, SAVED_IDS)),
    "qualifier_set_value": (_quals(_append("{'type': {'value': 212, 'displayName': 'L'}, "
                                           "'value': {1, 2}}")), (STEM, SAVED_IDS)),
    "qualifier_bytes_value": (_quals(_append("{'type': {'value': 212, 'displayName': 'L'}, "
                                             "'value': b'x'}")), (STEM, SAVED_IDS)),
    "qualifier_infinite_value": (_quals(_append("{'type': {'value': 212, 'displayName': 'L'}, "
                                                "'value': 1e999}")), (STEM, SAVED_IDS)),
    "q55_target_qualifiers": (_quals(lambda v: "[{'type': {'value': True}}]", wsc.PASS_B24),
                              (STEM,)),
    "q55_target_id": (_raw(wsc.PASS_B24, "id", "pass-24"), (STEM, "pass-24")),
    # Mapping rules (§2.2).
    "two_body_parts": (_add_qual(72), (STEM,)),
    "no_pattern": (_no_pattern, (STEM,)),
    "two_patterns": (_add_qual(23), (STEM,)),
    "own_goal_on_saved_shot": (_add_qual(28), (STEM,)),
    "q55_disagrees_with_related": (_raw(wsc.CROSS, "relatedEventId", "27.0"), (STEM,)),
    "q55_without_related": (_raw(wsc.CROSS, "relatedEventId", ""), (STEM,)),
    "related_without_q55": (_related_without_q55, (STEM,)),
    "q55_referenced_duplicate_key": (_referenced_duplicate_key, (STEM,)),
    # Teams.
    "third_team_id": (_third_team, (STEM,)),
    "team_name_on_two_ids": (_raw(wsc.DUPKEY_A50_1, "teamName", wsc.TEAM_NAMES[wsc.B]),
                             (STEM,)),
    # Location (paths and match_ids only, so also on dropped files).
    "unknown_competition_folder": (_set("main", "folder", "Bundesliga_2"),
                                   ("Bundesliga_2", "Championship")),
    "season_folder_not_consecutive": (_set("main", "season", "2025-2027"), ("2025-2027",)),
    "season_folder_short_form": (_set("main", "season", "2025-26"), ("2025-26",)),
    "stem_date_not_a_date": (_stem_date, ("31022026",)),
    "csv_with_another_stem": (_stray_csv, ("notes.csv",)),
    "row_from_another_match_ids_file": (_set("main", "ids_file", "Championship_2026-2027"),
                                        (str(wsc.MAIN),)),
    "competition_key_not_folder": (_set("main", "competition_key", "Serie_A"), ("Serie_A",)),
    "competition_key_on_dropped_row": (_set("pending", "competition_key", "Serie_A"),
                                       ("Serie_A",)),
    # Metadata (kept matches).
    "metadata_two_files": (_second_metadata, (str(wsc.MAIN),)),
    "metadata_not_json": (_meta_text("{"), (str(wsc.MAIN),)),
    "metadata_nan": (_nan_metadata, (str(wsc.MAIN),)),
    "metadata_top_level_list": (_meta_text("[]"), (str(wsc.MAIN),)),
    "metadata_match_id_int": (_meta(match_id=wsc.MAIN), (str(wsc.MAIN),)),
    "metadata_match_id_other": (_meta(match_id=str(wsc.OG)), (str(wsc.MAIN), str(wsc.OG))),
    "metadata_team_id_bool": (_meta(home_team_id=True), (str(wsc.MAIN),)),
    "metadata_team_id_str": (_meta(away_team_id=str(wsc.A)), (str(wsc.MAIN),)),
    "metadata_team_not_in_file": (_meta(home_team_id=9999), (str(wsc.MAIN), "9999")),
    "metadata_date": (_meta(start_time="2025-08-17T15:00:00"), (str(wsc.MAIN), "2025-08-17")),
    "metadata_competition_key": (_meta(competition_key="Serie_A"), ("Serie_A",)),
    "metadata_score_bool": (_meta(home_score=True), (str(wsc.MAIN),)),
    "metadata_score_str": (_meta(home_score="1"), (str(wsc.MAIN),)),
    "metadata_score_float": (_meta(home_score=1.0), (str(wsc.MAIN),)),
    "metadata_score_above_99": (_meta(away_score=100), (str(wsc.MAIN), "100")),
    "metadata_score_negative": (_meta(away_score=-1), (str(wsc.MAIN), "-1")),
    "metadata_score_one_null": (_meta(away_score=None), (str(wsc.MAIN),)),
    "extra_time_final_score": (_set("et", "scores", (1, 2)), (str(wsc.ET),)),
    # match_ids and the keep rule.
    "status_unknown": (_set("main", "status", "approved"), ("approved",)),
    "match_id_in_two_rows": (_match_id_twice, (str(wsc.OK_NOFILE[0]),)),
    "ok_without_home_away": (_set("names", "validated", False), (str(wsc.NAMES),)),
}  # fmt: skip


@pytest.mark.parametrize("case", list(CASES))
def test_load_raises(case: str, tmp_path: Path) -> None:
    mutate, needles = CASES[case]
    fx = wsc.standard()
    mutate(fx)
    wsc.write_archive(tmp_path, fx)
    with pytest.raises(ValueError) as info:
        whoscored.load(tmp_path)
    assert not isinstance(info.value, schema.SchemaValidationError), info.value
    message = str(info.value)
    for needle in needles:
        options = (needle,) if isinstance(needle, str) else needle
        assert any(o in message for o in options), (options, message)


def test_two_ok_copies_fail_validation(tmp_path: Path) -> None:
    """Two ``ok`` copies of one match never pass silently: ``shot_id`` must be unique."""
    fx = wsc.standard()
    _two_ok_copies(fx)
    wsc.write_archive(tmp_path, fx)
    with pytest.raises(schema.SchemaValidationError, match="shot_id"):
        whoscored.load(tmp_path)


def test_row_scope(tmp_path: Path) -> None:
    """Non-shot rows that no q55 references are read for team, period and minute only: odd
    qualifier text and coordinates there are legal (spec §3.1 row scope)."""
    fx = wsc.standard()
    main = fx.match("main")
    other = main.event(wsc.RECOVERY_B90)
    assert all(e.related != other.eid or e.team != other.team for e in main.events)
    other.raw |= {"qualifiers": "[{'type': (1, 2)}]", "x": "1_0", "y": "", "second": ""}
    main.event(wsc.DUPKEY_A50_2).raw["playerId"] = ""
    wsc.write_archive(tmp_path, fx)
    archive = whoscored.load(tmp_path)
    assert archive.shots.height == len(wsc.ALL_SHOTS)


@pytest.mark.parametrize(
    ("case", "needle"),
    [
        ("root", "nowhere"),
        ("events", "events"),
        ("match_ids", "match_ids"),
        ("match_ids_without_csv", "match_ids"),
    ],
)
def test_load_missing_paths(case: str, needle: str, tmp_path: Path) -> None:
    root = wsc.write_archive(tmp_path / "archive", wsc.standard())
    if case == "root":
        root = tmp_path / "nowhere"
    elif case == "events":
        shutil.rmtree(tmp_path / "archive" / "events")
    else:
        shutil.rmtree(tmp_path / "archive" / "match_ids")
        if case == "match_ids_without_csv":
            (root / "match_ids").mkdir()
            (root / "match_ids" / ".DS_Store").write_bytes(b"\x00")
    with pytest.raises(FileNotFoundError, match=needle):
        whoscored.load(root)
