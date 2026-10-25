"""``whoscored.load``: the C3 ``shots`` columns from Opta types and qualifiers, coordinates through
the ``pitch`` landmarks, and header layouts (spec ``runs/MS4/whoscored/spec.md`` §2.1, §2.2, §6).

Fixtures are the synthetic archive of ``whoscored_cases``; expected values come from the spec's
tables and from ``football_core.pitch``, never from the module under test.
"""

from __future__ import annotations

import datetime as dt
import json
import tempfile
from pathlib import Path
from typing import Any

import hypothesis.strategies as st
import polars as pl
import pytest
import whoscored_cases as wsc
from hypothesis import given, settings
from polars.testing import assert_frame_equal

from football_core import pitch, schema

TOL = 1e-9


def _row(loaded: Any, gid: int) -> dict[str, Any]:
    rows = loaded.archive.shots.filter(pl.col("shot_id") == f"ws:{gid}").to_dicts()
    assert len(rows) == 1, f"ws:{gid}: {len(rows)} row(s)"
    return rows[0]


def _canon(x: float, y: float) -> tuple[float, float]:
    cx, cy = pitch.to_canonical(x, y, "opta")
    return float(cx), float(cy)


def _mouth(gy: float, gz: float) -> tuple[float, float]:
    cy, cz = pitch.goal_mouth_to_canonical(gy, gz, "opta")
    return float(cy), float(cz)


def _assert_close(actual: object, expected: float | None, what: str) -> None:
    if expected is None:
        assert actual is None, f"{what}: {actual!r}, expected null"
    else:
        assert actual == pytest.approx(expected, abs=TOL), what


# --- Outcomes and flags (q82, q101, q214, q328, q89) -----------------------------------------

_NO_END: dict[str, Any] = {"end": None}
OUTCOME_CASES: dict[str, tuple[int, dict[str, Any]]] = {
    "saved": (wsc.SAVED, {"outcome": "saved", "on_target": True, "is_goal": False,
                          "end": (98.5, 49.0), "mouth": (48.0, 10.0), "big_chance": False,
                          "first_time": False, "one_on_one": False}),
    "blocked_q82": (wsc.BLOCKED, {"outcome": "blocked", "on_target": False, "is_goal": False,
                                  "end": (88.9, 45.0), "mouth": None}),
    "line_clearance_q101": (wsc.CLEARED, {"outcome": "blocked", "on_target": False,
                                          "end": (100.0, 50.0), "mouth": None}),
    "post": (wsc.POST, {"outcome": "post", "on_target": False, **_NO_END}),
    "off_target": (wsc.OFF, {"outcome": "off_target", "on_target": False, **_NO_END,
                             "mouth": (60.0, 15.0)}),
    "off_target_with_blocked_xy": (wsc.OFF_BXY, {"outcome": "off_target", "on_target": False,
                                                 **_NO_END, "mouth": (40.0, 5.0)}),
    "goal": (wsc.GOAL, {"outcome": "goal", "on_target": True, "is_goal": True,
                        "is_own_goal": False, **_NO_END, "mouth": (50.0, 5.0)}),
    "big_chance_q214": (wsc.GOAL, {"big_chance": True}),
    "first_touch": (wsc.GOAL, {"first_time": True}),
    "one_on_one": (wsc.GOAL, {"one_on_one": True}),
}  # fmt: skip


@pytest.mark.parametrize("case", list(OUTCOME_CASES))
def test_outcome_and_flags(ws_standard: Any, case: str) -> None:
    gid, expected = OUTCOME_CASES[case]
    ev = wsc.main_match().event(gid)
    # Preconditions: the near misses are in the fixture.
    if case == "blocked_q82":
        assert ev.mouth is not None and ev.blocked is not None  # Opta's projection is present
    if case == "off_target_with_blocked_xy":
        assert ev.type == "MissedShots" and ev.blocked is not None
    if case == "line_clearance_q101":
        assert {82, 101} <= ev.qids
    row = _row(ws_standard, gid)
    for col, value in expected.items():
        if col == "end":
            ex, ey = (None, None) if value is None else _canon(*value)
            _assert_close(row["end_x"], ex, "end_x")
            _assert_close(row["end_y"], ey, "end_y")
            assert row["end_z"] is None
        elif col == "mouth":
            gy, gz = (None, None) if value is None else _mouth(*value)
            _assert_close(row["goal_y"], gy, "goal_y")
            _assert_close(row["goal_z"], gz, "goal_z")
        else:
            assert row[col] == value, col
    if case == "saved":  # flags are recorded False, not left null
        assert row["is_own_goal"] is False and row["is_penalty"] is False


# --- Own goal (q28) --------------------------------------------------------------------------


def test_own_goal_q28(ws_standard: Any) -> None:
    ev = wsc.og_match().event(wsc.OWN_GOAL)
    assert 28 in ev.qids and ev.x == "4.0"  # recorded in the conceding team's frame
    assert ev.related is not None and {72, 22} <= ev.qids  # context present, must be nulled
    row = _row(ws_standard, wsc.OWN_GOAL)
    assert row["team_id"] == f"ws:{wsc.H}"  # the conceding player's team
    assert row["opponent_team_id"] == f"ws:{wsc.V}"
    assert row["is_home"] is True
    assert (row["is_goal"], row["is_own_goal"], row["outcome"]) == (True, True, "goal")
    assert row["on_target"] is False
    assert (row["is_home"] ^ row["is_own_goal"]) is False  # the away side scored
    fx, fy = pitch.flip(*pitch.to_canonical(4.0, 52.0, "opta"))
    assert row["x"] == pytest.approx(float(fx), abs=TOL)
    assert row["y"] == pytest.approx(float(fy), abs=TOL)
    assert row["x"] > 100.0  # ≈ 101.2 m: in front of the goal it entered
    assert row["distance_m"] == pytest.approx(float(pitch.shot_distance(fx, fy)), abs=TOL)
    assert row["angle_rad"] == pytest.approx(float(pitch.opening_angle(fx, fy)), abs=TOL)
    for col in ("body_part", "first_time", "one_on_one", "big_chance", "assist_type",
                "assist_event_id", "goal_y", "goal_z", "end_x", "end_y"):  # fmt: skip
        assert row[col] is None, col
    assert row["situation"] == "open_play"


# --- Penalties (q9) and the shoot-out --------------------------------------------------------


def test_penalty_q9_and_shootout(ws_standard: Any) -> None:
    pen = _row(ws_standard, wsc.PENALTY)
    assert (pen["situation"], pen["is_penalty"], pen["period"]) == ("penalty", True, 2)
    assert (pen["x"], pen["y"]) == pytest.approx((94.0, 34.0), abs=TOL)  # the penalty spot
    assert pen["distance_m"] == pytest.approx(11.0, abs=TOL)
    assert pen["big_chance"] is True  # every penalty carries q214 (spec §1.6)
    kick = _row(ws_standard, wsc.SO_GOAL)
    assert (kick["period"], kick["is_penalty"], kick["is_goal"]) == (5, True, True)
    assert kick["seconds_in_period"] == pytest.approx(60.0)  # (121 - 120) * 60 + 0
    miss = _row(ws_standard, wsc.SO_MISS)
    assert (miss["period"], miss["is_penalty"], miss["outcome"]) == (5, True, "off_target")
    shots = ws_standard.archive.shots
    assert shots.filter(pl.col("is_penalty"))["situation"].cast(pl.String).unique().to_list() == [
        "penalty"
    ]


# --- Assists (q55) ---------------------------------------------------------------------------

ASSIST_CASES: dict[str, tuple[int, str | None, int | None]] = {
    "cross": (wsc.CROSS, "cross", wsc.CROSS_B26),
    "through_ball": (wsc.THROUGH, "through_ball", wsc.THROUGH_A34),
    "pass": (wsc.GOAL, "pass", wsc.PASS_B24),
    "other_team_same_event_id": (wsc.OTHER_TEAM, "cross", wsc.CROSS_A36),
    "non_pass": (wsc.NON_PASS, "none", None),
    "unresolved": (wsc.UNRESOLVED, None, None),
    "absent": (wsc.SAVED, "none", None),
    "own_goal": (wsc.OWN_GOAL, None, None),
    "unreferenced_duplicate_key": (wsc.DFK, "none", None),
}


@pytest.mark.parametrize("case", list(ASSIST_CASES))
def test_assist_q55(ws_standard: Any, case: str) -> None:
    gid, kind, target = ASSIST_CASES[case]
    main = wsc.main_match()
    if case == "other_team_same_event_id":
        # Both teams have eventId 36: the shooter's (A) is a cross, the other team's a through ball.
        b36 = main.event(wsc.THROUGH_B36)
        assert (b36.eid, b36.team, 4 in b36.qids) == (36, wsc.B, True)
        assert main.event(wsc.CROSS_A36).eid == 36
    if case == "unresolved":
        # eventId 90 exists only for the other team.
        assert [e.team for e in main.events if e.eid == 90] == [wsc.B]
    if case == "unreferenced_duplicate_key":
        keys = [e.gid for e in main.events if (e.team, e.eid) == (wsc.A, 50)]
        assert len(keys) == 2  # one (match, team, eventId) key, two global ids
        assert all(e.related != 50 for e in main.events)
    if case == "absent":
        ev = main.event(gid)
        assert 55 not in ev.qids and ev.related is None
    row = _row(ws_standard, gid)
    assert row["assist_type"] == kind
    assert row["assist_event_id"] == (None if target is None else f"ws:{target}")
    if target is not None:
        assert target != main.event(target).eid  # the global id, never the eventId


# --- Body part and situation -----------------------------------------------------------------

BODY_SITUATION: dict[str, tuple[int, str, str | None]] = {
    "q20_right_foot": (wsc.SAVED, "body_part", "right_foot"),
    "q72_left_foot": (wsc.BLOCKED, "body_part", "left_foot"),
    "q15_head": (wsc.CLEARED, "body_part", "head"),
    "q21_other": (wsc.POST, "body_part", "other"),
    "no_body_part": (wsc.NO_BODY, "body_part", None),
    "q22_open_play": (wsc.SAVED, "situation", "open_play"),
    "q23_fast_break": (wsc.POST, "situation", "fast_break"),
    "q24_from_free_kick": (wsc.OFF, "situation", "from_free_kick"),
    "q25_from_corner": (wsc.CLEARED, "situation", "from_corner"),
    "q26_direct_free_kick": (wsc.DFK, "situation", "direct_free_kick"),
    "q160_from_throw_in": (wsc.OFF_BXY, "situation", "from_throw_in"),
    "q9_penalty": (wsc.PENALTY, "situation", "penalty"),
    "q263_direct_corner": (wsc.DCORNER, "situation", "direct_corner"),
}


@pytest.mark.parametrize("case", list(BODY_SITUATION))
def test_body_part_and_situation(ws_standard: Any, case: str) -> None:
    gid, col, expected = BODY_SITUATION[case]
    if case == "q263_direct_corner":
        assert {25, 263} <= wsc.main_match().event(gid).qids  # q25 is present too
    assert _row(ws_standard, gid)[col] == expected


# --- Columns Opta cannot fill ----------------------------------------------------------------

ALWAYS_NULL = ("end_z", "gk_x", "gk_y", "visible_area", "technique", "follows_dribble",
               "open_goal", "deflected", "provider_xg", "provider_psxg")  # fmt: skip


def test_null_columns(ws_standard: Any) -> None:
    archive = ws_standard.archive
    shots = archive.shots
    assert shots.schema == schema.SHOT_SCHEMA  # every C3 column, in order, with its dtype
    assert {108, 215} <= wsc.main_match().event(wsc.SAVED).qids  # volley, individual play
    for col in ALWAYS_NULL:
        assert shots[col].null_count() == shots.height, col
    assert archive.freeze_frames.height == 0
    assert archive.freeze_frames.schema == schema.empty_freeze_frames().schema
    assert archive.matches.schema == schema.MATCH_SCHEMA
    # The tables are validated: re-validation finds nothing.
    schema.validate_matches(archive.matches)
    schema.validate_shots(shots, matches=archive.matches)


# --- Ids, clock, lineage ---------------------------------------------------------------------


def test_ids_and_clock(ws_standard: Any) -> None:
    shots = ws_standard.archive.shots
    assert sorted(shots["shot_id"].to_list()) == sorted(f"ws:{g}" for g in wsc.ALL_SHOTS)
    saved = _row(ws_standard, wsc.SAVED)
    assert saved["player_id"] == f"ws:{wsc.B * 10 + 1}"
    assert (saved["team_id"], saved["opponent_team_id"]) == (f"ws:{wsc.B}", f"ws:{wsc.A}")
    assert (saved["match_id"], saved["competition_id"], saved["season_id"]) == (
        f"ws:{wsc.MAIN}",
        "ws:Championship",
        "ws:2025-2026",
    )
    assert saved["match_date"] == dt.date(2025, 8, 16)
    assert (saved["provider"], saved["schema_version"]) == ("whoscored", schema.SCHEMA_VERSION)
    assert saved["raw_type"] == "SavedShot"
    quals = wsc.main_match().event(wsc.SAVED).quals
    assert saved["raw_qualifiers"] == wsc.raw_json(quals)
    assert json.loads(saved["raw_qualifiers"]) == quals
    off = _row(ws_standard, wsc.OFF)
    assert 11112 in {e["type"]["value"] for e in json.loads(off["raw_qualifiers"])}
    # (gid, period, minute, second, seconds_in_period); "12" and "12.0" files alike.
    clock = [
        (wsc.SAVED, 1, 5, 12, 312.0),
        (wsc.CLEARED, 1, 46, 30, 2790.0),  # first-half stoppage time
        (wsc.CROSS, 2, 50, 20, 320.0),
        (wsc.OWN_GOAL, 1, 30, 12, 1812.0),  # second written "12.0"
        (wsc.ET_K_ET_GOAL, 3, 100, 0, 600.0),
    ]
    assert wsc.og_match().second_suffix == ".0" and wsc.main_match().second_suffix == ""
    for gid, period, minute, second, sip in clock:
        row = _row(ws_standard, gid)
        assert (row["period"], row["minute"], row["second"]) == (period, minute, second), gid
        assert row["seconds_in_period"] == pytest.approx(sip), gid


# --- Coordinates through the pitch landmarks -------------------------------------------------

LANDMARKS = {
    "penalty_spot": (wsc.PENALTY, (88.5, 50.0), (94.0, 34.0)),
    "goal_line_right_post": (wsc.POST, (100.0, 45.2), (105.0, 30.34)),
    "box_right_corner": (wsc.SAVED, (83.0, 21.1), (88.5, 13.84)),
    "box_left_corner": (wsc.NON_PASS, (83.0, 63.2), (88.5, 43.16)),
}


@pytest.mark.parametrize("case", list(LANDMARKS))
def test_landmarks(ws_standard: Any, case: str) -> None:
    gid, raw, metres = LANDMARKS[case]
    linear = (raw[0] * 1.05, raw[1] * 0.68)
    assert any(abs(a - b) > 0.1 for a, b in zip(linear, metres, strict=True))  # not a scale
    row = _row(ws_standard, gid)
    assert (row["x"], row["y"]) == pytest.approx(metres, abs=TOL)
    assert row["distance_m"] == pytest.approx(float(pitch.shot_distance(*metres)), abs=TOL)
    assert row["angle_rad"] == pytest.approx(float(pitch.opening_angle(*metres)), abs=TOL)


def test_goal_mouth_landmarks(ws_standard: Any) -> None:
    post = _row(ws_standard, wsc.POST)  # Opta (45.2, 38): the right post at the crossbar
    assert (post["goal_y"], post["goal_z"]) == pytest.approx((30.34, 2.44), abs=TOL)
    blocked = _row(ws_standard, wsc.BLOCKED)
    assert blocked["goal_y"] is None and blocked["goal_z"] is None


@settings(max_examples=25, deadline=None)
@given(
    x=st.integers(0, 10_000),
    y=st.integers(0, 10_000),
    style=st.sampled_from(["int", "dot0", "dec"]),
)
def test_coordinates_property(x: int, y: int, style: str) -> None:
    """Any COORD_TEXT in [0, 100] maps exactly as ``pitch.to_canonical`` maps its value."""

    def text(v: int) -> str:
        if style == "dec" or v % 100:
            return f"{v // 100}.{v % 100:02d}"
        return f"{v // 100}" + (".0" if style == "dot0" else "")

    fx = wsc.mini()
    ev = fx.match("main").event(wsc.NO_BODY)
    ev.x, ev.y = text(x), text(y)
    with tempfile.TemporaryDirectory() as tmp:
        loaded = wsc.load_fixture(Path(tmp), fx)
    row = _row(loaded, wsc.NO_BODY)
    cx, cy = _canon(x / 100, y / 100)
    assert (row["x"], row["y"]) == pytest.approx((cx, cy), abs=TOL)


# --- Header layouts --------------------------------------------------------------------------


def _mini_shots(layout: tuple[str, ...], second_suffix: str = "") -> pl.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        loaded = wsc.load_fixture(Path(tmp), wsc.mini(layout, second_suffix))
    return loaded.archive.shots.sort("shot_id")


@pytest.fixture(scope="module")
def reference_shots() -> pl.DataFrame:
    return _mini_shots(wsc.LAYOUT_A)


def test_header_layouts(reference_shots: pl.DataFrame) -> None:
    assert "isOwnGoal" not in wsc.LAYOUT_A and "isOwnGoal" in wsc.LAYOUT_B
    assert not {"isGoal", "isOwnGoal", "cardType"} & set(wsc.LAYOUT_C)
    assert reference_shots.height == len(wsc.MAIN_SHOTS)
    for layout in (wsc.LAYOUT_B, wsc.LAYOUT_C):
        assert_frame_equal(_mini_shots(layout), reference_shots)
    assert_frame_equal(_mini_shots(wsc.LAYOUT_A, ".0"), reference_shots)  # second "12.0"


@settings(max_examples=10, deadline=None)
@given(order=st.permutations(wsc.LAYOUT_A))
def test_column_order_invariant(reference_shots: pl.DataFrame, order: list[str]) -> None:
    assert_frame_equal(_mini_shots(tuple(order)), reference_shots)
