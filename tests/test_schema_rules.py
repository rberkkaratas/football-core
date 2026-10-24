"""Cross-column rules on the shot table and their totality contract (C3; spec §4)."""

from __future__ import annotations

import polars as pl
import pytest
import schema_cases as sc

from football_core import pitch, schema


def _err(df: pl.DataFrame) -> schema.SchemaValidationError:
    with pytest.raises(schema.SchemaValidationError) as info:
        schema.validate_shots(df)
    return info.value


def _pairs(err: schema.SchemaValidationError) -> set[tuple[tuple[str, ...], str]]:
    return {(f.columns, f.check) for f in err.failures}


def _rule(err: schema.SchemaValidationError, name: str) -> schema.Failure:
    found = [f for f in err.failures if f.check == name]
    assert len(found) == 1, (name, _pairs(err))
    return found[0]


# --- Rule metadata --------------------------------------------------------------------------


def test_shot_rules_are_pinned() -> None:
    assert [(r.name, r.columns) for r in schema.SHOT_RULES] == list(sc.SHOT_RULES)


def test_freeze_frame_rules_are_pinned() -> None:
    assert [(r.name, r.columns) for r in schema.FREEZE_FRAME_RULES] == list(sc.FREEZE_FRAME_RULES)


def test_rules_have_descriptions_and_shot_id_first() -> None:
    for rule in schema.SHOT_RULES + schema.FREEZE_FRAME_RULES:
        assert rule.description.strip(), rule.name
        assert rule.columns[0] == "shot_id", rule.name
        assert callable(rule.check)


# --- One violation per rule (spec §4.1) -----------------------------------------------------

# (rule, row, cells) where the cells break exactly that rule on that fixture row
VIOLATIONS = [
    ("ids_match_provider", 0, {"player_id": "ws:90001"}),
    ("ids_match_provider", 2, {"assist_event_id": "sb:900000002"}),
    ("ids_match_provider", 1, {"competition_id": "ws:9001"}),
    ("distinct_teams", 0, {"opponent_team_id": "sb:901"}),
    ("distance_matches_pitch", 0, {"distance_m": 12.0}),
    ("angle_matches_pitch", 0, {"angle_rad": 0.5}),
    ("is_goal_matches_outcome", 0, {"is_goal": False}),
    ("is_goal_matches_outcome", 1, {"is_goal": True}),
    ("own_goal_is_goal", 2, {"is_own_goal": True}),
    ("on_target_matches_outcome", 1, {"on_target": False}),
    ("on_target_matches_outcome", 4, {"on_target": True}),
    ("on_target_matches_outcome", 3, {"on_target": True}),  # own goal is never on target
    ("blocked_has_no_goal_mouth", 2, {"goal_y": 34.0}),
    ("blocked_has_no_goal_mouth", 2, {"goal_z": 0.0}),
    ("penalty_matches_situation", 0, {"is_penalty": True}),
    ("penalty_matches_situation", 1, {"situation": "direct_free_kick"}),
    ("shootout_is_penalty", 4, {"situation": "open_play", "is_penalty": False}),
    ("minute_within_period", 1, {"minute": 44, "second": 0}),
    ("clock_consistent", 0, {"seconds_in_period": 1500.0}),
    ("clock_consistent", 0, {"seconds_in_period": 1396.0}),  # exactly 1.0 off fails (< 1.0)
    ("unassisted_has_no_assist_event", 1, {"assist_event_id": "sb:00000000-0000-0000-0000-a2"}),
]


@pytest.mark.parametrize(
    ("rule", "row", "cells"),
    VIOLATIONS,
    ids=[f"{rule}-r{row}-{'-'.join(cells)}" for rule, row, cells in VIOLATIONS],
)
def test_rule_violation(valid_shots: pl.DataFrame, rule: str, row: int, cells: dict) -> None:
    err = _err(sc.with_values(valid_shots, row, **cells))
    failure = _rule(err, rule)
    assert row in failure.rows
    expected_columns = dict(sc.SHOT_RULES)[rule]
    assert failure.columns == expected_columns
    message = str(err)
    assert f"rule '{rule}' ({', '.join(expected_columns)}): " in message
    assert f"shot_id={sc.SHOT_IDS[row]!r}" in message
    assert "Error(" not in message


def test_rule_line_format(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "is_goal", 0, False))
    rule = "is_goal_matches_outcome"
    assert _pairs(err) == {(dict(sc.SHOT_RULES)[rule], rule)}
    failure = err.failures[0]
    assert failure.rows == (0,) and failure.count == 1
    message = str(err)
    assert "rule 'is_goal_matches_outcome' (shot_id, is_goal, outcome): " in message
    assert f"failed for shot_id={sc.SB_SHOT_1!r}, is_goal=False, outcome='goal' (rows 0)" in message


def test_rule_failure_lists_every_failing_row(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.with_columns(pl.col("distance_m") + 1.0)
    failure = _rule(_err(df), "distance_matches_pitch")
    assert failure.rows == (0, 1, 2, 3, 4)
    assert failure.count == 5


# --- Rule semantics: the cases that pass --------------------------------------------------


def test_own_goal_frame_distance(valid_shots: pl.DataFrame) -> None:
    # Own-goal row is in the frame where the entered goal is at x = 105 (spec §4.1).
    og = valid_shots.row(3, named=True)
    assert og["is_own_goal"] and og["is_goal"] and not og["on_target"]
    assert og["distance_m"] == float(pitch.shot_distance(og["x"], og["y"]))
    schema.validate_shots(valid_shots)
    # Unflipped coordinates with the stored distance no longer agree with the pitch.
    fx, fy = pitch.flip(og["x"], og["y"])
    df = sc.with_values(valid_shots, 3, x=float(fx), y=float(fy))
    _rule(_err(df), "distance_matches_pitch")


def test_distance_within_tolerance_passes(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.with_columns(pl.col("distance_m") + 5e-10, pl.col("angle_rad") - 5e-10)
    schema.validate_shots(df)


def test_distance_just_outside_tolerance_fails(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.with_columns(pl.col("distance_m") + 1e-7)
    _rule(_err(df), "distance_matches_pitch")


def test_angle_offset_fails(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.with_columns(pl.col("angle_rad") + 1e-7)
    _rule(_err(df), "angle_matches_pitch")


def test_clock_within_one_second_passes(valid_shots: pl.DataFrame) -> None:
    schema.validate_shots(sc.with_value(valid_shots, "seconds_in_period", 0, 1395.99))
    schema.validate_shots(sc.with_value(valid_shots, "seconds_in_period", 0, 1394.01))


def test_assisted_shot_may_lack_assist_event(valid_shots: pl.DataFrame) -> None:
    # Only assist_type 'none' forbids an id; 'cross' with a null id is allowed.
    schema.validate_shots(sc.with_value(valid_shots, "assist_event_id", 0, None))


def test_null_assist_type_with_assist_event_passes(valid_shots: pl.DataFrame) -> None:
    schema.validate_shots(sc.with_value(valid_shots, "assist_type", 0, None))


def test_blocked_with_end_location_passes(valid_shots: pl.DataFrame) -> None:
    blocked = valid_shots.row(2, named=True)
    assert blocked["outcome"] == "blocked" and blocked["end_x"] is not None
    schema.validate_shots(valid_shots)


def test_own_goal_needs_no_penalty_or_on_target(valid_shots: pl.DataFrame) -> None:
    # A penalty can be saved and still on target; a shoot-out penalty is not a goal.
    schema.validate_shots(valid_shots)
    assert valid_shots["on_target"].to_list() == [True, True, False, False, False]


@pytest.mark.parametrize("outcome", ["saved", "post", "off_target", "blocked"])
def test_non_goal_outcomes_validate_on_open_play_row(
    valid_shots: pl.DataFrame, outcome: str
) -> None:
    on_target = outcome == "saved"
    cells = {"outcome": outcome, "is_goal": False, "on_target": on_target}
    if outcome == "blocked":
        cells |= {"goal_y": None, "goal_z": None}
    schema.validate_shots(sc.with_values(valid_shots, 0, **cells))


# --- Totality (spec §4.0) -------------------------------------------------------------------


def test_rules_skip_out_of_domain_values(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "period", 0, 6))
    assert _pairs(err) == {(("period",), "allowed")}
    assert "rule '" not in str(err)
    assert "Error(" not in str(err)


def test_nan_location_reports_column_only(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "x", 0, float("nan")))
    assert {f.columns for f in err.failures} == {("x",)}
    assert {f.check for f in err.failures} <= {"nullable", "range", "finite"}
    assert "finite" in {f.check for f in err.failures}
    assert "distance_matches_pitch" not in str(err)
    assert "angle_matches_pitch" not in str(err)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinite_location_reports_column_only(valid_shots: pl.DataFrame, value: float) -> None:
    err = _err(sc.with_value(valid_shots, "y", 0, value))
    assert {f.columns for f in err.failures} == {("y",)}


def test_null_period_reports_nullable_only(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "period", 0, None))
    assert _pairs(err) == {(("period",), "nullable")}


@pytest.mark.parametrize(
    "column", ["outcome", "is_goal", "is_own_goal", "on_target", "situation", "is_penalty"]
)
def test_null_rule_input_reports_nullable_only(valid_shots: pl.DataFrame, column: str) -> None:
    err = _err(sc.with_value(valid_shots, column, 1, None))
    assert _pairs(err) == {((column,), "nullable")}


def test_null_provider_skips_prefix_rule(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "provider", 0, None))
    assert _pairs(err) == {(("provider",), "nullable")}


def test_clock_arithmetic_does_not_wrap(valid_shots: pl.DataFrame) -> None:
    # Int16 minute 1000: (1000 - 0) * 60 wraps to -5536 in Int16 (QA r2 m2). The row is
    # self-consistent, so only the column ranges may fail, never clock_consistent.
    df = sc.with_values(valid_shots, 0, minute=1000, second=0, seconds_in_period=60000.0)
    assert _pairs(_err(df)) == {(("minute",), "range"), (("seconds_in_period",), "range")}


def test_clock_arithmetic_at_int16_limits(valid_shots: pl.DataFrame) -> None:
    df = sc.with_values(valid_shots, 1, minute=32767, second=59, seconds_in_period=1963319.0)
    # (32767 - 45) * 60 + 59 = 1963379 != 1963319: inconsistent, judged on true arithmetic
    failure = _rule(_err(df), "clock_consistent")
    assert failure.rows == (1,)


def test_negative_minute_in_period_two(valid_shots: pl.DataFrame) -> None:
    df = sc.with_values(valid_shots, 1, minute=-32768)
    err = _err(df)
    assert (("minute",), "range") in _pairs(err)
    assert "Error(" not in str(err)


def test_finite_out_of_range_location_is_still_judged(valid_shots: pl.DataFrame) -> None:
    # Spec §4.0 point 3: x = 105.5 is finite, so the pitch rules judge it too.
    err = _err(sc.with_value(valid_shots, "x", 0, 105.5))
    checks = {f.check for f in err.failures}
    assert {"range", "distance_matches_pitch", "angle_matches_pitch"} <= checks


ALL_RULES = [("shots", r) for r, _ in sc.SHOT_RULES] + [
    ("freeze_frames", r) for r, _ in sc.FREEZE_FRAME_RULES
]


def _rule_by_name(table: str, name: str) -> schema.FrameRule:
    rules = schema.SHOT_RULES if table == "shots" else schema.FREEZE_FRAME_RULES
    (rule,) = [r for r in rules if r.name == name]
    return rule


def _assert_total_mask(mask: object, height: int) -> None:
    assert isinstance(mask, pl.Series)
    assert mask.dtype == pl.Boolean
    assert mask.len() == height
    assert mask.null_count() == 0


@pytest.mark.parametrize(("table", "name"), ALL_RULES, ids=[n for _, n in ALL_RULES])
def test_rule_checks_are_total(table: str, name: str) -> None:
    rule = _rule_by_name(table, name)
    columns = sc.SHOT_COLUMNS if table == "shots" else sc.FREEZE_FRAME_COLUMNS
    hostile = sc.hostile_frame(columns, n=12)
    _assert_total_mask(rule.check(hostile), hostile.height)
    empty = schema.empty_shots() if table == "shots" else schema.empty_freeze_frames()
    _assert_total_mask(rule.check(empty), 0)
    all_null = pl.DataFrame([pl.Series(n, [None] * 3, dtype=d) for n, d, _ in columns])
    assert all(_rule_by_name(table, name).check(all_null).to_list())


@pytest.mark.parametrize(("table", "name"), ALL_RULES, ids=[n for _, n in ALL_RULES])
def test_rule_checks_pass_on_valid_fixture(
    valid_shots: pl.DataFrame, valid_freeze_frames: pl.DataFrame, table: str, name: str
) -> None:
    df = valid_shots if table == "shots" else valid_freeze_frames
    assert _rule_by_name(table, name).check(df).all()


def test_rule_check_does_not_depend_on_column_order(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "is_goal", 0, False)
    reversed_df = df.select(list(reversed(df.columns)))
    for rule in schema.SHOT_RULES:
        assert rule.check(df).to_list() == rule.check(reversed_df).to_list(), rule.name


def test_hostile_frame_validate_raises_schema_error_only() -> None:
    hostile = sc.hostile_frame(sc.SHOT_COLUMNS, n=12)
    with pytest.raises(schema.SchemaValidationError) as info:
        schema.validate_shots(hostile)
    assert all(f.columns for f in info.value.failures)
    assert "Error(" not in str(info.value)
