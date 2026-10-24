"""Valid synthetic frames validate and come back canonical (C3; spec §1.1, §2.0, §7)."""

from __future__ import annotations

import random

import polars as pl
import pytest
import schema_cases as sc
from polars.testing import assert_frame_equal

from football_core import schema


def test_synthetic_shots_validate(valid_shots: pl.DataFrame) -> None:
    out = schema.validate_shots(valid_shots)
    assert isinstance(out, pl.DataFrame)
    assert_frame_equal(out, valid_shots)


def test_returns_canonical_column_order(valid_shots: pl.DataFrame) -> None:
    out = schema.validate_shots(valid_shots)
    assert tuple(out.columns) == sc.SHOT_NAMES
    assert out.schema == schema.SHOT_SCHEMA


def test_shuffled_input_columns_validate(valid_shots: pl.DataFrame) -> None:
    names = list(valid_shots.columns)
    random.Random(3).shuffle(names)
    shuffled = valid_shots.select(names)
    assert tuple(shuffled.columns) != sc.SHOT_NAMES
    out = schema.validate_shots(shuffled)
    assert tuple(out.columns) == sc.SHOT_NAMES
    assert_frame_equal(out, valid_shots)


def test_empty_shots_validate() -> None:
    empty = schema.empty_shots()
    assert empty.height == 0
    assert empty.schema == schema.SHOT_SCHEMA
    assert empty.schema == pl.Schema(sc.SHOT_DTYPES)
    assert schema.validate_shots(empty).height == 0


def test_empty_constructors_return_fresh_frames() -> None:
    a, b = schema.empty_shots(), schema.empty_shots()
    a = a.with_columns(pl.lit(1).alias("tmp"))
    assert "tmp" not in b.columns
    assert schema.empty_shots().columns == list(sc.SHOT_NAMES)


def test_empty_frame_with_no_rows_but_wrong_dtype_fails() -> None:
    bad = schema.empty_shots().with_columns(pl.col("x").cast(pl.Int64))
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_shots(bad)


def test_provider_only_fields_may_be_null(valid_shots: pl.DataFrame) -> None:
    # Guard against fixture drift: the fixture really carries the §2.0 nulls.
    sb = valid_shots.filter(pl.col("provider") == "statsbomb")
    ws = valid_shots.filter(pl.col("provider") == "whoscored")
    assert sb["big_chance"].is_null().all()
    assert ws["follows_dribble"].is_null().all()
    assert ws["deflected"].is_null().all()
    assert ws["gk_x"].is_null().all() and ws["gk_y"].is_null().all()
    assert ws["provider_xg"].is_null().all() and ws["provider_psxg"].is_null().all()
    assert ws["visible_area"].is_null().all()
    assert ws["technique"].is_null().all()
    assert sb["provider_psxg"].is_null().all()
    schema.validate_shots(valid_shots)


def test_saved_shot_keeps_goal_mouth_estimate(valid_shots: pl.DataFrame) -> None:
    # Spec §2.0: saved shots keep the provider's goal-plane estimate; never nulled.
    saved = valid_shots.filter(pl.col("outcome") == "saved")
    assert saved["goal_y"].is_not_null().all() and saved["goal_z"].is_not_null().all()
    schema.validate_shots(valid_shots)


def test_all_null_nullable_columns_validate(valid_shots: pl.DataFrame) -> None:
    nullable = [name for name, _, null in sc.SHOT_COLUMNS if null]
    df = valid_shots.with_columns(
        pl.lit(None, dtype=sc.SHOT_DTYPES[name]).alias(name) for name in nullable
    )
    assert df.schema == valid_shots.schema
    schema.validate_shots(df)


def test_negative_zero_coordinates_validate(valid_shots: pl.DataFrame) -> None:
    df = sc.with_location(valid_shots, 0, 94.5, -0.0)
    df = sc.with_values(df, 0, end_y=-0.0, gk_y=-0.0, goal_z=-0.0, end_z=-0.0)
    schema.validate_shots(df)


@pytest.mark.parametrize(
    ("x", "y"),
    [(0.0, 0.0), (105.0, 68.0), (0.0, 68.0), (105.0, 0.0), (105.0, 34.0), (105.0, 30.34)],
    ids=["origin", "far_corner", "x0_y68", "x105_y0", "goal_centre_angle_pi", "on_post"],
)
def test_boundary_locations_pass(valid_shots: pl.DataFrame, x: float, y: float) -> None:
    schema.validate_shots(sc.with_location(valid_shots, 0, x, y))


@pytest.mark.parametrize(
    ("row", "cells"),
    [
        (0, {"end_x": 0.0, "end_y": 0.0, "gk_x": 105.0, "gk_y": 68.0}),
        (0, {"end_x": 105.0, "end_y": 68.0, "gk_x": 0.0, "gk_y": 0.0, "end_z": 0.0}),
        # Off-target shoot-out row (non-blocked, period 5 skips clock rules): a goal-plane
        # point far outside the posts and on the ground is valid (QA r3 n1).
        (4, {"goal_y": 25.0, "goal_z": 0.0}),
        (4, {"goal_y": 0.0, "goal_z": 7.5}),  # no upper bound on goal_z in v1
        (4, {"goal_y": 68.0}),
        (0, {"provider_xg": 0.0, "provider_psxg": 1.0}),
        (0, {"provider_xg": 1.0, "provider_psxg": 0.0}),
        (0, {"second": 0, "seconds_in_period": 1380.0}),
        (0, {"second": 59, "seconds_in_period": 1439.0}),
        (4, {"minute": 150}),
        (4, {"seconds_in_period": 0.0}),
        (4, {"seconds_in_period": 4500.0}),
    ],
)
def test_boundary_values_pass(valid_shots: pl.DataFrame, row: int, cells: dict) -> None:
    schema.validate_shots(sc.with_values(valid_shots, row, **cells))


@pytest.mark.parametrize("period", [1, 2, 3, 4])
def test_every_period_validates(valid_shots: pl.DataFrame, period: int) -> None:
    start = {1: 0, 2: 45, 3: 90, 4: 105}[period]
    df = sc.with_values(
        valid_shots, 0, period=period, minute=start, second=0, seconds_in_period=0.4
    )
    schema.validate_shots(df)


def test_extra_time_minute_can_exceed_regulation(valid_shots: pl.DataFrame) -> None:
    # Stoppage time: minute 47 in period 1 is still period 1 (rule is minute >= start only).
    df = sc.with_values(valid_shots, 0, minute=47, second=30, seconds_in_period=2850.0)
    schema.validate_shots(df)


def test_shootout_skips_clock_rules(valid_shots: pl.DataFrame) -> None:
    df = sc.with_values(valid_shots, 4, minute=0, second=0, seconds_in_period=4000.0)
    schema.validate_shots(df)


def test_validate_returns_new_frame_not_alias(valid_shots: pl.DataFrame) -> None:
    out = schema.validate_shots(valid_shots)
    out = out.with_columns(pl.lit(0.0).alias("x"))
    assert valid_shots["x"][0] == 94.5


def test_validate_does_not_mutate_input_shots(valid_shots: pl.DataFrame) -> None:
    names = list(reversed(valid_shots.columns))
    shuffled = valid_shots.select(names)
    before = shuffled.clone()
    schema.validate_shots(shuffled)
    assert_frame_equal(shuffled, before)
    assert shuffled.columns == names

    bad = sc.with_value(shuffled, "x", 0, 200.0)
    bad_before = bad.clone()
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_shots(bad)
    assert_frame_equal(bad, bad_before)
    assert bad.columns == names


def test_validate_does_not_mutate_input_freeze_frames(
    valid_freeze_frames: pl.DataFrame, valid_shots: pl.DataFrame
) -> None:
    names = ["y", "x", "is_keeper", "is_teammate", "player_id", "shot_id"]
    shuffled = valid_freeze_frames.select(names)
    before, shots_before = shuffled.clone(), valid_shots.clone()
    out = schema.validate_freeze_frames(shuffled, shots=valid_shots)
    assert tuple(out.columns) == sc.FREEZE_FRAME_NAMES
    assert_frame_equal(shuffled, before)
    assert_frame_equal(valid_shots, shots_before)
    assert shuffled.columns == names

    bad = sc.with_value(shuffled, "x", 0, 200.0)
    bad_before = bad.clone()
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_freeze_frames(bad, shots=valid_shots)
    assert_frame_equal(bad, bad_before)


def test_parquet_round_trip_validates(
    tmp_path, valid_shots: pl.DataFrame, valid_freeze_frames: pl.DataFrame
) -> None:
    for name, df, validate in [
        ("shots", valid_shots, schema.validate_shots),
        ("freeze_frames", valid_freeze_frames, schema.validate_freeze_frames),
    ]:
        path = tmp_path / f"{name}.parquet"
        df.write_parquet(path, compression="zstd")
        back = pl.read_parquet(path)
        assert back.schema == df.schema
        assert_frame_equal(validate(back), df)


def test_parquet_writes_are_byte_identical(tmp_path, valid_shots: pl.DataFrame) -> None:
    # Spec §0.7: C6 hashes snapshots, so two writes of the same frame must match.
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    schema.validate_shots(valid_shots).write_parquet(a, compression="zstd")
    schema.validate_shots(valid_shots).write_parquet(b, compression="zstd")
    assert a.read_bytes() == b.read_bytes()


def test_empty_frames_round_trip(tmp_path) -> None:
    path = tmp_path / "empty.parquet"
    schema.empty_shots().write_parquet(path, compression="zstd")
    assert schema.validate_shots(pl.read_parquet(path)).schema == schema.SHOT_SCHEMA
