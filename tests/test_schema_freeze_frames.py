"""Long-format freeze-frame table: validation, rules, cross-table checks (C3; spec §2.2, §4.2)."""

from __future__ import annotations

import polars as pl
import pytest
import schema_cases as sc
from polars.testing import assert_frame_equal

from football_core import schema


def _err(df: pl.DataFrame, shots: pl.DataFrame | None = None) -> schema.SchemaValidationError:
    with pytest.raises(schema.SchemaValidationError) as info:
        schema.validate_freeze_frames(df, shots=shots)
    return info.value


def _pairs(err: schema.SchemaValidationError) -> set[tuple[tuple[str, ...], str]]:
    return {(f.columns, f.check) for f in err.failures}


def _one(err: schema.SchemaValidationError, column: str, check: str) -> schema.Failure:
    found = [f for f in err.failures if f.columns == (column,) and f.check == check]
    assert len(found) == 1, (column, check, _pairs(err))
    return found[0]


def _rule(err: schema.SchemaValidationError, name: str) -> schema.Failure:
    found = [f for f in err.failures if f.check == name]
    assert len(found) == 1, (name, _pairs(err))
    return found[0]


def _row(**cells: object) -> dict[str, object]:
    base = {
        "shot_id": sc.SB_SHOT_1,
        "player_id": "sb:90199",
        "is_teammate": False,
        "is_keeper": False,
        "x": 99.0,
        "y": 33.0,
    }
    return base | cells


# --- Valid frames ---------------------------------------------------------------------------


def test_synthetic_freeze_frames_validate(valid_freeze_frames: pl.DataFrame) -> None:
    out = schema.validate_freeze_frames(valid_freeze_frames)
    assert tuple(out.columns) == sc.FREEZE_FRAME_NAMES
    assert out.schema == schema.FREEZE_FRAME_SCHEMA
    assert_frame_equal(out, valid_freeze_frames)


def test_freeze_frames_validate_against_shots(
    valid_freeze_frames: pl.DataFrame, valid_shots: pl.DataFrame
) -> None:
    out = schema.validate_freeze_frames(valid_freeze_frames, shots=valid_shots)
    assert_frame_equal(out, valid_freeze_frames)


def test_empty_freeze_frames_validate(valid_shots: pl.DataFrame) -> None:
    empty = schema.empty_freeze_frames()
    assert empty.height == 0
    assert empty.schema == pl.Schema(sc.FREEZE_FRAME_DTYPES)
    assert schema.validate_freeze_frames(empty).height == 0
    assert schema.validate_freeze_frames(empty, shots=valid_shots).height == 0
    assert schema.validate_freeze_frames(empty, shots=schema.empty_shots()).height == 0


def test_freeze_frames_against_empty_shots_fail(valid_freeze_frames: pl.DataFrame) -> None:
    failure = _rule(_err(valid_freeze_frames, shots=schema.empty_shots()), "shot_exists")
    assert failure.count == 4


def test_shuffled_columns_come_back_canonical(valid_freeze_frames: pl.DataFrame) -> None:
    shuffled = valid_freeze_frames.select(list(reversed(valid_freeze_frames.columns)))
    out = schema.validate_freeze_frames(shuffled)
    assert tuple(out.columns) == sc.FREEZE_FRAME_NAMES


def test_null_player_ids_do_not_trip_unique(valid_freeze_frames: pl.DataFrame) -> None:
    extra = sc.freeze_frames_frame([_row(player_id=None, is_teammate=True, x=90.0, y=20.0)])
    df = pl.concat([valid_freeze_frames, extra])
    assert df["player_id"].null_count() == 2
    schema.validate_freeze_frames(df)


def test_one_keeper_on_each_side_is_fine(valid_freeze_frames: pl.DataFrame) -> None:
    own_keeper = sc.freeze_frames_frame(
        [_row(player_id="sb:90150", is_teammate=True, is_keeper=True, x=10.0, y=34.0)]
    )
    schema.validate_freeze_frames(pl.concat([valid_freeze_frames, own_keeper]))


def test_frames_for_several_shots(valid_freeze_frames: pl.DataFrame, valid_shots) -> None:
    # The same player and a keeper per shot may appear in frames of different shots.
    other = valid_freeze_frames.with_columns(pl.lit(sc.SB_SHOT_2).alias("shot_id"))
    df = pl.concat([valid_freeze_frames, other])
    schema.validate_freeze_frames(df, shots=valid_shots)


def test_whoscored_frame_with_ws_ids(valid_shots: pl.DataFrame) -> None:
    df = sc.freeze_frames_frame([_row(shot_id=sc.WS_SHOT_3, player_id="ws:91999")])
    schema.validate_freeze_frames(df, shots=valid_shots)


@pytest.mark.parametrize(("x", "y"), [(0.0, 0.0), (105.0, 68.0), (-0.0, -0.0)])
def test_boundary_values_pass(x: float, y: float) -> None:
    schema.validate_freeze_frames(sc.freeze_frames_frame([_row(x=x, y=y)]))


def test_no_order_is_implied(valid_freeze_frames: pl.DataFrame, valid_shots) -> None:
    out = schema.validate_freeze_frames(valid_freeze_frames.reverse(), shots=valid_shots)
    assert_frame_equal(out, valid_freeze_frames.reverse())


# --- Column-level failures ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "value", "check"),
    [
        ("x", 120.0, "range"),
        ("x", -0.01, "range"),
        ("y", -3.0, "range"),
        ("y", 68.1, "range"),
        ("x", float("nan"), "finite"),
        ("y", float("inf"), "finite"),
    ],
)
def test_out_of_bounds(
    valid_freeze_frames: pl.DataFrame, column: str, value: float, check: str
) -> None:
    err = _err(sc.with_value(valid_freeze_frames, column, 1, value))
    failure = _one(err, column, check)
    assert failure.rows == (1,)
    assert err.table == "freeze_frames"
    assert str(err).startswith("freeze_frames: ")
    assert f"column '{column}'" in str(err)
    assert repr(value) in failure.values
    assert "Error(" not in str(err)


@pytest.mark.parametrize(
    ("column", "make"),
    [
        ("x", lambda df: df.with_columns(pl.col("x").cast(pl.Int64))),
        ("is_keeper", lambda df: df.with_columns(pl.col("is_keeper").cast(pl.Int64))),
        ("shot_id", lambda df: df.with_columns(pl.Series("shot_id", [1, 2, 3, 4]))),
        ("player_id", lambda df: df.with_columns(pl.col("player_id").cast(pl.Categorical))),
        ("y", lambda df: df.with_columns(pl.col("y").cast(pl.Float32))),
    ],
    ids=["x_int64", "is_keeper_int64", "shot_id_int64", "player_id_categorical", "y_float32"],
)
def test_wrong_dtype(valid_freeze_frames: pl.DataFrame, column: str, make) -> None:
    df = make(valid_freeze_frames)
    err = _err(df)
    assert _pairs(err) == {((column,), "dtype")}
    assert f"column '{column}': dtype {df.schema[column]}, expected" in str(err)


@pytest.mark.parametrize("column", ["shot_id", "is_keeper", "x", "player_id"])
def test_missing_column(valid_freeze_frames: pl.DataFrame, column: str) -> None:
    err = _err(valid_freeze_frames.drop(column))
    assert _pairs(err) == {((column,), "missing")}
    assert f"column '{column}': missing required column" in str(err)


def test_extra_column(valid_freeze_frames: pl.DataFrame) -> None:
    err = _err(valid_freeze_frames.with_columns(pl.lit("GK").alias("position")))
    assert _pairs(err) == {(("position",), "extra")}
    assert "column 'position': not in the freeze_frames schema" in str(err)


@pytest.mark.parametrize("column", ["shot_id", "is_teammate", "is_keeper", "x", "y"])
def test_null_in_required_column(valid_freeze_frames: pl.DataFrame, column: str) -> None:
    err = _err(sc.with_value(valid_freeze_frames, column, 2, None))
    assert _one(err, column, "nullable").rows == (2,)


@pytest.mark.parametrize("bad", ["opta:1", "90101", "sb:", "sb: 1"])
def test_bad_player_id_format(valid_freeze_frames: pl.DataFrame, bad: str) -> None:
    err = _err(sc.with_value(valid_freeze_frames, "player_id", 1, bad))
    assert _one(err, "player_id", "id_format").values == (repr(bad),)


def test_shot_id_is_not_unique_here(valid_freeze_frames: pl.DataFrame) -> None:
    assert valid_freeze_frames["shot_id"].n_unique() == 1
    schema.validate_freeze_frames(valid_freeze_frames)


# --- Frame rules (spec §4.2) ----------------------------------------------------------------


def test_duplicate_player(valid_freeze_frames: pl.DataFrame) -> None:
    df = sc.with_value(valid_freeze_frames, "player_id", 2, "sb:90102")
    failure = _rule(_err(df), "unique_player_per_shot")
    assert set(failure.rows) <= {1, 2} and 2 in failure.rows
    assert "rule 'unique_player_per_shot' (shot_id, player_id)" in str(_err(df))
    assert "'sb:90102'" in str(_err(df))


def test_same_player_in_two_shots_is_not_a_duplicate(valid_freeze_frames, valid_shots) -> None:
    other = sc.freeze_frames_frame([_row(shot_id=sc.SB_SHOT_2, player_id="sb:90102")])
    schema.validate_freeze_frames(pl.concat([valid_freeze_frames, other]), shots=valid_shots)


def test_two_opponent_keepers(valid_freeze_frames: pl.DataFrame) -> None:
    df = sc.with_value(valid_freeze_frames, "is_keeper", 1, True)
    failure = _rule(_err(df), "one_keeper_per_side")
    assert 1 in failure.rows
    assert set(failure.rows) <= {0, 1}


def test_two_teammate_keepers(valid_freeze_frames: pl.DataFrame) -> None:
    keepers = sc.freeze_frames_frame(
        [
            _row(player_id="sb:90150", is_teammate=True, is_keeper=True),
            _row(player_id="sb:90151", is_teammate=True, is_keeper=True),
        ]
    )
    _rule(_err(pl.concat([valid_freeze_frames, keepers])), "one_keeper_per_side")


def test_keepers_in_different_shots_are_fine(valid_freeze_frames, valid_shots) -> None:
    other = valid_freeze_frames.with_columns(pl.lit(sc.SB_SHOT_5).alias("shot_id"))
    schema.validate_freeze_frames(pl.concat([valid_freeze_frames, other]), shots=valid_shots)


def test_prefix_mismatch(valid_freeze_frames: pl.DataFrame) -> None:
    df = sc.with_value(valid_freeze_frames, "player_id", 1, "ws:90102")
    err = _err(df)
    failure = _rule(err, "ids_share_prefix")
    assert failure.rows == (1,)
    assert f"shot_id={sc.SB_SHOT_1!r}" in str(err)
    assert "'ws:90102'" in str(err)


def test_rules_report_freeze_frame_row_positions(valid_freeze_frames: pl.DataFrame) -> None:
    df = pl.concat([valid_freeze_frames, valid_freeze_frames.slice(1, 1)])
    assert _rule(_err(df), "unique_player_per_shot").rows[-1] == 4


# --- Cross-table rules (only with shots=) ---------------------------------------------------


def test_unknown_shot_id(valid_freeze_frames: pl.DataFrame, valid_shots: pl.DataFrame) -> None:
    orphan = "sb:00000000-0000-0000-0000-00000000dead"
    df = sc.with_value(valid_freeze_frames, "shot_id", 3, orphan)
    err = _err(df, shots=valid_shots)
    failure = _rule(err, "shot_exists")
    assert failure.columns == ("shot_id",)
    assert failure.rows == (3,)
    assert repr(orphan) in str(err)


def test_unknown_shot_id_is_not_checked_without_shots(valid_freeze_frames) -> None:
    orphan = "sb:00000000-0000-0000-0000-00000000dead"
    schema.validate_freeze_frames(sc.with_value(valid_freeze_frames, "shot_id", 3, orphan))


def test_shooter_in_frame(valid_freeze_frames: pl.DataFrame, valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_freeze_frames, "player_id", 1, sc.SHOOTER_1)
    err = _err(df, shots=valid_shots)
    failure = _rule(err, "shooter_not_in_frame")
    assert failure.columns == ("shot_id", "player_id")
    assert failure.rows == (1,)
    assert repr(sc.SHOOTER_1) in str(err)


def test_shooter_of_another_shot_may_appear(valid_freeze_frames, valid_shots) -> None:
    # sb:90002 shoots shot 5; in shot 1's frame he is just a teammate.
    df = sc.with_value(valid_freeze_frames, "player_id", 3, "sb:90002")
    schema.validate_freeze_frames(df, shots=valid_shots)


def test_shooter_rule_is_not_checked_without_shots(valid_freeze_frames) -> None:
    schema.validate_freeze_frames(sc.with_value(valid_freeze_frames, "player_id", 1, sc.SHOOTER_1))


def test_duplicate_shots_do_not_multiply_rows(
    valid_freeze_frames: pl.DataFrame, valid_shots: pl.DataFrame
) -> None:
    # shots is not validated here; a duplicated shot table must not multiply failures.
    doubled = pl.concat([valid_shots, valid_shots, valid_shots])
    df = sc.with_value(valid_freeze_frames, "player_id", 1, sc.SHOOTER_1)
    failure = _rule(_err(df, shots=doubled), "shooter_not_in_frame")
    assert failure.rows == (1,)
    assert failure.count == 1


def test_cross_table_rules_come_after_frame_rules(
    valid_freeze_frames: pl.DataFrame, valid_shots: pl.DataFrame
) -> None:
    df = sc.with_value(valid_freeze_frames, "player_id", 1, sc.SHOOTER_1)
    df = sc.with_value(df, "is_keeper", 2, True)
    checks = [f.check for f in _err(df, shots=valid_shots).failures]
    assert checks == ["one_keeper_per_side", "shooter_not_in_frame"]


def test_shots_argument_is_not_validated(
    valid_freeze_frames: pl.DataFrame, valid_shots: pl.DataFrame
) -> None:
    bad_shots = sc.with_value(valid_shots, "x", 2, 500.0)
    schema.validate_freeze_frames(valid_freeze_frames, shots=bad_shots)


def test_null_player_never_matches_shooter(valid_shots: pl.DataFrame) -> None:
    shots = sc.with_value(valid_shots, "player_id", 0, None)  # not validated here
    df = sc.freeze_frames_frame([_row(player_id=None)])
    schema.validate_freeze_frames(df, shots=shots)


def test_column_checks_and_rules_reported_together(
    valid_freeze_frames: pl.DataFrame, valid_shots: pl.DataFrame
) -> None:
    df = sc.with_value(valid_freeze_frames, "x", 0, 120.0)
    df = sc.with_value(df, "player_id", 1, sc.SHOOTER_1)
    pairs = _pairs(_err(df, shots=valid_shots))
    assert (("x",), "range") in pairs
    assert (("shot_id", "player_id"), "shooter_not_in_frame") in pairs
