"""``cast_to_schema``: the only place a frame is cast, along safe paths only (C3; spec §1.2)."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
import schema_cases as sc
from polars.testing import assert_frame_equal

from football_core import schema


def _cast_err(df: pl.DataFrame, table: str = "shots") -> schema.SchemaValidationError:
    with pytest.raises(schema.SchemaValidationError) as info:
        schema.cast_to_schema(df, table)
    return info.value


def _one(err: schema.SchemaValidationError, column: str, check: str) -> schema.Failure:
    found = [f for f in err.failures if f.columns == (column,) and f.check == check]
    assert len(found) == 1, (column, check, [(f.columns, f.check) for f in err.failures])
    return found[0]


def _replace(name: str, values: list, dtype: pl.DataType):
    def mutate(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.Series(name, values, dtype=dtype, strict=True))

    return mutate


def _cast(name: str, dtype: pl.DataType):
    def mutate(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col(name).cast(dtype))

    return mutate


def _to_string_then(name: str, dtype: pl.DataType):
    def mutate(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col(name).cast(pl.String).cast(dtype))

    return mutate


DATES = [d.isoformat() for d in sc.shots_frame()["match_date"].to_list()]
NESTED = [None if v is None else [list(p) for p in v] for v in sc.shots_frame()["visible_area"]]
_SQUARE_INT = [[80, 10], [105, 10], [105, 58], [80, 58]]

# case id -> (mutation, column, None for accepted | (check, named value substring or None))
CAST_CASES = {
    # Enum
    "string_to_enum": (_cast("body_part", pl.String()), "body_part", None),
    "string_to_enum_invalid": (
        _replace("body_part", ["right_foot", "left_foot", "head", "chest", None], pl.String()),
        "body_part",
        ("allowed", "'chest'"),
    ),
    "categorical_to_enum": (_to_string_then("body_part", pl.Categorical()), "body_part", None),
    "foreign_enum_to_enum": (
        _to_string_then("situation", pl.Enum(["penalty", *sc.SITUATIONS[:-2], "kick_off", "x"])),
        "situation",
        None,
    ),
    "foreign_enum_invalid_value": (
        _replace(
            "outcome",
            ["goal", "saved", "blocked", "goal", "wide"],
            pl.Enum([*sc.OUTCOMES, "wide"]),
        ),
        "outcome",
        ("allowed", "'wide'"),
    ),
    "enum_from_int_refused": (
        _replace("provider", [0, 0, 1, 1, 0], pl.Int64()),
        "provider",
        ("cast", None),
    ),
    # Float64
    "int_to_float": (_replace("gk_x", [104, 105, None, None, 105], pl.Int32()), "gk_x", None),
    "float32_to_float": (
        _replace("end_z", [0.5, 0.5, None, None, 1.0], pl.Float32()),
        "end_z",
        None,
    ),
    "uint_to_float": (_replace("gk_y", [34, 34, None, None, 34], pl.UInt8()), "gk_y", None),
    "string_to_float_refused": (
        _replace("provider_xg", ["0.31", "0.76", None, None, "0.78"], pl.String()),
        "provider_xg",
        ("cast", None),
    ),
    # Int8 / Int16
    "int64_to_int8": (_cast("period", pl.Int64()), "period", None),
    "uint32_to_int16": (_cast("minute", pl.UInt32()), "minute", None),
    "int_overflow": (_replace("period", [1, 2, 300, 2, 5], pl.Int64()), "period", ("cast", "300")),
    "int_underflow": (
        _replace("second", [15, 3, -200, 0, 10], pl.Int64()),
        "second",
        ("cast", "-200"),
    ),
    "int16_overflow": (
        _replace("minute", [23, 67, 40000, 78, 121], pl.Int64()),
        "minute",
        ("cast", "40000"),
    ),
    "float_to_int_refused": (_cast("period", pl.Float64()), "period", ("cast", None)),
    "fractional_float_to_int_refused": (
        _replace("second", [15.0, 3.0, 30.5, 0.0, 10.0], pl.Float64()),
        "second",
        ("cast", None),
    ),
    # Boolean
    "bool_from_int_refused": (_cast("is_goal", pl.Int64()), "is_goal", ("cast", None)),
    "bool_from_string_refused": (_cast("is_home", pl.String()), "is_home", ("cast", None)),
    # String
    "string_from_int_refused": (
        _replace("shot_id", [1, 2, 3, 4, 5], pl.Int64()),
        "shot_id",
        ("cast", None),
    ),
    "string_from_categorical_refused": (
        _cast("raw_type", pl.Categorical()),
        "raw_type",
        ("cast", None),
    ),
    # Date
    "date_iso": (_replace("match_date", DATES, pl.String()), "match_date", None),
    "date_bad_string": (
        _replace("match_date", [*DATES[:2], "2024-13-01", *DATES[3:]], pl.String()),
        "match_date",
        ("cast", "'2024-13-01'"),
    ),
    "date_not_a_date": (
        _replace("match_date", [*DATES[:4], "yesterday"], pl.String()),
        "match_date",
        ("cast", "'yesterday'"),
    ),
    "date_from_datetime_refused": (
        _cast("match_date", pl.Datetime()),
        "match_date",
        ("cast", None),
    ),
    "date_from_int_refused": (_cast("match_date", pl.Int32()), "match_date", ("cast", None)),
    # List(Array(Float64, 2))
    "nested_lists_to_polygon": (
        _replace("visible_area", NESTED, pl.List(pl.List(pl.Float64()))),
        "visible_area",
        None,
    ),
    "nested_int_lists_to_polygon": (
        _replace(
            "visible_area", [_SQUARE_INT, None, None, None, None], pl.List(pl.List(pl.Int64()))
        ),
        "visible_area",
        None,
    ),
    "array_int_to_polygon": (
        _replace(
            "visible_area", [_SQUARE_INT, None, None, None, None], pl.List(pl.Array(pl.Int64(), 2))
        ),
        "visible_area",
        None,
    ),
    "vertex_length_3": (
        _replace(
            "visible_area",
            [[[80.0, 10.0], [1.0, 2.0, 3.0], [105.0, 58.0]], None, None, None, None],
            pl.List(pl.List(pl.Float64())),
        ),
        "visible_area",
        ("cast", "1.0, 2.0, 3.0"),
    ),
    "vertex_length_1": (
        _replace(
            "visible_area",
            [None, None, [[80.0, 10.0], [7.0], [105.0, 58.0]], None, None],
            pl.List(pl.List(pl.Float64())),
        ),
        "visible_area",
        ("cast", "7.0"),
    ),
    "flat_list_refused": (
        _replace("visible_area", [[80.0, 10.0, 105.0, 10.0]] * 5, pl.List(pl.Float64())),
        "visible_area",
        ("cast", None),
    ),
    "polygon_from_string_refused": (
        _replace("visible_area", ["[[80, 10]]"] * 5, pl.String()),
        "visible_area",
        ("cast", None),
    ),
    # Null dtype (all-null column)
    "null_to_enum": (
        lambda df: df.with_columns(pl.lit(None).alias("body_part")),
        "body_part",
        None,
    ),
    "null_to_polygon": (
        lambda df: df.with_columns(pl.lit(None).alias("visible_area")),
        "visible_area",
        None,
    ),
    "null_to_boolean": (
        lambda df: df.with_columns(pl.lit(None).alias("deflected")),
        "deflected",
        None,
    ),
    "null_to_float": (
        lambda df: df.with_columns(pl.lit(None).alias("provider_xg")),
        "provider_xg",
        None,
    ),
}


@pytest.mark.parametrize("case", list(CAST_CASES))
def test_cast_paths(valid_shots: pl.DataFrame, case: str) -> None:
    mutate, column, refused = CAST_CASES[case]
    df = mutate(valid_shots)
    if refused is None:
        out = schema.cast_to_schema(df)
        assert out.schema[column] == sc.SHOT_DTYPES[column]
        assert out.schema == schema.SHOT_SCHEMA
        schema.validate_shots(out)
    else:
        check, named = refused
        err = _cast_err(df)
        failure = _one(err, column, check)
        message = str(err)
        assert f"column '{column}'" in message
        if check == "cast":
            assert f"cannot cast {df.schema[column]} to {sc.SHOT_DTYPES[column]}" in message
        if named is not None:
            assert named in message
        if check == "allowed":
            assert named in failure.values
        assert "Error(" not in message


def test_cast_strings_to_enums_validates(valid_shots: pl.DataFrame) -> None:
    enums = [n for n, d, _ in sc.SHOT_COLUMNS if isinstance(d, pl.Enum)]
    df = valid_shots.with_columns(pl.col(enums).cast(pl.String))
    out = schema.cast_to_schema(df)
    assert_frame_equal(schema.validate_shots(out), valid_shots)


def test_cast_invalid_enum_names_value(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "body_part", 1, "chest", dtype=pl.String())
    failure = _one(_cast_err(df), "body_part", "allowed")
    assert failure.values == ("'chest'",)
    assert failure.rows == (1,)


def test_cast_rejects_float_to_int(valid_shots: pl.DataFrame) -> None:
    # polars' strict cast truncates 1.5 -> 1 silently; cast_to_schema refuses floats.
    df = valid_shots.with_columns(pl.Series("period", [1.5, 2.0, 1.0, 2.0, 5.0]))
    _one(_cast_err(df), "period", "cast")


def test_cast_drops_extra_and_reports_missing(valid_shots: pl.DataFrame) -> None:
    extra = valid_shots.with_columns(pl.lit("x").alias("note"), pl.lit(1).alias("eventId"))
    out = schema.cast_to_schema(extra)
    assert tuple(out.columns) == sc.SHOT_NAMES
    err = _cast_err(valid_shots.drop("y", "raw_type"))
    missing = {f.columns[0] for f in err.failures if f.check == "missing"}
    assert missing == {"y", "raw_type"}
    assert "column 'y': missing required column" in str(err)


def test_cast_returns_canonical_order(valid_shots: pl.DataFrame) -> None:
    shuffled = valid_shots.select(list(reversed(valid_shots.columns)))
    assert tuple(schema.cast_to_schema(shuffled).columns) == sc.SHOT_NAMES


def test_cast_is_identity_on_canonical_frame(valid_shots: pl.DataFrame) -> None:
    assert_frame_equal(schema.cast_to_schema(valid_shots), valid_shots)


def test_cast_iso_date_strings(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.with_columns(pl.col("match_date").cast(pl.String))
    out = schema.cast_to_schema(df)
    assert out["match_date"].to_list() == [
        dt.date(2024, 3, 2),
        dt.date(2024, 3, 2),
        dt.date(2024, 3, 9),
        dt.date(2024, 3, 9),
        dt.date(2024, 3, 16),
    ]


def test_cast_date_input_null_is_not_a_cast_failure(valid_shots: pl.DataFrame) -> None:
    df = _replace("match_date", [*DATES[:4], None], pl.String())(valid_shots)
    out = schema.cast_to_schema(df)
    assert out["match_date"].null_count() == 1
    with pytest.raises(schema.SchemaValidationError) as info:
        schema.validate_shots(out)
    assert {(f.columns, f.check) for f in info.value.failures} == {(("match_date",), "nullable")}


def test_cast_nested_lists_to_polygon(valid_shots: pl.DataFrame) -> None:
    df = _replace("visible_area", NESTED, pl.List(pl.List(pl.Float64())))(valid_shots)
    out = schema.cast_to_schema(df)
    assert out.schema["visible_area"] == sc.POLYGON
    assert_frame_equal(out, valid_shots)


def test_cast_does_not_check_ranges_or_rules(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "x", 0, 500.0).with_columns(pl.col("is_goal").not_())
    out = schema.cast_to_schema(df)
    assert out["x"][0] == 500.0


def test_cast_reports_every_bad_column(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.with_columns(
        pl.col("is_goal").cast(pl.Int64), pl.col("period").cast(pl.Float64)
    )
    df = sc.with_value(df, "technique", 0, "scissor", dtype=pl.String())
    pairs = {(f.columns, f.check) for f in _cast_err(df).failures}
    assert pairs == {
        (("is_goal",), "cast"),
        (("period",), "cast"),
        (("technique",), "allowed"),
    }


def test_cast_does_not_mutate_input(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.with_columns(pl.col("body_part").cast(pl.String)).select(
        list(reversed(valid_shots.columns))
    )
    before = df.clone()
    schema.cast_to_schema(df)
    assert_frame_equal(df, before)


def test_cast_freeze_frames(valid_freeze_frames: pl.DataFrame) -> None:
    df = valid_freeze_frames.with_columns(
        pl.col("x").cast(pl.Int64), pl.lit(None).alias("player_id"), pl.lit("z").alias("extra")
    )
    out = schema.cast_to_schema(df, "freeze_frames")
    assert out.schema == schema.FREEZE_FRAME_SCHEMA
    schema.validate_freeze_frames(out)


def test_cast_freeze_frames_refuses_int_booleans(valid_freeze_frames: pl.DataFrame) -> None:
    df = valid_freeze_frames.with_columns(pl.col("is_keeper").cast(pl.Int8))
    _one(_cast_err(df, "freeze_frames"), "is_keeper", "cast")


def test_cast_unknown_table(valid_shots: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="passes"):
        schema.cast_to_schema(valid_shots, "passes")  # type: ignore[arg-type]
