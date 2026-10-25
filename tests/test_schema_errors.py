"""Invalid shot frames fail with messages that name the column and the bad value (C3).

Spec §3 (two passes, message format) and §7 (acceptance clauses). Tests assert substrings and
``Failure`` fields, not whole messages.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandera.polars as pa
import polars as pl
import pytest
import schema_cases as sc
from pandera.config import ValidationDepth, config_context

from football_core import schema

TESTS_DIR = Path(__file__).parent


def _err(df: pl.DataFrame) -> schema.SchemaValidationError:
    with pytest.raises(schema.SchemaValidationError) as info:
        schema.validate_shots(df)
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


# --- The error type -------------------------------------------------------------------------


def test_error_is_value_error(valid_shots: pl.DataFrame) -> None:
    assert issubclass(schema.SchemaValidationError, ValueError)
    with pytest.raises(ValueError):
        schema.validate_shots(sc.with_value(valid_shots, "x", 0, 200.0))


def test_error_carries_table_and_failures(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "provider_xg", 0, 1.2))
    assert err.table == "shots"
    assert isinstance(err.failures, tuple) and err.failures
    for failure in err.failures:
        assert isinstance(failure, schema.Failure)
        assert failure.table == "shots"
        assert failure.columns, failure
        assert isinstance(failure.values, tuple)
        assert all(isinstance(v, str) for v in failure.values)
        assert isinstance(failure.rows, tuple)
        assert failure.count >= 1
        assert len(failure.values) <= schema.MAX_EXAMPLES
        assert len(failure.rows) <= schema.MAX_EXAMPLES


def test_message_header_and_one_line_per_failure(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "provider_xg", 0, 1.2))
    lines = str(err).splitlines()
    n = len(err.failures)
    assert lines[0] == f"shots: {n} schema problem(s) (schema 1.1)"
    assert len(lines) == n + 1
    assert all(line.startswith("  - ") for line in lines[1:])


def test_failure_is_frozen(valid_shots: pl.DataFrame) -> None:
    failure = _err(sc.with_value(valid_shots, "provider_xg", 0, 1.2)).failures[0]
    with pytest.raises(AttributeError):
        failure.check = "nope"  # type: ignore[misc]


def test_error_can_be_built_directly() -> None:
    failure = schema.Failure(
        table="shots", columns=("x",), check="range", values=("200.0",), rows=(0,), count=1
    )
    err = schema.SchemaValidationError("shots", [failure])
    assert err.failures == (failure,)
    assert "column 'x'" in str(err)


# --- Invalid enum values (acceptance clause) ------------------------------------------------

ENUM_CASES = [
    ("body_part", "chest"),
    ("situation", "throw"),
    ("outcome", "missed"),
    ("provider", "opta"),
    ("assist_type", "header"),
    ("technique", "bicycle"),
]


@pytest.mark.parametrize(("column", "bad"), ENUM_CASES)
def test_invalid_enum_value(valid_shots: pl.DataFrame, column: str, bad: str) -> None:
    df = sc.with_value(valid_shots, column, 3, bad, dtype=pl.String())
    err = _err(df)
    message = str(err)
    assert f"column '{column}'" in message
    assert repr(bad) in message
    allowed = _one(err, column, "allowed")
    assert repr(bad) in allowed.values
    assert allowed.rows == (3,)
    _one(err, column, "dtype")
    vocab = sc.SHOT_DTYPES[column].categories.to_list()
    assert f"allowed: {', '.join(vocab)}" in message
    assert "Error(" not in message


def test_invalid_enum_line_format(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "body_part", 3, "chest", dtype=pl.String()))
    assert (
        "column 'body_part': value(s) 'chest' not allowed; "
        "allowed: right_foot, left_foot, head, other (rows 3)"
    ) in str(err)
    assert (
        "column 'body_part': dtype String, expected "
        "Enum(categories=['right_foot', 'left_foot', 'head', 'other'])"
    ) in str(err)


def test_invalid_enum_lists_every_distinct_bad_value(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "body_part", 0, "chest", dtype=pl.String())
    df = sc.with_value(df, "body_part", 1, "knee")
    df = sc.with_value(df, "body_part", 4, "chest")
    allowed = _one(_err(df), "body_part", "allowed")
    assert set(allowed.values) == {"'chest'", "'knee'"}
    assert allowed.rows == (0, 1, 4)
    assert allowed.count == 3


def test_enum_case_matters(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "body_part", 0, "Head", dtype=pl.String())
    assert "'Head'" in _one(_err(df), "body_part", "allowed").values


def test_null_in_string_enum_is_not_an_allowed_failure(valid_shots: pl.DataFrame) -> None:
    # A String body_part with a null and only valid values: dtype failure only.
    df = valid_shots.with_columns(pl.col("body_part").cast(pl.String))
    err = _err(df)
    assert _pairs(err) == {(("body_part",), "dtype")}


def test_enum_with_foreign_categories(valid_shots: pl.DataFrame) -> None:
    foreign = pl.Enum([*sc.BODY_PARTS, "chest"])
    df = valid_shots.with_columns(pl.col("body_part").cast(pl.String).cast(foreign))
    df = sc.with_value(df, "body_part", 0, "chest")
    err = _err(df)
    _one(err, "body_part", "dtype")
    assert "'chest'" in _one(err, "body_part", "allowed").values


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Enum(["other", "head", "left_foot", "right_foot"]),
        pl.Enum([*sc.BODY_PARTS, "chest"]),
        pl.Categorical(),
        pl.String(),
    ],
    ids=["same_categories_other_order", "superset_valid_values", "categorical", "string"],
)
def test_wrong_enum_dtype_with_valid_values_has_no_allowed_line(
    valid_shots: pl.DataFrame, dtype: pl.DataType
) -> None:
    df = valid_shots.with_columns(pl.col("body_part").cast(pl.String).cast(dtype))
    err = _err(df)
    assert _pairs(err) == {(("body_part",), "dtype")}
    assert "not allowed" not in str(err)


def test_categorical_enum_value_named(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "situation", 2, "throw", dtype=pl.String())
    df = df.with_columns(pl.col("situation").cast(pl.Categorical))
    assert "'throw'" in _one(_err(df), "situation", "allowed").values


# --- Out-of-bounds coordinates and ranges (acceptance clause) -------------------------------

POLYGON_BAD_VERTEX = [[80.0, 10.0], [106.0, 30.0], [105.0, 58.0], [80.0, 58.0]]
OUT_OF_BOUNDS = [
    ("x", 0, 105.5, "range"),
    ("x", 0, -0.1, "range"),
    ("y", 0, 68.01, "range"),
    ("y", 0, -1e-9, "range"),
    ("end_x", 0, -1.0, "range"),
    ("end_x", 0, 105.000001, "range"),
    ("end_y", 0, 70.0, "range"),
    ("end_z", 0, -0.5, "range"),
    ("goal_y", 0, -1.0, "range"),  # row 0 is non-blocked: the bound fires, not the rule
    ("goal_y", 0, 68.5, "range"),
    ("goal_z", 0, -0.2, "range"),
    ("gk_x", 0, 106.0, "range"),
    ("gk_y", 0, 68.5, "range"),
    ("gk_y", 0, -0.5, "range"),
    ("provider_xg", 0, 1.2, "range"),
    ("provider_xg", 0, -0.01, "range"),
    ("provider_psxg", 0, 1.5, "range"),
    ("distance_m", 0, -1.0, "range"),
    ("angle_rad", 0, 3.2, "range"),
    ("angle_rad", 0, -0.1, "range"),
    ("minute", 0, 151, "range"),
    ("minute", 4, -1, "range"),
    ("second", 0, 60, "range"),
    ("second", 0, -1, "range"),
    ("seconds_in_period", 0, 4500.5, "range"),
    ("seconds_in_period", 0, -0.5, "range"),
    ("period", 0, 6, "allowed"),
    ("period", 0, 0, "allowed"),
    ("period", 0, -1, "allowed"),
    ("visible_area", 0, POLYGON_BAD_VERTEX, "polygon"),
    ("visible_area", 0, [[80.0, 10.0], [105.0, 10.0]], "polygon"),
    ("visible_area", 0, [], "polygon"),
    ("visible_area", 0, [[80.0, 10.0], [float("nan"), 30.0], [105.0, 58.0]], "polygon"),
    ("visible_area", 0, [[80.0, 10.0], [float("inf"), 30.0], [105.0, 58.0]], "polygon"),
    ("visible_area", 0, [[80.0, 10.0], None, [105.0, 58.0], [80.0, 58.0]], "polygon"),
    ("visible_area", 0, [[80.0, None], [105.0, 10.0], [105.0, 58.0]], "polygon"),
    ("visible_area", 0, [[80.0, -0.5], [105.0, 10.0], [105.0, 58.0]], "polygon"),
    ("visible_area", 0, [[80.0, 10.0], [105.0, 10.0], [105.0, 68.5]], "polygon"),
]


def _case_id(case: tuple) -> str:
    column, row, value, _ = case
    text = f"{len(value)}v" if isinstance(value, list) else repr(value)
    return f"{column}-{text}-r{row}"


@pytest.mark.parametrize(
    ("column", "row", "value", "check"),
    OUT_OF_BOUNDS,
    ids=[_case_id(c) + f"-{i}" for i, c in enumerate(OUT_OF_BOUNDS)],
)
def test_out_of_bounds(
    valid_shots: pl.DataFrame, column: str, row: int, value: object, check: str
) -> None:
    err = _err(sc.with_value(valid_shots, column, row, value))
    failure = _one(err, column, check)
    assert row in failure.rows
    message = str(err)
    assert f"column '{column}'" in message
    assert "Error(" not in message
    if column != "visible_area":
        assert repr(value) in failure.values or str(value) in failure.values


def test_range_line_format(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "x", 0, 105.5))
    message = str(err)
    assert "column 'x': value(s) 105.5 outside [" in message
    line = next(line for line in message.splitlines() if "column 'x'" in line)
    assert "105" in line and line.rstrip().endswith("(rows 0)")


def test_range_line_without_upper_bound_says_inf(valid_shots: pl.DataFrame) -> None:
    message = str(_err(sc.with_value(valid_shots, "end_z", 0, -0.5)))
    line = next(line for line in message.splitlines() if "column 'end_z'" in line)
    assert "-0.5" in line and "inf)" in line


def test_polygon_value_rendering(valid_shots: pl.DataFrame) -> None:
    failure = _one(
        _err(sc.with_value(valid_shots, "visible_area", 0, POLYGON_BAD_VERTEX)),
        "visible_area",
        "polygon",
    )
    assert "4 vertices, first bad (106.0, 30.0)" in failure.values


def test_out_of_bounds_reports_every_bad_row(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "gk_x", 0, 106.0)
    df = sc.with_value(df, "gk_x", 4, 120.0)
    failure = _one(_err(df), "gk_x", "range")
    assert failure.rows == (0, 4)
    assert failure.count == 2
    assert set(failure.values) == {"106.0", "120.0"}


NON_FINITE = [
    ("x", 0, float("nan"), {"nullable", "finite"}),
    ("x", 0, float("inf"), {"finite"}),
    ("provider_xg", 0, float("inf"), {"finite"}),
    ("gk_x", 0, float("nan"), {"finite"}),  # NaN in a *nullable* column is not a null
    ("end_z", 0, float("inf"), {"finite"}),  # ge alone passes +inf: finite must catch it
    ("goal_z", 0, float("inf"), {"finite"}),
    ("distance_m", 0, float("inf"), {"finite"}),
    ("seconds_in_period", 0, float("nan"), {"finite"}),
    ("end_y", 0, float("-inf"), {"finite"}),
]


@pytest.mark.parametrize(("column", "row", "value", "checks"), NON_FINITE)
def test_non_finite(
    valid_shots: pl.DataFrame, column: str, row: int, value: float, checks: set[str]
) -> None:
    err = _err(sc.with_value(valid_shots, column, row, value))
    got = {f.check for f in err.failures if f.columns == (column,)}
    assert checks <= got, got
    finite = _one(err, column, "finite")
    assert repr(value) in finite.values  # 'nan' / 'inf' / '-inf'
    assert "Error(" not in str(err)


# --- Wrong dtypes (acceptance clause) -------------------------------------------------------


def _int_series(name: str, n: int = 5) -> pl.Series:
    return pl.Series(name, list(range(n)), dtype=pl.Int64)


WRONG_DTYPES = {
    "x_int64": lambda df: df.with_columns(pl.col("x").cast(pl.Int64)),
    "x_float32": lambda df: df.with_columns(pl.col("x").cast(pl.Float32)),
    "period_int64": lambda df: df.with_columns(pl.col("period").cast(pl.Int64)),
    "period_int16": lambda df: df.with_columns(pl.col("period").cast(pl.Int16)),
    "minute_int8": lambda df: df.with_columns(pl.col("minute").cast(pl.Int8)),
    "second_uint8": lambda df: df.with_columns(pl.col("second").cast(pl.UInt8)),
    "match_date_string": lambda df: df.with_columns(pl.col("match_date").cast(pl.String)),
    "match_date_datetime": lambda df: df.with_columns(pl.col("match_date").cast(pl.Datetime)),
    "is_goal_int64": lambda df: df.with_columns(pl.col("is_goal").cast(pl.Int64)),
    "is_home_string": lambda df: df.with_columns(pl.col("is_home").cast(pl.String)),
    "shot_id_int64": lambda df: df.with_columns(_int_series("shot_id")),
    "visible_area_list_float": lambda df: df.with_columns(
        pl.Series("visible_area", [[1.0, 2.0]] * 5, dtype=pl.List(pl.Float64))
    ),
    "visible_area_list_list": lambda df: df.with_columns(
        pl.col("visible_area").cast(pl.List(pl.List(pl.Float64)))
    ),
    "provider_xg_null_dtype": lambda df: df.with_columns(pl.lit(None).alias("provider_xg")),
}
WRONG_DTYPE_COLUMNS = {
    "x_int64": "x",
    "x_float32": "x",
    "period_int64": "period",
    "period_int16": "period",
    "minute_int8": "minute",
    "second_uint8": "second",
    "match_date_string": "match_date",
    "match_date_datetime": "match_date",
    "is_goal_int64": "is_goal",
    "is_home_string": "is_home",
    "shot_id_int64": "shot_id",
    "visible_area_list_float": "visible_area",
    "visible_area_list_list": "visible_area",
    "provider_xg_null_dtype": "provider_xg",
}


@pytest.mark.parametrize("case", sorted(WRONG_DTYPES))
def test_wrong_dtype(valid_shots: pl.DataFrame, case: str) -> None:
    column = WRONG_DTYPE_COLUMNS[case]
    df = WRONG_DTYPES[case](valid_shots)
    got = str(df.schema[column])
    err = _err(df)
    failure = _one(err, column, "dtype")
    assert failure.rows == ()
    message = str(err)
    assert f"column '{column}': dtype {got}, expected {sc.SHOT_DTYPES[column]}" in message
    # validate_* never casts: the wrong dtype is never "fixed" silently
    assert "Error(" not in message


def test_structural_failures_skip_data_checks(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "shot_id", 1, sc.SB_SHOT_1)  # duplicate id (data)
    df = df.with_columns(pl.col("x").cast(pl.Int64))  # wrong dtype (structure)
    err = _err(df)
    assert _pairs(err) == {(("x",), "dtype")}


def test_all_structural_failures_reported_together(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.drop("y").with_columns(
        pl.col("period").cast(pl.Int64), pl.lit(1).alias("xg_model")
    )
    err = _err(df)
    assert _pairs(err) == {
        (("y",), "missing"),
        (("period",), "dtype"),
        (("xg_model",), "extra"),
    }


# --- Missing and extra columns (acceptance clause) ------------------------------------------


@pytest.mark.parametrize("column", ["y", "shot_id", "schema_version", "visible_area", "outcome"])
def test_missing_column(valid_shots: pl.DataFrame, column: str) -> None:
    err = _err(valid_shots.drop(column))
    failure = _one(err, column, "missing")
    assert failure.rows == ()
    assert f"column '{column}': missing required column" in str(err)


def test_missing_nullable_column_is_still_missing(valid_shots: pl.DataFrame) -> None:
    # Nullable means "may hold null", not "may be absent".
    err = _err(valid_shots.drop("big_chance", "provider_psxg"))
    assert _pairs(err) == {(("big_chance",), "missing"), (("provider_psxg",), "missing")}


def test_every_missing_column_is_listed(valid_shots: pl.DataFrame) -> None:
    err = _err(valid_shots.select("shot_id"))
    missing = {f.columns[0] for f in err.failures if f.check == "missing"}
    assert missing == set(sc.SHOT_NAMES) - {"shot_id"}


def test_extra_column(valid_shots: pl.DataFrame) -> None:
    err = _err(valid_shots.with_columns(pl.lit(0.1).alias("my_xg")))
    _one(err, "my_xg", "extra")
    assert "column 'my_xg': not in the shots schema" in str(err)


def test_every_extra_column_is_listed(valid_shots: pl.DataFrame) -> None:
    # pandera 0.34.1 (polars backend) reports only the first extra column of a strict schema,
    # so the implementation must list extras itself (spec §1.1: every failure found).
    df = valid_shots.with_columns(pl.lit(1).alias("eventId"), pl.lit(2).alias("xT"))
    extras = [f.columns[0] for f in _err(df).failures if f.check == "extra"]
    assert extras == ["eventId", "xT"]


def test_column_name_case_matters(valid_shots: pl.DataFrame) -> None:
    err = _err(valid_shots.rename({"x": "X"}))
    assert _pairs(err) == {(("x",), "missing"), (("X",), "extra")}


def test_extra_columns_come_after_canonical_columns(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.drop("schema_version").with_columns(
        pl.lit(1).alias("zz_extra"), pl.lit(2).alias("aa_extra")
    )
    df = df.select(["zz_extra", "aa_extra", *[c for c in df.columns if "extra" not in c]])
    columns = [f.columns[0] for f in _err(df).failures]
    assert columns == ["schema_version", "zz_extra", "aa_extra"]


# --- Duplicate shot_id (acceptance clause) --------------------------------------------------


def test_duplicate_shot_id(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "shot_id", 1, sc.SB_SHOT_1))
    failure = _one(err, "shot_id", "unique")
    assert failure.rows == (0, 1)
    assert failure.values == (repr(sc.SB_SHOT_1),)
    assert f"column 'shot_id': duplicate value(s) {sc.SB_SHOT_1!r} (rows 0, 1)" in str(err)


def test_duplicate_shot_id_across_providers_is_reported(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "shot_id", 3, sc.WS_SHOT_3)
    df = sc.with_value(df, "shot_id", 1, sc.SB_SHOT_5)
    failure = _one(_err(df), "shot_id", "unique")
    assert set(failure.values) == {repr(sc.WS_SHOT_3), repr(sc.SB_SHOT_5)}
    assert failure.rows == (1, 2, 3, 4)
    assert failure.count == 4


def test_triplicate_shot_id(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "shot_id", 1, sc.SB_SHOT_1)
    df = sc.with_value(df, "shot_id", 4, sc.SB_SHOT_1)
    assert _one(_err(df), "shot_id", "unique").rows == (0, 1, 4)


def test_null_shot_id_reported_once(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "shot_id", 0, None)
    df = sc.with_value(df, "shot_id", 1, None)
    err = _err(df)
    assert _one(err, "shot_id", "nullable").rows == (0, 1)
    assert not [f for f in err.failures if f.check == "unique"]


def test_shot_ids_differing_only_by_whitespace_are_not_ids(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "shot_id", 1, sc.SB_SHOT_1 + " ")
    failure = _one(_err(df), "shot_id", "id_format")
    assert failure.rows == (1,)


# --- Other clear messages -------------------------------------------------------------------


@pytest.mark.parametrize(
    "column", ["minute", "situation", "match_date", "raw_type", "is_goal", "x", "provider"]
)
def test_null_in_required_column(valid_shots: pl.DataFrame, column: str) -> None:
    err = _err(sc.with_value(valid_shots, column, 2, None))
    failure = _one(err, column, "nullable")
    assert failure.rows == (2,)
    assert f"column '{column}': null/NaN in non-nullable column (rows 2)" in str(err)
    assert "Error(" not in str(err)


@pytest.mark.parametrize(
    ("column", "bad"),
    [
        ("player_id", "opta:1"),
        ("player_id", "SB:90001"),
        ("player_id", "sb:"),
        ("player_id", "sb:9 0"),
        ("player_id", "90001"),
        ("team_id", "xsb:901"),
        ("match_id", "sb:1\nws:2"),
        ("assist_event_id", "ws:"),
        ("competition_id", ""),
    ],
)
def test_bad_id_prefix(valid_shots: pl.DataFrame, column: str, bad: str) -> None:
    err = _err(sc.with_value(valid_shots, column, 0, bad))
    failure = _one(err, column, "id_format")
    assert failure.values == (repr(bad),)
    assert f"column '{column}': value(s) {bad!r} not a provider-prefixed id" in str(err)


def test_wrong_schema_version(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.with_columns(pl.lit("0.9").alias("schema_version"))
    err = _err(df)
    failure = _one(err, "schema_version", "schema_version")
    assert failure.values == ("'0.9'",)
    assert failure.count == 5
    assert "column 'schema_version'" in str(err) and "'0.9'" in str(err)


@pytest.mark.parametrize("version", ["1", "1.0", "1.0 ", "v1.0", "1.0.0", "2.0"])
def test_schema_version_is_exact(valid_shots: pl.DataFrame, version: str) -> None:
    err = _err(sc.with_value(valid_shots, "schema_version", 2, version))
    _one(err, "schema_version", "schema_version")


def test_empty_raw_type(valid_shots: pl.DataFrame) -> None:
    err = _err(sc.with_value(valid_shots, "raw_type", 1, ""))
    assert _one(err, "raw_type", "non_empty").rows == (1,)
    assert "column 'raw_type'" in str(err)


@pytest.mark.parametrize(
    "bad",
    [
        "nope",
        "{",
        "",
        '"just a string"',  # valid JSON, but the top level is not an object / array
        "42",
        "null",
        "true",
        "[NaN]",  # not JSON (RFC 8259); Python's json accepts it by default (QA r2 m5)
        '{"xg": Infinity}',
        "[" * 100_000,  # json.loads raises RecursionError: still bad data (QA r2 m1)
        "{'single': 'quotes'}",
    ],
    ids=[
        "word",
        "open_brace",
        "empty",
        "scalar_string",
        "scalar_number",
        "scalar_null",
        "scalar_bool",
        "nan_constant",
        "infinity_constant",
        "deep_nesting",
        "single_quotes",
    ],
)
def test_invalid_json_qualifiers(valid_shots: pl.DataFrame, bad: str) -> None:
    err = _err(sc.with_value(valid_shots, "raw_qualifiers", 1, bad))
    failure = _one(err, "raw_qualifiers", "json")
    assert failure.rows == (1,)
    assert "column 'raw_qualifiers'" in str(err)


def test_deep_json_does_not_hide_other_bad_rows(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "raw_qualifiers", 1, "[" * 100_000)
    df = sc.with_value(df, "raw_qualifiers", 3, "nope")
    assert _one(_err(df), "raw_qualifiers", "json").rows == (1, 3)


@pytest.mark.parametrize("good", ["[]", "{}", '[{"type": 82}]', '{"a": [1, 2.5, null]}'])
def test_valid_json_qualifiers(valid_shots: pl.DataFrame, good: str) -> None:
    schema.validate_shots(sc.with_value(valid_shots, "raw_qualifiers", 1, good))


def test_all_problems_reported_together(valid_shots: pl.DataFrame) -> None:
    # Three independent data errors that trip no rule (second 60 on the shoot-out row, where
    # period 5 skips clock_consistent; QA r2 m4).
    df = sc.with_value(valid_shots, "provider_xg", 0, 1.2)
    df = sc.with_value(df, "second", 4, 60)
    df = sc.with_value(df, "raw_type", 1, "")
    err = _err(df)
    assert _pairs(err) == {
        (("second",), "range"),
        (("provider_xg",), "range"),
        (("raw_type",), "non_empty"),
    }
    lines = str(err).splitlines()
    assert lines[0].startswith("shots: 3 schema problem(s)")
    assert len(lines) == 4


def test_line_order_is_canonical_then_rules(valid_shots: pl.DataFrame) -> None:
    df = sc.with_value(valid_shots, "schema_version", 0, "0.9")  # column 47
    df = sc.with_value(df, "x", 0, 105.5)  # column 15, also trips distance/angle rules
    df = sc.with_value(df, "is_goal", 0, False)  # rule is_goal_matches_outcome
    order = [(f.columns[0] if len(f.columns) == 1 else None, f.check) for f in _err(df).failures]
    assert order == [
        ("x", "range"),
        ("schema_version", "schema_version"),
        (None, "distance_matches_pitch"),
        (None, "angle_matches_pitch"),
        (None, "is_goal_matches_outcome"),
    ]


def test_line_order_within_a_column(valid_shots: pl.DataFrame) -> None:
    # §3.3 order: nullable < range < finite.
    err = _err(sc.with_value(valid_shots, "x", 0, float("nan")))
    checks = [f.check for f in err.failures if f.columns == ("x",)]
    assert checks == sorted(checks, key=["nullable", "range", "finite"].index)


def test_message_truncates_examples(valid_shots: pl.DataFrame) -> None:
    copies = [valid_shots.with_columns(pl.col("shot_id") + f"-copy{i}") for i in range(3)]
    df = pl.concat(copies).with_columns(pl.lit(1.2).alias("provider_xg"))
    assert df.height == 15
    err = _err(df)
    failure = _one(err, "provider_xg", "range")
    assert failure.count == 15
    assert failure.rows == (0, 1, 2, 3, 4)
    assert failure.values == ("1.2",)
    assert "and 10 more" in str(err)


def test_no_truncation_note_at_exactly_max_examples(valid_shots: pl.DataFrame) -> None:
    df = valid_shots.with_columns(pl.lit(1.2).alias("provider_xg"))
    err = _err(df)
    assert _one(err, "provider_xg", "range").count == 5
    assert "more" not in str(err)


def test_long_values_are_capped_at_max_examples(valid_shots: pl.DataFrame) -> None:
    copies = [valid_shots.with_columns(pl.col("shot_id") + f"-c{i}") for i in range(2)]
    df = pl.concat(copies)
    df = df.with_columns(pl.Series("gk_x", [106.0 + i for i in range(10)], dtype=pl.Float64))
    failure = _one(_err(df), "gk_x", "range")
    assert len(failure.values) == schema.MAX_EXAMPLES
    assert failure.count == 10


# --- Input types ----------------------------------------------------------------------------

NOT_FRAMES = {
    "lazyframe": lambda df: df.lazy(),
    "dict": lambda df: df.to_dict(as_series=False),
    "series": lambda df: df["shot_id"],
    "none": lambda df: None,
}
TYPE_NAMES = {"lazyframe": "LazyFrame", "dict": "dict", "series": "Series", "none": "None"}


@pytest.mark.parametrize("kind", sorted(NOT_FRAMES))
@pytest.mark.parametrize(
    "call",
    [
        lambda obj: schema.validate_shots(obj),
        lambda obj: schema.validate_freeze_frames(obj),
        lambda obj: schema.cast_to_schema(obj),
        lambda obj: schema.cast_to_schema(obj, "freeze_frames"),
    ],
    ids=["validate_shots", "validate_freeze_frames", "cast_shots", "cast_freeze_frames"],
)
def test_rejects_non_dataframe(valid_shots: pl.DataFrame, call, kind: str) -> None:
    with pytest.raises(TypeError) as info:
        call(NOT_FRAMES[kind](valid_shots))
    assert TYPE_NAMES[kind] in str(info.value)


@pytest.mark.parametrize("kind", ["lazyframe", "dict", "series"])
def test_rejects_non_dataframe_shots_argument(
    valid_shots: pl.DataFrame, valid_freeze_frames: pl.DataFrame, kind: str
) -> None:
    with pytest.raises(TypeError) as info:
        schema.validate_freeze_frames(valid_freeze_frames, shots=NOT_FRAMES[kind](valid_shots))
    assert TYPE_NAMES[kind] in str(info.value)


@pytest.mark.parametrize(
    ("mutate", "column"),
    [
        (lambda s: s.drop("player_id"), "player_id"),
        (lambda s: s.drop("shot_id"), "shot_id"),
        (lambda s: s.with_columns(pl.Series("player_id", range(5))), "player_id"),
        (lambda s: s.with_columns(pl.col("shot_id").cast(pl.Categorical)), "shot_id"),
    ],
    ids=["no_player_id", "no_shot_id", "player_id_int", "shot_id_categorical"],
)
def test_shots_argument_missing_columns(
    valid_shots: pl.DataFrame, valid_freeze_frames: pl.DataFrame, mutate, column: str
) -> None:
    with pytest.raises(ValueError, match=column):
        schema.validate_freeze_frames(valid_freeze_frames, shots=mutate(valid_shots))


def test_shots_argument_needs_only_the_key_columns(
    valid_shots: pl.DataFrame, valid_freeze_frames: pl.DataFrame
) -> None:
    # `shots` is not validated here (spec §1.1): a frame with just the keys is enough.
    keys = valid_shots.select("shot_id", "player_id")
    schema.validate_freeze_frames(valid_freeze_frames, shots=keys)


# --- Validation cannot be switched off ------------------------------------------------------

OUTER_CONTEXTS = {
    "enabled_false": {"validation_enabled": False},
    "depth_schema_only": {"validation_depth": ValidationDepth.SCHEMA_ONLY},
    "both": {"validation_enabled": False, "validation_depth": ValidationDepth.SCHEMA_ONLY},
}


@pytest.mark.parametrize("context", sorted(OUTER_CONTEXTS))
@pytest.mark.parametrize("table", ["shots", "freeze_frames"])
def test_validation_ignores_global_pandera_config(
    valid_shots: pl.DataFrame, valid_freeze_frames: pl.DataFrame, context: str, table: str
) -> None:
    outer = config_context(**OUTER_CONTEXTS[context])
    with outer, pytest.raises(schema.SchemaValidationError) as info:
        if table == "shots":
            schema.validate_shots(sc.with_value(valid_shots, "x", 0, 200.0))
        else:
            schema.validate_freeze_frames(sc.with_value(valid_freeze_frames, "x", 0, 200.0))
    assert "column 'x'" in str(info.value)
    assert any(f.columns == ("x",) and f.check == "range" for f in info.value.failures)


_SUBPROCESS_PROBE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, sys.argv[1])
    import schema_cases as sc
    from football_core import schema
    shots = sc.with_value(sc.shots_frame(), "x", 0, 200.0)
    frames = sc.with_value(sc.freeze_frames_frame(), "x", 0, 200.0)
    for call, df in ((schema.validate_shots, shots), (schema.validate_freeze_frames, frames)):
        try:
            call(df)
        except schema.SchemaValidationError as err:
            assert any(f.columns == ("x",) and f.check == "range" for f in err.failures)
        else:
            sys.exit(3)
    """
)


@pytest.mark.parametrize(
    "env",
    [
        {"PANDERA_VALIDATION_ENABLED": "False"},
        {"PANDERA_VALIDATION_DEPTH": "SCHEMA_ONLY"},
    ],
    ids=["env_enabled_false", "env_depth_schema_only"],
)
def test_validation_ignores_pandera_env_vars(env: dict[str, str]) -> None:
    # pandera reads these at import, so the probe runs in a fresh interpreter (QA r2 n1).
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_PROBE, str(TESTS_DIR)],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- Broken checks and rules are internal errors, never schema messages ---------------------


def _raising_check(data):  # pandera custom check that blows up
    raise ZeroDivisionError("boom")


def _broken_column_model(model: pa.DataFrameSchema, column: str) -> pa.DataFrameSchema:
    existing = list(model.columns[column].checks)
    check = pa.Check(_raising_check, name="boom", error="boom")
    return model.update_column(column, checks=[*existing, check])


def _frame_rule(name: str, check) -> schema.FrameRule:
    return schema.FrameRule(name=name, columns=("shot_id",), description="broken", check=check)


def _raise(df: pl.DataFrame) -> pl.Series:
    raise KeyError("boom")


BROKEN_RULES = {
    "raises": _raise,
    "short_mask": lambda df: pl.Series([True]),
    "mask_with_nulls": lambda df: pl.Series([None] * df.height, dtype=pl.Boolean),
    "int_mask": lambda df: pl.Series([1] * df.height, dtype=pl.Int64),
    "not_a_series": lambda df: [True] * df.height,
}


def test_broken_column_check_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, valid_shots: pl.DataFrame
) -> None:
    monkeypatch.setattr(schema, "SHOT_MODEL", _broken_column_model(schema.SHOT_MODEL, "x"))
    with pytest.raises(RuntimeError, match=r"^internal error in") as info:
        schema.validate_shots(valid_shots)
    assert not isinstance(info.value, schema.SchemaValidationError)
    assert "boom" in str(info.value)


def test_broken_freeze_frame_check_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, valid_freeze_frames: pl.DataFrame
) -> None:
    broken = _broken_column_model(schema.FREEZE_FRAME_MODEL, "y")
    monkeypatch.setattr(schema, "FREEZE_FRAME_MODEL", broken)
    with pytest.raises(RuntimeError, match=r"^internal error in"):
        schema.validate_freeze_frames(valid_freeze_frames)


@pytest.mark.parametrize("kind", sorted(BROKEN_RULES))
def test_broken_rule_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, valid_shots: pl.DataFrame, kind: str
) -> None:
    rules = (*schema.SHOT_RULES, _frame_rule("broken_rule", BROKEN_RULES[kind]))
    monkeypatch.setattr(schema, "SHOT_RULES", rules)
    with pytest.raises(RuntimeError, match=r"^internal error in rule 'broken_rule'") as info:
        schema.validate_shots(valid_shots)
    assert not isinstance(info.value, schema.SchemaValidationError)


@pytest.mark.parametrize("kind", ["raises", "short_mask"])
def test_broken_freeze_frame_rule_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, valid_freeze_frames: pl.DataFrame, kind: str
) -> None:
    rules = (*schema.FREEZE_FRAME_RULES, _frame_rule("broken_ff_rule", BROKEN_RULES[kind]))
    monkeypatch.setattr(schema, "FREEZE_FRAME_RULES", rules)
    with pytest.raises(RuntimeError, match=r"^internal error in rule 'broken_ff_rule'"):
        schema.validate_freeze_frames(valid_freeze_frames)


def test_rule_exception_is_chained(
    monkeypatch: pytest.MonkeyPatch, valid_shots: pl.DataFrame
) -> None:
    monkeypatch.setattr(schema, "SHOT_RULES", (_frame_rule("broken_rule", _raise),))
    with pytest.raises(RuntimeError) as info:
        schema.validate_shots(valid_shots)
    assert isinstance(info.value.__cause__, KeyError)


def test_monkeypatched_rule_is_read_at_call_time(
    monkeypatch: pytest.MonkeyPatch, valid_shots: pl.DataFrame
) -> None:
    # A rule that fails every row proves validate_shots reads SHOT_RULES from the module.
    always_fail = _frame_rule("always_fail", lambda df: pl.Series([False] * df.height))
    monkeypatch.setattr(schema, "SHOT_RULES", (always_fail,))
    err = _err(valid_shots)
    failure = _rule(err, "always_fail")
    assert failure.rows == (0, 1, 2, 3, 4)
    assert "rule 'always_fail' (shot_id): broken" in str(err)
