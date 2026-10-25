"""The canonical matches table and the shots-against-matches rules (MS3.matches spec §2-§4.1, §6).

Fixtures are the hand-written ``schema_cases.base_match_rows`` (three matches of the five C3
fixture shots plus a women's match without shots). Every near miss changes one element of one
row, on a row that is otherwise valid football, so each test fails for its own rule only.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

import hypothesis.strategies as st
import polars as pl
import pytest
import schema_cases as sc
from hypothesis import given, settings
from polars.testing import assert_frame_equal

from football_core import schema

Frame = pl.DataFrame
KEYS = ("match_id", "competition_id", "season_id", "match_date", "home_team_id", "away_team_id")
RULE_TEXT = {
    "ids_match_provider": "match, competition, season and team ids must start with the "
    "provider's prefix",
    "distinct_teams": "home_team_id must differ from away_team_id",
    "scores_both_or_neither": "home_score and away_score must both be set or both be null",
    "has_360_matches_provider": "has_360 must be set for statsbomb rows and null for every "
    "other provider",
    "match_exists": "match_id must exist in matches.match_id",
    "shot_agrees_with_match": "competition, season, date and the team pair (home/away from "
    "is_home) must equal the shot's matches row",
}


def _err(call: Callable[[], object]) -> schema.SchemaValidationError:
    with pytest.raises(schema.SchemaValidationError) as info:
        call()
    return info.value


def _pairs(err: schema.SchemaValidationError) -> set[tuple[tuple[str, ...], str]]:
    return {(f.columns, f.check) for f in err.failures}


# --- Columns, roles, version ------------------------------------------------------------------


def test_match_columns() -> None:
    assert schema.TABLES == ("shots", "freeze_frames", "matches")
    assert schema.GENDERS == ("male", "female")
    assert pl.Enum(("male", "female")) == schema.GENDER
    specs = schema.MATCH_COLUMNS
    assert tuple(s.name for s in specs) == sc.MATCH_NAMES
    assert len(specs) == 14
    assert {s.name: s.dtype for s in specs} == sc.MATCH_DTYPES
    assert {s.name: s.nullable for s in specs} == {n: null for n, _, null in sc.MATCH_COLUMNS}
    assert {s.name: s.role for s in specs} == sc.MATCH_ROLES
    assert {s.name for s in specs if s.is_id} == sc.MATCH_ID_COLUMNS
    assert {s.name for s in specs if s.unique} == {"match_id"}
    bounds = {s.name: (s.ge, s.le) for s in specs if s.ge is not None or s.le is not None}
    assert bounds == sc.MATCH_BOUNDS
    assert pl.Schema(sc.MATCH_DTYPES) == schema.MATCH_SCHEMA
    assert tuple(schema.MATCH_SCHEMA.names()) == sc.MATCH_NAMES
    model = schema.MATCH_MODEL
    assert list(model.columns) == list(sc.MATCH_NAMES)
    assert model.strict is True and not model.coerce
    for spec in specs:
        assert model.columns[spec.name].nullable == spec.nullable, spec.name
        assert model.columns[spec.name].metadata == {"role": spec.role}, spec.name
        assert spec.description.strip(), spec.name
    assert schema.column_names("matches") == sc.MATCH_NAMES
    assert schema.column_names("matches", roles="pre_shot") == ("gender",)
    assert schema.column_names("matches", roles="outcome") == ("home_score", "away_score")
    assert [(r.name, r.columns) for r in schema.MATCH_RULES] == list(sc.MATCH_RULES)
    assert {r.name: r.description for r in schema.MATCH_RULES} == {
        name: RULE_TEXT[name] for name, _ in sc.MATCH_RULES
    }


def test_shared_columns_agree_with_shots() -> None:
    """Columns 1-5 and 14 share name, dtype and role with shots (the guard relies on it); no
    pre_shot matches column shares a name with a shots column (the load join relies on it)."""
    shots = {s.name: s for s in schema.SHOT_COLUMNS}
    shared = [s for s in schema.MATCH_COLUMNS if s.name in shots]
    assert [s.name for s in shared] == [
        "match_id",
        "provider",
        "competition_id",
        "season_id",
        "match_date",
        "schema_version",
    ]
    for spec in shared:
        assert spec.dtype == shots[spec.name].dtype, spec.name
        assert spec.role == shots[spec.name].role, spec.name
    pre_shot = schema.column_names("matches", roles="pre_shot")
    assert set(pre_shot).isdisjoint(schema.column_names("shots"))
    assert "nominal at neutral venues" in shots["is_home"].description


# --- validate_matches -------------------------------------------------------------------------


def test_valid_matches_pass() -> None:
    matches = sc.matches_frame()
    assert matches["provider"].to_list().count("whoscored") == 1  # null scores and has_360
    assert_frame_equal(schema.validate_matches(matches), matches)
    reversed_order = matches.select(list(reversed(matches.columns)))
    assert schema.validate_matches(reversed_order).columns == list(sc.MATCH_NAMES)
    edge = sc.with_values(matches, 0, home_score=99, away_score=0)  # inclusive bounds
    schema.validate_matches(edge)
    empty = schema.empty_matches()
    assert empty.schema == schema.MATCH_SCHEMA and empty.height == 0
    assert schema.validate_matches(empty).height == 0


COLUMN_FAILURES: dict[str, tuple[Callable[[Frame], Frame], set[tuple[str, str]], tuple[int, ...]]]
COLUMN_FAILURES = {
    "duplicate_id": (
        lambda m: sc.with_value(m, "match_id", 2, "sb:9000001"),
        {("match_id", "unique")},
        (0, 2),
    ),
    "null_gender": (lambda m: sc.with_value(m, "gender", 0, None), {("gender", "nullable")}, (0,)),
    "null_match_date": (
        lambda m: sc.with_value(m, "match_date", 3, None),
        {("match_date", "nullable")},
        (3,),
    ),
    # a null input passes the rules: distinct_teams does not fire on it
    "null_home_team": (
        lambda m: sc.with_value(m, "home_team_id", 0, None),
        {("home_team_id", "nullable")},
        (0,),
    ),
    "string_gender_men": (
        lambda m: sc.with_value(m, "gender", 1, "men", dtype=pl.String()),
        {("gender", "dtype"), ("gender", "allowed")},
        (1,),
    ),
    "home_score_minus_1": (
        lambda m: sc.with_value(m, "home_score", 2, -1),
        {("home_score", "range")},
        (2,),
    ),
    "away_score_100": (
        lambda m: sc.with_value(m, "away_score", 0, 100),
        {("away_score", "range")},
        (0,),
    ),
    "score_int64_not_cast": (
        lambda m: m.with_columns(pl.col("home_score").cast(pl.Int64)),
        {("home_score", "dtype")},
        (),
    ),
    "empty_competition_name": (
        lambda m: sc.with_value(m, "competition_name", 3, ""),
        {("competition_name", "non_empty")},
        (3,),
    ),
    "empty_season_name": (
        lambda m: sc.with_value(m, "season_name", 1, ""),
        {("season_name", "non_empty")},
        (1,),
    ),
    "bad_id": (
        lambda m: sc.with_value(m, "home_team_id", 0, "901"),
        {("home_team_id", "id_format")},
        (0,),
    ),
    "old_version": (
        lambda m: sc.with_value(m, "schema_version", 2, "1.0"),
        {("schema_version", "schema_version")},
        (2,),
    ),
    "missing_column": (lambda m: m.drop("has_360"), {("has_360", "missing")}, ()),
    "extra_column": (
        lambda m: m.with_columns(pl.lit("Fixture Ground").alias("venue")),
        {("venue", "extra")},
        (),
    ),
}


@pytest.mark.parametrize("case", sorted(COLUMN_FAILURES))
def test_match_column_failures(case: str) -> None:
    mutate, expected, rows = COLUMN_FAILURES[case]
    err = _err(lambda: schema.validate_matches(mutate(sc.matches_frame())))
    assert err.table == "matches"
    assert {(f.columns[0], f.check) for f in err.failures} == expected
    assert all(f.table == "matches" for f in err.failures)
    assert all(f.rows == rows for f in err.failures if f.check != "dtype")
    header = str(err).splitlines()[0]
    assert header == f"matches: {len(err.failures)} schema problem(s) (schema 1.1)"
    if case == "string_gender_men":
        assert "'men'" in str(err) and "allowed: male, female" in str(err)


RULE_CASES = [
    ("match_id_prefix", 0, {"match_id": "ws:9000001"}, "ids_match_provider"),
    ("competition_id_prefix", 0, {"competition_id": "ws:9001"}, "ids_match_provider"),
    ("season_id_prefix", 3, {"season_id": "ws:9002"}, "ids_match_provider"),
    ("home_team_id_prefix", 0, {"home_team_id": "ws:901"}, "ids_match_provider"),
    ("away_team_id_prefix_on_whoscored", 1, {"away_team_id": "sb:911"}, "ids_match_provider"),
    ("equal_teams", 0, {"away_team_id": "sb:901"}, "distinct_teams"),
    ("home_score_null_only", 0, {"home_score": None}, "scores_both_or_neither"),
    ("away_score_null_only", 2, {"away_score": None}, "scores_both_or_neither"),
    ("whoscored_home_score_only", 1, {"home_score": 2}, "scores_both_or_neither"),
    ("has_360_null_on_statsbomb", 0, {"has_360": None}, "has_360_matches_provider"),
    ("has_360_false_on_whoscored", 1, {"has_360": False}, "has_360_matches_provider"),
    ("has_360_true_on_whoscored", 1, {"has_360": True}, "has_360_matches_provider"),
]


@pytest.mark.parametrize(
    ("row", "cells", "rule"), [c[1:] for c in RULE_CASES], ids=[c[0] for c in RULE_CASES]
)
def test_match_rules_fail(row: int, cells: dict[str, Any], rule: str) -> None:
    err = _err(lambda: schema.validate_matches(sc.with_values(sc.matches_frame(), row, **cells)))
    columns = dict(sc.MATCH_RULES)[rule]
    assert _pairs(err) == {(columns, rule)}
    (failure,) = err.failures
    assert failure.table == "matches"
    assert failure.rows == (row,)
    # the description is looked up in the matches rules, never the shots rule of that name
    assert f"rule '{rule}' ({', '.join(columns)}): {RULE_TEXT[rule]}" in str(err)


# --- validate_shots(df, matches=...) ----------------------------------------------------------

AGREES, EXISTS = "shot_agrees_with_match", "match_exists"


def _empty(m: Frame) -> Frame:
    return schema.empty_matches()


def _duplicate_with_other_date(m: Frame) -> Frame:
    copy = m.filter(pl.col("match_id") == "sb:9000001").with_columns(
        pl.lit(dt.date(2024, 3, 3)).alias("match_date")
    )
    return pl.concat([m, copy])


def _same(m: Frame) -> Frame:
    return m


# name: (cells changed on shot row 0, matches transform, failure checks in order, rows)
SHOT_CASES: dict[str, tuple[dict[str, Any], Callable[[Frame], Frame], list[str], tuple[int, ...]]]
SHOT_CASES = {
    # rows 1 (away side) and 3 (a WhoScored own goal by the home side) agree too
    "agrees": ({}, _same, [], ()),
    "is_home_flipped": ({"is_home": False}, _same, [AGREES], (0,)),
    "date_of_another_match": ({"match_date": dt.date(2024, 3, 16)}, _same, [AGREES], (0,)),
    "competition_of_another_match": ({"competition_id": "sb:9003"}, _same, [AGREES], (0,)),
    "other_season": ({"season_id": "sb:9004"}, _same, [AGREES], (0,)),
    "opponent_from_another_match": ({"opponent_team_id": "sb:903"}, _same, [AGREES], (0,)),
    "team_from_another_match": ({"team_id": "sb:951"}, _same, [AGREES], (0,)),
    "match_id_of_another_match": ({"match_id": "sb:9000002"}, _same, [AGREES], (0,)),
    "unknown_match": ({"match_id": "sb:9000009"}, _same, [EXISTS], (0,)),
    # a null key is the column checks' job: both rules pass the row
    "null_is_home": ({"is_home": None}, _same, ["nullable"], (0,)),
    "null_match_id": ({"match_id": None}, _same, ["nullable"], (0,)),
    "shot_rule_before_cross_rule": (
        {"match_id": "sb:9000009", "is_goal": False},
        _same,
        ["is_goal_matches_outcome", EXISTS],
        (0,),
    ),
    "empty_matches": ({}, _empty, [EXISTS], (0, 1, 2, 3, 4)),
    # uniqueness is validate_matches' job; one agreeing row is enough and rows do not multiply
    "duplicated_match_id": ({}, _duplicate_with_other_date, [], ()),
}


@pytest.mark.parametrize("case", list(SHOT_CASES))
def test_shots_against_matches(case: str) -> None:
    cells, transform, checks, rows = SHOT_CASES[case]
    shots = sc.with_values(sc.shots_frame(), 0, **cells)
    matches = transform(sc.matches_frame())
    if not checks:
        out = schema.validate_shots(shots, matches=matches)
        assert_frame_equal(out, shots)
        return
    err = _err(lambda: schema.validate_shots(shots, matches=matches))
    assert err.table == "shots"
    assert [f.check for f in err.failures] == checks
    for failure in err.failures:
        assert failure.rows == rows
        if failure.check in sc.CROSS_SHOT_RULES:
            assert failure.columns == sc.CROSS_SHOT_RULES[failure.check]
            assert RULE_TEXT[failure.check] in str(err)
    # without matches= the cross-table rules do not run
    without = [f.check for f in err.failures if f.check not in sc.CROSS_SHOT_RULES]
    if without:
        assert [f.check for f in _err(lambda: schema.validate_shots(shots)).failures] == without
    else:
        schema.validate_shots(shots)


def _drop_key(column: str) -> Callable[[Frame], Frame]:
    return lambda m: m.drop(column)


ARG_ERRORS: dict[str, tuple[Callable[[Frame], object], type[Exception], tuple[str, ...]]] = {
    "lazyframe": (lambda m: m.lazy(), TypeError, ("LazyFrame",)),
    "dict": (lambda m: m.to_dict(as_series=False), TypeError, ("dict",)),
    "series": (lambda m: m["match_id"], TypeError, ("Series",)),
    **{f"no_{k}": (_drop_key(k), ValueError, (k,)) for k in KEYS},
    "match_date_string": (
        lambda m: m.with_columns(pl.col("match_date").cast(pl.String)),
        ValueError,
        ("match_date", "Date"),
    ),
    "home_team_id_categorical": (
        lambda m: m.with_columns(pl.col("home_team_id").cast(pl.Categorical)),
        ValueError,
        ("home_team_id", "String"),
    ),
    "match_id_int": (
        lambda m: m.with_columns(pl.Series("match_id", [1, 2, 3, 4])),
        ValueError,
        ("match_id", "String"),
    ),
}


@pytest.mark.parametrize("case", list(ARG_ERRORS))
def test_validate_shots_matches_arg(case: str) -> None:
    mutate, error, needles = ARG_ERRORS[case]
    with pytest.raises(error) as info:
        schema.validate_shots(sc.shots_frame(), matches=mutate(sc.matches_frame()))
    assert not isinstance(info.value, schema.SchemaValidationError)
    assert all(n in str(info.value) for n in needles), str(info.value)
    if error is TypeError:
        with pytest.raises(TypeError, match=needles[0]):
            schema.validate_matches(mutate(sc.matches_frame()))


def test_matches_arg_needs_only_its_keys() -> None:
    """``matches`` is not validated by validate_shots: the six keys are enough, and a bad
    non-key column (a String gender with 'men') is not its concern."""
    matches = sc.with_value(sc.matches_frame(), "gender", 0, "men", dtype=pl.String())
    schema.validate_shots(sc.shots_frame(), matches=matches.select(KEYS))
    schema.validate_shots(sc.shots_frame(), matches=matches)


# --- cast_to_schema("matches") and the Date pre-check -----------------------------------------


def _stringly(m: Frame) -> Frame:
    """The matches as a loader builds them: Enums and dates as strings, scores as Int64."""
    return m.with_columns(
        pl.col("provider", "gender").cast(pl.String),
        pl.col("match_date").dt.strftime("%Y-%m-%d"),
        pl.col("home_score", "away_score").cast(pl.Int64),
        pl.lit("dropped").alias("note"),
    )


BAD_DATES = ["24-01-05", "2024-1-5", "2024-01-05\n", "2024-02-30", "10000-01-01", "05/01/2024"]


CAST_CASES = [
    ("home_score", pl.Series("home_score", [40000, None, 0, 2], dtype=pl.Int64), "cast",
     "40000", (0,)),
    ("away_score", pl.Series("away_score", [0.0, None, 0.0, 2.0]), "cast", "Float64", ()),
    ("home_score", pl.Series("home_score", [True, None, False, True]), "cast", "Boolean", ()),
    ("gender", pl.Series("gender", ["male", "male", "men", "female"]), "allowed", "'men'", (2,)),
    *(
        ("match_date", pl.Series("match_date", [bad, "2024-03-09", None, "2024-03-02"]), "cast",
         repr(bad), (0,))
        for bad in BAD_DATES
    ),
]  # fmt: skip


@pytest.mark.parametrize(
    ("column", "series", "check", "shown", "rows"),
    CAST_CASES,
    ids=["int64_40000", "float", "bool", "men", *(repr(b) for b in BAD_DATES)],
)
def test_cast_matches(
    column: str, series: pl.Series, check: str, shown: str, rows: tuple[int, ...]
) -> None:
    matches = sc.matches_frame()
    assert_frame_equal(schema.cast_to_schema(_stringly(matches), "matches"), matches)
    bad = _stringly(matches).with_columns(series)
    err = _err(lambda: schema.cast_to_schema(bad, "matches"))
    assert err.table == "matches"
    (failure,) = err.failures
    assert (failure.columns, failure.check, failure.rows) == ((column,), check, rows)
    assert shown in failure.values


@pytest.mark.parametrize("bad", BAD_DATES)
def test_cast_shots_date_is_iso_only(bad: str) -> None:
    """The pre-check covers every table's Date column: before 1.1 '24-01-05' became year 24."""
    shots = sc.shots_frame()
    text = shots.with_columns(pl.col("match_date").dt.strftime("%Y-%m-%d"))
    assert_frame_equal(schema.cast_to_schema(text), shots)
    with_null = sc.with_value(text, "match_date", 1, None)
    assert schema.cast_to_schema(with_null)["match_date"][1] is None  # null is not a cast failure
    err = _err(lambda: schema.cast_to_schema(sc.with_value(text, "match_date", 2, bad)))
    (failure,) = err.failures
    assert (failure.columns, failure.check, failure.rows) == (("match_date",), "cast", (2,))
    assert repr(bad) in failure.values


# --- Property ---------------------------------------------------------------------------------

_NAME = st.text(alphabet="abcdefghij KLMNOP/0123456789'", min_size=1, max_size=12)


@st.composite
def _match_rows(draw: st.DrawFn) -> list[dict[str, Any]]:
    rows = []
    for i in range(draw(st.integers(0, 6))):
        provider = draw(st.sampled_from(sc.PROVIDERS))
        p = "sb:" if provider == "statsbomb" else "ws:"
        home = draw(st.integers(0, 20))
        scored = draw(st.booleans())
        rows.append(
            {
                "match_id": f"{p}{i}",
                "provider": provider,
                "competition_id": f"{p}c{draw(st.integers(0, 3))}",
                "season_id": f"{p}s{draw(st.integers(0, 3))}",
                "match_date": draw(st.dates(dt.date(1900, 1, 1), dt.date(2100, 12, 31))),
                "home_team_id": f"{p}t{home}",
                "away_team_id": f"{p}t{home + draw(st.integers(1, 5))}",
                "competition_name": draw(_NAME),
                "season_name": draw(_NAME),
                "gender": draw(st.sampled_from(sc.GENDERS)),
                "home_score": draw(st.integers(0, 99)) if scored else None,
                "away_score": draw(st.integers(0, 99)) if scored else None,
                "has_360": draw(st.booleans()) if provider == "statsbomb" else None,
                "schema_version": schema.SCHEMA_VERSION,
            }
        )
    return rows


@settings(deadline=None, max_examples=40)
@given(_match_rows())
def test_property_valid_matches_validate_and_round_trip(rows: list[dict[str, Any]]) -> None:
    matches = sc.matches_frame(rows)
    assert_frame_equal(schema.validate_matches(matches), matches)
    assert_frame_equal(schema.cast_to_schema(_stringly(matches), "matches"), matches)
