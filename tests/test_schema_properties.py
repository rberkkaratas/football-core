"""Properties of the schema validators over generated frames (C3; spec §4.0, §7)."""

from __future__ import annotations

from typing import Any

import polars as pl
import schema_cases as sc
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from football_core import schema

SETTINGS = settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
PERIOD_START = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}

canon_x = st.floats(min_value=0.0, max_value=105.0, allow_nan=False)
canon_y = st.floats(min_value=0.0, max_value=68.0, allow_nan=False)


def _validates(df: pl.DataFrame) -> None:
    try:
        schema.validate_shots(df)
    except schema.SchemaValidationError as err:  # pragma: no cover - report the frame
        raise AssertionError(f"{err}\n{df.row(0, named=True)}") from err


# --- Valid frames always validate -----------------------------------------------------------


@SETTINGS
@given(x=canon_x, y=canon_y)
def test_any_pitch_location_validates(x: float, y: float) -> None:
    _validates(sc.with_location(sc.shots_frame(), 0, x, y))


@SETTINGS
@given(x=canon_x, y=canon_y, offset=st.floats(min_value=1e-6, max_value=50.0))
def test_distance_offset_fails(x: float, y: float, offset: float) -> None:
    df = sc.with_location(sc.shots_frame(), 0, x, y)
    df = sc.with_value(df, "distance_m", 0, df["distance_m"][0] + offset)
    try:
        schema.validate_shots(df)
    except schema.SchemaValidationError as err:
        failures = [f for f in err.failures if f.check == "distance_matches_pitch"]
        assert len(failures) == 1 and failures[0].rows == (0,)
        assert {f.check for f in err.failures} == {"distance_matches_pitch"}
    else:
        raise AssertionError("distance offset was accepted")


@st.composite
def consistent_shot(draw: st.DrawFn) -> dict[str, Any]:
    row = sc.base_shot_rows()[0]
    provider = draw(st.sampled_from(sc.PROVIDERS))
    prefix = {"statsbomb": "sb:", "whoscored": "ws:"}[provider]
    ids = ("shot_id", "competition_id", "season_id", "match_id", "team_id", "player_id")
    for name in ids:
        row[name] = prefix + row[name].split(":", 1)[1]
    row["opponent_team_id"] = prefix + "999"
    row["provider"] = provider

    outcome = draw(st.sampled_from(sc.OUTCOMES))
    situation = draw(st.sampled_from(sc.SITUATIONS))
    own_goal = outcome == "goal" and draw(st.booleans())
    is_penalty = situation == "penalty"
    period = draw(st.sampled_from([1, 2, 3, 4, 5] if is_penalty else [1, 2, 3, 4]))
    if period == 5:
        minute = draw(st.integers(120, 150))
        second = draw(st.integers(0, 59))
        seconds = draw(st.floats(0.0, 4500.0))
    else:
        minute = PERIOD_START[period] + draw(st.integers(0, 45))
        second = draw(st.integers(0, 59))
        seconds = (minute - PERIOD_START[period]) * 60 + second + draw(st.floats(0.0, 0.99))
    assist_type = draw(st.none() | st.sampled_from(sc.ASSIST_TYPES))
    x, y = draw(canon_x), draw(canon_y)
    row |= sc.geometry(x, y)
    row |= {
        "outcome": outcome,
        "situation": situation,
        "body_part": draw(st.none() | st.sampled_from(sc.BODY_PARTS)),
        "technique": draw(st.none() | st.sampled_from(sc.TECHNIQUES)),
        "assist_type": assist_type,
        "assist_event_id": None if assist_type == "none" else prefix + "777",
        "is_goal": outcome == "goal",
        "is_own_goal": own_goal,
        "on_target": outcome in ("goal", "saved") and not own_goal,
        "is_penalty": is_penalty,
        "period": period,
        "minute": minute,
        "second": second,
        "seconds_in_period": seconds,
        "big_chance": draw(st.none() | st.booleans()),
        "deflected": draw(st.none() | st.booleans()),
    }
    if outcome == "blocked":
        row |= {"goal_y": None, "goal_z": None}
    return row


@SETTINGS
@given(row=consistent_shot())
def test_any_consistent_category_combination_validates(row: dict[str, Any]) -> None:
    _validates(sc.shots_frame([row]))


@SETTINGS
@given(rows=st.lists(consistent_shot(), min_size=1, max_size=4))
def test_unique_consistent_rows_validate_together(rows: list[dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        row["shot_id"] = f"{row['shot_id']}-{i}"
    _validates(sc.shots_frame(rows))


@SETTINGS
@given(x=canon_x, y=canon_y, keeper=st.booleans(), teammate=st.booleans())
def test_any_freeze_frame_location_validates(
    x: float, y: float, keeper: bool, teammate: bool
) -> None:
    rows = [
        {
            "shot_id": sc.SB_SHOT_1,
            "player_id": "sb:95000",
            "is_teammate": teammate,
            "is_keeper": keeper,
            "x": x,
            "y": y,
        }
    ]
    schema.validate_freeze_frames(sc.freeze_frames_frame(rows), shots=sc.shots_frame())


# --- Arbitrary values never escape as anything but SchemaValidationError --------------------

_float = (
    st.none()
    | st.floats(allow_nan=True, allow_infinity=True)
    | st.sampled_from([0.0, -0.0, 105.0, 68.0, 1e-300, -1e-300])
)
_string = (
    st.none()
    | st.text(max_size=12)
    | st.sampled_from(
        ["", "sb:1", "ws:1", "opta:1", "1.0", "[]", "{}", "[NaN]", "[1, Infinity]", "[" * 100_000]
    )
)
_vertex = st.none() | st.lists(st.none() | st.floats(allow_nan=True), min_size=2, max_size=2)
_polygon = st.none() | st.lists(_vertex, max_size=5)


def _values(dtype: pl.DataType) -> st.SearchStrategy[Any]:
    if isinstance(dtype, pl.Enum):
        return st.none() | st.sampled_from(dtype.categories.to_list())
    if dtype == sc.POLYGON:
        return _polygon
    return {
        "Float64": _float,
        "Int8": st.none() | st.integers(-128, 127),
        "Int16": st.none() | st.integers(-32768, 32767),
        "String": _string,
        "Boolean": st.none() | st.booleans(),
        "Date": st.none() | st.dates(),
    }[str(dtype)]


@st.composite
def hostile_frames(
    draw: st.DrawFn, columns: tuple[tuple[str, pl.DataType, bool], ...]
) -> pl.DataFrame:
    n = draw(st.integers(0, 4))
    return pl.DataFrame(
        [
            pl.Series(name, draw(st.lists(_values(dtype), min_size=n, max_size=n)), dtype=dtype)
            for name, dtype, _ in columns
        ]
    )


def _assert_clean(err: schema.SchemaValidationError) -> None:
    assert err.failures
    for failure in err.failures:
        assert failure.columns, failure
        assert all(isinstance(v, str) for v in failure.values)
    assert len(str(err).splitlines()) == len(err.failures) + 1


@SETTINGS
@given(df=hostile_frames(sc.SHOT_COLUMNS))
def test_validate_only_raises_schema_error(df: pl.DataFrame) -> None:
    try:
        out = schema.validate_shots(df)
    except schema.SchemaValidationError as err:
        _assert_clean(err)
    else:
        assert out.schema == schema.SHOT_SCHEMA


@SETTINGS
@given(
    df=hostile_frames(sc.FREEZE_FRAME_COLUMNS),
    shots=st.none() | st.just(sc.shots_frame()) | hostile_frames(sc.SHOT_COLUMNS),
)
def test_validate_freeze_frames_only_raises_schema_error(
    df: pl.DataFrame, shots: pl.DataFrame | None
) -> None:
    try:
        out = schema.validate_freeze_frames(df, shots=shots)
    except schema.SchemaValidationError as err:
        _assert_clean(err)
    else:
        assert out.schema == schema.FREEZE_FRAME_SCHEMA


@SETTINGS
@given(df=hostile_frames(sc.SHOT_COLUMNS))
def test_rules_are_total_on_generated_frames(df: pl.DataFrame) -> None:
    for rule in schema.SHOT_RULES:
        mask = rule.check(df)
        assert isinstance(mask, pl.Series) and mask.dtype == pl.Boolean, rule.name
        assert mask.len() == df.height and mask.null_count() == 0, rule.name
