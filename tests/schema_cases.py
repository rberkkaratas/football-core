"""Hand-written synthetic cases for the ``football_core.schema`` tests (C3).

Everything here is written from the C3 spec (``.claude/team/runs/C3/spec.md``), independently
of ``schema.py``, so the tests can hold the module to the spec instead of to itself. No value
is copied from a provider file: ids are obviously fake (``sb:00000000-...``, ``ws:9000000NN``)
and every number was typed by hand.

Frames are always built column by column with ``strict=True`` (spec §0.6): building canonical
frames from row dicts with ``schema=`` silently nulls invalid Enum values.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from football_core import pitch

# --- Vocabularies (spec §1; tuple order is the Enum physical order) -------------------------

PROVIDERS = ("statsbomb", "whoscored")
BODY_PARTS = ("right_foot", "left_foot", "head", "other")
SITUATIONS = (
    "open_play",
    "fast_break",
    "from_corner",
    "from_free_kick",
    "from_throw_in",
    "direct_free_kick",
    "direct_corner",
    "penalty",
    "kick_off",
)
ASSIST_TYPES = ("none", "pass", "cross", "through_ball", "cutback")
TECHNIQUES = (
    "normal",
    "volley",
    "half_volley",
    "lob",
    "backheel",
    "diving_header",
    "overhead_kick",
)
OUTCOMES = ("goal", "saved", "blocked", "post", "off_target")
VOCABULARIES: dict[str, tuple[str, ...]] = {
    "PROVIDERS": PROVIDERS,
    "BODY_PARTS": BODY_PARTS,
    "SITUATIONS": SITUATIONS,
    "ASSIST_TYPES": ASSIST_TYPES,
    "TECHNIQUES": TECHNIQUES,
    "OUTCOMES": OUTCOMES,
}
# Enum dtype constant name -> vocabulary constant name
ENUM_CONSTANTS = {
    "PROVIDER": "PROVIDERS",
    "BODY_PART": "BODY_PARTS",
    "SITUATION": "SITUATIONS",
    "ASSIST_TYPE": "ASSIST_TYPES",
    "TECHNIQUE": "TECHNIQUES",
    "OUTCOME": "OUTCOMES",
}

POLYGON = pl.List(pl.Array(pl.Float64, 2))

# --- Expected column tables (spec §2.1 / §2.2): name, dtype, nullable ------------------------

SHOT_COLUMNS: tuple[tuple[str, pl.DataType, bool], ...] = (
    ("shot_id", pl.String(), False),
    ("provider", pl.Enum(PROVIDERS), False),
    ("competition_id", pl.String(), False),
    ("season_id", pl.String(), False),
    ("match_id", pl.String(), False),
    ("match_date", pl.Date(), False),
    ("team_id", pl.String(), False),
    ("opponent_team_id", pl.String(), False),
    ("player_id", pl.String(), False),
    ("is_home", pl.Boolean(), False),
    ("period", pl.Int8(), False),
    ("minute", pl.Int16(), False),
    ("second", pl.Int8(), False),
    ("seconds_in_period", pl.Float64(), False),
    ("x", pl.Float64(), False),
    ("y", pl.Float64(), False),
    ("end_x", pl.Float64(), True),
    ("end_y", pl.Float64(), True),
    ("end_z", pl.Float64(), True),
    ("goal_y", pl.Float64(), True),
    ("goal_z", pl.Float64(), True),
    ("gk_x", pl.Float64(), True),
    ("gk_y", pl.Float64(), True),
    ("distance_m", pl.Float64(), False),
    ("angle_rad", pl.Float64(), False),
    ("visible_area", POLYGON, True),
    ("body_part", pl.Enum(BODY_PARTS), True),
    ("situation", pl.Enum(SITUATIONS), False),
    ("assist_type", pl.Enum(ASSIST_TYPES), True),
    ("technique", pl.Enum(TECHNIQUES), True),
    ("outcome", pl.Enum(OUTCOMES), False),
    ("is_goal", pl.Boolean(), False),
    ("is_own_goal", pl.Boolean(), False),
    ("is_penalty", pl.Boolean(), False),
    ("on_target", pl.Boolean(), False),
    ("first_time", pl.Boolean(), True),
    ("follows_dribble", pl.Boolean(), True),
    ("open_goal", pl.Boolean(), True),
    ("one_on_one", pl.Boolean(), True),
    ("big_chance", pl.Boolean(), True),
    ("deflected", pl.Boolean(), True),
    ("provider_xg", pl.Float64(), True),
    ("provider_psxg", pl.Float64(), True),
    ("assist_event_id", pl.String(), True),
    ("raw_type", pl.String(), False),
    ("raw_qualifiers", pl.String(), False),
    ("schema_version", pl.String(), False),
)
FREEZE_FRAME_COLUMNS: tuple[tuple[str, pl.DataType, bool], ...] = (
    ("shot_id", pl.String(), False),
    ("player_id", pl.String(), True),
    ("is_teammate", pl.Boolean(), False),
    ("is_keeper", pl.Boolean(), False),
    ("x", pl.Float64(), False),
    ("y", pl.Float64(), False),
)
SHOT_NAMES = tuple(name for name, _, _ in SHOT_COLUMNS)
FREEZE_FRAME_NAMES = tuple(name for name, _, _ in FREEZE_FRAME_COLUMNS)
SHOT_DTYPES = {name: dtype for name, dtype, _ in SHOT_COLUMNS}
FREEZE_FRAME_DTYPES = {name: dtype for name, dtype, _ in FREEZE_FRAME_COLUMNS}

# Bounds (spec §2.1 "Allowed / range"); None = unbounded on that side.
SHOT_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "minute": (0, 150),
    "second": (0, 59),
    "seconds_in_period": (0, 4500),
    "x": (0, 105),
    "y": (0, 68),
    "end_x": (0, 105),
    "end_y": (0, 68),
    "end_z": (0, None),
    "goal_y": (0, 68),
    "goal_z": (0, None),
    "gk_x": (0, 105),
    "gk_y": (0, 68),
    "distance_m": (0, None),
    "angle_rad": (0, math.pi),
    "provider_xg": (0, 1),
    "provider_psxg": (0, 1),
}
FREEZE_FRAME_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "x": (0, 105),
    "y": (0, 68),
}
SHOT_ID_COLUMNS = frozenset(
    {
        "shot_id",
        "competition_id",
        "season_id",
        "match_id",
        "team_id",
        "opponent_team_id",
        "player_id",
        "assist_event_id",
    }
)
FREEZE_FRAME_ID_COLUMNS = frozenset({"shot_id", "player_id"})

# --- Roles (spec §2.1 role sets) ------------------------------------------------------------
# These four sets are not touched by the open review issue on end_x / end_y / end_z.
ROLE_SETS_FIXED: dict[str, tuple[str, ...]] = {
    "identity": (
        "shot_id",
        "provider",
        "competition_id",
        "season_id",
        "match_id",
        "match_date",
        "team_id",
        "opponent_team_id",
        "player_id",
    ),
    "pre_shot": (
        "is_home",
        "period",
        "minute",
        "second",
        "seconds_in_period",
        "x",
        "y",
        "gk_x",
        "gk_y",
        "distance_m",
        "angle_rad",
        "visible_area",
        "body_part",
        "situation",
        "assist_type",
        "technique",
        "is_penalty",
        "first_time",
        "follows_dribble",
        "open_goal",
        "one_on_one",
    ),
    "restricted": ("big_chance",),
    "lineage": (
        "provider_xg",
        "provider_psxg",
        "assist_event_id",
        "raw_type",
        "raw_qualifiers",
        "schema_version",
    ),
}
BASE_ROLES = ("identity", "pre_shot", "post_shot", "outcome", "restricted", "lineage")
# Open issue 1 (football r3 M1): the ball's stop point encodes the outcome on P2's population.
STOP_POINT_COLUMNS = ("end_x", "end_y", "end_z")
OUTCOME_LABELS = ("outcome", "is_goal", "is_own_goal", "on_target")
P2_POST_SHOT = ("goal_y", "goal_z", "deflected")
FEATURE_ROLES = ("pre_shot", "post_shot", "restricted")
FREEZE_FRAME_ROLES = {
    "shot_id": "identity",
    "player_id": "identity",
    "is_teammate": "pre_shot",
    "is_keeper": "pre_shot",
    "x": "pre_shot",
    "y": "pre_shot",
}

# --- Rules (spec §4.1 / §4.2, in order) -----------------------------------------------------

SHOT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "ids_match_provider",
        (
            "shot_id",
            "provider",
            "competition_id",
            "season_id",
            "match_id",
            "team_id",
            "opponent_team_id",
            "player_id",
            "assist_event_id",
        ),
    ),
    ("distinct_teams", ("shot_id", "team_id", "opponent_team_id")),
    ("distance_matches_pitch", ("shot_id", "x", "y", "distance_m")),
    ("angle_matches_pitch", ("shot_id", "x", "y", "angle_rad")),
    ("is_goal_matches_outcome", ("shot_id", "is_goal", "outcome")),
    ("own_goal_is_goal", ("shot_id", "is_own_goal", "is_goal")),
    ("on_target_matches_outcome", ("shot_id", "on_target", "outcome", "is_own_goal")),
    ("blocked_has_no_goal_mouth", ("shot_id", "outcome", "goal_y", "goal_z")),
    ("penalty_matches_situation", ("shot_id", "is_penalty", "situation")),
    ("shootout_is_penalty", ("shot_id", "period", "is_penalty")),
    ("minute_within_period", ("shot_id", "period", "minute")),
    ("clock_consistent", ("shot_id", "period", "minute", "second", "seconds_in_period")),
    ("unassisted_has_no_assist_event", ("shot_id", "assist_type", "assist_event_id")),
)
FREEZE_FRAME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ids_share_prefix", ("shot_id", "player_id")),
    ("unique_player_per_shot", ("shot_id", "player_id")),
    ("one_keeper_per_side", ("shot_id", "is_teammate", "is_keeper")),
)

# --- Synthetic fixture rows (spec §7) -------------------------------------------------------

SB_SHOT_1 = "sb:00000000-0000-0000-0000-000000000001"
SB_SHOT_2 = "sb:00000000-0000-0000-0000-000000000002"
WS_SHOT_3 = "ws:900000003"
WS_SHOT_4 = "ws:900000004"
SB_SHOT_5 = "sb:00000000-0000-0000-0000-000000000005"
SHOT_IDS = (SB_SHOT_1, SB_SHOT_2, WS_SHOT_3, WS_SHOT_4, SB_SHOT_5)
SHOOTER_1 = "sb:90001"
SQUARE = [[80.0, 10.0], [105.0, 10.0], [105.0, 58.0], [80.0, 58.0]]


def geometry(x: float, y: float) -> dict[str, float]:
    return {
        "x": x,
        "y": y,
        "distance_m": float(pitch.shot_distance(x, y)),
        "angle_rad": float(pitch.opening_angle(x, y)),
    }


def _false_flags() -> dict[str, Any]:
    # StatsBomb: the attributes are defined for every shot, absent -> False (spec §2.0).
    return {
        "first_time": False,
        "follows_dribble": False,
        "open_goal": False,
        "one_on_one": False,
        "big_chance": None,  # StatsBomb has no big-chance attribute
        "deflected": False,
    }


def base_shot_rows() -> list[dict[str, Any]]:
    """The five §7 fixture rows, as fresh dicts (index 0..4 = fixture rows 1..5)."""
    sb_goal = {
        "shot_id": SB_SHOT_1,
        "provider": "statsbomb",
        "competition_id": "sb:9001",
        "season_id": "sb:9002",
        "match_id": "sb:9000001",
        "match_date": dt.date(2024, 3, 2),
        "team_id": "sb:901",
        "opponent_team_id": "sb:902",
        "player_id": SHOOTER_1,
        "is_home": True,
        "period": 1,
        "minute": 23,
        "second": 15,
        "seconds_in_period": 1395.4,
        **geometry(94.5, 36.0),
        "end_x": 105.0,
        "end_y": 35.2,
        "end_z": 0.6,
        "goal_y": 35.2,
        "goal_z": 0.6,
        "gk_x": 103.8,
        "gk_y": 34.5,
        "visible_area": [list(v) for v in SQUARE],
        "body_part": "right_foot",
        "situation": "open_play",
        "assist_type": "cross",
        "technique": "normal",
        "outcome": "goal",
        "is_goal": True,
        "is_own_goal": False,
        "is_penalty": False,
        "on_target": True,
        **_false_flags(),
        "provider_xg": 0.31,
        "provider_psxg": None,
        "assist_event_id": "sb:00000000-0000-0000-0000-0000000000a1",
        "raw_type": "Shot",
        "raw_qualifiers": '{"outcome": {"name": "Goal"}, "first_time": false}',
        "schema_version": "1.0",
    }
    sb_penalty_saved = {
        **sb_goal,
        "shot_id": SB_SHOT_2,
        "team_id": "sb:902",
        "opponent_team_id": "sb:901",
        "player_id": "sb:90201",
        "is_home": False,
        "period": 2,
        "minute": 67,
        "second": 3,
        "seconds_in_period": 1323.0,
        **geometry(94.0, 34.0),
        # Save point; goal_y / goal_z are the provider's estimate (spec §2.0), never nulled.
        "end_x": 104.2,
        "end_y": 33.0,
        "end_z": 0.5,
        "goal_y": 33.0,
        "goal_z": 0.5,
        "gk_x": 104.6,
        "gk_y": 34.0,
        "visible_area": None,
        "body_part": "left_foot",
        "situation": "penalty",
        "assist_type": "none",
        "technique": "normal",
        "outcome": "saved",
        "is_goal": False,
        "is_penalty": True,
        "on_target": True,
        "provider_xg": 0.76,
        "assist_event_id": None,
        "raw_qualifiers": '{"type": {"name": "Penalty"}}',
    }
    ws_blocked_header = {
        "shot_id": WS_SHOT_3,
        "provider": "whoscored",
        "competition_id": "ws:9101",
        "season_id": "ws:9102",
        "match_id": "ws:9100001",
        "match_date": dt.date(2024, 3, 9),
        "team_id": "ws:911",
        "opponent_team_id": "ws:912",
        "player_id": "ws:91001",
        "is_home": False,
        "period": 1,
        "minute": 40,
        "second": 30,
        "seconds_in_period": 2430.0,
        **geometry(99.0, 30.0),
        "end_x": 100.5,  # block location
        "end_y": 31.0,
        "end_z": None,
        "goal_y": None,  # placeholder after a block -> null (blocked_has_no_goal_mouth)
        "goal_z": None,
        "gk_x": None,
        "gk_y": None,
        "visible_area": None,
        "body_part": "head",
        "situation": "from_corner",
        "assist_type": "cross",
        "technique": None,
        "outcome": "blocked",
        "is_goal": False,
        "is_own_goal": False,
        "is_penalty": False,
        "on_target": False,
        "first_time": False,
        "follows_dribble": None,
        "open_goal": None,
        "one_on_one": False,
        "big_chance": False,
        "deflected": None,
        "provider_xg": None,
        "provider_psxg": None,
        "assist_event_id": "ws:900000002",
        "raw_type": "SavedShot",
        "raw_qualifiers": '[{"type": "Blocked"}, {"type": "Head"}]',
        "schema_version": "1.0",
    }
    ws_own_goal = {
        **ws_blocked_header,
        "shot_id": WS_SHOT_4,
        "team_id": "ws:912",
        "opponent_team_id": "ws:911",
        "player_id": "ws:92001",
        "is_home": True,
        "period": 2,
        "minute": 78,
        "second": 0,
        "seconds_in_period": 1980.0,
        # Frame in which the goal the ball entered is at x = 105 (spec §4.1).
        **geometry(101.0, 33.0),
        "end_x": None,
        "end_y": None,
        "end_z": None,
        "goal_y": 35.0,
        "goal_z": 0.3,
        "body_part": None,
        "situation": "open_play",
        "assist_type": None,
        "technique": None,
        "outcome": "goal",
        "is_goal": True,
        "is_own_goal": True,
        "on_target": False,
        "assist_event_id": None,
        "raw_type": "Goal",
        "raw_qualifiers": '[{"type": "OwnGoal"}]',
    }
    sb_shootout_miss = {
        **sb_goal,
        "shot_id": SB_SHOT_5,
        "match_id": "sb:9000002",
        "match_date": dt.date(2024, 3, 16),
        "team_id": "sb:901",
        "opponent_team_id": "sb:903",
        "player_id": "sb:90002",
        "is_home": True,
        "period": 5,
        "minute": 121,
        "second": 10,
        "seconds_in_period": 10.0,
        **geometry(94.0, 34.0),
        "end_x": 105.0,
        "end_y": 38.5,
        "end_z": 1.2,
        "goal_y": 38.5,  # outside the left post (37.66): off target, still a valid point
        "goal_z": 1.2,
        "gk_x": 105.0,
        "gk_y": 34.0,
        "visible_area": None,
        "body_part": "right_foot",
        "situation": "penalty",
        "assist_type": "none",
        "technique": "normal",
        "outcome": "off_target",
        "is_goal": False,
        "is_penalty": True,
        "on_target": False,
        "provider_xg": 0.78,
        "assist_event_id": None,
        "raw_qualifiers": '{"type": {"name": "Penalty"}}',
    }
    return [sb_goal, sb_penalty_saved, ws_blocked_header, ws_own_goal, sb_shootout_miss]


def base_freeze_frame_rows() -> list[dict[str, Any]]:
    """Four players visible at shot 1 (the shooter is not in the frame)."""
    return [
        {
            "shot_id": SB_SHOT_1,
            "player_id": "sb:90101",
            "is_teammate": False,
            "is_keeper": True,
            "x": 103.8,
            "y": 34.5,
        },
        {
            "shot_id": SB_SHOT_1,
            "player_id": "sb:90102",
            "is_teammate": False,
            "is_keeper": False,
            "x": 100.0,
            "y": 36.5,
        },
        {
            "shot_id": SB_SHOT_1,
            "player_id": "sb:90103",
            "is_teammate": False,
            "is_keeper": False,
            "x": 98.0,
            "y": 30.0,
        },
        {
            "shot_id": SB_SHOT_1,
            "player_id": None,
            "is_teammate": True,
            "is_keeper": False,
            "x": 97.0,
            "y": 40.0,
        },
    ]


# --- Frame builders and one-cell mutation helpers -------------------------------------------


def build_frame(
    rows: Sequence[Mapping[str, Any]], columns: tuple[tuple[str, pl.DataType, bool], ...]
) -> pl.DataFrame:
    """Build a frame column-wise with strict dtypes (never from row dicts with ``schema=``)."""
    return pl.DataFrame(
        [
            pl.Series(name, [row[name] for row in rows], dtype=dtype, strict=True)
            for name, dtype, _ in columns
        ]
    )


def shots_frame(rows: Sequence[Mapping[str, Any]] | None = None) -> pl.DataFrame:
    return build_frame(base_shot_rows() if rows is None else rows, SHOT_COLUMNS)


def freeze_frames_frame(rows: Sequence[Mapping[str, Any]] | None = None) -> pl.DataFrame:
    return build_frame(base_freeze_frame_rows() if rows is None else rows, FREEZE_FRAME_COLUMNS)


def with_value(
    df: pl.DataFrame, column: str, row: int, value: Any, dtype: pl.DataType | None = None
) -> pl.DataFrame:
    """Replace one cell, keeping the column's dtype (or switching it to ``dtype``)."""
    values = df[column].to_list()
    values[row] = value
    target = df.schema[column] if dtype is None else dtype
    return df.with_columns(pl.Series(column, values, dtype=target, strict=True))


def with_values(df: pl.DataFrame, row: int, **cells: Any) -> pl.DataFrame:
    for column, value in cells.items():
        df = with_value(df, column, row, value)
    return df


def with_location(df: pl.DataFrame, row: int, x: float, y: float) -> pl.DataFrame:
    """Move a shot and keep ``distance_m`` / ``angle_rad`` consistent with ``pitch``."""
    return with_values(df, row, **geometry(x, y))


def replace_column(df: pl.DataFrame, series: pl.Series) -> pl.DataFrame:
    """Swap a column for ``series`` (any dtype) in the same position."""
    return df.with_columns(series)


# --- Hostile canonical-dtype frames for totality tests (spec §4.0) --------------------------

_NAN, _INF = float("nan"), float("inf")
_HOSTILE: dict[str, list[Any]] = {
    "Float64": [None, _NAN, _INF, -_INF, -0.0, 1e308, -1.0, 105.5, 0.0, 34.0],
    "Int8": [None, 0, 6, 127, -128, 5, 1, 2, 3, 4],
    "Int16": [None, 0, 32767, -32768, 150, 151, 1000, 45, 90, 121],
    "String": [None, "", "sb:", "ws:1", "xx", "sb:1", "opta:1", "[", "{}", "\n"],
    "Boolean": [None, True, False, True, None, False, True, False, None, True],
    "Date": [None, dt.date(1900, 1, 1), dt.date(2024, 2, 29), None, dt.date(2100, 12, 31)],
    "Polygon": [None, [], [[_NAN, _NAN]], [None], [[1.0, None]], [list(v) for v in SQUARE]],
}


def _hostile_values(dtype: pl.DataType, n: int, offset: int) -> list[Any]:
    if isinstance(dtype, pl.Enum):
        pool: list[Any] = [None, *dtype.categories.to_list()]
    elif dtype == POLYGON:
        pool = _HOSTILE["Polygon"]
    else:
        pool = _HOSTILE[str(dtype)]
    return [pool[(i + offset) % len(pool)] for i in range(n)]


def hostile_frame(columns: tuple[tuple[str, pl.DataType, bool], ...], n: int = 12) -> pl.DataFrame:
    """Canonical dtypes, values chosen to hurt: nulls, NaN, ±inf, out-of-vocabulary ints, ''."""
    return pl.DataFrame(
        [
            pl.Series(name, _hostile_values(dtype, n, offset), dtype=dtype, strict=True)
            for offset, (name, dtype, _) in enumerate(columns)
        ]
    )
