"""Canonical shot, freeze-frame and match tables: polars dtypes, pandera models and roles.

Every loader (StatsBomb, WhoScored / Opta) writes the same three tables, so models, charts and
snapshots never care where a shot came from. This module is the contract between them. The
matches table holds one row per match: the keys every shot repeats, the labels the cards
print, and match-level inputs known before kick-off (``gender``). ``validate_shots(df,
matches=...)`` checks that each shot agrees with its match.

**One source of truth.** Each table is a tuple of :class:`ColumnSpec` records. The polars
schema, the pandera model, the docs tables and the leakage roles are all derived from it.

**Roles.** Every column carries a role, so a model can build its feature allow-list from the
schema itself (``column_names("shots", roles="pre_shot")``) and a new column is never a
feature by accident:

- ``identity``: keys and grouping (who, which match, when). Never a feature; used for joins
  and cross-validation groups.
- ``pre_shot``: fully determined at the instant the ball is struck, **both the value and
  whether it is null**. Usable by xG and everything after it.
- ``post_shot``: ball flight after the strike (goal-mouth point, deflection). xGOT / PSxG
  only, never xG.
- ``outcome``: labels, population definitions and where the ball was stopped (``end_x``,
  ``end_y``, ``end_z``: the save, the block or the line). Never a feature of the same shot:
  the stop point separates goals from saves on the on-target population.
- ``restricted``: pre-shot in time but an annotator judgement that leaks the outcome
  (``big_chance``). Labelled ablations only.
- ``lineage``: provenance, provider model outputs, raw payloads and the schema version.
  Never a feature.

**Missing values have one meaning.** ``True`` = the provider tagged the attribute; ``False`` =
the provider records it for that competition-season and did not tag it; ``null`` = not
recorded. Loaders never fill a default, and null is the only missing marker (no NaN).

**Validation runs in two passes** and raises one :class:`SchemaValidationError` listing every
problem, one line per (column, check), naming the column and the bad values. Pass 1 checks
structure (missing / extra columns, dtypes, invalid values of mis-typed enum columns). Pass 2
runs the pandera column checks and then the cross-column rules, which this module evaluates
itself. Both passes pin pandera's config, so ``PANDERA_VALIDATION_ENABLED`` and friends cannot
switch validation off. Checks and rules are total: bad data never makes them raise, and a
check that does raise is an internal error (``RuntimeError``), never a schema message.

**No silent coercion.** ``validate_*`` never casts. Build frames column-wise with strict
dtypes, or with :func:`cast_to_schema`: ``pl.DataFrame(row_dicts, schema=...)`` silently
turns an invalid enum value into null and truncates ``1.5`` to ``1`` in an integer column.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import partial
from types import MappingProxyType
from typing import Final, Literal, get_args

import numpy as np
import pandera.polars as pa
import polars as pl
from pandera.api.polars.types import PolarsData
from pandera.config import ValidationDepth, config_context
from pandera.errors import SchemaErrors

from football_core import pitch

Table = Literal["shots", "freeze_frames", "matches"]  # TABLES order is message order
Role = Literal["identity", "pre_shot", "post_shot", "outcome", "restricted", "lineage"]
RuleCheck = Callable[[pl.DataFrame], pl.Series]

SCHEMA_VERSION: Final[str] = "1.2"
TABLES: Final[tuple[Table, ...]] = get_args(Table)
ROLES: Final[tuple[Role, ...]] = get_args(Role)

# Category vocabularies. Tuple order is the Enum physical order: never reorder, only append
# with a MAJOR version bump (polars Enums with different categories do not concat).
PROVIDERS: Final = ("statsbomb", "whoscored")
BODY_PARTS: Final = ("right_foot", "left_foot", "head", "other")
SITUATIONS: Final = (
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
ASSIST_TYPES: Final = ("none", "pass", "cross", "through_ball", "cutback")
TECHNIQUES: Final = (
    "normal",
    "volley",
    "half_volley",
    "lob",
    "backheel",
    "diving_header",
    "overhead_kick",
)
OUTCOMES: Final = ("goal", "saved", "blocked", "post", "off_target")
GENDERS: Final = ("male", "female")  # male first: the indicator reference

PROVIDER: Final = pl.Enum(PROVIDERS)
BODY_PART: Final = pl.Enum(BODY_PARTS)
SITUATION: Final = pl.Enum(SITUATIONS)
ASSIST_TYPE: Final = pl.Enum(ASSIST_TYPES)
TECHNIQUE: Final = pl.Enum(TECHNIQUES)
OUTCOME: Final = pl.Enum(OUTCOMES)
GENDER: Final = pl.Enum(GENDERS)
POLYGON: Final = pl.List(pl.Array(pl.Float64(), 2))

ID_PREFIXES: Final[Mapping[str, str]] = MappingProxyType({"statsbomb": "sb:", "whoscored": "ws:"})
ID_PATTERN: Final[str] = r"^(sb|ws):\S+$"
ON_TARGET_OUTCOMES: Final[frozenset[str]] = frozenset({"goal", "saved"})
PERIOD_START_MINUTE: Final[Mapping[int, int]] = MappingProxyType(
    {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
)
DISTANCE_TOLERANCE: Final[float] = 1e-9  # metres; distance_m vs pitch.shot_distance
ANGLE_TOLERANCE: Final[float] = 1e-9  # radians; angle_rad vs pitch.opening_angle
MAX_EXAMPLES: Final[int] = 5  # values / rows shown per failure line

# Failure.check vocabulary, in message order within one column (rule names come after).
CHECKS: Final[tuple[str, ...]] = (
    "missing",
    "extra",
    "dtype",
    "allowed",
    "nullable",
    "range",
    "finite",
    "unique",
    "id_format",
    "non_empty",
    "json",
    "schema_version",
    "polygon",
    "cast",
)
_PANDERA_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {"not_nullable": "nullable", "field_uniqueness": "unique"}
)
_LONG_STRING: Final = 60  # longer strings are cut in messages (a raw payload can be huge)


@dataclass(frozen=True)
class ColumnSpec:
    """One column of a canonical table.

    Attributes
    ----------
    name : str
        Column name.
    dtype : polars.DataType
        Canonical dtype, always an instance (``pl.Float64()``, not ``pl.Float64``).
    role : str
        Leakage role, one of ``ROLES``.
    nullable : bool
        Whether null is allowed (null means "not recorded", never a default).
    description : str
        One sentence, rendered in the docs.
    ge, le : float or None
        Inclusive lower / upper bound for numeric columns.
    allowed : tuple of int or None
        Integer vocabulary (``period``).
    is_id : bool
        Provider-prefixed id: matches ``ID_PATTERN`` and the provider's prefix.
    unique : bool
        Values must be unique (nulls are reported as nulls, not duplicates).
    """

    name: str
    dtype: pl.DataType
    role: Role
    nullable: bool
    description: str
    ge: float | None = None
    le: float | None = None
    allowed: tuple[int, ...] | None = None
    is_id: bool = False
    unique: bool = False


@dataclass(frozen=True)
class FrameRule:
    """A row-wise rule over several columns.

    Attributes
    ----------
    name : str
        Rule name, used as ``Failure.check``.
    columns : tuple of str
        Columns shown in the failure message, the table's key first.
    description : str
        The rule in words.
    check : callable
        ``check(df)`` returns a Boolean Series of length ``df.height`` without nulls; True
        means the row passes. It is total: it never raises on a frame with canonical dtypes,
        and a null, non-finite or out-of-domain input makes the row pass (the column checks
        report those values).
    """

    name: str
    columns: tuple[str, ...]
    description: str
    check: RuleCheck


@dataclass(frozen=True)
class Failure:
    """One failed check on one column (or one rule), with example values.

    Attributes
    ----------
    table : str
        One of ``TABLES``.
    columns : tuple of str
        ``(column,)`` for a column check, the rule's columns for a rule. Never empty.
    check : str
        A name from ``CHECKS`` or a rule name.
    values : tuple of str
        Up to ``MAX_EXAMPLES`` distinct bad values, rendered. For a rule: the rule's columns
        in the first failing row. For ``dtype`` and ``cast``: the received dtype first, then
        (``cast`` only) the values that could not be cast.
    rows : tuple of int
        Up to ``MAX_EXAMPLES`` 0-based row positions; empty for structural failures.
    count : int
        Total number of failing rows (1 for structural failures).
    """

    table: Table
    columns: tuple[str, ...]
    check: str
    values: tuple[str, ...]
    rows: tuple[int, ...]
    count: int


class SchemaValidationError(ValueError):
    """Raised when a frame does not match a canonical table schema.

    ``str(err)`` is a header line followed by one line per failure, each naming the column
    (or rule) and the bad values.

    Attributes
    ----------
    table : str
        The table that failed.
    failures : tuple of Failure
        Every problem found, in message order.
    """

    table: Table
    failures: tuple[Failure, ...]

    def __init__(self, table: Table, failures: Collection[Failure]) -> None:
        self.table = table
        self.failures = tuple(failures)
        lines = [f"{table}: {len(self.failures)} schema problem(s) (schema {SCHEMA_VERSION})"]
        lines += [f"  - {_format_failure(failure)}" for failure in self.failures]
        super().__init__("\n".join(lines))

    def __reduce__(self) -> tuple[type[SchemaValidationError], tuple[Table, tuple[Failure, ...]]]:
        return type(self), (self.table, self.failures)


# --- Column tables ---------------------------------------------------------------------------

_X: Final = (0.0, pitch.PITCH_LENGTH)
_Y: Final = (0.0, pitch.PITCH_WIDTH)

# fmt: off
SHOT_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec("shot_id", pl.String(), "identity", False,
               "Provider event id, the primary key (Opta: the global id, never eventId).",
               is_id=True, unique=True),
    ColumnSpec("provider", PROVIDER, "identity", False,
               "Data source (whoscored uses the opta pitch grid)."),
    ColumnSpec("competition_id", pl.String(), "identity", False,
               "Competition id.", is_id=True),
    ColumnSpec("season_id", pl.String(), "identity", False, "Season id.", is_id=True),
    ColumnSpec("match_id", pl.String(), "identity", False,
               "Match id, the group key for match-level cross-validation.", is_id=True),
    ColumnSpec("match_date", pl.Date(), "identity", False, "Match date."),
    ColumnSpec("team_id", pl.String(), "identity", False,
               "Team of the player who struck the ball (also for own goals).", is_id=True),
    ColumnSpec("opponent_team_id", pl.String(), "identity", False,
               "The other team.", is_id=True),
    ColumnSpec("player_id", pl.String(), "identity", False,
               "Player who struck the ball.", is_id=True),
    ColumnSpec("gk_id", pl.String(), "identity", True,
               "Goalkeeper of the defending team on the pitch at the strike (lineup state); "
               "null for own goals.", is_id=True),
    ColumnSpec("is_home", pl.Boolean(), "pre_shot", False,
               "Whether team_id is the home side (nominal at neutral venues)."),
    ColumnSpec("period", pl.Int8(), "pre_shot", False,
               "Period: 1-2 regulation, 3-4 extra time, 5 penalty shoot-out.",
               allowed=(1, 2, 3, 4, 5)),
    ColumnSpec("minute", pl.Int16(), "pre_shot", False,
               "Provider match-clock minute (period 2 starts at 45).", ge=0.0, le=150.0),
    ColumnSpec("second", pl.Int8(), "pre_shot", False,
               "Second within the minute.", ge=0.0, le=59.0),
    ColumnSpec("seconds_in_period", pl.Float64(), "pre_shot", False,
               "Seconds since the period kicked off.", ge=0.0, le=4500.0),
    ColumnSpec("x", pl.Float64(), "pre_shot", False, "Shot location, metres.", *_X),
    ColumnSpec("y", pl.Float64(), "pre_shot", False, "Shot location, metres.", *_Y),
    ColumnSpec("end_x", pl.Float64(), "outcome", True,
               "Where the ball was stopped (save, block or line); null when not given.", *_X),
    ColumnSpec("end_y", pl.Float64(), "outcome", True, "As end_x.", *_Y),
    ColumnSpec("end_z", pl.Float64(), "outcome", True,
               "Height at the end location, metres.", ge=0.0),
    ColumnSpec("goal_y", pl.Float64(), "post_shot", True,
               "Where the ball crossed or would cross the goal line, pitch-width metres (posts "
               "30.34 / 37.66; `pitch.to_goal_frame` for the goal frame); StatsBomb stop points "
               "more than one grid cell before the line projected by "
               "`pitch.project_to_goal_line`; null if blocked.", *_Y),
    ColumnSpec("goal_z", pl.Float64(), "post_shot", True,
               "Height on the goal plane, metres (crossbar 2.44); the recorded height at a "
               "StatsBomb stop point; null if blocked.", ge=0.0),
    ColumnSpec("gk_x", pl.Float64(), "pre_shot", True,
               "Goalkeeper defending the target goal at the strike.", *_X),
    ColumnSpec("gk_y", pl.Float64(), "pre_shot", True, "As gk_x.", *_Y),
    ColumnSpec("distance_m", pl.Float64(), "pre_shot", False,
               "pitch.shot_distance(x, y), stored so nobody recomputes it differently.",
               ge=0.0),
    ColumnSpec("angle_rad", pl.Float64(), "pre_shot", False,
               "pitch.opening_angle(x, y).", ge=0.0, le=math.pi),
    ColumnSpec("visible_area", POLYGON, "pre_shot", True,
               "StatsBomb 360 camera polygon in metres; null without a 360 frame."),
    ColumnSpec("body_part", BODY_PART, "pre_shot", True, "Body part used."),
    ColumnSpec("situation", SITUATION, "pre_shot", False,
               "Open play, fast break, set-piece phase or direct set piece."),
    ColumnSpec("assist_type", ASSIST_TYPE, "pre_shot", True,
               "none = recorded and unassisted; null = the provider cannot tell."),
    ColumnSpec("technique", TECHNIQUE, "pre_shot", True,
               "Strike technique; normal only where the full vocabulary is recorded."),
    ColumnSpec("outcome", OUTCOME, "outcome", False, "What happened to the shot."),
    ColumnSpec("is_goal", pl.Boolean(), "outcome", False,
               "The ball went in (own goals included)."),
    ColumnSpec("is_own_goal", pl.Boolean(), "outcome", False,
               "Own goal (xG trains on the other shots)."),
    ColumnSpec("is_penalty", pl.Boolean(), "pre_shot", False,
               "Penalty kick (situation == penalty)."),
    ColumnSpec("on_target", pl.Boolean(), "outcome", False,
               "Outcome goal or saved, and not an own goal."),
    ColumnSpec("first_time", pl.Boolean(), "pre_shot", True, "Tri-state: first-time shot."),
    ColumnSpec("follows_dribble", pl.Boolean(), "pre_shot", True,
               "Tri-state: shot follows a dribble (StatsBomb only)."),
    ColumnSpec("open_goal", pl.Boolean(), "pre_shot", True, "Tri-state: open goal."),
    ColumnSpec("one_on_one", pl.Boolean(), "pre_shot", True, "Tri-state: one on one."),
    ColumnSpec("big_chance", pl.Boolean(), "restricted", True,
               "Tri-state: Opta big chance; null for StatsBomb."),
    ColumnSpec("deflected", pl.Boolean(), "post_shot", True, "Tri-state: shot deflected."),
    ColumnSpec("provider_xg", pl.Float64(), "lineage", True,
               "Provider xG, a benchmark only.", ge=0.0, le=1.0),
    ColumnSpec("provider_psxg", pl.Float64(), "lineage", True,
               "Provider post-shot xG, a benchmark only.", ge=0.0, le=1.0),
    ColumnSpec("assist_event_id", pl.String(), "lineage", True,
               "Event id of the assisting pass (Opta: resolved to the global id).",
               is_id=True),
    ColumnSpec("raw_type", pl.String(), "lineage", False,
               "Provider event type (encodes the outcome, so never a feature)."),
    ColumnSpec("raw_qualifiers", pl.String(), "lineage", False,
               "Raw provider shot object / qualifier list as JSON text."),
    ColumnSpec("schema_version", pl.String(), "lineage", False,
               "Schema version the row was written with."),
)
# fmt: on

# fmt: off
FREEZE_FRAME_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec("shot_id", pl.String(), "identity", False,
               "The shot this frame belongs to (not unique here).", is_id=True),
    ColumnSpec("player_id", pl.String(), "identity", True,
               "Visible player; null for id-less sources.", is_id=True),
    ColumnSpec("is_teammate", pl.Boolean(), "pre_shot", False, "Teammate of the shooter."),
    ColumnSpec("is_keeper", pl.Boolean(), "pre_shot", False, "Goalkeeper."),
    ColumnSpec("x", pl.Float64(), "pre_shot", False, "Player location, metres.", *_X),
    ColumnSpec("y", pl.Float64(), "pre_shot", False, "Player location, metres.", *_Y),
)
# fmt: on

# fmt: off
MATCH_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec("match_id", pl.String(), "identity", False,
               "Provider match id, the primary key.", is_id=True, unique=True),
    ColumnSpec("provider", PROVIDER, "identity", False, "Data source."),
    ColumnSpec("competition_id", pl.String(), "identity", False,
               "Competition id.", is_id=True),
    ColumnSpec("season_id", pl.String(), "identity", False, "Season id.", is_id=True),
    ColumnSpec("match_date", pl.Date(), "identity", False, "Match date, as in the shots table."),
    ColumnSpec("home_team_id", pl.String(), "identity", False,
               "Home side as the provider lists it (nominal at neutral venues); "
               "shots.is_home refers to it.", is_id=True),
    ColumnSpec("away_team_id", pl.String(), "identity", False, "The other team.", is_id=True),
    ColumnSpec("competition_name", pl.String(), "identity", False,
               "Competition name, a label for cards."),
    ColumnSpec("season_name", pl.String(), "identity", False,
               "Season as football writes it ('2015/2016', '2026'), a label for cards."),
    ColumnSpec("gender", GENDER, "pre_shot", False,
               "Men's or women's football, known before kick-off."),
    ColumnSpec("home_score", pl.Int16(), "outcome", True,
               "Recorded final score without the shoot-out; null when the provider gives none.",
               ge=0.0, le=99.0),
    ColumnSpec("away_score", pl.Int16(), "outcome", True, "As home_score.", ge=0.0, le=99.0),
    ColumnSpec("has_360", pl.Boolean(), "lineage", True,
               "A usable StatsBomb 360 file; null for every other provider "
               "(rule has_360_matches_provider)."),
    ColumnSpec("schema_version", pl.String(), "lineage", False,
               "Schema version the row was written with."),
)
# fmt: on

_SPECS: Final[Mapping[Table, tuple[ColumnSpec, ...]]] = MappingProxyType(
    {"shots": SHOT_COLUMNS, "freeze_frames": FREEZE_FRAME_COLUMNS, "matches": MATCH_COLUMNS}
)
_SPECS_BY_NAME: Final[Mapping[Table, Mapping[str, ColumnSpec]]] = MappingProxyType(
    {table: {spec.name: spec for spec in specs} for table, specs in _SPECS.items()}
)
# Column-specific checks that a ColumnSpec field cannot express.
_SPECIAL_CHECKS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "raw_type": "non_empty",
        "competition_name": "non_empty",
        "season_name": "non_empty",
        "raw_qualifiers": "json",
        "schema_version": "schema_version",
        "visible_area": "polygon",
    }
)
_SPECIAL_TEXT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "non_empty": "must not be empty",
        "json": "not JSON text with an object or array at the top level",
        "schema_version": f"not schema version {SCHEMA_VERSION!r}",
        "polygon": "not a polygon of >= 3 finite vertices on the pitch",
    }
)


# --- Pandera models (column checks only) -----------------------------------------------------


def _finite(data: PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(data.key).is_finite().fill_null(True))


def _non_empty(data: PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select((pl.col(data.key).str.len_chars() > 0).fill_null(True))


def _reject_constant(name: str) -> float:
    raise ValueError(f"non-standard JSON constant {name}")


def _is_json_container(text: str) -> bool:
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError):  # RecursionError: absurdly deep nesting
        return False
    return isinstance(value, dict | list)


def _json(data: PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(
        pl.col(data.key)
        .map_elements(_is_json_container, return_dtype=pl.Boolean, skip_nulls=True)
        .fill_null(True)
    )


def _polygon_ok(column: str) -> pl.Expr:
    vertex = pl.element()
    on_pitch = (
        vertex.is_not_null()
        & vertex.arr.get(0).is_finite()
        & vertex.arr.get(1).is_finite()
        & vertex.arr.get(0).is_between(*_X)
        & vertex.arr.get(1).is_between(*_Y)
    ).fill_null(False)
    vertices_ok = pl.col(column).list.eval(on_pitch).list.all()
    return ((pl.col(column).list.len() >= 3) & vertices_ok).fill_null(True)


def _polygon(data: PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(_polygon_ok(data.key))


_CUSTOM_CHECKS: Final[Mapping[str, Callable[[PolarsData], pl.LazyFrame]]] = MappingProxyType(
    {"non_empty": _non_empty, "json": _json, "polygon": _polygon}
)


def _column_checks(spec: ColumnSpec) -> list[pa.Check]:
    # name= and error= both carry the vocabulary name: pandera reports a built-in check
    # under its error text, not its name.
    checks: list[pa.Check] = []
    if spec.dtype == pl.Float64():
        checks.append(pa.Check(_finite, name="finite", error="finite"))
    if spec.ge is not None and spec.le is not None:
        checks.append(pa.Check.in_range(spec.ge, spec.le, name="range", error="range"))
    elif spec.ge is not None:
        checks.append(pa.Check.ge(spec.ge, name="range", error="range"))
    elif spec.le is not None:
        checks.append(pa.Check.le(spec.le, name="range", error="range"))
    if spec.allowed is not None:
        checks.append(pa.Check.isin(list(spec.allowed), name="allowed", error="allowed"))
    if spec.is_id:
        checks.append(pa.Check.str_matches(ID_PATTERN, name="id_format", error="id_format"))
    special = _SPECIAL_CHECKS.get(spec.name)
    if special == "schema_version":
        checks.append(pa.Check.eq(SCHEMA_VERSION, name=special, error=special))
    elif special is not None:
        checks.append(pa.Check(_CUSTOM_CHECKS[special], name=special, error=special))
    return checks


def _model(table: Table) -> pa.DataFrameSchema:
    columns = {
        spec.name: pa.Column(
            spec.dtype,
            _column_checks(spec),
            nullable=spec.nullable,
            unique=spec.unique,
            description=spec.description,
            metadata={"role": spec.role},
        )
        for spec in _SPECS[table]
    }
    return pa.DataFrameSchema(columns, strict=True, name=table)


SHOT_SCHEMA: Final[pl.Schema] = pl.Schema({spec.name: spec.dtype for spec in SHOT_COLUMNS})
FREEZE_FRAME_SCHEMA: Final[pl.Schema] = pl.Schema(
    {spec.name: spec.dtype for spec in FREEZE_FRAME_COLUMNS}
)
MATCH_SCHEMA: Final[pl.Schema] = pl.Schema({spec.name: spec.dtype for spec in MATCH_COLUMNS})
# Column-level only: use the validate_* functions, which add the frame rules and pin the
# pandera config.
SHOT_MODEL: Final[pa.DataFrameSchema] = _model("shots")
FREEZE_FRAME_MODEL: Final[pa.DataFrameSchema] = _model("freeze_frames")
MATCH_MODEL: Final[pa.DataFrameSchema] = _model("matches")


# --- Frame rules (evaluated here, not inside pandera) ----------------------------------------


def _mask(df: pl.DataFrame, expr: pl.Expr) -> pl.Series:
    """Evaluate a row-wise rule; a null result (some input was null) passes."""
    return df.select(expr.fill_null(True).alias("ok")).to_series()


def _text(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String)


def _ids_match_provider(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.Series:
    prefix = _text("provider").replace_strict(
        dict(ID_PREFIXES), default=None, return_dtype=pl.String
    )
    ok = pl.lit(True)
    for column in columns:
        # Malformed ids are id_format's job; this rule judges well-formed ones only.
        judged = pl.col(column).str.contains(ID_PATTERN) & prefix.is_not_null()
        ok = ok & (~judged | pl.col(column).str.starts_with(prefix)).fill_null(True)
    return _mask(df, ok)


def _distinct_teams(df: pl.DataFrame) -> pl.Series:
    return _mask(df, pl.col("team_id") != pl.col("opponent_team_id"))


def _matches_pitch(
    df: pl.DataFrame,
    column: str,
    geometry: Callable[[pitch.FloatArray, pitch.FloatArray], pitch.FloatArray],
    tolerance: float,
) -> pl.Series:
    x = df["x"].cast(pl.Float64).to_numpy()
    y = df["y"].cast(pl.Float64).to_numpy()
    stored = df[column].cast(pl.Float64).to_numpy()
    with np.errstate(over="ignore", invalid="ignore"):
        expected = geometry(x, y)
        judged = np.isfinite(x) & np.isfinite(y) & np.isfinite(stored)
        agrees = np.abs(stored - expected) <= tolerance
    return pl.Series("ok", ~judged | agrees, dtype=pl.Boolean)


def _period_start() -> pl.Expr:
    return pl.col("period").replace_strict(
        dict(PERIOD_START_MINUTE), default=None, return_dtype=pl.Int64
    )


def _clock_consistent(df: pl.DataFrame) -> pl.Series:
    # Int64 arithmetic: Int16 minutes times 60 would wrap silently.
    expected = (pl.col("minute").cast(pl.Int64) - _period_start()) * 60 + pl.col("second").cast(
        pl.Int64
    )
    seconds = pl.col("seconds_in_period")
    gap_ok = (seconds - expected.cast(pl.Float64)).abs() < 1.0
    return _mask(df, (pl.col("period") == 5) | ~seconds.is_finite() | gap_ok)


def _rule(name: str, columns: tuple[str, ...], description: str, check: RuleCheck) -> FrameRule:
    return FrameRule(name=name, columns=columns, description=description, check=check)


def _expr_rule(name: str, columns: tuple[str, ...], description: str, expr: pl.Expr) -> FrameRule:
    return _rule(name, columns, description, partial(_mask, expr=expr))


_SHOT_ID_COLUMNS: Final = tuple(spec.name for spec in SHOT_COLUMNS if spec.is_id)
_OUTCOME = _text("outcome")
_ON_TARGET_EXPECTED = _OUTCOME.is_in(sorted(ON_TARGET_OUTCOMES)) & ~pl.col("is_own_goal")

# fmt: off
SHOT_RULES: Final[tuple[FrameRule, ...]] = (
    _rule("ids_match_provider", ("shot_id", "provider", *_SHOT_ID_COLUMNS[1:]),
          "every id must start with the provider's prefix",
          partial(_ids_match_provider, columns=_SHOT_ID_COLUMNS)),
    _rule("distinct_teams", ("shot_id", "team_id", "opponent_team_id"),
          "team_id must differ from opponent_team_id", _distinct_teams),
    _rule("distance_matches_pitch", ("shot_id", "x", "y", "distance_m"),
          "distance_m must equal pitch.shot_distance(x, y)",
          partial(_matches_pitch, column="distance_m", geometry=pitch.shot_distance,
                  tolerance=DISTANCE_TOLERANCE)),
    _rule("angle_matches_pitch", ("shot_id", "x", "y", "angle_rad"),
          "angle_rad must equal pitch.opening_angle(x, y)",
          partial(_matches_pitch, column="angle_rad", geometry=pitch.opening_angle,
                  tolerance=ANGLE_TOLERANCE)),
    _expr_rule("is_goal_matches_outcome", ("shot_id", "is_goal", "outcome"),
               "is_goal must equal outcome == 'goal'",
               pl.col("is_goal") == (_OUTCOME == "goal")),
    _expr_rule("own_goal_is_goal", ("shot_id", "is_own_goal", "is_goal"),
               "an own goal must be a goal", ~pl.col("is_own_goal") | pl.col("is_goal")),
    _expr_rule("on_target_matches_outcome", ("shot_id", "on_target", "outcome", "is_own_goal"),
               "on_target must equal outcome in (goal, saved) and not an own goal",
               pl.col("on_target") == _ON_TARGET_EXPECTED),
    _expr_rule("blocked_has_no_goal_mouth", ("shot_id", "outcome", "goal_y", "goal_z"),
               "a blocked shot must have null goal_y and goal_z",
               (_OUTCOME != "blocked")
               | (pl.col("goal_y").is_null() & pl.col("goal_z").is_null())),
    _expr_rule("penalty_matches_situation", ("shot_id", "is_penalty", "situation"),
               "is_penalty must equal situation == 'penalty'",
               pl.col("is_penalty") == (_text("situation") == "penalty")),
    _expr_rule("shootout_is_penalty", ("shot_id", "period", "is_penalty"),
               "a shoot-out kick (period 5) must be a penalty",
               (pl.col("period") != 5) | pl.col("is_penalty")),
    _expr_rule("minute_within_period", ("shot_id", "period", "minute"),
               "minute must not be before the period's start minute",
               (pl.col("period") == 5) | (pl.col("minute").cast(pl.Int64) >= _period_start())),
    _rule("clock_consistent", ("shot_id", "period", "minute", "second", "seconds_in_period"),
          "seconds_in_period must match minute and second within 1 s", _clock_consistent),
    _expr_rule("unassisted_has_no_assist_event", ("shot_id", "assist_type", "assist_event_id"),
               "assist_type 'none' must have a null assist_event_id",
               (_text("assist_type") != "none") | pl.col("assist_event_id").is_null()),
    _expr_rule("gk_id_matches_own_goal", ("shot_id", "gk_id", "is_own_goal", "player_id"),
               "gk_id must be null exactly for own goals and never equal player_id",
               (pl.col("gk_id").is_null() == pl.col("is_own_goal"))
               & (pl.col("gk_id") != pl.col("player_id"))),
)
# fmt: on

_FF_KEYS_OK = pl.col("shot_id").is_not_null() & pl.col("player_id").is_not_null()

# fmt: off
FREEZE_FRAME_RULES: Final[tuple[FrameRule, ...]] = (
    _expr_rule("ids_share_prefix", ("shot_id", "player_id"),
               "player_id must have the same provider prefix as shot_id",
               ~(pl.col("shot_id").str.contains(ID_PATTERN)
                 & pl.col("player_id").str.contains(ID_PATTERN))
               | (pl.col("shot_id").str.slice(0, 3) == pl.col("player_id").str.slice(0, 3))),
    _expr_rule("unique_player_per_shot", ("shot_id", "player_id"),
               "a player may appear only once per shot",
               ~_FF_KEYS_OK | (pl.len().over("shot_id", "player_id") == 1)),
    _expr_rule("one_keeper_per_side", ("shot_id", "is_teammate", "is_keeper"),
               "at most one keeper per side per shot",
               ~pl.col("is_keeper")
               | (pl.col("is_keeper").cast(pl.Int64).sum().over("shot_id", "is_teammate") <= 1)),
)
# fmt: on

_MATCH_ID_COLUMNS: Final = tuple(spec.name for spec in MATCH_COLUMNS if spec.is_id)
_HAS_360_OK = pl.col("has_360").is_not_null() == (pl.col("provider") == "statsbomb")

# fmt: off
MATCH_RULES: Final[tuple[FrameRule, ...]] = (
    _rule("ids_match_provider", ("match_id", "provider", *_MATCH_ID_COLUMNS[1:]),
          "match, competition, season and team ids must start with the provider's prefix",
          partial(_ids_match_provider, columns=_MATCH_ID_COLUMNS)),
    _expr_rule("distinct_teams", ("match_id", "home_team_id", "away_team_id"),
               "home_team_id must differ from away_team_id",
               pl.col("home_team_id") != pl.col("away_team_id")),
    _expr_rule("scores_both_or_neither", ("match_id", "home_score", "away_score"),
               "home_score and away_score must both be set or both be null",
               pl.col("home_score").is_null() == pl.col("away_score").is_null()),
    _expr_rule("has_360_matches_provider", ("match_id", "provider", "has_360"),
               "has_360 must be set for statsbomb rows and null for every other provider",
               _HAS_360_OK),
)
# fmt: on

# Cross-table rules, keyed by the table they validate: name -> (columns, description).
# validate_freeze_frames runs its own with shots=, validate_shots its own with matches=.
_AGREES_COLUMNS: Final = (
    *("shot_id", "match_id", "competition_id", "season_id", "match_date"),
    *("team_id", "opponent_team_id", "is_home"),
)
# fmt: off
_CROSS_TABLE_RULES: Final[Mapping[Table, Mapping[str, tuple[tuple[str, ...], str]]]] = (
    MappingProxyType({
        "freeze_frames": MappingProxyType({
            "shot_exists": (("shot_id",), "shot_id must exist in shots.shot_id"),
            "shooter_not_in_frame": (("shot_id", "player_id"),
                                     "the shooter must not appear in the shot's own freeze frame"),
        }),
        "shots": MappingProxyType({
            "match_exists": (("shot_id", "match_id"), "match_id must exist in matches.match_id"),
            "shot_agrees_with_match": (_AGREES_COLUMNS,
                                       "competition, season, date and the team pair (home/away "
                                       "from is_home) must equal the shot's matches row"),
        }),
    })
)
# fmt: on


def _shot_exists(df: pl.DataFrame, keys: pl.DataFrame) -> pl.Series:
    known = keys["shot_id"].drop_nulls().unique()
    return _mask(df, pl.col("shot_id").is_in(known.implode()))


def _shooter_not_in_frame(df: pl.DataFrame, keys: pl.DataFrame) -> pl.Series:
    shooters = keys.drop_nulls().select(pl.struct("shot_id", "player_id")).to_series()
    in_shots = pl.struct("shot_id", "player_id").is_in(shooters.implode())
    return _mask(df, ~_FF_KEYS_OK | ~in_shots)


_MATCH_KEYS: Final = ("match_id", "competition_id", "season_id", "match_date")
_HOME = pl.when(pl.col("is_home")).then(pl.col("team_id")).otherwise(pl.col("opponent_team_id"))
_AWAY = pl.when(pl.col("is_home")).then(pl.col("opponent_team_id")).otherwise(pl.col("team_id"))
_SHOT_MATCH = pl.struct(*_MATCH_KEYS, _HOME.alias("home_team_id"), _AWAY.alias("away_team_id"))
# A struct is_in with a null field is False, not null: mask null keys so they pass.
_SHOT_KEYS_OK = pl.all_horizontal(
    pl.col(*_MATCH_KEYS, "team_id", "opponent_team_id", "is_home").is_not_null()
)


def _match_exists(df: pl.DataFrame, keys: pl.DataFrame) -> pl.Series:
    known = keys["match_id"].drop_nulls().unique()
    return _mask(df, pl.col("match_id").is_in(known.implode()))


def _shot_agrees_with_match(df: pl.DataFrame, keys: pl.DataFrame) -> pl.Series:
    known = keys.select(pl.struct(*_MATCH_KEYS, "home_team_id", "away_team_id")).to_series()
    exists = pl.col("match_id").is_in(keys["match_id"].drop_nulls().implode())
    return _mask(df, ~_SHOT_KEYS_OK | ~exists | _SHOT_MATCH.is_in(known.implode()))


# --- Rendering -------------------------------------------------------------------------------


def _render(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        if len(value) > _LONG_STRING:
            return f"{value[:_LONG_STRING]!r}... ({len(value)} chars)"
        return repr(value)
    if isinstance(value, bool | int | float):
        return repr(value)
    if isinstance(value, date):
        return repr(value.isoformat())
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_render(v) for v in value) + "]"
    return repr(value)


def _vertex_ok(vertex: object) -> bool:
    if not isinstance(vertex, list | tuple) or len(vertex) != 2:
        return False
    x, y = vertex
    return (
        isinstance(x, float)
        and isinstance(y, float)
        and _X[0] <= x <= _X[1]
        and _Y[0] <= y <= _Y[1]
    )


def _render_polygon(value: object) -> str:
    if not isinstance(value, list):
        return _render(value)
    n = len(value)
    text = f"{n} vertex" if n == 1 else f"{n} vertices"
    bad = [v for v in value if not _vertex_ok(v)]
    if not bad:
        return text
    first = bad[0]
    shown = "null" if first is None else "(" + ", ".join(_render(c) for c in first) + ")"
    return f"{text}, first bad {shown}"


def _spec_of(table: Table, column: str) -> ColumnSpec | None:
    return _SPECS_BY_NAME.get(table, {}).get(column)


def _rows_text(failure: Failure) -> str:
    rows = ", ".join(str(r) for r in failure.rows)
    hidden = failure.count - len(failure.rows)
    return f"(rows {rows}, ... and {hidden} more)" if hidden > 0 else f"(rows {rows})"


def _bound(value: float | None, default: str) -> str:
    return default if value is None else repr(value)


def _rule_description(table: Table, name: str) -> str:
    # Only this table's rules: shots and matches both have a distinct_teams rule.
    for rule in globals()[_RULES_NAME[table]]:
        if rule.name == name:
            return str(rule.description)
    cross = _CROSS_TABLE_RULES.get(table, {})
    if name in cross:
        return cross[name][1]
    return "rule failed"


def _format_failure(failure: Failure) -> str:
    check, values = failure.check, ", ".join(failure.values)
    if check not in CHECKS:
        row = ", ".join(f"{c}={v}" for c, v in zip(failure.columns, failure.values, strict=False))
        columns = ", ".join(failure.columns)
        return (
            f"rule '{check}' ({columns}): {_rule_description(failure.table, check)}; "
            f"failed for {row} {_rows_text(failure)}"
        )
    column = failure.columns[0]
    head = f"column '{column}': "
    spec = _spec_of(failure.table, column)
    if check == "missing":
        return head + "missing required column"
    if check == "extra":
        return head + f"not in the {failure.table} schema"
    if check == "dtype":
        expected = "?" if spec is None else str(spec.dtype)
        return head + f"dtype {failure.values[0] if failure.values else '?'}, expected {expected}"
    if check == "cast":
        source = failure.values[0] if failure.values else "?"
        target = "?" if spec is None else str(spec.dtype)
        bad = failure.values[1:]
        text = head + f"cannot cast {source} to {target}"
        return text + (f"; bad value(s) {', '.join(bad)} {_rows_text(failure)}" if bad else "")
    rows = _rows_text(failure)
    if check == "allowed":
        if spec is not None and isinstance(spec.dtype, pl.Enum):
            vocab = ", ".join(spec.dtype.categories.to_list())
        elif spec is not None and spec.allowed is not None:
            vocab = ", ".join(str(v) for v in spec.allowed)
        else:
            vocab = "?"
        return head + f"value(s) {values} not allowed; allowed: {vocab} {rows}"
    if check == "nullable":
        return head + f"null/NaN in non-nullable column {rows}"
    if check == "range":
        ge = _bound(None if spec is None else spec.ge, "-inf")
        le = _bound(None if spec is None else spec.le, "inf")
        closing = ")" if spec is None or spec.le is None else "]"
        opening = "(" if spec is None or spec.ge is None else "["
        return head + f"value(s) {values} outside {opening}{ge}, {le}{closing} {rows}"
    if check == "finite":
        return head + f"non-finite value(s) {values} {rows}"
    if check == "unique":
        return head + f"duplicate value(s) {values} {rows}"
    if check == "id_format":
        return head + f"value(s) {values} not a provider-prefixed id like 'sb:…' or 'ws:…' {rows}"
    return head + f"value(s) {values} {_SPECIAL_TEXT[check]} {rows}"


# --- Failure assembly ------------------------------------------------------------------------


def _row_failure(
    table: Table,
    column: str,
    check: str,
    series: pl.Series,
    indices: Sequence[int],
    render: Callable[[object], str] = _render,
    leading: tuple[str, ...] = (),
) -> Failure:
    """A failure on ``indices`` of ``series``: distinct rendered values, capped examples."""
    shown: list[str] = []
    for value in series.gather(list(indices)).to_list():
        text = render(value)
        if text not in shown:
            shown.append(text)
        if len(shown) == MAX_EXAMPLES - len(leading):
            break
    return Failure(
        table=table,
        columns=(column,),
        check=check,
        values=(*leading, *shown),
        rows=tuple(indices[:MAX_EXAMPLES]),
        count=len(indices),
    )


def _structural(table: Table, column: str, check: str, values: tuple[str, ...] = ()) -> Failure:
    return Failure(table=table, columns=(column,), check=check, values=values, rows=(), count=1)


def _renderer(table: Table, column: str) -> Callable[[object], str]:
    spec = _spec_of(table, column)
    return _render_polygon if spec is not None and spec.dtype == POLYGON else _render


def _enum_values_not_allowed(
    table: Table, column: str, series: pl.Series, enum: pl.Enum
) -> Failure | None:
    text = series.cast(pl.String)
    bad = (text.is_not_null() & ~text.is_in(enum.categories.implode())).fill_null(False)
    indices = bad.arg_true().to_list()
    if not indices:
        return None
    return _row_failure(table, column, "allowed", text, indices)


def _is_texty(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.String | pl.Categorical | pl.Enum)


def _run_pandera(
    model: pa.DataFrameSchema, df: pl.DataFrame, depth: ValidationDepth
) -> pl.DataFrame | None:
    # Pin both flags: PANDERA_VALIDATION_ENABLED / _DEPTH must not switch validation off.
    with config_context(validation_enabled=True, validation_depth=depth):
        try:
            model.validate(df, lazy=True)
        except SchemaErrors as err:
            cases: pl.DataFrame = err.failure_cases
            return cases
    return None


def _structure_failures(table: Table, df: pl.DataFrame) -> list[Failure]:
    """Pass 1: missing / extra columns, dtypes, and invalid values of mis-typed enums."""
    names = column_names(table)
    failures = [_structural(table, n, "missing") for n in names if n not in df.columns]
    # pandera reports only the first extra column of a strict schema: list them here.
    failures += [_structural(table, c, "extra") for c in df.columns if c not in names]
    cases = _run_pandera(_current_model(table), df, ValidationDepth.SCHEMA_ONLY)
    checks = [] if cases is None else cases.select("column", "check").rows()
    for column, check in checks:
        if check in ("column_in_dataframe", "column_in_schema"):
            continue
        if not str(check).startswith("dtype("):
            raise RuntimeError(f"internal error in check '{check}' on column '{column}'")
        got = df.schema[column]
        failures.append(_structural(table, column, "dtype", (str(got),)))
        expected = _SPECS_BY_NAME[table][column].dtype
        if isinstance(expected, pl.Enum) and _is_texty(got):
            allowed = _enum_values_not_allowed(table, column, df[column], expected)
            if allowed is not None:
                failures.append(allowed)
    return failures


def _column_failures(table: Table, df: pl.DataFrame, model: pa.DataFrameSchema) -> list[Failure]:
    """Pass 2, column part: pandera nulls, uniqueness and column checks."""
    cases = _run_pandera(model, df, ValidationDepth.SCHEMA_AND_DATA)
    if cases is None:
        return []
    failures: list[Failure] = []
    for column, check in cases.select("column", "check").unique(maintain_order=True).rows():
        group = cases.filter((pl.col("column") == column) & (pl.col("check") == check))
        broken = group.filter(pl.col("index").is_null())
        if broken.height:
            # pandera turns an exception inside a check into a row without an index.
            detail = broken["failure_case"][0]
            raise RuntimeError(f"internal error in check '{check}' on column '{column}': {detail}")
        name = _PANDERA_NAMES.get(check, check)
        indices = sorted(set(group["index"].to_list()))
        if name == "unique":  # pandera counts nulls as duplicates; they are `nullable` here
            indices = [i for i in indices if df[column][i] is not None]
        if indices:
            render = _renderer(table, column)
            failures.append(_row_failure(table, column, name, df[column], indices, render))
    return failures


def _apply_rule(table: Table, df: pl.DataFrame, rule: FrameRule) -> Failure | None:
    try:
        mask = rule.check(df)
    except Exception as exc:
        raise RuntimeError(f"internal error in rule '{rule.name}': {exc!r}") from exc
    if not (
        isinstance(mask, pl.Series)
        and mask.dtype == pl.Boolean
        and mask.len() == df.height
        and mask.null_count() == 0
    ):
        raise RuntimeError(
            f"internal error in rule '{rule.name}': check must return a Boolean Series of "
            f"length {df.height} without nulls, got {mask!r:.200}"
        )
    indices = (~mask).arg_true().to_list()
    if not indices:
        return None
    first = df.select(rule.columns).row(indices[0])
    return Failure(
        table=table,
        columns=rule.columns,
        check=rule.name,
        values=tuple(_render(v) for v in first),
        rows=tuple(indices[:MAX_EXAMPLES]),
        count=len(indices),
    )


def _apply_rules(table: Table, df: pl.DataFrame, rules: Iterable[FrameRule]) -> list[Failure]:
    return [f for rule in rules if (f := _apply_rule(table, df, rule)) is not None]


def _ordered(table: Table, df: pl.DataFrame, failures: list[Failure]) -> list[Failure]:
    """Canonical column order (extras after, in input order), then §3.3 check order."""
    position = {name: i for i, name in enumerate(column_names(table))}
    extras = {c: len(position) + i for i, c in enumerate(df.columns) if c not in position}

    def key(failure: Failure) -> tuple[int, int]:
        column = failure.columns[0]
        rank = CHECKS.index(failure.check) if failure.check in CHECKS else len(CHECKS)
        return position.get(column, extras.get(column, len(position) + len(extras))), rank

    return sorted(failures, key=key)


# Module globals read at call time, so a monkeypatched model or rule tuple is the one that runs.
_MODEL_NAME: Final[Mapping[Table, str]] = MappingProxyType(
    {"shots": "SHOT_MODEL", "freeze_frames": "FREEZE_FRAME_MODEL", "matches": "MATCH_MODEL"}
)
_RULES_NAME: Final[Mapping[Table, str]] = MappingProxyType(
    {"shots": "SHOT_RULES", "freeze_frames": "FREEZE_FRAME_RULES", "matches": "MATCH_RULES"}
)


def _current_model(table: Table) -> pa.DataFrameSchema:
    model: pa.DataFrameSchema = globals()[_MODEL_NAME[table]]
    return model


def _require_frame(df: object, function: str) -> None:
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"{function} expects a polars DataFrame, got {type(df).__name__}")


def _validate(
    table: Table, df: pl.DataFrame, rules: Iterable[FrameRule], extra: Iterable[FrameRule] = ()
) -> pl.DataFrame:
    structure = _structure_failures(table, df)
    if structure:
        raise SchemaValidationError(table, _ordered(table, df, structure))
    failures = _ordered(table, df, _column_failures(table, df, _current_model(table)))
    failures += _apply_rules(table, df, rules)
    failures += _apply_rules(table, df, extra)
    if failures:
        raise SchemaValidationError(table, failures)
    return df.select(column_names(table))


# --- Public functions ------------------------------------------------------------------------


def _check_table(table: str) -> Table:
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}; expected one of: {', '.join(TABLES)}")
    return table


def column_names(
    table: Table = "shots", *, roles: Role | Collection[Role] | None = None
) -> tuple[str, ...]:
    """Column names of a canonical table in canonical order, optionally filtered by role.

    Parameters
    ----------
    table : {"shots", "freeze_frames", "matches"}, default "shots"
        The table.
    roles : str or collection of str, optional
        One role, or several (their union, still in canonical order). ``None`` returns every
        column; an empty collection returns ``()``.

    Returns
    -------
    tuple of str
        The matching column names.

    Raises
    ------
    ValueError
        If the table or a role is unknown; the message lists the allowed ones.

    Examples
    --------
    >>> column_names("freeze_frames", roles="pre_shot")
    ('is_teammate', 'is_keeper', 'x', 'y')
    >>> column_names("shots", roles="restricted")
    ('big_chance',)
    >>> column_names("matches", roles="pre_shot")
    ('gender',)
    """
    specs = _SPECS[_check_table(table)]
    if roles is None:
        return tuple(spec.name for spec in specs)
    wanted = {roles} if isinstance(roles, str) else set(roles)
    unknown = sorted(str(role) for role in wanted if role not in ROLES)
    if unknown:
        raise ValueError(
            f"unknown role(s) {', '.join(repr(r) for r in unknown)}; "
            f"expected one of: {', '.join(ROLES)}"
        )
    return tuple(spec.name for spec in specs if spec.role in wanted)


def empty_shots() -> pl.DataFrame:
    """An empty shot table with the canonical columns and dtypes.

    Returns
    -------
    polars.DataFrame
        Zero rows; validates.

    Examples
    --------
    >>> empty_shots().shape
    (0, 48)
    >>> validate_shots(empty_shots()).height
    0
    """
    return pl.DataFrame(schema=SHOT_SCHEMA)


def empty_freeze_frames() -> pl.DataFrame:
    """An empty freeze-frame table with the canonical columns and dtypes.

    Returns
    -------
    polars.DataFrame
        Zero rows; validates.

    Examples
    --------
    >>> empty_freeze_frames().columns
    ['shot_id', 'player_id', 'is_teammate', 'is_keeper', 'x', 'y']
    """
    return pl.DataFrame(schema=FREEZE_FRAME_SCHEMA)


def empty_matches() -> pl.DataFrame:
    """An empty matches table with the canonical columns and dtypes.

    Returns
    -------
    polars.DataFrame
        Zero rows; validates.

    Examples
    --------
    >>> empty_matches().shape
    (0, 14)
    >>> validate_matches(empty_matches()).height
    0
    """
    return pl.DataFrame(schema=MATCH_SCHEMA)


_INT_RANGES: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {"Int8": (-(2**7), 2**7 - 1), "Int16": (-(2**15), 2**15 - 1)}
)
_ISO_DATE: Final[str] = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"


def _is_numeric_list(dtype: pl.DataType) -> bool:
    """``List(List(<numeric>))`` or ``List(Array(<numeric>, 2))``, i.e. vertex pairs."""
    if not isinstance(dtype, pl.List):
        return False
    inner = dtype.inner
    if isinstance(inner, pl.List):
        return inner.inner.is_numeric()
    return isinstance(inner, pl.Array) and inner.shape == (2,) and inner.inner.is_numeric()


def _cast_column(table: Table, series: pl.Series, target: pl.DataType) -> pl.Series | Failure:
    """Cast one column along the safe paths only, or describe why it cannot be cast."""
    name, source = series.name, series.dtype
    refused = _structural(table, name, "cast", (str(source),))
    if source == target:
        return series
    if source == pl.Null():
        return series.cast(target)
    if isinstance(target, pl.Enum):
        if not _is_texty(source):
            return refused
        allowed = _enum_values_not_allowed(table, name, series, target)
        return allowed if allowed is not None else series.cast(pl.String).cast(target)
    if target == pl.Float64():
        ok = source.is_integer() or source.is_float()
        return series.cast(pl.Float64) if ok else refused
    if str(target) in _INT_RANGES:
        # Floats are refused: polars' strict cast truncates 1.5 to 1 without complaint.
        if not source.is_integer():
            return refused
        low, high = _INT_RANGES[str(target)]
        wide = series.cast(pl.Float64)
        indices = ((wide < low) | (wide > high)).fill_null(False).arg_true().to_list()
        if indices:
            return _row_failure(table, name, "cast", series, indices, leading=(str(source),))
        return series.cast(target, strict=True)
    if target == pl.Date() and source == pl.String():
        # The pattern too: a lenient parse reads '24-01-05' as the year 24.
        parsed = series.str.to_date("%Y-%m-%d", strict=False)
        bad = series.is_not_null() & (parsed.is_null() | ~series.str.contains(_ISO_DATE))
        indices = bad.arg_true().to_list()
        if indices:
            return _row_failure(table, name, "cast", series, indices, leading=(str(source),))
        return parsed
    if target == POLYGON and _is_numeric_list(source):
        if isinstance(source, pl.List) and isinstance(source.inner, pl.List):
            # A direct cast raises on a vertex of the wrong length: find and name it first.
            bad_vertex = pl.col(name).list.eval(pl.element().list.len() != 2).list.any()
            bad = series.to_frame().select(bad_vertex.fill_null(False)).to_series()
            rows = bad.arg_true().to_list()
            if rows:
                polygon = series[rows[0]].to_list()
                first = next(v for v in polygon if v is not None and len(v) != 2)
                values = (str(source), _render(first))
                shown = tuple(rows[:MAX_EXAMPLES])
                return Failure(table, (name,), "cast", values, shown, len(rows))
            series = series.cast(pl.List(pl.List(pl.Float64)))
        return series.cast(POLYGON)
    return refused


def cast_to_schema(df: pl.DataFrame, table: Table = "shots") -> pl.DataFrame:
    """Select a table's columns in canonical order and cast them along safe paths only.

    For loaders and test fixtures that build frames from strings or plain lists. Extra
    columns are dropped. Accepted casts: String / Categorical / Enum to an Enum (values outside
    the vocabulary are named); any integer or float to Float64; integers to Int8 / Int16
    after a range check (floats are refused); ISO ``YYYY-MM-DD`` strings to Date (anything
    else, such as ``'24-01-05'`` or ``'2024-02-30'``, is a ``cast`` failure); nested
    numeric lists of vertex pairs to the polygon dtype; an all-null (``Null``) column to
    anything. Ranges, nulls and rules are **not** checked: call ``validate_*`` afterwards.

    Parameters
    ----------
    df : polars.DataFrame
        Input frame; not modified.
    table : {"shots", "freeze_frames", "matches"}, default "shots"
        Target table.

    Returns
    -------
    polars.DataFrame
        The canonical columns with canonical dtypes.

    Raises
    ------
    TypeError
        If ``df`` is not a polars DataFrame.
    ValueError
        If ``table`` is unknown.
    SchemaValidationError
        Listing every missing column and every column that could not be cast.

    Examples
    --------
    >>> frames = pl.DataFrame({"shot_id": ["sb:1"], "player_id": [None], "is_teammate": [True],
    ...                        "is_keeper": [False], "x": [100], "y": [34], "note": ["z"]})
    >>> out = cast_to_schema(frames, "freeze_frames")
    >>> out.schema == FREEZE_FRAME_SCHEMA
    True
    """
    _require_frame(df, "cast_to_schema")
    table = _check_table(table)
    columns: list[pl.Series] = []
    failures: list[Failure] = []
    for spec in _SPECS[table]:
        if spec.name not in df.columns:
            failures.append(_structural(table, spec.name, "missing"))
            continue
        result = _cast_column(table, df[spec.name], spec.dtype)
        if isinstance(result, Failure):
            failures.append(result)
        else:
            columns.append(result)
    if failures:
        raise SchemaValidationError(table, _ordered(table, df, failures))
    return pl.DataFrame(columns)


def validate_shots(df: pl.DataFrame, matches: pl.DataFrame | None = None) -> pl.DataFrame:
    """Validate a shot table and return it in canonical column order.

    Pass 1 checks structure (missing / extra columns, dtypes and the invalid values of
    mis-typed enum columns). Only if it is clean, pass 2 checks nulls, ranges, finiteness,
    uniqueness, id format and the other column checks, then every rule in ``SHOT_RULES``.
    The input is never cast or modified, and pandera's global or environment config cannot
    switch any of this off.

    When ``matches`` is given, two cross-table rules follow: every ``match_id`` exists there
    (``match_exists``), and each shot's competition, season, date and team pair equal its
    match's (``shot_agrees_with_match``; home and away come from ``is_home``). ``matches``
    itself is not validated (use :func:`validate_matches` first); only its six key columns
    are read.

    Parameters
    ----------
    df : polars.DataFrame
        Shot table; any column order.
    matches : polars.DataFrame, optional
        The matching matches table, for the cross-table rules.

    Returns
    -------
    polars.DataFrame
        ``df`` with its columns in canonical order.

    Raises
    ------
    TypeError
        If ``df`` is not a polars DataFrame, or ``matches`` is neither None nor one.
    ValueError
        If ``matches`` lacks a key column or has it with the wrong dtype (String ids, Date
        ``match_date``).
    SchemaValidationError
        Listing every failure found by the pass that failed; rule rows are shot row positions.
    RuntimeError
        Only if a check or rule itself is broken; never for bad data.

    Examples
    --------
    >>> validate_shots(empty_shots(), matches=empty_matches()).height
    0
    >>> try:
    ...     validate_shots(empty_shots().drop("x"))
    ... except SchemaValidationError as err:
    ...     print(err.failures[0].check, err.failures[0].columns)
    missing ('x',)
    """
    _require_frame(df, "validate_shots")
    cross: list[FrameRule] = []
    if matches is not None:
        keys = _match_keys(matches)
        checks = {"match_exists": _match_exists, "shot_agrees_with_match": _shot_agrees_with_match}
        for name, (columns, description) in _CROSS_TABLE_RULES["shots"].items():
            cross.append(_rule(name, columns, description, partial(checks[name], keys=keys)))
    return _validate("shots", df, globals()["SHOT_RULES"], cross)


def _match_keys(matches: object) -> pl.DataFrame:
    if not isinstance(matches, pl.DataFrame):
        raise TypeError(
            f"validate_shots expects matches to be a polars DataFrame or None, "
            f"got {type(matches).__name__}"
        )
    columns = (*_MATCH_KEYS, "home_team_id", "away_team_id")
    for column in columns:
        expected = pl.Date() if column == "match_date" else pl.String()
        if column not in matches.columns:
            raise ValueError(f"matches has no column '{column}'; it needs {', '.join(columns)}")
        if matches.schema[column] != expected:
            raise ValueError(
                f"matches column '{column}' has dtype {matches.schema[column]}, expected {expected}"
            )
    return matches.select(columns).unique()


def validate_matches(df: pl.DataFrame) -> pl.DataFrame:
    """Validate a matches table and return it in canonical column order.

    The same two passes as :func:`validate_shots`, with ``MATCH_RULES``: ids under the
    provider's prefix, two different teams, both scores or neither, and ``has_360`` set
    exactly on StatsBomb rows. ``match_id`` is unique.

    Parameters
    ----------
    df : polars.DataFrame
        Matches table; any column order.

    Returns
    -------
    polars.DataFrame
        ``df`` with its columns in canonical order.

    Raises
    ------
    TypeError
        If ``df`` is not a polars DataFrame.
    SchemaValidationError
        Listing every failure found by the pass that failed.
    RuntimeError
        Only if a check or rule itself is broken; never for bad data.

    Examples
    --------
    >>> from datetime import date
    >>> row = {"match_id": ["ws:1"], "provider": ["whoscored"], "competition_id": ["ws:2"],
    ...        "season_id": ["ws:3"], "match_date": [date(2026, 8, 7)],
    ...        "home_team_id": ["ws:4"], "away_team_id": ["ws:4"],
    ...        "competition_name": ["Championship"], "season_name": ["2026/2027"],
    ...        "gender": ["male"], "home_score": [None], "away_score": [None],
    ...        "has_360": [None], "schema_version": [SCHEMA_VERSION]}
    >>> try:
    ...     validate_matches(cast_to_schema(pl.DataFrame(row), "matches"))
    ... except SchemaValidationError as err:
    ...     print([f.check for f in err.failures])
    ['distinct_teams']
    """
    _require_frame(df, "validate_matches")
    return _validate("matches", df, globals()["MATCH_RULES"])


def _shot_keys(shots: object) -> pl.DataFrame:
    if not isinstance(shots, pl.DataFrame):
        raise TypeError(
            f"validate_freeze_frames expects shots to be a polars DataFrame or None, "
            f"got {type(shots).__name__}"
        )
    for column in ("shot_id", "player_id"):
        if column not in shots.columns:
            raise ValueError(f"shots has no column '{column}'; it needs shot_id and player_id")
        if shots.schema[column] != pl.String():
            raise ValueError(
                f"shots column '{column}' has dtype {shots.schema[column]}, expected String"
            )
    return shots.select("shot_id", "player_id").unique()


def validate_freeze_frames(df: pl.DataFrame, shots: pl.DataFrame | None = None) -> pl.DataFrame:
    """Validate a long-format freeze-frame table and return it in canonical column order.

    Same two passes as :func:`validate_shots`, with ``FREEZE_FRAME_RULES``. When ``shots`` is
    given, also checks that every ``shot_id`` exists there and that no shooter appears in
    their own shot's frame. ``shots`` itself is not validated (validate it first); only its
    ``shot_id`` and ``player_id`` columns are read.

    Parameters
    ----------
    df : polars.DataFrame
        Freeze-frame table; any column order.
    shots : polars.DataFrame, optional
        The matching shot table, for the cross-table rules.

    Returns
    -------
    polars.DataFrame
        ``df`` with its columns in canonical order.

    Raises
    ------
    TypeError
        If ``df`` is not a polars DataFrame, or ``shots`` is neither None nor one.
    ValueError
        If ``shots`` lacks a String ``shot_id`` or ``player_id`` column.
    SchemaValidationError
        Listing every failure found by the pass that failed; rule rows are freeze-frame
        row positions.
    RuntimeError
        Only if a check or rule itself is broken; never for bad data.

    Examples
    --------
    >>> validate_freeze_frames(empty_freeze_frames(), shots=empty_shots()).height
    0
    """
    _require_frame(df, "validate_freeze_frames")
    cross: list[FrameRule] = []
    if shots is not None:
        keys = _shot_keys(shots)
        checks = {"shot_exists": _shot_exists, "shooter_not_in_frame": _shooter_not_in_frame}
        for name, (columns, description) in _CROSS_TABLE_RULES["freeze_frames"].items():
            cross.append(_rule(name, columns, description, partial(checks[name], keys=keys)))
    return _validate("freeze_frames", df, globals()["FREEZE_FRAME_RULES"], cross)


# A null score never equals a goal count, so an unscored match is always reported.
_DIFFERS: Final = pl.col("home_goals").ne_missing(pl.col("home_score").cast(pl.Int64)) | pl.col(
    "away_goals"
).ne_missing(pl.col("away_score").cast(pl.Int64))


def reconcile_goals(shots: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """Matches whose goals in ``shots`` differ from the recorded score.

    Shoot-out kicks (period 5) are excluded; an own goal counts for the other side.

    Parameters
    ----------
    shots : polars.DataFrame
        Needs ``match_id``, ``is_goal``, ``is_own_goal``, ``is_home`` and ``period``.
    matches : polars.DataFrame
        Needs ``match_id``, ``home_score`` and ``away_score``.

    Returns
    -------
    polars.DataFrame
        ``match_id, home_score, away_score, home_goals, away_goals``, sorted by match; empty
        when every match reconciles. A match without a recorded score is always reported, so
        callers filter unscored matches explicitly.

    Examples
    --------
    >>> shots = pl.DataFrame({"match_id": ["sb:1", "sb:1"], "is_goal": [True, True],
    ...                       "is_own_goal": [False, True], "is_home": [True, True],
    ...                       "period": [1, 2]})
    >>> matches = pl.DataFrame({"match_id": ["sb:1", "sb:2"], "home_score": [1, 0],
    ...                         "away_score": [1, 0]})
    >>> reconcile_goals(shots, matches).height
    0
    """
    home_side = pl.col("is_home") ^ pl.col("is_own_goal")  # an own goal counts for the other side
    goals = (
        shots.filter(pl.col("is_goal") & (pl.col("period") != 5))
        .group_by("match_id")
        .agg(
            home_side.sum().cast(pl.Int64).alias("home_goals"),
            (~home_side).sum().cast(pl.Int64).alias("away_goals"),
        )
    )
    return (
        matches.join(goals, on="match_id", how="left")
        .with_columns(pl.col("home_goals", "away_goals").fill_null(0))
        .filter(_DIFFERS)
        .select("match_id", "home_score", "away_score", "home_goals", "away_goals")
        .sort("match_id")
    )
