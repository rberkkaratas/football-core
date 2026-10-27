"""Goal-mouth census: where each provider's goal-mouth points lie against the goal frame.

The xGOT target population is the on-target shots, and its main input is where the ball
crossed the goal line. Before a model reads ``shots.goal_y`` / ``goal_z``, two facts must hold
for both providers: goals lie inside the goal opening (within ``GOAL_TOLERANCE``) and woodwork
hits lie on the woodwork (within ``WOODWORK_TOLERANCE``). :func:`census` counts both per
provider and outcome group, through :func:`football_core.pitch.to_goal_frame`, the one goal
frame every check reads.

**Opta direction.** Opta's ``goalMouthY`` runs 0-100 across the goal, and nothing in its
documentation says which side is the shooter's left. Its zone qualifiers name sides (LowLeft,
HighRight, ...), so :func:`zone_sides` counts on which side of the goal centre the shots of
each zone land, and :func:`confirms_direction` says whether the sides agree with no exception.

**The criterion.** :func:`criterion_problems` judges both MS5 elements on the tables: every
provider has at least :data:`MIN_GOALS_IN_FRAME` of its goals inside the frame (within
``GOAL_TOLERANCE``), and the Opta direction is confirmed. ``fc goalmouth --check`` exits 1
when it does not hold, so a loader change that flips the Opta direction or pushes goals out of
the frame fails a command, not only a reader of its counts.

Counts only: nothing here returns a row, an id or a player, so the output may be quoted for
WhoScored data too.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType
from typing import Final

import numpy as np
import polars as pl

from football_core import pitch, schema

GOAL_TOLERANCE: Final[float] = 0.15  # metres outside the opening that still count as in
WOODWORK_TOLERANCE: Final[float] = 0.3  # metres from the woodwork that still count as on it
MIN_GOALS_IN_FRAME: Final[float] = 0.99  # MS5: share of a provider's goals inside the frame
GROUPS: Final = ("goal", "saved", "post", "off_target", "blocked", "own_goal")
# Opta zone qualifier -> the side it names, from the shooter's view: 73 MissLeft, 75 MissRight,
# 76 LowLeft, 77 HighLeft, 80 LowRight, 81 HighRight, 83 CloseLeft, 84 CloseRight,
# 86 CloseLeftAndHigh, 87 CloseRightAndHigh. 74, 78, 79 and 85 name no side; 82 is Blocked.
# fmt: off
OPTA_ZONES: Final[Mapping[int, str]] = MappingProxyType({
    73: "left", 75: "right", 76: "left", 77: "left", 80: "right", 81: "right", 83: "left",
    84: "right", 86: "left", 87: "right",
})
# fmt: on
_CENTRE: Final[float] = 1e-9  # metres; raw 50 maps to the centre up to rounding
_COUNTS: Final = ("shots", "with_point", "in_frame", "near_woodwork")
_ZONE_COUNTS: Final = ("shots", "shooter_left", "shooter_right", "centre")


def _require(shots: pl.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in shots.columns]
    if missing:
        raise ValueError(f"{what}: missing column(s) {', '.join(missing)}")


def _goal_frame(shots: pl.DataFrame) -> tuple[pitch.FloatArray, pitch.FloatArray]:
    """The goal-frame point of every row; null and NaN become NaN."""
    return pitch.to_goal_frame(
        shots["goal_y"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        shots["goal_z"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
    )


def census(shots: pl.DataFrame) -> pl.DataFrame:
    """Count goal-mouth points inside the frame and on the woodwork, per provider and group.

    Parameters
    ----------
    shots : polars.DataFrame
        C3 shots; reads ``provider``, ``outcome``, ``is_own_goal``, ``goal_y`` and ``goal_z``.

    Returns
    -------
    polars.DataFrame
        One row per (provider, group) present, sorted by provider then ``GROUPS`` order.
        ``group`` is ``own_goal`` for own goals, else the outcome. Int64 counts: ``shots``,
        ``with_point`` (goal_y and goal_z not null), ``in_frame`` (``frame_distance`` ≤
        ``GOAL_TOLERANCE``) and ``near_woodwork`` (``woodwork_distance`` ≤
        ``WOODWORK_TOLERANCE``), both up to ``schema.DISTANCE_TOLERANCE``; NaN never counts.
        Shares are left to the caller (divide by ``shots``).

    Raises
    ------
    ValueError
        A missing column, named.

    Examples
    --------
    >>> shots = pl.DataFrame({"provider": ["whoscored"] * 3, "outcome": ["goal", "post", "goal"],
    ...                       "is_own_goal": [False, False, True], "goal_y": [34.0, 30.25, None],
    ...                       "goal_z": [1.0, 1.0, None]})
    >>> census(shots).rows()
    [('whoscored', 'goal', 1, 1, 1, 0), ('whoscored', 'post', 1, 1, 1, 1), \
('whoscored', 'own_goal', 1, 0, 0, 0)]
    """
    _require(shots, ("provider", "outcome", "is_own_goal", "goal_y", "goal_z"), "census")
    gy, gz = _goal_frame(shots)
    with np.errstate(invalid="ignore"):  # NaN compares False: it never counts
        in_frame = pitch.frame_distance(gy, gz) <= GOAL_TOLERANCE + schema.DISTANCE_TOLERANCE
        wood = pitch.woodwork_distance(gy, gz) <= WOODWORK_TOLERANCE + schema.DISTANCE_TOLERANCE
    outcome = pl.col("outcome").cast(pl.String)
    group = pl.when("is_own_goal").then(pl.lit("own_goal")).otherwise(outcome)
    flags = shots.select(
        pl.col("provider").cast(pl.String),
        group.alias("group"),
        shots=pl.lit(True),
        with_point=pl.col("goal_y").is_not_null() & pl.col("goal_z").is_not_null(),
        in_frame=pl.Series(in_frame, dtype=pl.Boolean),
        near_woodwork=pl.Series(wood, dtype=pl.Boolean),
    )
    table = flags.group_by("provider", "group").agg(pl.col(_COUNTS).sum().cast(pl.Int64))
    rank = {g: i for i, g in enumerate(GROUPS)}
    order = pl.col("group").replace_strict(rank, default=None, return_dtype=pl.Int64)
    return table.sort(pl.col("provider"), order, nulls_last=True)


def _zones(text: str) -> list[int]:
    """The ``OPTA_ZONES`` qualifier ids in a stored ``raw_qualifiers`` JSON list."""
    try:
        ids = {q["type"]["value"] for q in json.loads(text)}
    except (KeyError, TypeError) as e:
        raise ValueError(f"raw_qualifiers {text[:60]!r} is not a qualifier list: {e!r}") from None
    return sorted(i for i in ids if i in OPTA_ZONES)


def zone_sides(shots: pl.DataFrame) -> pl.DataFrame:
    """Per Opta zone qualifier, on which side of the goal centre its shots' points lie.

    Parameters
    ----------
    shots : polars.DataFrame
        C3 shots; reads ``provider``, ``goal_y``, ``goal_z`` and ``raw_qualifiers``. Only
        whoscored rows with a goal-mouth point count.

    Returns
    -------
    polars.DataFrame
        One row per zone of ``OPTA_ZONES`` that some shot carries, sorted by ``zone`` (Int64):
        ``named_side`` (``left`` / ``right``, the shooter's view) and Int64 counts ``shots``,
        ``shooter_left`` (goal-frame y < 0), ``shooter_right`` (y > 0) and ``centre``
        (|y| ≤ 1e-9). A shot with two zone qualifiers counts in both rows.

    Raises
    ------
    ValueError
        A missing column, named, or a ``raw_qualifiers`` that is not a qualifier list.

    Examples
    --------
    >>> q = '[{"type":{"value":76,"displayName":"LowLeft"}}]'
    >>> shots = pl.DataFrame({"provider": ["whoscored"], "goal_y": [36.0], "goal_z": [0.5],
    ...                       "raw_qualifiers": [q]})
    >>> zone_sides(shots).rows()
    [(76, 'left', 1, 1, 0, 0)]
    """
    _require(shots, ("provider", "goal_y", "goal_z", "raw_qualifiers"), "zone_sides")
    ws = shots.filter(
        (pl.col("provider").cast(pl.String) == "whoscored")
        & pl.col("goal_y").is_not_null()
        & pl.col("goal_z").is_not_null()
    )
    gy, _ = _goal_frame(ws)
    zones = [_zones(text) for text in ws["raw_qualifiers"]]
    sides = {"shooter_left": gy < -_CENTRE, "shooter_right": gy > _CENTRE}
    flags = pl.DataFrame(
        {"shots": np.ones(ws.height, dtype=bool), **sides, "centre": np.abs(gy) <= _CENTRE}
    ).with_columns(zone=pl.Series(zones, dtype=pl.List(pl.Int64)))
    exploded = flags.explode("zone").drop_nulls("zone")
    table = exploded.group_by("zone").agg(pl.col(_ZONE_COUNTS).sum().cast(pl.Int64))
    side = pl.col("zone").replace_strict(dict(OPTA_ZONES), return_dtype=pl.String)
    return table.select("zone", side.alias("named_side"), *_ZONE_COUNTS).sort("zone")


def confirms_direction(table: pl.DataFrame, zones: Collection[int] = range(76, 88)) -> bool:
    """Whether the zone sides confirm the goal-mouth y direction, with no exception.

    True when, over the rows of ``table`` (:func:`zone_sides`) whose zone is in ``zones``,
    every ``left`` zone has no shot on the shooter's right, every ``right`` zone none on the
    left, and both sides have shots.

    Examples
    --------
    >>> table = pl.DataFrame({"zone": [76, 80], "named_side": ["left", "right"],
    ...                       "shots": [5, 4], "shooter_left": [5, 0], "shooter_right": [0, 4],
    ...                       "centre": [0, 0]})
    >>> confirms_direction(table), confirms_direction(table, zones=[76])
    (True, False)
    """
    rows = table.filter(pl.col("zone").is_in(list(zones)))
    left = rows.filter(pl.col("named_side") == "left")
    right = rows.filter(pl.col("named_side") == "right")
    return bool(
        left["shooter_right"].sum() == 0
        and right["shooter_left"].sum() == 0
        and left["shots"].sum() > 0
        and right["shots"].sum() > 0
    )


def goal_shares(table: pl.DataFrame) -> dict[str, tuple[int, int]]:
    """Per provider of a :func:`census` table, ``(goals in frame, goals)``; (0, 0) without goals.

    Own goals are their own group, so they never count.

    Examples
    --------
    >>> table = pl.DataFrame({"provider": ["statsbomb", "statsbomb", "whoscored"],
    ...                       "group": ["goal", "saved", "saved"], "shots": [200, 50, 9],
    ...                       "with_point": [200, 50, 9], "in_frame": [198, 50, 9],
    ...                       "near_woodwork": [3, 1, 0]})
    >>> goal_shares(table)
    {'statsbomb': (198, 200), 'whoscored': (0, 0)}
    """
    out = {}
    for provider in table["provider"].unique(maintain_order=True).to_list():
        goals = table.filter((pl.col("provider") == provider) & (pl.col("group") == "goal"))
        out[str(provider)] = (int(goals["in_frame"].sum()), int(goals["shots"].sum()))
    return out


def criterion_problems(table: pl.DataFrame, zones: pl.DataFrame | None) -> tuple[str, ...]:
    """The MS5 goal-mouth criterion on the census and zone tables: one message per failure.

    Per provider of ``table``: ``in_frame / shots`` of its goals is at least
    :data:`MIN_GOALS_IN_FRAME` (a provider with no goal fails: the share cannot be judged).
    When ``table`` has whoscored rows: ``zones`` (:func:`zone_sides`) is given and
    :func:`confirms_direction` holds. An empty tuple means the criterion holds.

    Examples
    --------
    >>> table = pl.DataFrame({"provider": ["statsbomb"], "group": ["goal"], "shots": [200],
    ...                       "with_point": [200], "in_frame": [197], "near_woodwork": [3]})
    >>> criterion_problems(table, None)
    ('statsbomb: 197 of 200 goals inside the frame (98.50%), below 99%',)
    >>> criterion_problems(table.with_columns(in_frame=pl.lit(198)), None)
    ()
    """
    problems = []
    for provider, (inside, goals) in goal_shares(table).items():
        if goals == 0:
            problems.append(f"{provider}: no goal, so the goals-in-frame share cannot be judged")
        elif inside < MIN_GOALS_IN_FRAME * goals:
            problems.append(
                f"{provider}: {inside} of {goals} goals inside the frame ({inside / goals:.2%}), "
                f"below {MIN_GOALS_IN_FRAME:.0%}"
            )
    if "whoscored" in goal_shares(table):
        if zones is None:
            problems.append("whoscored: no zone table, so the Opta direction is not checked")
        elif not confirms_direction(zones):
            problems.append("whoscored: the zone qualifiers 76-87 do not confirm the direction")
    return tuple(problems)
