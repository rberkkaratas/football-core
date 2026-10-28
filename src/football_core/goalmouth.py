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

**StatsBomb grid (MS6).** :func:`grid_census` counts StatsBomb end locations on whole yards
and on tenths, by the match's grid, outcome and where the ball stopped, and
:func:`lattice_problems` checks that every stored ``goal_y`` sits on the 0.1 yd goal-mouth
lattice that ``io.statsbomb.snap_goal_y`` puts projected points on.

Counts only: nothing here returns a row, an id or a player, so the output may be quoted for
WhoScored data too.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import polars as pl

from football_core import pitch, schema
from football_core.io import statsbomb

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
STOPS: Final = ("line", "near line", "projected", "no mouth")  # grid_census, in this order
_GRID_COUNTS: Final = ("shots", "x_whole", "y_whole", "y_odd_tenth", "with_z", "z_whole")
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


def _numbers(value: object) -> list[float] | None:
    """A JSON list of 2 or 3 numbers as floats, else None."""
    if not isinstance(value, list) or len(value) not in (2, 3):
        return None
    if any(not isinstance(v, int | float) or isinstance(v, bool) for v in value):
        return None
    return [float(v) for v in value]


def _raw_ends(texts: Sequence[str | None], own: Sequence[bool]) -> tuple[list[Any], int]:
    """Per row ``(location, end_location)`` from ``raw_qualifiers`` and the count of bad rows.

    An own goal has no end location (``None``) and a bad own-goal location is left out of the
    grid inference; a shot needs both."""
    rows: list[Any] = []
    bad = 0
    for text, is_own in zip(texts, own, strict=True):
        try:
            event = json.loads(text) if isinstance(text, str) else None
        except ValueError:
            event = None
        event = event if isinstance(event, dict) else {}
        shot = event.get("shot")
        loc = _numbers(event.get("location"))
        end = (
            None
            if is_own
            else _numbers(shot.get("end_location") if isinstance(shot, dict) else None)
        )
        if not is_own and (loc is None or end is None):
            bad += 1
        rows.append((loc, end))
    return rows, bad


def grid_census(shots: pl.DataFrame) -> pl.DataFrame:
    """Count StatsBomb end locations on whole yards and on tenths, by grid, outcome and stop.

    Parameters
    ----------
    shots : polars.DataFrame
        C3 shots; reads ``provider``, ``match_id``, ``outcome``, ``is_own_goal`` and
        ``raw_qualifiers`` (the raw ``location`` and ``shot.end_location``, in yards). Only
        StatsBomb rows count, and own goals are left out (they have no end location).

    Returns
    -------
    polars.DataFrame
        One row per (``fidelity``, ``group``, ``stop``) present, sorted by fidelity, ``GROUPS``
        order and ``STOPS`` order. ``fidelity`` is the match's grid by the loader's own rule
        (``io.statsbomb.cell_offset`` over the raw ``location`` of all its StatsBomb rows):
        ``"fine"`` equals ``metadata.shot_fidelity_version`` "2" and ``"coarse"`` "1" wherever
        the matches file states one (the loader refuses a disagreement), and is inferred
        otherwise. ``stop`` is ``"line"`` (raw end x 120), ``"near line"`` (a goal-mouth point
        within one grid cell of the line, kept as recorded), ``"projected"`` (a goal-mouth
        point ``io.statsbomb.is_projected`` moves: exactly the rows the snap moves) or ``"no
        mouth"`` (a blocked shot or a 2-value end). Int64 counts: ``shots``, ``x_whole`` /
        ``y_whole`` (the raw end x / y equals its rounding), ``y_odd_tenth`` (``round(y · 10)``
        is odd), ``with_z`` (a 3-value end) and ``z_whole``. No StatsBomb row gives an empty
        table with these columns.

    Raises
    ------
    ValueError
        A missing column, named; a null ``match_id`` on a StatsBomb row; a StatsBomb shot whose
        ``raw_qualifiers`` lacks a 2- or 3-number ``location`` or ``shot.end_location`` (the
        count in the message).

    Examples
    --------
    >>> raw = ['{"location":[100.0,40.0],"shot":{"end_location":[120.0,39.9,1.0]}}',
    ...        '{"location":[100.0,40.0],"shot":{"end_location":[118.0,41.0,0.5]}}']
    >>> shots = pl.DataFrame({"provider": ["statsbomb"] * 2, "match_id": ["sb:1"] * 2,
    ...                       "outcome": ["goal", "saved"], "is_own_goal": [False, False],
    ...                       "raw_qualifiers": raw})
    >>> grid_census(shots).rows()
    [('coarse', 'goal', 'line', 1, 1, 0, 1, 1, 1), ('coarse', 'saved', 'projected', 1, 1, 1, 0, \
1, 0)]
    """
    _require(shots, ("provider", "match_id", "outcome", "is_own_goal", "raw_qualifiers"),
             "grid_census")  # fmt: skip
    sb = shots.filter(pl.col("provider").cast(pl.String) == "statsbomb")
    if nulls := sb["match_id"].null_count():
        raise ValueError(f"grid_census: {nulls} StatsBomb row(s) have a null match_id")
    own = sb["is_own_goal"].fill_null(False).to_list()
    ends, bad = _raw_ends(sb["raw_qualifiers"].to_list(), own)
    if bad:
        raise ValueError(
            f"grid_census: {bad} StatsBomb shot(s) lack a 2- or 3-number location or "
            "shot.end_location in raw_qualifiers"
        )
    by_match: dict[str, list[list[float]]] = {}
    for mid, (loc, _) in zip(sb["match_id"].to_list(), ends, strict=True):
        by_match.setdefault(mid, []).extend([] if loc is None else [loc])
    coarse = {m: statsbomb.cell_offset(0, locs, None) == statsbomb.COARSE_OFFSET
              for m, locs in by_match.items()}  # fmt: skip
    keep = [not o for o in own]
    rows = sb.filter(pl.Series(keep, dtype=pl.Boolean))
    end = [e for e, k in zip(ends, keep, strict=True) if k]
    offset = np.array([statsbomb.COARSE_OFFSET if coarse[m] else statsbomb.FINE_OFFSET
                       for m in rows["match_id"].to_list()], dtype=np.float64)  # fmt: skip
    ex = np.array([e[1][0] for e in end], dtype=np.float64)
    ey = np.array([e[1][1] for e in end], dtype=np.float64)
    ez = np.array([e[1][2] if len(e[1]) == 3 else np.nan for e in end], dtype=np.float64)
    blocked = (rows["outcome"].cast(pl.String) == "blocked").fill_null(False).to_numpy()
    mouth = ~np.isnan(ez) & ~blocked
    line, projected = ex == 120.0, mouth & statsbomb.is_projected(ex, offset)
    stop = np.where(line, STOPS[0], np.where(projected, STOPS[2],
                    np.where(mouth, STOPS[1], STOPS[3])))  # fmt: skip
    flags = pl.DataFrame({
        "fidelity": ["coarse" if coarse[m] else "fine" for m in rows["match_id"].to_list()],
        "group": rows["outcome"].cast(pl.String),
        "stop": stop.astype(str),
        "shots": np.ones(rows.height, dtype=bool),
        "x_whole": ex == np.round(ex),
        "y_whole": ey == np.round(ey),
        "y_odd_tenth": np.round(ey * 10) % 2 == 1,
        "with_z": ~np.isnan(ez),
        "z_whole": ez == np.round(ez),
    }, schema_overrides={"fidelity": pl.String, "group": pl.String, "stop": pl.String})  # fmt: skip
    table = flags.group_by("fidelity", "group", "stop").agg(
        pl.col(_GRID_COUNTS).sum().cast(pl.Int64)
    )
    rank = {g: i for i, g in enumerate(GROUPS)}
    order = pl.col("group").replace_strict(rank, default=None, return_dtype=pl.Int64)
    stops = pl.col("stop").replace_strict({s: i for i, s in enumerate(STOPS)},
                                          return_dtype=pl.Int64)  # fmt: skip
    return table.sort(pl.col("fidelity"), order, stops, nulls_last=True)


def lattice_problems(shots: pl.DataFrame) -> tuple[str, ...]:
    """Whether every StatsBomb ``goal_y`` sits on the 0.1 yd goal-mouth lattice.

    Counts the StatsBomb rows with a non-null ``goal_y`` that ``io.statsbomb.snap_goal_y``
    moves by more than ``schema.DISTANCE_TOLERANCE``; a NaN counts as off (a loader bug, so the
    check fails closed). WhoScored rows are ignored: Opta records the goal-plane point.

    Returns
    -------
    tuple of str
        One problem message when a value is off, else ``()``.

    Raises
    ------
    ValueError
        A missing column, named.

    Examples
    --------
    >>> on = float(statsbomb.snap_goal_y([33.0])[0])
    >>> shots = pl.DataFrame({"provider": ["statsbomb"] * 2, "goal_y": [on, None]})
    >>> lattice_problems(shots)
    ()
    >>> lattice_problems(shots.with_columns(goal_y=pl.lit(on + 1e-6)))
    ('statsbomb: 2 of 2 goal_y values are off the 0.1 yd goal-mouth lattice',)
    """
    off, points = lattice_counts(shots)
    if off:
        return (
            f"statsbomb: {off} of {points} goal_y values are off the "
            f"{statsbomb.MOUTH_CELL:g} yd goal-mouth lattice",
        )
    return ()


def lattice_counts(shots: pl.DataFrame) -> tuple[int, int]:
    """``(off, points)``: StatsBomb ``goal_y`` values off the lattice, of those not null.

    The rule of :func:`lattice_problems` (NaN counts as off; WhoScored rows are ignored).

    Raises
    ------
    ValueError
        A missing column, named.

    Examples
    --------
    >>> lattice_counts(pl.DataFrame({"provider": ["statsbomb", "whoscored"],
    ...                              "goal_y": [float("nan"), 33.0]}))
    (1, 1)
    """
    _require(shots, ("provider", "goal_y"), "lattice_problems")
    sb = shots.filter(
        (pl.col("provider").cast(pl.String) == "statsbomb") & pl.col("goal_y").is_not_null()
    )
    gy = sb["goal_y"].cast(pl.Float64).to_numpy()
    with np.errstate(invalid="ignore"):  # NaN compares False: it counts as off
        on = np.abs(statsbomb.snap_goal_y(gy) - gy) <= schema.DISTANCE_TOLERANCE
    return int((~on).sum()), int(gy.size)
