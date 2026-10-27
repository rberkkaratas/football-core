"""``goalmouth.census`` (spec ``runs/MS5/goalmouth`` §5.1, §10.2-3): goals, saves, woodwork and
misses per provider inside the goal frame and near the woodwork.

The fixture is typed column by column in pitch-width metres (posts 30.34 / 37.66), rows out of
key order, groups of unequal size on two providers. Every expected count below is worked out by
hand from the goal-frame arithmetic in the comments, not by calling ``pitch``.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from football_core import schema

PROVIDER = pl.Enum(("statsbomb", "whoscored"))
OUTCOME = pl.Enum(("goal", "saved", "blocked", "post", "off_target"))
NAN, INF = float("nan"), float("inf")
COUNT_COLUMNS = ("shots", "with_point", "in_frame", "near_woodwork")
CENSUS_SCHEMA = pl.Schema(
    {
        "provider": pl.String(),
        "group": pl.String(),
        **{c: pl.Int64() for c in COUNT_COLUMNS},
    }
)
READ_COLUMNS = ("provider", "outcome", "is_own_goal", "goal_y", "goal_z")


def _gm() -> Any:
    from football_core import goalmouth

    return goalmouth


# shot_id, provider, outcome, is_own_goal, goal_y, goal_z   (goal frame y = 34 - goal_y)
# fmt: off
ROWS = [
    ("s07", "statsbomb", "post", False, 38.2, 1.0),         # y -4.2: frame 0.54, post 0.42
    ("w02", "whoscored", "goal", False, 37.66, 0.0),        # y -3.66: in; on the post face
    ("s01", "statsbomb", "goal", False, 34.0, 1.0),         # centre: in; woodwork 1.44
    ("s11", "statsbomb", "goal", True, 35.0, 0.3),          # own goal: in; woodwork 2.14
    ("s05", "statsbomb", "saved", False, 36.0, 2.5901),     # z 0.1501 above the bar: out; near
    ("w05", "whoscored", "post", False, 34.0, INF),         # infinite height: never counts
    ("s02", "statsbomb", "goal", False, 30.19, 1.0),        # y 3.81: 0.15 - 1e-15 in float: in
    ("s09", "statsbomb", "off_target", False, None, None),  # no point
    ("w04", "whoscored", "off_target", False, NAN, 1.0),    # NaN: a point, never counts
    ("s03", "statsbomb", "goal", False, 30.1899, 1.0),      # y 3.8101: 0.1501 -> out; near
    ("s10", "statsbomb", "blocked", False, None, None),
    ("w03", "whoscored", "saved", False, 33.0, 1.0),        # y 1.0: in; woodwork 1.44
    ("s04", "statsbomb", "saved", False, 36.0, 2.59),       # z 0.15 - 1e-16 above the bar: in
    ("s12", "statsbomb", "off_target", False, 36.0, None),  # half a point: not with_point
    ("s06", "statsbomb", "post", False, 37.70, 1.2),        # y -3.70: in (0.04); in the post
    ("w01", "whoscored", "goal", False, 34.5, 0.5),         # in; woodwork 1.94
    ("s08", "statsbomb", "off_target", False, 40.0, 0.5),   # y -6: frame 2.34, post 2.22
    ("s13", "statsbomb", "goal", False, 30.1899999995, 1.0),   # frame 0.15 + 5e-10: in; near
    ("s14", "statsbomb", "goal", False, 30.1899999985, 1.0),   # 0.15 + 1.5e-9: out; near
    ("s15", "statsbomb", "saved", False, 34.0, 2.8600000005),  # woodwork 0.3 + 5e-10: near
    ("s16", "statsbomb", "saved", False, 34.0, 2.8600000015),  # 0.3 + 1.5e-9: not near
]
# provider, group, shots, with_point, in_frame, near_woodwork (provider, then GROUPS order)
EXPECTED = [
    ("statsbomb", "goal", 5, 5, 3, 4),
    ("statsbomb", "saved", 4, 4, 1, 3),
    ("statsbomb", "post", 2, 2, 1, 1),
    ("statsbomb", "off_target", 3, 1, 0, 0),
    ("statsbomb", "blocked", 1, 0, 0, 0),
    ("statsbomb", "own_goal", 1, 1, 1, 0),
    ("whoscored", "goal", 2, 2, 2, 1),
    ("whoscored", "saved", 1, 1, 1, 0),
    ("whoscored", "post", 1, 1, 0, 0),
    ("whoscored", "off_target", 1, 1, 0, 0),
]
# fmt: on


def _frame(rows: list[tuple[object, ...]]) -> pl.DataFrame:
    cols = list(zip(*rows, strict=True)) if rows else [[]] * 6
    return pl.DataFrame(
        [
            pl.Series("shot_id", cols[0], dtype=pl.String(), strict=True),
            pl.Series("provider", cols[1], dtype=PROVIDER, strict=True),
            pl.Series("outcome", cols[2], dtype=OUTCOME, strict=True),
            pl.Series("is_own_goal", cols[3], dtype=pl.Boolean(), strict=True),
            pl.Series("goal_y", cols[4], dtype=pl.Float64(), strict=True),
            pl.Series("goal_z", cols[5], dtype=pl.Float64(), strict=True),
        ]
    )


def _expected(rows: list[tuple[object, ...]]) -> pl.DataFrame:
    cols = list(zip(*rows, strict=True))
    return pl.DataFrame(
        {name: list(col) for name, col in zip(CENSUS_SCHEMA, cols, strict=True)},
        schema=CENSUS_SCHEMA,
    )


def test_constants() -> None:
    gm = _gm()
    assert gm.GOAL_TOLERANCE == 0.15
    assert gm.WOODWORK_TOLERANCE == 0.3
    assert gm.GROUPS == ("goal", "saved", "post", "off_target", "blocked", "own_goal")
    assert schema.DISTANCE_TOLERANCE == 1e-9


def test_census_counts() -> None:
    # Preconditions: rows out of key order; unequal groups; both boundary goals present.
    ids = [r[0] for r in ROWS]
    assert ids != sorted(ids)
    assert {r[0] for r in ROWS if r[4] in (30.19, 30.1899)} == {"s02", "s03"}
    out = _gm().census(_frame(ROWS))
    assert out.schema == CENSUS_SCHEMA
    assert out.to_dicts() == _expected(EXPECTED).to_dicts()


def test_census_reads_only_its_columns() -> None:
    """§5.1: only provider, outcome, is_own_goal, goal_y, goal_z are read; other columns and
    their order change nothing."""
    base = _gm().census(_frame(ROWS))
    shuffled = (
        _frame(ROWS)
        .with_columns(pl.lit(1.5).alias("x"))
        .select("goal_z", "x", "outcome", "goal_y", "provider", "is_own_goal")
    )
    assert _gm().census(shuffled).to_dicts() == base.to_dicts()


def test_census_empty() -> None:
    out = _gm().census(_frame([]))
    assert out.height == 0
    assert out.schema == CENSUS_SCHEMA


@pytest.mark.parametrize("column", READ_COLUMNS)
def test_census_missing_column(column: str) -> None:
    with pytest.raises(ValueError, match=column):
        _gm().census(_frame(ROWS).drop(column))


def test_census_own_goal_is_its_own_group() -> None:
    """§5.1: ``group`` is ``own_goal`` for an own goal (outcome goal), never ``goal``; an
    absent group has no row."""
    rows = [r for r in ROWS if r[1] == "statsbomb" and r[2] == "goal"]
    out = _gm().census(_frame(rows))
    assert out.select("group", "shots").rows() == [("goal", 5), ("own_goal", 1)]


def test_census_null_own_goal_and_unknown_group() -> None:
    """Build ratify 8: a null ``is_own_goal`` puts the row in its outcome's group, and a group
    outside ``GROUPS`` sorts after every known group (neither occurs in validated data)."""
    shots = pl.DataFrame(
        [
            pl.Series("provider", ["whoscored"] * 4, dtype=PROVIDER),
            pl.Series("outcome", ["weird", "goal", "goal", "goal"], dtype=pl.String()),
            pl.Series("is_own_goal", [False, None, True, False], dtype=pl.Boolean()),
            pl.Series("goal_y", [34.0, 34.0, None, 34.0], dtype=pl.Float64()),
            pl.Series("goal_z", [1.0, 1.0, None, 1.0], dtype=pl.Float64()),
        ]
    )
    out = _gm().census(shots)
    assert out.select("group", "shots").rows() == [("goal", 2), ("own_goal", 1), ("weird", 1)]


@pytest.mark.parametrize(
    ("shot_id", "distance", "tolerance", "column", "counted"),
    [
        ("s13", "frame", 0.15, "in_frame", True),
        ("s14", "frame", 0.15, "in_frame", False),
        ("s15", "woodwork", 0.3, "near_woodwork", True),
        ("s16", "woodwork", 0.3, "near_woodwork", False),
    ],
)
def test_census_tolerance_boundaries(
    shot_id: str, distance: str, tolerance: float, column: str, counted: bool
) -> None:
    """The criterion's "within 0.15 m" (and the 0.3 m woodwork band) is two-sided and inclusive
    up to ``schema.DISTANCE_TOLERANCE``: a point whose float distance lies in [tol, tol + 1e-9]
    counts, one just beyond it does not (run5-ms5 Q3: the "exactly 0.15" rows sat inside in
    float, so a strict ``<`` survived)."""
    from football_core import pitch

    row = next(r for r in ROWS if r[0] == shot_id)
    gy, gz = pitch.to_goal_frame([row[4]], [row[5]])
    fn = pitch.frame_distance if distance == "frame" else pitch.woodwork_distance
    d = float(fn(gy, gz)[0])
    if counted:  # precondition: on the boundary in float, not inside it
        assert tolerance <= d <= tolerance + schema.DISTANCE_TOLERANCE
    else:
        assert tolerance + schema.DISTANCE_TOLERANCE < d < tolerance + 1e-8
    out = _gm().census(_frame([row]))
    assert out[column].to_list() == [int(counted)]


def test_criterion_problems() -> None:
    """``criterion_problems``: each provider needs >= 99 % of its goals inside the frame (99 %
    exactly passes, one goal fewer fails, no goal fails); whoscored also needs a zone table that
    confirms the direction."""
    gm = _gm()
    assert gm.MIN_GOALS_IN_FRAME == 0.99

    def table(sb_in: int, ws_goals: int = 100) -> pl.DataFrame:
        rows = [("statsbomb", "goal", 200, 200, sb_in, 0), ("statsbomb", "saved", 9, 9, 1, 0),
                ("whoscored", "goal", ws_goals, ws_goals, ws_goals, 0),
                ("whoscored", "saved", 5, 5, 5, 0)]  # fmt: skip
        return _expected(rows)

    zones = pl.DataFrame({
        "zone": [76, 80], "named_side": ["left", "right"], "shots": [5, 4],
        "shooter_left": [5, 0], "shooter_right": [0, 4], "centre": [0, 0],
    })  # fmt: skip
    assert gm.criterion_problems(table(198), zones) == ()
    (miss,) = gm.criterion_problems(table(197), zones)
    assert miss.startswith("statsbomb: 197 of 200")
    (no_goal,) = gm.criterion_problems(table(198, ws_goals=0), zones)
    assert no_goal.startswith("whoscored: no goal")
    flipped = zones.with_columns(shooter_left=pl.Series([4, 1]), shooter_right=pl.Series([1, 3]))
    assert gm.criterion_problems(table(198), flipped) == (
        "whoscored: the zone qualifiers 76-87 do not confirm the direction",
    )
    assert "not checked" in gm.criterion_problems(table(198), None)[0]
    assert gm.criterion_problems(table(198).filter(pl.col("provider") == "statsbomb"), None) == ()
