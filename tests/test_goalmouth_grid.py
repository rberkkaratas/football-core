"""``goalmouth.grid_census`` and ``goalmouth.lattice_problems`` (spec ``runs/MS6/xgot-ready`` §2.3,
§2.4, §6, §8): where StatsBomb end locations sit on the whole-yard and the 0.1 yd grid, by
fidelity, outcome and stop point, and whether every stored StatsBomb ``goal_y`` is on the 0.1 yd
goal-mouth lattice.

The census fixture is typed column by column with invented raw values: a coarse match (every
shot ``location`` a whole yard, though its end locations carry tenths, so the grid is read from
``location`` only), a fine match, and a third match whose shots sit on whole yards but whose own
goal does not (so it is fine: own goals count for the grid, never in the table). Every expected
count is worked out by hand in the comments. Rows are out of key order.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest
import statsbomb_cases as sbc

from football_core import goalmouth, schema

READ = ("provider", "match_id", "outcome", "is_own_goal", "raw_qualifiers")
COUNTS = ("shots", "x_whole", "y_whole", "y_odd_tenth", "with_z", "z_whole")
CENSUS_SCHEMA = pl.Schema(
    {"fidelity": pl.String(), "group": pl.String(), "stop": pl.String(),
     **{c: pl.Int64() for c in COUNTS}}
)  # fmt: skip
OUTCOME_NAMES = {"goal": "Goal", "saved": "Saved", "post": "Post", "off_target": "Off T",
                 "blocked": "Blocked"}  # fmt: skip


def _raw(location: Sequence[float] | None, end: Sequence[float] | None, outcome: str) -> str:
    """A StatsBomb event as ``raw_qualifiers`` stores it (canonical JSON, no freeze frame)."""
    ev: dict[str, object] = {"type": {"name": "Shot"}}
    if location is not None:
        ev["location"] = list(location)
    if outcome == "own_goal":
        ev["type"] = {"name": "Own Goal Against"}
    else:
        shot: dict[str, object] = {"outcome": {"name": OUTCOME_NAMES[outcome]}}
        if end is not None:
            shot["end_location"] = list(end)
        ev["shot"] = shot
    return json.dumps(ev, sort_keys=True, separators=(",", ":"))


def _frame(rows: Sequence[tuple[str, str, str, Sequence[float] | None, Sequence[float] | None]]
           ) -> pl.DataFrame:  # fmt: skip
    """``(provider, match_id, outcome, location, end_location)`` rows as C3-typed columns."""
    provider = [r[0] for r in rows]
    outcome = [r[2] for r in rows]
    return pl.DataFrame(
        {
            "provider": pl.Series(provider, dtype=schema.PROVIDER),
            "match_id": pl.Series([r[1] for r in rows], dtype=pl.String()),
            "outcome": pl.Series(["goal" if o == "own_goal" else o for o in outcome],
                                 dtype=schema.OUTCOME),
            "is_own_goal": pl.Series([o == "own_goal" for o in outcome], dtype=pl.Boolean()),
            "raw_qualifiers": pl.Series(
                ["[]" if r[0] == "whoscored" else _raw(r[3], r[4], r[2]) for r in rows],
                dtype=pl.String(),
            ),
        }
    )  # fmt: skip


C, F, W = "sb:3001", "sb:3002", "sb:3003"  # coarse; fine; whole-yard shots + a tenths own goal
# fmt: off
ROWS = [
    # provider, match, outcome, raw location, raw end_location            stop        x y odd z zw
    ("statsbomb", F, "saved", (102.0, 36.4), (119.8, 41.2, 0.4)),       # projected  0 0 0  1 0
    ("statsbomb", C, "saved", (102.0, 38.0), (119.0, 40.1, 1.0)),       # near line  1 0 1  1 1
    ("whoscored", "ws:9", "saved", None, None),                          # ignored
    ("statsbomb", W, "saved", (100.0, 40.0), (119.0, 40.0, 1.0)),       # projected  1 1 0  1 1
    ("statsbomb", C, "blocked", (99.0, 35.0), (110.0, 37.0)),           # no mouth   1 1 0  0 0
    ("statsbomb", F, "goal", (100.5, 40.2), (120.0, 38.3, 0.85)),       # line       1 0 1  1 0
    ("statsbomb", C, "off_target", (98.0, 33.0), (120.0, 30.3, 2.7)),   # line       1 0 1  1 0
    ("statsbomb", F, "blocked", (104.1, 33.3), (106.2, 35.5)),          # no mouth   0 0 1  0 0
    ("statsbomb", C, "goal", (100.0, 40.0), (120.0, 38.4, 0.5)),        # line       1 0 0  1 0
    ("statsbomb", F, "saved", (101.3, 39.0), (119.9, 40.0, 1.2)),       # near line  0 1 0  1 0
    ("statsbomb", W, "own_goal", (5.5, 30.0), None),                     # not counted; W is fine
    ("statsbomb", C, "saved", (101.0, 41.0), (118.0, 39.0, 2.0)),       # projected  1 1 0  1 1
    ("statsbomb", F, "off_target", (95.5, 30.1), (115.0, 60.0)),        # no mouth   1 1 0  0 0
    ("statsbomb", C, "post", (100.0, 44.0), (120.0, 36.0, 2.3)),        # line       1 1 0  1 0
    ("statsbomb", F, "saved", (95.5, 30.1), (110.0, 38.7, 0.6)),        # projected  1 0 1  1 0
    ("statsbomb", C, "goal", (103.0, 37.0), (117.0, 41.7, 0.2)),        # projected  1 0 1  1 0
    ("statsbomb", C, "own_goal", (6.0, 31.0), None),                     # not counted
]
# fidelity, group, stop, shots, x_whole, y_whole, y_odd_tenth, with_z, z_whole
EXPECTED = [
    ("coarse", "goal", "line", 1, 1, 0, 0, 1, 0),
    ("coarse", "goal", "projected", 1, 1, 0, 1, 1, 0),
    ("coarse", "saved", "near line", 1, 1, 0, 1, 1, 1),
    ("coarse", "saved", "projected", 1, 1, 1, 0, 1, 1),
    ("coarse", "post", "line", 1, 1, 1, 0, 1, 0),
    ("coarse", "off_target", "line", 1, 1, 0, 1, 1, 0),
    ("coarse", "blocked", "no mouth", 1, 1, 1, 0, 0, 0),
    ("fine", "goal", "line", 1, 1, 0, 1, 1, 0),
    ("fine", "saved", "near line", 1, 0, 1, 0, 1, 0),
    ("fine", "saved", "projected", 3, 2, 1, 1, 3, 1),   # F 119.8 / F 110.0 / W 119.0 (fine grid)
    ("fine", "off_target", "no mouth", 1, 1, 1, 0, 0, 0),
    ("fine", "blocked", "no mouth", 1, 0, 0, 1, 0, 0),
]
# fmt: on


def test_grid_census_counts() -> None:
    """§2.3: one row per (fidelity, group, stop) present, sorted by fidelity, ``GROUPS`` order
    and stop order (line, near line, projected, no mouth); whole means the raw value equals its
    rounding exactly; ``y_odd_tenth`` counts odd ``round(y · 10)``. The grid is the match's, from
    the ``location`` of all its StatsBomb rows (own goals included, end locations not): coarse
    119 is near the line and fine 119.9 too, while 119 on a fine match is projected. Own goals
    and WhoScored rows are not counted."""
    got = goalmouth.grid_census(_frame(ROWS))
    assert got.schema == CENSUS_SCHEMA
    assert got.rows() == EXPECTED
    assert set(got["group"]) <= set(goalmouth.GROUPS)
    # the near miss on the grid: match W's shots alone would read coarse (119.0 near the line)
    alone = goalmouth.grid_census(_frame([r for r in ROWS if r[1] == W and r[2] != "own_goal"]))
    assert alone.rows() == [("coarse", "saved", "near line", 1, 1, 1, 0, 1, 1)]


@pytest.mark.parametrize("column", READ)
def test_grid_census_missing_column(column: str) -> None:
    """§2.3 Raises: a missing column, named."""
    with pytest.raises(ValueError, match=column):
        goalmouth.grid_census(_frame(ROWS).drop(column))


# fmt: off
BAD = {
    "no location": [("statsbomb", C, "saved", None, (118.0, 39.0, 2.0))],
    "no end_location": [("statsbomb", C, "saved", (101.0, 41.0), None)],
    "one-number end": [("statsbomb", C, "saved", (101.0, 41.0), (118.0,))],
    "four-number end": [("statsbomb", C, "saved", (101.0, 41.0), (118.0, 39.0, 2.0, 1.0))],
    "one-number location": [("statsbomb", C, "goal", (101.0,), (120.0, 39.0, 2.0))],
    "text in end": [("statsbomb", C, "goal", (101.0, 41.0), (120.0, "x", 2.0))],
}
# fmt: on


@pytest.mark.parametrize("case", list(BAD))
def test_grid_census_errors(case: str) -> None:
    """§2.3 Raises: a StatsBomb non-own-goal row whose ``raw_qualifiers`` lacks a 2- or 3-number
    ``location`` / ``shot.end_location``; the message carries the count (here 5: the bad row
    five times, beside good rows)."""
    bad = BAD[case] * 5
    with pytest.raises(ValueError, match=r"\b5\b"):
        goalmouth.grid_census(_frame([*ROWS, *bad]))


def test_grid_census_null_match_id() -> None:
    """§6: a null ``match_id`` (refused upstream by C3) raises if one reaches the census."""
    frame = _frame(ROWS).with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("match_id"))
        .alias("match_id")
    )  # fmt: skip
    with pytest.raises(ValueError, match="match_id"):
        goalmouth.grid_census(frame)


def test_grid_census_empty() -> None:
    """§2.3 / §6: no StatsBomb row (an empty frame, or WhoScored rows only, or StatsBomb own goals
    only) gives an empty table with the census schema."""
    for rows in ([], [r for r in ROWS if r[0] == "whoscored"],
                 [r for r in ROWS if r[2] == "own_goal"]):  # fmt: skip
        frame = _frame(rows) if rows else _frame(ROWS).clear()
        got = goalmouth.grid_census(frame)
        assert got.height == 0 and got.schema == CENSUS_SCHEMA


# --- §2.4 the lattice check -----------------------------------------------------------------


def _goal_y(provider: Sequence[str], goal_y: Sequence[float | None]) -> pl.DataFrame:
    return pl.DataFrame({
        "provider": pl.Series(list(provider), dtype=schema.PROVIDER),
        "goal_y": pl.Series(list(goal_y), dtype=pl.Float64()),
    })  # fmt: skip


def test_lattice_problems() -> None:
    """§2.4: StatsBomb rows with a non-null ``goal_y`` off the 0.1 yd lattice by more than
    ``DISTANCE_TOLERANCE`` are counted in one problem string; on-lattice values (cell centres,
    through the metre map), WhoScored rows and nulls are not; NaN counts as off (fails
    closed)."""
    on = [sbc.mouth_metres(38.35), sbc.mouth_metres(0.05), sbc.mouth_metres(62.05)]
    off = sbc.mouth_metres(41.05) + 1e-6
    half = sbc.mouth_metres(41.05) + schema.DISTANCE_TOLERANCE / 2  # inside the tolerance: on
    clean = _goal_y(["statsbomb"] * 5 + ["whoscored"], [*on, half, None, 33.33333])
    assert goalmouth.lattice_problems(clean) == ()
    dirty = clean.vstack(_goal_y(["statsbomb", "statsbomb"], [off, float("nan")]))
    assert goalmouth.lattice_problems(dirty) == (
        "statsbomb: 2 of 6 goal_y values are off the 0.1 yd goal-mouth lattice",
    )
    assert goalmouth.lattice_problems(dirty.clear()) == ()
    ws_only = _goal_y(["whoscored"], [33.33333])
    assert goalmouth.lattice_problems(ws_only) == ()
    assert np.isfinite(off) and abs(off - sbc.snapped(off)) > 1e-7  # precondition: truly off
