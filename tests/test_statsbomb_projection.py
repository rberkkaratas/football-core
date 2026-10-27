"""StatsBomb stop points projected to the goal line (spec ``runs/MS5/goalmouth`` §3.1, §10.5).

Outcome-blind and grid-aware: a stop point with raw ``end_x >= 120 - 2c`` (one grid cell: 1 yd on
coarse matches, 0.1 yd on fine ones) is on the line and keeps its recorded ``goal_y``; any other
goal-mouth point gets the y where the line from the shot location through the stop point meets
x = 105 (metres). ``goal_z`` stays the recorded height. Expected values are computed here from
the stored ``x, y, end_x, end_y`` with the straight-line formula, not with ``pitch``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import statsbomb_cases as sbc
from statsbomb_cases import FINE, YARD

TOL = 1e-9
# Shot locations on a fine / an integer grid: an angled run-in, so one grid cell of projection
# moves the point by more than a centimetre.
FINE_AT, COARSE_AT = (114.05, 45.05), (114, 45)


def _line(r: dict[str, Any]) -> float:
    """The goal-line crossing of the line from (x, y) through (end_x, end_y), clipped."""
    k = (105.0 - r["x"]) / (r["end_x"] - r["x"])
    return min(max(r["y"] + (r["end_y"] - r["y"]) * k, 0.0), 68.0)


def _load(tmp_path: Path, *events: sbc.Event, coarse: bool = False) -> Any:
    match = sbc.one_match(*events, fidelity="1" if coarse else "2")
    return sbc.load_matches(tmp_path / "cache", [match]).shots


def _shot(
    eid: str,
    outcome: str,
    end: tuple[float, ...],
    *,
    coarse: bool,
    minute: int,
    location: tuple[float, float] | None = None,
) -> Any:
    return sbc.shot(
        eid,
        location=location or (COARSE_AT if coarse else FINE_AT),
        outcome=outcome,
        end_location=end,
        freeze_frame=sbc.default_frame(sbc.HOME, coarse=coarse),
        minute=minute,
    )


def test_saved_projected(tmp_path: Path) -> None:
    """A save stopped at 114 yd: goal_y on the goal line; stop point and height as recorded."""
    shots = _load(tmp_path, _shot("fake-shot-p01", "Saved", (114.05, 39.05, 0.55), coarse=False,
                                  minute=10, location=(102.35, 35.65)))  # fmt: skip
    r = sbc.row(shots, "fake-shot-p01")
    ex, ey = sbc.metres(114.05, 39.05, FINE)
    assert (r["end_x"], r["end_y"]) == pytest.approx((ex, ey), abs=TOL)  # the stop point stays
    assert abs(_line(r) - r["end_y"]) > 0.1  # precondition: projecting moves the point
    assert r["goal_y"] == pytest.approx(_line(r), abs=TOL)
    assert r["goal_z"] == pytest.approx(0.55 * YARD, abs=TOL)


def test_projection_outcome_blind(tmp_path: Path) -> None:
    """§3.1: the mask reads raw end_x and the cell offset, never the outcome: a goal, a save, a
    post and a miss with one geometry get one goal_y."""
    end = (116.05, 39.05, 0.55)
    outcomes = {"fake-shot-o01": "Goal", "fake-shot-o02": "Saved", "fake-shot-o03": "Post",
                "fake-shot-o04": "Off T"}  # fmt: skip
    events = [
        _shot(eid, outcome, end, coarse=False, minute=10 + i, location=(102.35, 35.65))
        for i, (eid, outcome) in enumerate(outcomes.items())
    ]
    shots = _load(tmp_path, *events)
    rows = [sbc.row(shots, eid) for eid in outcomes]
    assert abs(_line(rows[0]) - rows[0]["end_y"]) > 0.1
    for r in rows:
        assert r["goal_y"] == pytest.approx(_line(rows[0]), abs=TOL), r["shot_id"]
        assert r["goal_z"] == pytest.approx(0.55 * YARD, abs=TOL), r["shot_id"]


# grid, outcome, raw 3-value end location, projected?
# fmt: off
GRID_CASES = [
    ("coarse_goal_at_119_on_line", True, "Goal", (119.0, 40.1, 0.5), False),
    ("coarse_save_at_118_projected", True, "Saved", (118.0, 40.1, 0.5), True),
    ("fine_save_at_119.9_on_line", False, "Saved", (119.9, 40.05, 0.55), False),
    ("fine_save_at_119.8_projected", False, "Saved", (119.8, 40.05, 0.55), True),
    ("fine_save_at_118_projected", False, "Saved", (118.0, 40.05, 0.55), True),
]
# fmt: on


@pytest.mark.parametrize(
    ("coarse", "outcome", "end", "projected"),
    [c[1:] for c in GRID_CASES],
    ids=[c[0] for c in GRID_CASES],
)
def test_grid_cell_not_projected(
    tmp_path: Path, coarse: bool, outcome: str, end: tuple[float, ...], projected: bool
) -> None:
    shots = _load(tmp_path, _shot("fake-shot-g01", outcome, end, coarse=coarse, minute=10),
                  coarse=coarse)  # fmt: skip
    r = sbc.row(shots, "fake-shot-g01")
    recorded = sbc.metres_y(end[1] - FINE)  # the recorded goal-mouth point (tenths on both grids)
    assert abs(_line(r) - recorded) > 0.01  # precondition: the two rules disagree here
    expected = _line(r) if projected else recorded
    assert r["goal_y"] == pytest.approx(expected, abs=TOL)
    assert r["goal_z"] == pytest.approx(end[2] * YARD, abs=TOL)
    assert math.isfinite(r["end_x"]) and r["end_x"] < 105.0  # a stop point before the line
    assert 30.34 <= expected <= 37.66  # valid football: every case is on target
