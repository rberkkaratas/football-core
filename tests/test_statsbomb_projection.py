"""StatsBomb stop points projected to the goal line (spec ``runs/MS5/goalmouth`` §3.1, §10.5) and
snapped to the 0.1 yd goal-mouth lattice (spec ``runs/MS6/xgot-ready`` §2.2, §6, §8).

Outcome-blind and grid-aware: a stop point with raw ``end_x >= 120 - 2c`` (one grid cell: 1 yd on
coarse matches, 0.1 yd on fine ones) is on the line and keeps its recorded ``goal_y``; any other
goal-mouth point gets the y where the line from the shot location through the stop point meets
x = 105 (metres), moved to the centre of the 0.1 yd cell that holds it (MS6: goal-mouth y is
recorded in tenths in every match, so a projected save sits on the lattice the recorded goals sit
on). ``goal_z`` stays the recorded height. Expected values are computed here from the stored
``x, y, end_x, end_y`` with the straight-line formula and the cell rule written out in
``statsbomb_cases.snapped``, not with the loader's helpers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import hypothesis.strategies as st
import numpy as np
import pytest
import statsbomb_cases as sbc
from hypothesis import given, settings
from statsbomb_cases import FINE, YARD

from football_core.io import statsbomb

TOL = 1e-9
NAN, INF = float("nan"), float("inf")
# Shot locations on a fine / an integer grid: an angled run-in, so one grid cell of projection
# moves the point by more than a centimetre.
FINE_AT, COARSE_AT = (114.05, 45.05), (114, 45)
# the largest metres-per-yard slope of the StatsBomb goal-mouth map (11 m over 12 yd)
HALF_CELL_M = 0.05 * 11.0 / 12.0


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
    """A save stopped at 114 yd: goal_y on the goal line, in its 0.1 yd cell; the stop point and
    the height as recorded."""
    shots = _load(tmp_path, _shot("fake-shot-p01", "Saved", (114.05, 39.05, 0.55), coarse=False,
                                  minute=10, location=(102.35, 35.65)))  # fmt: skip
    r = sbc.row(shots, "fake-shot-p01")
    ex, ey = sbc.metres(114.05, 39.05, FINE)
    assert (r["end_x"], r["end_y"]) == pytest.approx((ex, ey), abs=TOL)  # the stop point stays
    assert abs(_line(r) - r["end_y"]) > 0.1  # precondition: projecting moves the point
    assert abs(sbc.snapped(_line(r)) - _line(r)) > 0.001  # precondition: so does the snap
    assert r["goal_y"] == pytest.approx(sbc.snapped(_line(r)), abs=TOL)
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
        assert r["goal_y"] == pytest.approx(sbc.snapped(_line(rows[0])), abs=TOL), r["shot_id"]
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
    expected = sbc.snapped(_line(r)) if projected else recorded
    assert r["goal_y"] == pytest.approx(expected, abs=TOL)
    assert r["goal_z"] == pytest.approx(end[2] * YARD, abs=TOL)
    assert math.isfinite(r["end_x"]) and r["end_x"] < 105.0  # a stop point before the line
    assert 30.34 <= expected <= 37.66  # valid football: every case is on target


# ---- MS6 xgot-ready §2.2: the snap ----------------------------------------------------------


def test_snap_constants() -> None:
    """§2.2: the goal-mouth cell is 0.1 yd, 800 cells span the 80 yd width, the tolerance is in
    cells, and the fine half-cell is half a goal-mouth cell (value unchanged)."""
    assert (statsbomb.MOUTH_CELL, statsbomb.MOUTH_CELLS, statsbomb.SNAP_TOL) == (0.1, 800, 1e-4)
    assert statsbomb.FINE_OFFSET == statsbomb.MOUTH_CELL / 2 == 0.05


# raw StatsBomb yards in -> the cell centre out (a value names the cell that ends at it)
# fmt: off
SNAP_CASES = [
    ("inside a cell", 39.97, 39.95),
    ("lower half of a cell (ceil, not round)", 39.93, 39.95),
    ("on a cell end", 40.0, 39.95),
    ("the cell below", 39.9, 39.85),
    ("tolerance: 0.5e-4 cells above an end", 39.9 + 5e-6, 39.85),  # 399.00005 cells
    ("past tolerance: 2e-4 cells above", 39.9 + 2e-5, 39.95),      # 399.0002 cells
    ("right post", 44.0, 43.95),
    ("beyond the box side", 62.03, 62.05),
    ("edge 0 yd -> first cell", 0.0, 0.05),
    ("edge 80 yd -> last cell", 80.0, 79.95),
]
# fmt: on


@pytest.mark.parametrize(
    ("yd", "want"), [c[1:] for c in SNAP_CASES], ids=[c[0] for c in SNAP_CASES]
)
def test_snap_goal_y_cells(yd: float, want: float) -> None:
    """§2.2: ``k = clip(ceil(yd / 0.1 − 1e-4), 1, 800)``, snapped to ``(k − 0.5) · 0.1`` yd, in
    canonical metres in and out."""
    got = statsbomb.snap_goal_y(np.array([sbc.mouth_metres(yd)]))
    assert got.dtype == np.float64 and got.shape == (1,)
    assert float(got[0]) == pytest.approx(sbc.mouth_metres(want), abs=TOL)


def test_snap_goal_y_totality() -> None:
    """§6: NaN stays NaN; ±inf metres go to the clip ends (+inf is past the 0 yd touchline, −inf
    past the 80 yd one); an empty array gives an empty array; a list is accepted."""
    got = statsbomb.snap_goal_y([NAN, INF, -INF, sbc.mouth_metres(40.0)])
    assert math.isnan(got[0])
    assert got[1:] == pytest.approx(
        [sbc.mouth_metres(0.05), sbc.mouth_metres(79.95), sbc.mouth_metres(39.95)], abs=TOL
    )
    empty = statsbomb.snap_goal_y(np.array([], dtype=np.float64))
    assert empty.shape == (0,)


@settings(deadline=None, max_examples=200)
@given(k=st.integers(1, 800), m=st.floats(0.0, 68.0, allow_nan=False))
def test_snap_idempotent(k: int, m: float) -> None:
    """A cell centre snaps to itself; any snapped value snaps to itself; a snap moves a point by
    at most half a cell (0.05 yd at the steepest metres-per-yard segment)."""
    centre = sbc.mouth_metres((k - 0.5) * 0.1)
    assert float(statsbomb.snap_goal_y([centre])[0]) == pytest.approx(centre, abs=TOL)
    once = statsbomb.snap_goal_y([m])
    assert statsbomb.snap_goal_y(once) == pytest.approx(once, abs=TOL)
    assert abs(float(once[0]) - m) <= HALF_CELL_M + TOL
    assert float(once[0]) == pytest.approx(sbc.snapped(m), abs=TOL)


# raw end_x, half-cell offset, projected?
# fmt: off
PROJECTED_CASES = [
    ("coarse 118", 118.0, 0.5, True),
    ("coarse 119 (one cell)", 119.0, 0.5, False),
    ("coarse 120", 120.0, 0.5, False),
    ("coarse 2e-9 short of 119", 119.0 - 2e-9, 0.5, True),
    ("coarse 5e-10 short of 119", 119.0 - 5e-10, 0.5, False),
    ("fine 119.8", 119.8, 0.05, True),
    ("fine 119.9 (one cell)", 119.9, 0.05, False),
    ("fine 120", 120.0, 0.05, False),
    ("fine 119 (coarse would keep it)", 119.0, 0.05, True),
    ("NaN", NAN, 0.05, False),
]
# fmt: on


def test_is_projected() -> None:
    """§2.2: ``end_x < 120 − 2·offset − 1e-9``, vectorised, a bool array; NaN is False (§6)."""
    end_x = np.array([c[1] for c in PROJECTED_CASES])
    offset = np.array([c[2] for c in PROJECTED_CASES])
    got = statsbomb.is_projected(end_x, offset)
    assert isinstance(got, np.ndarray) and got.dtype == np.bool_
    assert dict(zip([c[0] for c in PROJECTED_CASES], got.tolist(), strict=True)) == {
        c[0]: c[3] for c in PROJECTED_CASES
    }


def test_loader_snaps_projected_outcome_blind(tmp_path: Path) -> None:
    """§2.2 / §8: in one coarse and one fine match, a goal and a save with one geometry, both
    stopped short of the line, get the same goal_y: the snapped projection. A point on the line
    keeps its recorded value (already a cell centre). The snap reads the stop point and the grid,
    never the outcome."""
    fine_end, coarse_end = (116.05, 39.05, 0.55), (116.0, 39.1, 0.5)
    fine = sbc.one_match(
        sbc.shot("fake-shot-f01", location=(102.35, 35.65), outcome="Goal", end_location=fine_end,
                 minute=10),
        sbc.shot("fake-shot-f02", location=(102.35, 35.65), outcome="Saved",
                 end_location=fine_end, minute=11),
        sbc.shot("fake-shot-f03", location=(102.35, 35.65), outcome="Goal",
                 end_location=(120.0, 38.3, 0.85), minute=12),
        match_id=sbc.A, fidelity="2",
    )  # fmt: skip
    frame = sbc.default_frame(sbc.HOME, coarse=True)
    coarse = sbc.one_match(
        sbc.shot("fake-shot-c01", location=(102, 35), outcome="Saved", end_location=coarse_end,
                 freeze_frame=frame, minute=10),
        sbc.shot("fake-shot-c02", location=(102, 35), outcome="Goal", end_location=coarse_end,
                 freeze_frame=frame, minute=11),
        sbc.shot("fake-shot-c03", location=(102, 35), outcome="Goal",
                 end_location=(120.0, 38.4, 0.5), freeze_frame=frame, minute=12),
        match_id=sbc.B, fidelity="1",
    )  # fmt: skip
    shots = sbc.load_matches(tmp_path / "cache", [fine, coarse]).shots
    for goal_id, save_id, line_id, line_yd in (
        ("fake-shot-f01", "fake-shot-f02", "fake-shot-f03", 38.3),
        ("fake-shot-c02", "fake-shot-c01", "fake-shot-c03", 38.4),
    ):
        goal, save = sbc.row(shots, goal_id), sbc.row(shots, save_id)
        line = _line(save)
        assert abs(sbc.snapped(line) - line) > 0.005  # precondition: the snap visibly moves it
        assert goal["goal_y"] == save["goal_y"], goal_id
        assert save["goal_y"] == pytest.approx(sbc.snapped(line), abs=TOL), save_id
        assert 30.34 <= save["goal_y"] <= 37.66  # valid football: on target
        on_line = sbc.row(shots, line_id)
        recorded = sbc.mouth_metres(line_yd - FINE)
        assert on_line["goal_y"] == pytest.approx(recorded, abs=TOL), line_id
        assert sbc.snapped(recorded) == pytest.approx(recorded, abs=TOL)  # already on the lattice
