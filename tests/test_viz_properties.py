"""Hypothesis: on any frame, every drawn marker has area ``area_per_xg * xg`` of its own row, and
drawn + hidden == rows (C7 spec §2 "Markers", "Note"). WhoScored rows only, so no logo is read."""

from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import numpy as np
import pytest
import viz_cases as vc
from hypothesis import given, settings

from football_core import viz

KINDS = ("open", "own_goal", "shootout")


@st.composite
def shot_rows(draw: st.DrawFn) -> dict[str, Any]:
    kind = draw(st.sampled_from(KINDS))
    if kind == "own_goal":  # attacker's frame flipped: own goals sit at x 47.8..105
        x, y = draw(st.floats(47.8, 105.0)), draw(st.floats(0.0, 68.0))
        return {"x": x, "y": y, "is_goal": True, "is_own_goal": True,
                "period": draw(st.sampled_from([1, 2])), "goal_y": None, "goal_z": None,
                "xg": None}  # fmt: skip
    has_point = draw(st.booleans())
    return {
        "x": 94.0 if kind == "shootout" else draw(st.floats(0.0, 105.0)),
        "y": 34.0 if kind == "shootout" else draw(st.floats(0.0, 68.0)),
        "is_goal": draw(st.booleans()),
        "is_own_goal": False,
        "period": 5 if kind == "shootout" else draw(st.sampled_from([1, 2])),
        "goal_y": draw(st.floats(3.0, 67.0)) if has_point else None,
        "goal_z": draw(st.floats(0.0, 7.2)) if has_point else None,
        "xg": draw(st.floats(0.0, 1.0)),
    }


def _frame(rows: list[dict[str, Any]]) -> Any:
    cols = {k: [r[k] for r in rows] for k in ("x", "y", "xg")}
    return vc.frame(
        **cols,
        is_goal=[r["is_goal"] for r in rows],
        is_own_goal=[r["is_own_goal"] for r in rows],
        period=[r["period"] for r in rows],
        goal_y=[r["goal_y"] for r in rows],
        goal_z=[r["goal_z"] for r in rows],
    )


@pytest.mark.parametrize("chart", ["shot_map", "goal_mouth"])
@settings(deadline=None, max_examples=40)
@given(rows=st.lists(shot_rows(), max_size=12), area=st.floats(1.0, 1000.0))
def test_area_property(chart: str, rows: list[dict[str, Any]], area: float) -> None:
    shots = _frame(rows)
    ax = vc.render(chart, shots, area_per_xg=area)
    h, v = vc.COORDS[chart]
    for gid, is_goal in ((viz.GOALS_GID, True), (viz.SHOTS_GID, False)):
        for (ph, pv), size in zip(vc.offsets(ax, gid), vc.sizes(ax, gid), strict=True):
            # the rows this marker can be: same point, same goal flag, drawable
            match = [
                r for r in rows
                if r[h] == ph and r[v] == pv and r["is_goal"] is is_goal
                and not r["is_own_goal"] and r["period"] != 5
            ]  # fmt: skip
            assert match, (gid, ph, pv)
            assert any(np.isclose(size, area * r["xg"], rtol=1e-12, atol=0) for r in match)
    assert vc.n_drawn(ax) + vc.note_count(vc.note(ax)) == len(rows)
    if chart == "shot_map":  # in view: x >= HALFWAY_X
        drawable = [r for r in rows if not r["is_own_goal"] and r["period"] != 5]
        assert vc.n_drawn(ax) == sum(r["x"] >= viz.HALFWAY_X for r in drawable)
