"""Fixed layout below the axes, the legend's size key and the deterministic Agg canvas (C7 spec
§1 "Deterministic backend and fixed layout", §2 "Layout", "Legend")."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from itertools import combinations
from pathlib import Path

import pytest
import viz_cases as vc
from matplotlib.colors import to_rgba

from football_core import viz

CHARTS = ("shot_map", "goal_mouth")
LABELS = ["Goal", "No goal", "xG 0.05", "xG 0.2", "xG 0.5"]


@pytest.mark.parametrize("chart", CHARTS)
@pytest.mark.parametrize("shape", [(10, 60), (94, 588)], ids=["synthetic", "real-aspect"])
def test_attribution_visible(chart: str, shape: tuple[int, int], tmp_path: Path) -> None:
    logo = vc.write_logo(tmp_path / "logo.png", *shape)
    ax = vc.render(chart, vc.edge_frame("statsbomb"), logo=logo)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    (text,) = [t for t in ax.texts if t.get_gid() == viz.TEXT_GID]
    (note,) = [t for t in ax.texts if t.get_gid() == viz.NOTE_GID]
    (logo_box,) = vc.logos(ax)
    extents = {
        "legend": ax.get_legend().get_window_extent(renderer),
        "text": text.get_window_extent(renderer),
        "logo": logo_box.get_window_extent(renderer),
        "note": note.get_window_extent(renderer),
    }
    axes_box = ax.get_window_extent(renderer)
    for name, box in extents.items():
        assert fig.bbox.x0 <= box.x0 and box.x1 <= fig.bbox.x1, name
        assert fig.bbox.y0 <= box.y0 and box.y1 <= fig.bbox.y1, name
        assert box.y1 <= axes_box.y0 + 1, f"{name} must sit below the axes"
    for a, b in combinations(extents, 2):
        assert not extents[a].overlaps(extents[b]), (a, b)
    assert extents["logo"].width == pytest.approx(viz.LOGO_WIDTH_PT * fig.dpi / 72, abs=1)
    # the legend is one row: every label on the same baseline
    rows = {round(t.get_window_extent(renderer).y0) for t in ax.get_legend().get_texts()}
    assert len(rows) == 1


@pytest.mark.parametrize("chart", CHARTS)
@pytest.mark.parametrize("empty", [False, True], ids=["edge", "empty"])
@pytest.mark.parametrize("area", [viz.AREA_PER_XG, 150.0])
def test_legend_key(chart: str, empty: bool, area: float, synthetic_logo: Path) -> None:
    shots = vc.edge_frame("statsbomb")
    ax = vc.render(chart, shots.head(0) if empty else shots, logo=synthetic_logo, area_per_xg=area)
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == LABELS
    handles = legend.legend_handles
    assert len(handles) == 5
    assert viz.SIZE_KEY == (0.05, 0.2, 0.5)
    for handle, v in zip(handles[2:], viz.SIZE_KEY, strict=True):
        # a Line2D marker size is a diameter in points: its square is the scatter's area
        assert handle.get_markersize() ** 2 == pytest.approx(area * v, rel=1e-9)
    for handle in handles[:2]:
        assert handle.get_markersize() == pytest.approx(math.sqrt(area * 0.2), rel=1e-9)
    goal, *others = handles
    assert to_rgba(goal.get_markerfacecolor()) == pytest.approx(to_rgba(viz.GOAL_COLOUR))
    for handle in others:
        assert to_rgba(handle.get_markerfacecolor())[3] == 0.0  # hollow
        assert to_rgba(handle.get_markeredgecolor())[:3] == pytest.approx(
            to_rgba(viz.SHOT_COLOUR)[:3]
        )


def test_canvas_is_agg(tmp_path: Path) -> None:
    # A fresh interpreter with MPLBACKEND=svg: the chart's own figure still has an Agg canvas,
    # so the PNG bytes never depend on the user's backend.
    script = textwrap.dedent(
        """
        import polars as pl
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from football_core import schema, viz
        shots = pl.DataFrame({
            "provider": pl.Series(["whoscored"], dtype=schema.PROVIDER),
            "x": [95.0], "y": [34.0], "is_goal": [True], "is_own_goal": [False],
            "period": pl.Series([1], dtype=pl.Int8), "goal_y": [34.0], "goal_z": [1.0],
            "xg": [0.3],
        })
        for chart in (viz.shot_map, viz.goal_mouth):
            ax = chart(shots, xg="xg")
            print(chart.__name__, isinstance(ax.figure.canvas, FigureCanvasAgg))
        """
    )
    env = os.environ | {"MPLBACKEND": "svg", "MPLCONFIGDIR": str(tmp_path)}
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["shot_map", "True", "goal_mouth", "True"]
