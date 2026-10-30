"""Image tests: both charts from the edge fixture against reviewed baselines at ``IMAGE_TOL``, and
byte-identical renders (C7 spec §1 "Deterministic backend", §7).

Baselines live in ``tests/baseline_images/viz/`` and contain only the synthetic fixture and the
synthetic logo, never StatsBomb data or the StatsBomb logo. To regenerate them after an intended
change: ``FC_UPDATE_BASELINES=1 uv run pytest tests/test_viz_images.py``, then look at the PNGs.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import matplotlib.style
import pytest
import viz_cases as vc


def test_shot_map_image(synthetic_logo: Path, tmp_path: Path) -> None:
    ax = vc.render("shot_map", vc.edge_frame("statsbomb"), logo=synthetic_logo)
    vc.check_image(ax, "shot_map", tmp_path)


def test_goal_mouth_image(synthetic_logo: Path, tmp_path: Path) -> None:
    ax = vc.render("goal_mouth", vc.edge_frame("statsbomb"), logo=synthetic_logo)
    vc.check_image(ax, "goal_mouth", tmp_path)


def _png_sha256(chart: str, logo: Path) -> str:
    ax = vc.render(chart, vc.edge_frame("statsbomb"), logo=logo)
    buf = io.BytesIO()
    with matplotlib.style.context("default"):
        ax.figure.savefig(buf, format="png", dpi=100, metadata={"Software": None})
    return hashlib.sha256(buf.getvalue()).hexdigest()


@pytest.mark.parametrize("chart", ["shot_map", "goal_mouth"])
def test_render_is_deterministic(chart: str, synthetic_logo: Path) -> None:
    assert _png_sha256(chart, synthetic_logo) == _png_sha256(chart, synthetic_logo)


# --- CH1 themed canvas (spec runs/CH1/style-shots §6 "Image baselines") ------------------------


def _themed_canvas(theme: str, size: str, logo: Path) -> object:
    """A synthetic themed chart: strip, cropped pitch with shots and keys, footer with logo."""
    import canvas_cases as cc
    import numpy as np

    from football_core import viz

    c = cc.make(
        theme, size, logo=logo, provider="statsbomb", title="7 of 9 goals came from inside the box",
        subtitle="Synthetic FC · Test League 2015/2016", stats=cc.FIVE_STATS[:3],
        model="xG: own model", note="Not shown: 1 penalty (1 scored)",
    )  # fmt: skip
    t = viz.THEMES[theme]
    ax = c.axes((0.0, 0.12, 1.0, 0.88), gid="map")
    viz.draw_half_pitch(ax, colour=t.line, box_colour=t.box_line)
    x = np.array([94.0, 99.0, 88.0, 102.5, 97.0, 80.0, 101.0, 91.0, 85.0])
    y = np.array([34.0, 30.0, 45.0, 36.0, 28.0, 20.0, 33.0, 40.0, 52.0])
    xg = np.array([0.76, 0.12, 0.04, 0.45, 0.006, 0.03, 0.31, 0.09, 0.02])
    goal = np.array([True, False, False, True, True, False, True, False, False])
    viz.crop_pitch(ax, x, y)
    scale = viz.ink_scale(float(xg.sum()), ax)
    viz.draw_shots(ax, x, y, xg, goal, t, area_per_xg=scale)
    c.legend(viz.shot_keys(scale, t), ax=ax)
    return c


@pytest.mark.parametrize("case", ["light-portrait", "dark-square"])
def test_canvas_image(case: str, synthetic_logo: Path, tmp_path: Path) -> None:
    """The themed layout against reviewed baselines (Inter, synthetic shots and logo); the canvas
    passes every machine check. ``FC_UPDATE_BASELINES=1`` writes the baseline and skips."""
    import os
    import shutil

    from matplotlib.testing.compare import compare_images

    from football_core import viz

    theme, size = case.split("-")
    c = _themed_canvas(theme, size, synthetic_logo)
    assert viz.check(c) == []
    actual = viz.save(c, tmp_path / f"canvas-{case}.png")
    baseline = vc.BASELINE_DIR / f"canvas-{case}.png"
    if os.environ.get("FC_UPDATE_BASELINES") == "1":
        shutil.copyfile(actual, baseline)
        pytest.skip(f"baseline written; review it: {baseline}")
    assert baseline.is_file(), f"no baseline at {baseline}; run with FC_UPDATE_BASELINES=1"
    result = compare_images(str(baseline), str(actual), tol=vc.IMAGE_TOL)
    assert result is None, result
