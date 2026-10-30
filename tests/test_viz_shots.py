"""Generic pitch and shot helpers: crop, ink budget, draw order, goal floor, legend keys (spec
``runs/CH1/style-shots`` §2.4). Data coordinates on the vertical half pitch are ``(y, x)``."""

from __future__ import annotations

import math

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure

from football_core import pitch, viz

LIGHT = viz.THEMES["light"]
CANVAS_PX = 1080


def axes_px(w: float, h: float) -> Axes:
    """An axes exactly ``w`` x ``h`` px on a 1080 x 1080 px figure at ``DPI``."""
    fig = Figure(figsize=(CANVAS_PX / viz.DPI, CANVAS_PX / viz.DPI), dpi=viz.DPI)
    FigureCanvasAgg(fig)
    return fig.add_axes((0.0, 0.0, w / CANVAS_PX, h / CANVAS_PX))


def view(ax: Axes) -> tuple[float, float, float, float]:
    """``(x_lo, x_hi, y_lo, y_hi)`` in pitch metres (x along the length, drawn vertically)."""
    h0, h1 = ax.get_xlim()
    v0, v1 = ax.get_ylim()
    return min(v0, v1), max(v0, v1), min(h0, h1), max(h0, h1)


def inside(ax: Axes, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_lo, x_hi, y_lo, y_hi = view(ax)
    return (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)


def cropped(
    w: float, h: float, x: list[float], y: list[float], **kw: object
) -> tuple[Axes, np.ndarray]:
    ax = axes_px(w, h)
    viz.draw_half_pitch(ax, colour=LIGHT.line, box_colour=LIGHT.box_line)
    mask = viz.crop_pitch(ax, np.asarray(x, dtype=float), np.asarray(y, dtype=float), **kw)
    return ax, np.asarray(mask)


def fills(ax: Axes, w: float, h: float) -> bool:
    x_lo, x_hi, y_lo, y_hi = view(ax)
    return math.isclose((y_hi - y_lo) / (x_hi - x_lo), w / h, rel_tol=1e-6)


def collection(ax: Axes, gid: str) -> object:
    (found,) = [c for c in ax.collections if c.get_gid() == gid]
    return found


# rows: (x, y, xg, goal); goals include one at 0.006 xG (below the floor)
ROWS = [
    (94.0, 34.0, 0.05, False),
    (99.0, 30.0, 0.40, False),
    (88.0, 45.0, 0.10, False),
    (101.0, 36.0, 0.30, True),
    (97.0, 28.0, 0.006, True),
    (103.0, 34.5, 0.70, True),
]


def _draw(area: float) -> Axes:
    x, y, xg, goal = (np.asarray(v) for v in zip(*ROWS, strict=True))
    ax = viz.draw_half_pitch()
    viz.draw_shots(ax, x.astype(float), y.astype(float), xg.astype(float), goal.astype(bool),
                   LIGHT, area_per_xg=area)  # fmt: skip
    return ax


def test_draw_order() -> None:
    """Goals above misses; within each group xG-descending (small on top); rings in ``mark`` at
    ``mark_alpha``, goals filled ``accent`` with a ``background`` halo; never clipped."""
    ax = _draw(250.0)
    shots, goals = collection(ax, viz.MARK_SHOTS), collection(ax, viz.MARK_GOALS)
    assert (viz.MARK_SHOTS, viz.MARK_GOALS) == ("mark-shots", "mark-goals")
    assert goals.get_zorder() > shots.get_zorder()
    for coll, is_goal in ((shots, False), (goals, True)):
        rows = sorted((r for r in ROWS if r[3] is is_goal), key=lambda r: -r[2])
        expected = np.array([[r[1], r[0]] for r in rows])  # (y, x): shooter's view
        assert np.array_equal(np.asarray(coll.get_offsets()), expected)
        assert not coll.get_clip_on()
        assert np.allclose(coll.get_linewidths(), 0.9)
    assert to_hex(goals.get_facecolors()[0]) == LIGHT.accent
    assert to_hex(goals.get_edgecolors()[0]) == LIGHT.background
    edge = to_rgba(shots.get_edgecolors()[0])
    assert to_hex(edge[:3]) == LIGHT.mark
    alpha = shots.get_alpha() if shots.get_alpha() is not None else edge[3]
    assert alpha == pytest.approx(LIGHT.mark_alpha)
    faces = shots.get_facecolors()
    assert len(faces) == 0 or all(f[3] == 0 for f in faces)  # rings: no fill


def test_goal_floor_misses_exact() -> None:
    """Misses: ``s == area_per_xg · xg`` exactly; goals: ``max(area_per_xg · xg, GOAL_MIN_S)``."""
    assert viz.GOAL_MIN_S == 36.0
    area = 250.0
    ax = _draw(area)
    misses = sorted((r[2] for r in ROWS if not r[3]), reverse=True)
    goals = sorted((r[2] for r in ROWS if r[3]), reverse=True)
    got_m = np.asarray(collection(ax, viz.MARK_SHOTS).get_sizes())
    got_g = np.asarray(collection(ax, viz.MARK_GOALS).get_sizes())
    assert np.array_equal(got_m, area * np.asarray(misses))
    assert np.array_equal(got_g, np.maximum(area * np.asarray(goals), viz.GOAL_MIN_S))
    assert area * 0.006 < viz.GOAL_MIN_S and viz.GOAL_MIN_S in got_g.tolist()  # floor acted


@settings(deadline=None, max_examples=80)
@given(
    xg_sum=st.one_of(st.just(0.0), st.floats(1e-6, 400.0)),
    w=st.floats(400.0, 1000.0),
    h=st.floats(400.0, 1000.0),
)
def test_ink_budget(xg_sum: float, w: float, h: float) -> None:
    """``ink_scale``: Σ s = INK_SHARE · A unless the scale sits at a bound; 0 xG -> AREA_MAX."""
    assert (viz.INK_SHARE, viz.AREA_MAX, viz.AREA_PER_XG_MIN) == (0.12, 900.0, 190.0)
    ax = axes_px(w, h)
    a_pt2 = w * h * (72.0 / viz.DPI) ** 2
    s = viz.ink_scale(xg_sum, ax)
    assert viz.AREA_PER_XG_MIN <= s <= viz.AREA_MAX
    if xg_sum == 0:
        assert s == viz.AREA_MAX
    elif viz.AREA_PER_XG_MIN < s < viz.AREA_MAX:
        assert s * xg_sum == pytest.approx(viz.INK_SHARE * a_pt2, rel=1e-6)
    else:
        target = viz.INK_SHARE * a_pt2 / xg_sum
        assert s == (viz.AREA_MAX if target >= viz.AREA_MAX else viz.AREA_PER_XG_MIN)


def test_crop_depth_cap_and_mask() -> None:
    """A shot 60 m out and one in the own half fall outside the capped view and the mask;
    ``x_min=None`` keeps them."""
    assert (viz.CROP_PAD, viz.CROP_MAX_DEPTH) == (2.0, 35.0)
    x, y = [100.0, 45.0, 30.0], [34.0, 34.0, 34.0]
    ax, mask = cropped(600, 400, x, y)
    assert mask.tolist() == [True, False, False]
    assert np.array_equal(mask, inside(ax, np.asarray(x), np.asarray(y)))
    x_lo, x_hi, _, _ = view(ax)
    assert x_lo == pytest.approx(pitch.PITCH_LENGTH - viz.CROP_MAX_DEPTH)
    assert x_hi == pytest.approx(pitch.PITCH_LENGTH + viz.GOAL_DEPTH + 1.0)
    assert fills(ax, 600, 400)
    ax, mask = cropped(600, 400, x, y, x_min=None)
    assert mask.tolist() == [True, True, True]
    assert view(ax)[0] <= 30.0 - viz.CROP_PAD + 1e-9


def test_crop_fills_aspect() -> None:
    """Tall axes: extra height extends the depth (top fixed at the goal back + 1). Wide axes:
    y widens symmetrically and is shifted inside [-1, 69]. Empty input: the box region."""
    ax, mask = cropped(300, 600, [100.0, 96.0], [34.0, 30.0])
    x_lo, x_hi, y_lo, y_hi = view(ax)
    assert mask.all() and fills(ax, 300, 600)
    assert x_hi == pytest.approx(108.0) and x_lo < pitch.PITCH_LENGTH - viz.CROP_MAX_DEPTH
    assert y_lo <= pitch.METRE_Y[1] - viz.CROP_PAD and y_hi >= pitch.METRE_Y[6] + viz.CROP_PAD
    # y extent [-1, 56.16] (centre 27.58); x extent [78, 108]; 880 x 400 needs 66 m of width
    ax, mask = cropped(880, 400, [80.0, 80.0], [1.0, 40.0])
    x_lo, x_hi, y_lo, y_hi = view(ax)
    assert mask.all() and fills(ax, 880, 400)
    assert (x_lo, x_hi) == (pytest.approx(78.0), pytest.approx(108.0))
    assert y_lo == pytest.approx(-1.0) and y_hi == pytest.approx(65.0)
    ax, mask = cropped(600, 400, [], [])
    x_lo, x_hi, y_lo, y_hi = view(ax)
    assert mask.shape == (0,) and fills(ax, 600, 400)
    assert x_lo <= pitch.METRE_X[5] and x_hi >= pitch.PITCH_LENGTH
    assert y_lo <= pitch.METRE_Y[1] and y_hi >= pitch.METRE_Y[6]


@settings(deadline=None, max_examples=60)
@given(
    pts=st.lists(st.tuples(st.floats(70.0, 105.0), st.floats(0.0, 68.0)), max_size=25),
    w=st.floats(300.0, 900.0),
    ratio=st.floats(0.75, 1.5),
)
def test_crop_keeps_attacking_shots(pts: list[tuple[float, float]], w: float, ratio: float) -> None:
    """Any shot within 35 m of the goal line stays in view; the mask is the final view; the view
    fills the axes with its top at the goal back + 1."""
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    ax, mask = cropped(w, w / ratio, x, y)
    assert mask.shape == (len(pts),) and bool(np.all(mask))
    assert np.array_equal(mask, inside(ax, np.asarray(x), np.asarray(y)))
    x_lo, x_hi, y_lo, y_hi = view(ax)
    assert x_hi == pytest.approx(108.0)
    assert y_lo >= -1.0 - viz.CROP_PAD and y_hi <= 69.0 + viz.CROP_PAD  # widening clips to
    assert fills(ax, w, w / ratio)  # [-1, 69]; an unwidened extent keeps its pad


def test_shot_keys_sizes() -> None:
    """Goal key at the goal floor with its caveat, No goal, one ring per ``LEGEND_XG``; the
    smallest ring at the scale's lower bound is still ``MIN_KEY_PX`` across."""
    assert viz.LEGEND_XG == (0.1, 0.3, 0.6) and viz.MIN_KEY_PX == 9
    keys = viz.shot_keys(250.0, LIGHT)
    assert keys[0] == viz.Key("goal", "Goal (small xG drawn at this size)", viz.GOAL_MIN_S)
    assert (keys[1].kind, keys[1].label) == ("shot", "No goal")
    rings = keys[2:]
    assert [k.label for k in rings] == ["0.1 xG", "0.3 xG", "0.6 xG"]
    assert [k.kind for k in rings] == ["shot"] * 3
    assert [k.size for k in rings] == [250.0 * v for v in viz.LEGEND_XG]
    smallest = viz.shot_keys(viz.AREA_PER_XG_MIN, LIGHT)[2]
    assert math.sqrt(smallest.size) * viz.DPI / 72 >= viz.MIN_KEY_PX
