"""Pins for the Engineer's ratify items of CH1.style-shots that no earlier test held (``runs/CH1/
style-shots/build.md``, R4, R6, R7): the legend's width, float noise in the overlap check, mark
contrast on an artist's union box, and the vertical layout rules."""

from __future__ import annotations

import dataclasses

import canvas_cases as cc
import pytest
from matplotlib.patches import Rectangle
from matplotlib.transforms import IdentityTransform

from football_core import viz


def test_legend_width_runs_from_drawn_left_to_box_right() -> None:
    """R4: an equal-aspect map drawn narrower than its box (anchored right here) gets a legend as
    wide as from its drawn left edge to the right edge of the box it was given."""
    c = cc.make("light", "square")
    ax = c.axes((0.0, 0.3, 1.0, 0.7), gid="map")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_aspect("equal")
    ax.set_anchor("E")
    ax.apply_aspect()
    w = viz.SIZES["square"][0]
    box, drawn = ax.get_position(original=True), ax.get_position()
    assert drawn.x0 > box.x0 + 0.1 and drawn.x1 == pytest.approx(box.x1)  # precondition
    assert c.legend_width(ax) == pytest.approx((box.x1 - drawn.x0) * w)
    ax.set_anchor("W")  # drawn at the left: the legend still runs to the box's right edge
    ax.apply_aspect()
    assert c.legend_width(ax) == pytest.approx((box.x1 - box.x0) * w)
    assert c.legend_width(ax) > ax.get_position().width * w + 100  # precondition: wider


def test_float_noise_is_not_overlap() -> None:
    """R6: boxes overlapping by less than ``EPS_PX`` (float noise of boxes laid edge to edge) are
    not an overlap; a visible overlap of 0.5 px is."""
    assert viz.EPS_PX == 1e-6
    for shift, flagged in ((5e-7, False), (0.5, True)):
        c = cc.make()
        r = cc.renderer(c)
        left, bottom, _, _ = cc.body_px(c).bounds
        kw = {"transform": IdentityTransform(), "ha": "left", "va": "baseline", "fontsize": 12,
              "color": cc.theme_of(c).ink, "family": cc.theme_of(c).family}  # fmt: skip
        a = c.fig.text(left + 100.0, bottom + 200.0, "Copperbeech", **kw)
        edge = a.get_window_extent(r).x1
        b = c.fig.text(edge - shift, bottom + 200.0, "Dogwood", **kw)
        r = cc.renderer(c)
        overlap = a.get_window_extent(r).x1 - b.get_window_extent(r).x0
        assert overlap == pytest.approx(shift, rel=1e-3)  # precondition: the overlap in float
        assert any("text overlap" in p for p in viz.check(c)) is flagged, shift


def test_mark_contrast_judged_on_union_box() -> None:
    """R6: one artist's mark contrast is judged against the median surface under the union of
    its markers' boxes, not marker by marker: a ring of a collection over a small light patch
    passes when most of the union is the dark background."""
    c = cc.make("dark", "square")
    t = cc.theme_of(c)
    ax = c.axes((0.0, 0.1, 1.0, 0.8), gid="map")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.add_patch(Rectangle((0.06, 0.46), 0.08, 0.08, facecolor="#f8fafc", zorder=0,
                           gid="decor-light"))  # fmt: skip
    ax.scatter([0.1, 0.5, 0.9], [0.5, 0.5, 0.5], s=400, facecolors="none", edgecolors=t.mark,
               linewidths=1.5, gid="mark-row")  # fmt: skip
    assert cc.wcag(t.mark, t.background) >= 3.0 > cc.wcag(t.mark, "#f8fafc")  # precondition
    assert viz.check(c) == []
    alone = cc.make("dark", "square")  # the same light patch under a one-ring artist fails
    ax = alone.axes((0.0, 0.1, 1.0, 0.8), gid="map")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.add_patch(Rectangle((0.06, 0.46), 0.08, 0.08, facecolor="#f8fafc", zorder=0))
    ax.scatter([0.1], [0.5], s=400, facecolors="none", edgecolors=t.mark, linewidths=1.5,
               gid="mark-one")  # fmt: skip
    assert any("mark-one" in p for p in viz.check(alone))


def test_vertical_layout_constants() -> None:
    """R7: the title's top sits ``TOP_PX`` under the canvas top, "Prepared by" ``BOTTOM_PX`` above
    its bottom; the constants as built."""
    assert (viz.TOP_PX, viz.BOTTOM_PX, viz.LINE_GAP_PX) == (48, 32, 4)
    assert (viz.KEY_GAP_PX, viz.KEY_SPACING_PX, viz.PANEL_RADIUS_PX) == (8, 16, 12.0)
    for size in viz.SIZES:
        c = cc.make("light", size)
        r = cc.renderer(c)
        h = viz.SIZES[size][1]
        (title,) = cc.styled(c, "title", cc.TITLE_SIZES)
        assert title.get_window_extent(r).y1 == pytest.approx(h - viz.TOP_PX, abs=1.0)
        (prepared,) = cc.styled(c, "footer_strong")
        assert prepared.get_window_extent(r).y0 == pytest.approx(viz.BOTTOM_PX, abs=1.0)


def test_strip_rows_stack_without_gap() -> None:
    """R7: the stat strip's rows stack value, label, note with no gap, every column on the same
    row lines (a row is as tall as its tallest text), in Inter and in the fallback family."""
    stats = (
        viz.Stat("512", "Shots"),
        viz.Stat("−4.0", "Goals − xG", "SD 2.7"),
        viz.Stat("xg", "Q"),
    )
    for family in (viz.FAMILY, viz.FALLBACK_FAMILY):
        theme = dataclasses.replace(viz.THEMES["light"], family=family)
        c = cc.make(theme, "portrait", stats=stats)
        r = cc.renderer(c)
        rows = [cc.styled(c, s) for s in ("stat_value", "stat_label", "stat_note")]
        assert [len(x) for x in rows] == [3, 3, 1]  # precondition
        tops = [{round(t.get_window_extent(r).y1, 3) for t in row} for row in rows]
        assert all(len(x) == 1 for x in tops)  # one top per row across the columns
        for upper, lower in zip(rows, rows[1:], strict=False):
            bottom = min(t.get_window_extent(r).y0 for t in upper)
            assert lower[0].get_window_extent(r).y1 == pytest.approx(bottom, abs=0.5)
        assert viz.check(c) == [], family
