"""The machine checks of ``viz.check``: overlap, text on marks, clipping, panel containment, text
and mark contrast on the painted surface, key size, totality (spec ``runs/CH1/style-shots``
§2.5). Each test breaks one clean canvas in one way; labels are distinct words so a problem
sentence can be matched to the artist it names."""

from __future__ import annotations

from typing import Any

import canvas_cases as cc
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.transforms import IdentityTransform

from football_core import viz


def _named(problems: list[str], name: str) -> bool:
    return any(name in p for p in problems)


def _body_text(c: viz.Canvas, fx: float, fy: float, s: str, **kw: Any) -> Any:
    x, y = cc.body_point(c, fx, fy)
    kw.setdefault("fontsize", 12)
    kw.setdefault("color", cc.theme_of(c).ink)
    return c.fig.text(x, y, s, family=cc.theme_of(c).family, **kw)


def test_overlap_found() -> None:
    c = cc.make()
    _body_text(c, 0.3, 0.5, "Alderwood")
    _body_text(c, 0.3, 0.5, "Blackthorn")
    problems = viz.check(c)
    assert _named(problems, "Alderwood") and _named(problems, "Blackthorn")


def test_touching_is_not_overlap() -> None:
    """Two extents that share an edge (zero-area intersection) are not an overlap."""
    c = cc.make()
    r = cc.renderer(c)
    left, bottom, _, _ = cc.body_px(c).bounds
    kw = {"transform": IdentityTransform(), "ha": "left", "va": "baseline", "fontsize": 12,
          "color": cc.theme_of(c).ink, "family": cc.theme_of(c).family}  # fmt: skip
    a = c.fig.text(left + 100.0, bottom + 200.0, "Copperbeech", **kw)
    edge = a.get_window_extent(r).x1
    b = c.fig.text(edge, bottom + 200.0, "Dogwood", **kw)
    r = cc.renderer(c)
    assert a.get_window_extent(r).x1 == b.get_window_extent(r).x0  # precondition: touching
    assert viz.check(c) == []


@pytest.mark.parametrize("mark", ["marker", "tick"])
def test_text_on_mark_found(mark: str) -> None:
    """A text over a marker, or over a tick (a zero-width ``Line2D``: its box is its stroke;
    integrate census: no core test reached the line branch)."""
    c = cc.make()
    ax = c.axes((0.1, 0.1, 0.8, 0.8), gid="map")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if mark == "marker":
        ax.scatter([0.5], [0.5], s=400, color=cc.theme_of(c).accent, gid="mark-elm")
    else:
        ax.plot([0.52, 0.52], [0.45, 0.55], lw=2.0, color=cc.theme_of(c).ink, gid="mark-elm")
    ax.text(0.5, 0.5, "Fieldmaple", fontsize=12, color=cc.theme_of(c).ink,
            family=cc.theme_of(c).family)  # fmt: skip
    problems = viz.check(c)
    assert _named(problems, "Fieldmaple") and _named(problems, "mark-elm")


@pytest.mark.parametrize("what", ["text", "marker"])
def test_clipped_found(what: str) -> None:
    c = cc.make()
    if what == "text":
        c.fig.text(-0.01, 0.5, "Ginkgo", fontsize=12, color=cc.theme_of(c).ink)
        name = "Ginkgo"
    else:
        ax = c.axes((0.0, 0.1, 0.5, 0.8), gid="map")
        ax.set_xlim(0.0, 1.0)
        ax.scatter([-0.5], [0.5], s=100, color=cc.theme_of(c).accent, gid="mark-hornbeam",
                   clip_on=False)  # fmt: skip
        name = "mark-hornbeam"
    assert _named(viz.check(c), name)


def test_panel_containment_found() -> None:
    c = cc.make()
    ax = c.axes((0.0, 0.0, 0.4, 0.4), gid="zones", panel=True)
    ax.text(0.1, 0.5, "Inside", transform=ax.transAxes, fontsize=12,
            color=cc.theme_of(c).ink, family=cc.theme_of(c).family)  # fmt: skip
    assert viz.check(c) == []  # precondition: a text inside the panel passes
    ax.text(1.3, 0.5, "Juniper", transform=ax.transAxes, fontsize=12,
            color=cc.theme_of(c).ink, family=cc.theme_of(c).family)  # fmt: skip
    problems = viz.check(c)
    assert _named(problems, "Juniper") and not _named(problems, "Inside")


# (label, colour, size pt, weight, passes) on the light background #f8fafc
CONTRAST_CASES = [
    ("Kapok", cc.GREY_PASS_45, 12.0, 400, True),  # 4.53 >= 4.5
    ("Larch", cc.GREY_FAIL_45, 12.0, 400, False),  # 4.47 < 4.5
    ("Magnolia", cc.GREY_PASS_30, 18.0, 400, True),  # large at exactly 18 pt; 3.01 >= 3.0
    ("Nutmeg", cc.GREY_FAIL_30, 18.0, 400, False),  # large, 2.97 < 3.0
    ("Olive", cc.GREY_MID, 18.0, 400, True),  # large: 3.0 suffices
    ("Pine", cc.GREY_MID, 17.5, 400, False),  # regular under 18 pt: not large
    ("Quince", cc.GREY_MID, 14.0, 600, True),  # exactly 14 pt at exactly weight 600: large
    ("Rowan", cc.GREY_MID, 14.0, 500, False),  # 14 pt at weight 500: not large
    ("Sumac", cc.GREY_MID, 13.5, 700, False),  # bold under 14 pt: not large
]


def test_text_contrast_threshold() -> None:
    """4.5:1, or 3:1 when large (>= 18 pt, or >= 14 pt with weight >= 600), each threshold
    and each size/weight boundary at its exact value."""
    bg = viz.THEMES["light"].background
    for _, colour, _, _, _ in CONTRAST_CASES:  # precondition: the greys sit where claimed
        assert viz.contrast_ratio(colour, bg) == pytest.approx(cc.wcag(colour, bg), rel=1e-9)
    assert 3.0 < cc.wcag(cc.GREY_MID, bg) < 4.5
    assert cc.wcag(cc.GREY_PASS_45, bg) >= 4.5 > cc.wcag(cc.GREY_FAIL_45, bg)
    assert cc.wcag(cc.GREY_PASS_30, bg) >= 3.0 > cc.wcag(cc.GREY_FAIL_30, bg)
    c = cc.make("light", "portrait")
    for i, (label, colour, size, weight, _) in enumerate(CONTRAST_CASES):
        _body_text(c, 0.05, 0.92 - i * 0.1, label, fontsize=size, weight=weight, color=colour)
    problems = viz.check(c)
    for label, _, _, _, passes in CONTRAST_CASES:
        assert _named(problems, label) is not passes, (label, problems)
    assert len(problems) == sum(1 for case in CONTRAST_CASES if not case[4])


def test_mark_contrast_uses_painted_surface() -> None:
    """A ring in the dark ``mark`` colour on an axis-off axes over the dark background passes
    (the axes' unpainted white facecolour is not the surface); the same ring over a light patch
    (decor) fails."""
    c = cc.make("dark", "square")
    t = cc.theme_of(c)
    left = c.axes((0.0, 0.1, 0.45, 0.8), gid="map-left")
    left.scatter([0.5], [0.5], s=400, facecolors="none", edgecolors=t.mark, linewidths=1.5,
                 gid="mark-on-dark")  # fmt: skip
    right = c.axes((0.55, 0.1, 0.45, 0.8), gid="map-right")
    right.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="#f8fafc", gid="decor-light",
                              zorder=0))  # fmt: skip
    right.scatter([0.5], [0.5], s=400, facecolors="none", edgecolors=t.mark, linewidths=1.5,
                  gid="mark-on-light")  # fmt: skip
    # ticks (Line2D marks, judged by their line colour; integrate census: no core test drew one)
    left.plot([0.2, 0.2], [0.4, 0.6], lw=2.0, color=t.ink, gid="mark-tick-ink")
    left.plot([0.8, 0.8], [0.4, 0.6], lw=2.0, color="#1e293b", gid="mark-tick-slate")
    assert cc.wcag(t.mark, t.background) >= 3.0 > cc.wcag(t.mark, "#f8fafc")  # precondition
    assert cc.wcag(t.ink, t.background) >= 3.0 > cc.wcag("#1e293b", t.background)  # precondition
    problems = viz.check(c)
    assert _named(problems, "mark-on-light")
    assert not _named(problems, "mark-on-dark")
    assert _named(problems, "mark-tick-slate")
    assert not _named(problems, "mark-tick-ink")


def test_key_size_found() -> None:
    """Every ``mark-key`` marker must be >= ``MIN_KEY_PX`` across (diameter √s pt)."""
    tiny = (viz.MIN_KEY_PX * 72 / viz.DPI) ** 2 * 0.9  # 90 % of the area: 0.95 x the diameter
    ok = (viz.MIN_KEY_PX * 72 / viz.DPI) ** 2 * 1.1
    for size, flagged in ((ok, False), (tiny, True)):
        c = cc.make()
        ax = c.axes((0.0, 0.3, 1.0, 0.7), gid="map")
        c.legend([viz.Key("shot", "Small ring", size)], ax=ax)
        assert cc.by_gid(c, "mark-key")  # precondition
        problems = viz.check(c)
        assert any("key" in p for p in problems) is flagged, problems


def _bare_canvas() -> viz.Canvas:
    fig = Figure(figsize=(1080 / viz.DPI, 1080 / viz.DPI), dpi=viz.DPI,
                 facecolor=viz.THEMES["light"].background)  # fmt: skip
    FigureCanvasAgg(fig)
    return viz.Canvas(fig=fig, theme=viz.THEMES["light"], size="square",
                      body=(0.1, 0.1, 0.8, 0.8))  # fmt: skip


@pytest.mark.parametrize("case", ["empty", "alpha-none", "zero-size", "empty-collection"])
def test_check_totality(case: str) -> None:
    """No text and no marks -> []; ``alpha=None`` counts as 1; a zero-size marker is skipped for
    overlap but still caught when it leaves the canvas; an empty collection adds nothing."""
    c = _bare_canvas()
    ink = viz.THEMES["light"].ink
    if case == "empty":
        assert viz.check(c) == []
        return
    ax = c.fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if case == "alpha-none":
        coll = ax.scatter([0.5], [0.5], s=200, color=ink, gid="mark-quiet")
        coll.set_alpha(None)
        assert viz.check(c) == []  # as alpha 1: ink on the background passes (as 0 it fails)
    elif case == "zero-size":
        ax.text(0.5, 0.5, "Tamarind", fontsize=12, color=ink)
        ax.scatter([0.5], [0.5], s=0.0, color=ink, gid="mark-dot")
        assert viz.check(c) == []  # zero area: no text-on-mark overlap
        ax.scatter([-0.3], [0.5], s=0.0, color=ink, gid="mark-gone", clip_on=False)
        assert _named(viz.check(c), "mark-gone")
    else:
        ax.scatter(np.zeros(0), np.zeros(0), s=50, color=ink, gid="mark-none")
        assert viz.check(c) == []
