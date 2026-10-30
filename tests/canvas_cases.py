"""Builders and artist lookups for the CH1 theme/layout tests (spec ``runs/CH1/style-shots`` §2).

Everything here reads the public API only: ``viz.canvas``, ``viz.Canvas``, ``THEMES``,
``TEXT_STYLES`` and the gids the spec names (``mark-*``, ``panel-*``). Texts are found by their
style (size and colour token), because the spec names styles, not gids, for the bands.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgb
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from football_core import viz

# worst-case strings of spec §6 (3-digit counts, a negative 2-digit gap with U+2212, a 22-char
# name, the longest league label); made up, no provider row
NAME_22 = "Alexandre Oxlade-Duval"
LEAGUE = "1. Bundesliga 2015/2016"
# every size a title may be drawn at (21 pt, then the steps; D1/R1 ratified the last one)
TITLE_SIZES = (21.0, viz.TITLE_STEP_PT, viz.TITLE_MIN_PT, viz.TITLE_FALLBACK_PT)
TITLE = "104 non-penalty goals, 25.6 more than the model's 78.4 xG (SD 7.9)"
SUBTITLE = f"{NAME_22} · Borussia Mönchengladbach · {LEAGUE}"
MODEL = "xG: own model, not trained on Women's Super League 2018/2019"
NOTE = (
    "Not shown: 13 penalties (10 scored) · 2 direct corners (1 scored) · "
    "34 shots outside the drawn area (0 goals, 1.0 xG)"
)
FIVE_STATS = (
    viz.Stat("512", "Shots"),
    viz.Stat("104", "Goals"),
    viz.Stat("116.4", "xG"),
    viz.Stat("−12.4", "Goals − xG", "SD 7.9"),
    viz.Stat("0.95", "League goals per xG"),
)
GREY_PASS_45, GREY_FAIL_45 = "#737373", "#747474"  # 4.53 / 4.47 on #f8fafc (light background)
GREY_PASS_30, GREY_FAIL_30 = "#919191", "#929292"  # 3.01 / 2.97 on #f8fafc
GREY_MID = "#858585"  # between 3.0 and 4.5 on #f8fafc


def make(
    theme: str | viz.Theme = "light",
    size: str = "portrait",
    *,
    logo: Path | None = None,
    title: str = "Short statement title",
    subtitle: str = "Team · League 2015/2016",
    stats: Sequence[viz.Stat] = (),
    provider: str = "whoscored",
    model: str | None = None,
    note: str | None = None,
) -> viz.Canvas:
    """A canvas with small defaults; WhoScored by default so no logo file is needed."""
    footer = viz.Footer((provider,), model=model, note=note)
    return viz.canvas(
        theme, size, title=title, subtitle=subtitle, footer=footer, stats=stats, logo=logo
    )


def renderer(c: viz.Canvas) -> Any:
    c.fig.canvas.draw()
    return c.fig.canvas.get_renderer()


def texts(c: viz.Canvas) -> list[Text]:
    """Every visible, non-blank Text of the figure (as check() collects them)."""
    found = c.fig.findobj(lambda a: isinstance(a, Text))
    return [t for t in found if t.get_visible() and t.get_text().strip()]


def theme_of(c: viz.Canvas) -> viz.Theme:
    return c.theme if isinstance(c.theme, viz.Theme) else viz.THEMES[c.theme]


def _weight(w: object) -> int:
    return int(w) if isinstance(w, int | float) else int(font_manager.weight_dict[str(w).lower()])


def styled(c: viz.Canvas, style: str, sizes: Iterable[float] | None = None) -> list[Text]:
    """Texts drawn in ``style``: its colour token, weight and size (or one of ``sizes``)."""
    size, weight, token = viz.TEXT_STYLES[style]
    colour = getattr(theme_of(c), token)
    allowed = {float(s) for s in (sizes if sizes is not None else (size,))}
    return [
        t
        for t in texts(c)
        if float(t.get_fontsize()) in allowed
        and to_hex(t.get_color()) == to_hex(colour)
        and _weight(t.get_weight()) == _weight(weight)
    ]


def top_down(ts: Sequence[Text], r: Any) -> list[Text]:
    return sorted(ts, key=lambda t: -t.get_window_extent(r).y1)


def lines_of(ts: Sequence[Text]) -> int:
    return sum(len(t.get_text().split("\n")) for t in ts)


def words(ts: Sequence[Text], r: Any) -> str:
    """The texts top-down, joined and whitespace-collapsed (a wrap may split one Text or many)."""
    joined = " ".join(t.get_text() for t in top_down(ts, r))
    return re.sub(r"\s+", " ", joined).replace("- ", "-").strip()


def union(ts: Sequence[Any], r: Any) -> Bbox:
    return Bbox.union([t.get_window_extent(r) for t in ts])


def by_gid(c: viz.Canvas, gid: str) -> list[Any]:
    return c.fig.findobj(lambda a: a.get_gid() == gid)


def gids(c: viz.Canvas, prefix: str) -> set[str]:
    return {
        a.get_gid()
        for a in c.fig.findobj(lambda a: isinstance(a.get_gid(), str))
        if a.get_gid().startswith(prefix)
    }


def marker_centres(artist: Any) -> np.ndarray:
    """Display-pixel centres of a Line2D's or a collection's markers."""
    if hasattr(artist, "get_offsets") and len(artist.get_offsets()):
        return np.asarray(artist.get_offset_transform().transform(artist.get_offsets()))
    if hasattr(artist, "get_xydata"):
        return np.asarray(artist.get_transform().transform(artist.get_xydata()))
    return np.zeros((0, 2))


def body_px(c: viz.Canvas) -> Bbox:
    """``Canvas.body`` (figure fraction, ``(left, bottom, width, height)``) in px."""
    w, h = (c.fig.get_size_inches() * c.fig.dpi).tolist()
    left, bottom, width, height = c.body
    return Bbox.from_bounds(left * w, bottom * h, width * w, height * h)


def body_point(c: viz.Canvas, fx: float, fy: float) -> tuple[float, float]:
    """Figure-fraction coordinates of the point ``(fx, fy)`` given as a fraction of the body."""
    left, bottom, width, height = c.body
    return left + fx * width, bottom + fy * height


def relative_luminance(rgb: Sequence[float]) -> float:
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def wcag(a: str | Sequence[float], b: str | Sequence[float]) -> float:
    """WCAG 2.x contrast, written out here as the independent route to ``viz.contrast_ratio``."""
    la, lb = (relative_luminance(to_rgb(v) if isinstance(v, str) else v) for v in (a, b))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def composite(fg: str, alpha: float, bg: str) -> tuple[float, float, float]:
    """``fg`` at ``alpha`` over ``bg``, rounded to 8 bits per channel as the rendered pixel is
    (spec p7 composites this way; unrounded, dark mark@0.7 on band is 2.998)."""
    f, b = to_rgb(fg), to_rgb(bg)
    out = [round(255 * (alpha * x + (1 - alpha) * y)) / 255 for x, y in zip(f, b, strict=True)]
    return out[0], out[1], out[2]


def words_to_width(c: viz.Canvas, style: str, lo: float, hi: float) -> str:
    """Words whose measured one-line width lies in ``[lo, hi]`` px (``hi - lo`` >= 20): whole
    words up to just under ``lo``, then a last token grown one letter at a time."""
    vocab = ("goals", "from", "inside", "the", "box", "and", "shots", "came", "late", "on")
    out: list[str] = []
    i = 0
    while c.measure(" ".join([*out, vocab[i % len(vocab)]]), style)[0] < lo:
        out.append(vocab[i % len(vocab)])
        i += 1
    letters = "abcdefghijklmnopqrstuvwxyz"
    for n in range(1, len(letters) + 1):
        text = " ".join([*out, letters[:n]])
        width = c.measure(text, style)[0]
        if width >= lo:
            assert width <= hi, f"cannot land in [{lo}, {hi}] px: {width}"
            return text
    raise AssertionError(f"cannot reach {lo} px")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
