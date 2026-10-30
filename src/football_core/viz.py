"""Shot charts on the canonical frame: a half-pitch shot map and a front-on goal-mouth plot.

One place draws every chart, so every chart reads canonical metres from the C3 shot table
(no provider transform here), sizes markers the same way and credits its data source
automatically: StatsBomb rows add the StatsBomb logo and "Data: StatsBomb" (the open-data
licence asks for both), WhoScored rows add "Data: Opta via WhoScored".

**Orientation: as the shooter sees it.** The shot map is a vertical half pitch with the goal
at the top and the attacker's right touchline (``y = 0``) on the right. The goal mouth is the
7.32 x 2.44 m frame seen from the pitch, right post (``y = 30.34``) on the right. Marker area
is ``area_per_xg * xG`` exactly, goals filled, non-goals hollow and drawn on top.

**Licence.** The StatsBomb logo is never committed: ``statsbomb.fetch_logo()`` downloads it,
pinned and verified, into the git-ignored data cache. A chart of StatsBomb rows without the
logo file raises before anything is drawn, rather than producing an uncredited chart.

**Goal-mouth points** are the provider's goal-plane point. For a StatsBomb save it is where
the keeper stopped the ball, not a projection onto the goal line, so a save can sit outside
the frame.

Needs the ``viz`` extra: ``uv add 'football-core[viz]'``.
"""

from __future__ import annotations

import filecmp
import math
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import matplotlib.image as mpimg
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib import font_manager
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import Collection, PathCollection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform, ScaledTranslation

from football_core import pitch, schema

FloatArray = npt.NDArray[np.float64]
StrPath = str | os.PathLike[str]
BoolArray = npt.NDArray[np.bool_]
Image = npt.NDArray[np.generic]

# statsbomb.logo_file() spelled out (a test keeps them equal), so charts need no provider loader
DEFAULT_LOGO: Final[str] = (
    "data/raw/statsbomb/open-data/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/"
    "img/SB - Icon Lockup - Colour positive.png"
)
AREA_PER_XG: Final[float] = 600.0  # marker area in pt² at xG = 1 (matplotlib's `s` is area)
SIZE_KEY: Final[tuple[float, ...]] = (0.05, 0.2, 0.5)
CENTRE_CIRCLE_RADIUS: Final[float] = 9.15
CORNER_RADIUS: Final[float] = 1.0
GOAL_DEPTH: Final[float] = 2.0
VIEW_PAD_Y: Final[float] = 2.0  # goal-mouth view beyond each post: every goal and post shot
VIEW_PAD_Z: Final[float] = 1.5  # and 96.5 % of StatsBomb saves fall inside it
HALFWAY_X: Final[float] = pitch.PITCH_LENGTH / 2
PITCH_LW: Final[float] = 1.2
FRAME_LW: Final[float] = 3.0
SHOT_MAP_SIZE: Final[tuple[float, float]] = (6.0, 5.9)
GOAL_MOUTH_SIZE: Final[tuple[float, float]] = (6.0, 3.45)
MARGIN_IN: Final[tuple[float, float, float, float]] = (0.2, 1.0, 0.2, 0.35)  # l, b, r, t
LEGEND_OFFSET_PT: Final[float] = 4.0
TEXT_OFFSET_PT: Final[float] = 34.0
NOTE_OFFSET_PT: Final[float] = 52.0
LOGO_WIDTH_PT: Final[float] = 72.0
GOAL_COLOUR: Final[str] = "#eb6834"
SHOT_COLOUR: Final[str] = "#52514e"
INK: Final[str] = "#0b0b0b"
SURFACE: Final[str] = "#fcfcfb"
LINE_COLOUR: Final[str] = "#9a9993"
ATTRIBUTION: Final = MappingProxyType(
    {"statsbomb": "Data: StatsBomb", "whoscored": "Data: Opta via WhoScored"}
)
GOALS_GID: Final[str] = "goals"
SHOTS_GID: Final[str] = "shots"
FRAME_GID: Final[str] = "goal-frame"
NOTE_GID: Final[str] = "not-shown"
TEXT_GID: Final[str] = "attribution"
LOGO_GID: Final[str] = "statsbomb-logo"

_FLAGS: Final = ("provider", "is_goal", "is_own_goal", "period")


# --- Drawing the background -------------------------------------------------------------------


def _new_axes(ax: Axes | None, size: tuple[float, float]) -> Axes:
    if ax is not None:
        return ax
    fig = Figure(figsize=size, dpi=100, facecolor=SURFACE)
    FigureCanvasAgg(fig)  # Agg whatever pyplot's backend or MPLBACKEND say: same bytes everywhere
    w, h = size
    left, bottom, right, top = MARGIN_IN
    return fig.add_axes((left / w, bottom / h, 1 - (left + right) / w, 1 - (bottom + top) / h))


def _line(
    ax: Axes, gid: str, h: npt.ArrayLike, v: npt.ArrayLike, colour: str = LINE_COLOUR,
    lw: float = PITCH_LW,
) -> None:  # fmt: skip
    ax.plot(h, v, color=colour, lw=lw, zorder=1, gid=gid)


def _arc(
    cy: float, cx: float, r: float, a0: float, a1: float, n: int
) -> tuple[FloatArray, FloatArray]:
    """Points ``(y, x) = (cy + r sin a, cx + r cos a)`` for ``a`` from ``a0`` to ``a1``."""
    a = np.linspace(a0, a1, n)
    y: FloatArray = cy + r * np.sin(a)
    x: FloatArray = cx + r * np.cos(a)
    return y, x


def draw_half_pitch(
    ax: Axes | None = None,
    *,
    colour: str = LINE_COLOUR,
    lw: float = PITCH_LW,
    box_colour: str | None = None,
) -> Axes:
    """Draw the attacking half, goal at the top, as the shooter sees it.

    Every line is built from the ``football_core.pitch`` landmarks, so the markings sit on
    the same IFAB points the provider maps use. Data coordinates are ``(y, x)`` in canonical
    metres; the horizontal axis is inverted so that ``y = 0`` (the attacker's right) is on the
    right.

    Parameters
    ----------
    ax : matplotlib Axes, optional
        Axes to draw on; by default a new 6.0 x 5.9 in figure on the Agg canvas.
    colour : str, default LINE_COLOUR
        Line colour (a theme passes its ``line`` token).
    lw : float, default PITCH_LW
        Line width in points.
    box_colour : str, optional
        Colour of the penalty box and six-yard box lines; ``None`` is ``colour``. A themed
        chart draws them darker, because its title talks about the box.

    Returns
    -------
    matplotlib Axes
        Each marking is one ``Line2D`` with a ``pitch-*`` gid.

    Examples
    --------
    >>> ax = draw_half_pitch()
    >>> [float(v) for v in (*ax.get_xlim(), *ax.get_ylim())]
    [69.0, -1.0, 51.5, 108.0]
    """
    ax = _new_axes(ax, SHOT_MAP_SIZE)
    length, width = pitch.PITCH_LENGTH, pitch.PITCH_WIDTH
    box_x, spot_x, six_x = pitch.METRE_X[5], pitch.METRE_X[6], pitch.METRE_X[7]
    box_y0, six_y0, six_y1, box_y1 = (pitch.METRE_Y[i] for i in (1, 2, 5, 6))
    post_r, post_l = pitch.RIGHT_POST_Y, pitch.LEFT_POST_Y
    back = length + GOAL_DEPTH
    d = math.acos((spot_x - box_x) / CENTRE_CIRCLE_RADIUS)  # where the D meets the box line
    r, cy, pi = CENTRE_CIRCLE_RADIUS, pitch.GOAL_CENTRE_Y, math.pi
    c, b = colour, colour if box_colour is None else box_colour
    # fmt: off
    _line(ax, "pitch-outline", [0.0, 0.0, width, width, 0.0],
          [HALFWAY_X, length, length, HALFWAY_X, HALFWAY_X], c, lw)
    _line(ax, "pitch-box", [box_y0, box_y0, box_y1, box_y1], [length, box_x, box_x, length], b, lw)
    _line(ax, "pitch-six-yard-box", [six_y0, six_y0, six_y1, six_y1],
          [length, six_x, six_x, length], b, lw)
    _line(ax, "pitch-goal", [post_r, post_r, post_l, post_l], [length, back, back, length], c, lw)
    ax.plot([cy], [spot_x], ls="none", marker="o", ms=2.5, lw=lw, color=c,
            zorder=1, gid="pitch-spot")
    _line(ax, "pitch-arc", *_arc(cy, spot_x, r, pi - d, pi + d, 61), c, lw)
    _line(ax, "pitch-centre-circle", *_arc(cy, HALFWAY_X, r, -pi / 2, pi / 2, 91), c, lw)
    _line(ax, "pitch-corner-right", *_arc(0.0, length, CORNER_RADIUS, pi / 2, pi, 16), c, lw)
    _line(ax, "pitch-corner-left", *_arc(width, length, CORNER_RADIUS, -pi, -pi / 2, 16), c, lw)
    # fmt: on
    ax.set_xlim(width + 1.0, -1.0)
    ax.set_ylim(HALFWAY_X - 1.0, back + 1.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def draw_goal_mouth(ax: Axes | None = None) -> Axes:
    """Draw the 7.32 x 2.44 m goal frame front-on, from the pitch, right post on the right.

    Data coordinates are ``(goal_y, goal_z)`` in metres. The frame line is centred on the
    inner edges of the posts and bar, so it covers about 5 cm of the mouth (real posts stand
    outside the 7.32 m); at chart resolution that is invisible.

    Parameters
    ----------
    ax : matplotlib Axes, optional
        Axes to draw on; by default a new 6.0 x 3.45 in figure on the Agg canvas.

    Returns
    -------
    matplotlib Axes
        The frame is one ``Line2D`` with gid ``FRAME_GID``; the ground line has gid
        ``"ground"``.

    Examples
    --------
    >>> ax = draw_goal_mouth()
    >>> [round(float(v), 2) for v in ax.get_xlim()]
    [39.66, 28.34]
    """
    ax = _new_axes(ax, GOAL_MOUTH_SIZE)
    y0, y1 = pitch.RIGHT_POST_Y - VIEW_PAD_Y, pitch.LEFT_POST_Y + VIEW_PAD_Y
    ax.plot([y0, y1], [0.0, 0.0], color=LINE_COLOUR, lw=PITCH_LW, zorder=1, gid="ground")
    frame = Line2D(
        [pitch.RIGHT_POST_Y, pitch.RIGHT_POST_Y, pitch.LEFT_POST_Y, pitch.LEFT_POST_Y],
        [0.0, pitch.GOAL_HEIGHT, pitch.GOAL_HEIGHT, 0.0],
        color=INK, lw=FRAME_LW, solid_joinstyle="miter", zorder=2, gid=FRAME_GID,
    )  # fmt: skip
    ax.add_line(frame)
    ax.set_xlim(y1, y0)
    ax.set_ylim(-0.3, pitch.GOAL_HEIGHT + VIEW_PAD_Z)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


# --- Input checks -----------------------------------------------------------------------------


def _read_logo(logo: StrPath | None) -> Image:
    path = Path(DEFAULT_LOGO if logo is None else logo)
    if not path.is_file():  # missing, a directory, a dangling symlink, a loop, a FIFO
        raise FileNotFoundError(
            f"no StatsBomb logo at '{path}'; run statsbomb.fetch_logo() or pass logo="
        )
    try:
        img: Image = mpimg.imread(path)
    except (OSError, SyntaxError, ValueError) as e:  # PIL reports a non-PNG as SyntaxError
        raise ValueError(f"cannot read the logo '{path}': {e}") from None
    return img


def _count(mask: pl.Series) -> int:
    return int(mask.sum())


def _check(
    shots: object, xg: str, point: tuple[str, str], area_per_xg: float, logo: StrPath | None
) -> Image | None:
    """Reject anything the charts cannot draw honestly; return the logo if one is needed.

    Only the required columns are read. ``point`` is ``("x", "y")`` or ``("goal_y", "goal_z")``;
    only the goal-mouth point may be null (both together).
    """
    if not isinstance(shots, pl.DataFrame):
        raise TypeError(f"shots must be a polars DataFrame, not {type(shots).__name__}")
    if not (math.isfinite(area_per_xg) and area_per_xg > 0):
        raise ValueError(f"area_per_xg must be finite and > 0, not {area_per_xg!r}")
    required = (*_FLAGS, *point, xg)
    missing = [c for c in dict.fromkeys(required) if c not in shots.columns]
    if missing:
        raise ValueError(f"shots is missing required column(s): {', '.join(missing)}")
    expected = {c: schema.SHOT_SCHEMA[c] for c in (*_FLAGS, *point)} | {xg: pl.Float64()}
    for col, dtype in expected.items():
        if shots.schema[col] != dtype:
            raise ValueError(f"column '{col}' has dtype {shots.schema[col]}, expected {dtype}")
    for col in _FLAGS:
        if n := shots[col].null_count():
            raise ValueError(f"column '{col}' has {n} null value(s)")
    scope = ~(shots["is_own_goal"] | (shots["period"] == 5))  # own goals have no shot xG
    p = shots[xg]
    bad_xg = (p.is_null() | ~p.is_finite() | (p < 0) | (p > 1)) & scope
    if n := _count(bad_xg):
        raise ValueError(
            f"column '{xg}' has {n} value(s) that are null, NaN, infinite or outside [0, 1] "
            "on rows that are neither own goals nor shoot-out kicks"
        )
    a, b = shots[point[0]], shots[point[1]]
    nullable = point[0] == "goal_y"
    for col, s in ((point[0], a), (point[1], b)):
        bad = s.is_not_null() & ~s.is_finite() if nullable else s.is_null() | ~s.is_finite()
        if n := _count(bad):
            what = "NaN or infinite" if nullable else "null, NaN or infinite"
            raise ValueError(f"column '{col}' has {n} {what} value(s)")
    if nullable and (n := _count(a.is_null() ^ b.is_null())):
        raise ValueError(f"columns '{point[0]}' and '{point[1]}' are null apart on {n} row(s)")
    if (shots["provider"] == "statsbomb").any():
        return _read_logo(logo)  # before anything is drawn: never a chart without the logo
    return None


# --- Shots, legend, notes ---------------------------------------------------------------------


def _hidden(shots: pl.DataFrame) -> tuple[BoolArray, list[str]]:
    """Rows never drawn (own goals, then shoot-out kicks) and the note parts counting them."""
    own_goal = shots["is_own_goal"].to_numpy()
    shootout = (shots["period"] == 5).to_numpy() & ~own_goal
    parts = [
        f"{n} {label}"
        for n, label in ((own_goal.sum(), "own goal(s)"), (shootout.sum(), "shoot-out kick(s)"))
        if n
    ]
    keep: BoolArray = ~(own_goal | shootout)
    return keep, parts


def _scatter(
    ax: Axes, h: FloatArray, v: FloatArray, xg: FloatArray, goal: BoolArray, area_per_xg: float
) -> None:
    # xG-descending within each group, so small markers sit on top of large ones
    g = np.flatnonzero(goal)[np.argsort(-xg[goal], kind="stable")]
    n = np.flatnonzero(~goal)[np.argsort(-xg[~goal], kind="stable")]
    # fmt: off
    ax.scatter(h[g], v[g], s=area_per_xg * xg[g], c=GOAL_COLOUR, edgecolors=SURFACE,
               linewidths=0.8, zorder=3, gid=GOALS_GID)
    ax.scatter(h[n], v[n], s=area_per_xg * xg[n], facecolors="none", edgecolors=SHOT_COLOUR,
               linewidths=0.8, alpha=0.8, zorder=4, gid=SHOTS_GID)
    # fmt: on


def _legend(ax: Axes, area_per_xg: float) -> None:
    def key(label: str, v: float, face: str, edge: str) -> Line2D:
        # a Line2D marker size is a diameter in points, so its area matches the scatter's
        # fmt: off
        return Line2D([], [], ls="none", marker="o", markersize=math.sqrt(area_per_xg * v),
                      markerfacecolor=face, markeredgecolor=edge, label=label)
        # fmt: on

    handles = [key("Goal", 0.2, GOAL_COLOUR, SURFACE), key("No goal", 0.2, "none", SHOT_COLOUR)]
    handles += [key(f"xG {v:g}", v, "none", SHOT_COLOUR) for v in SIZE_KEY]
    down = ScaledTranslation(0.0, -LEGEND_OFFSET_PT / 72, ax.figure.dpi_scale_trans)
    # fmt: off
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              bbox_transform=ax.transAxes + down, ncol=5, frameon=False, fontsize=8,
              borderaxespad=0.0)
    # fmt: on


def _below(ax: Axes, text: str, offset_pt: float, size: float, colour: str, gid: str) -> None:
    """Text under the axes' left edge, ``offset_pt`` points down whatever the axes height."""
    # fmt: off
    ax.annotate(text, (0.0, 0.0), xycoords="axes fraction", xytext=(0.0, -offset_pt),
                textcoords="offset points", ha="left", va="top", fontsize=size, color=colour,
                gid=gid, annotation_clip=False)
    # fmt: on


def _attribute(ax: Axes, providers: tuple[str, ...], img: Image | None) -> None:
    present = [p for p in schema.PROVIDERS if p in providers]
    _below(ax, " · ".join(ATTRIBUTION[p] for p in present), TEXT_OFFSET_PT, 8, INK, TEXT_GID)
    if img is None:
        return
    box = AnnotationBbox(
        OffsetImage(img, zoom=LOGO_WIDTH_PT / img.shape[1]),  # zoom is per point: dpi-proof
        (1.0, 0.0), xycoords="axes fraction", xybox=(0.0, -(TEXT_OFFSET_PT - 2.0)),
        boxcoords="offset points", box_alignment=(1.0, 1.0), frameon=False, pad=0.0,
        annotation_clip=False,
    )  # fmt: skip
    box.set_gid(LOGO_GID)
    ax.add_artist(box)


def _attributed(ax: Axes) -> bool:
    return any(t.get_gid() == TEXT_GID for t in ax.texts)


def add_attribution(ax: Axes, *providers: str, logo: StrPath | None = None) -> None:
    """Credit the data source under the axes: text, plus the StatsBomb logo for StatsBomb.

    The charts call this themselves with the providers in the frame; call it directly only
    for a chart of your own. Nothing is added when no provider is given or the axes already
    carry an attribution (a second chart drawn on the same axes).

    Parameters
    ----------
    ax : matplotlib Axes
    *providers : str
        Values of ``schema.PROVIDERS``; a mixed chart credits each in one line.
    logo : str or path, optional
        The StatsBomb logo PNG; default ``DEFAULT_LOGO`` (``statsbomb.logo_file()``),
        relative to the working directory (fetch it once with ``statsbomb.fetch_logo()``).

    Raises
    ------
    ValueError
        An unknown provider, or a logo file that is not a readable image.
    FileNotFoundError
        ``"statsbomb"`` is given and there is no logo file.

    Examples
    --------
    >>> ax = draw_goal_mouth()
    >>> add_attribution(ax, "whoscored")
    >>> [t.get_text() for t in ax.texts]
    ['Data: Opta via WhoScored']
    """
    bad = [p for p in providers if p not in schema.PROVIDERS]
    if bad:
        raise ValueError(f"unknown provider(s) {bad}; expected some of {list(schema.PROVIDERS)}")
    if not providers or _attributed(ax):
        return
    _attribute(ax, providers, _read_logo(logo) if "statsbomb" in providers else None)


def _finish(
    ax: Axes, shots: pl.DataFrame, parts: list[str], img: Image | None, area_per_xg: float
) -> None:
    _legend(ax, area_per_xg)
    if parts:
        _below(ax, "not shown: " + ", ".join(parts), NOTE_OFFSET_PT, 7, SHOT_COLOUR, NOTE_GID)
    providers = tuple(shots["provider"].cast(pl.String).unique().sort().to_list())
    if providers and not _attributed(ax):
        _attribute(ax, providers, img)


# --- Charts -----------------------------------------------------------------------------------


def shot_map(
    shots: pl.DataFrame,
    *,
    xg: str,
    ax: Axes | None = None,
    logo: StrPath | None = None,
    area_per_xg: float = AREA_PER_XG,
) -> Axes:
    """Shot map on the attacking half: marker area proportional to xG, goals highlighted.

    Filtering (team, player, penalties) is the caller's job. In-match penalties are drawn;
    own goals and shoot-out kicks (period 5) never are, and neither are shots outside the half
    pitch: a note under the chart counts each group. Markers for xG near 0 are drawn but
    invisible, because area stays exactly proportional (no floor).

    Parameters
    ----------
    shots : polars.DataFrame
        C3 shot columns ``provider``, ``is_goal``, ``is_own_goal``, ``period``, ``x``, ``y``
        (canonical metres, no nulls) with their C3 dtypes, plus the ``xg`` column. Other
        columns are not read.
    xg : str
        A ``Float64`` column of P(goal) in [0, 1], e.g. ``"provider_xg"`` or a model's output;
        null only on own goals and shoot-out kicks. No default, so a chart names its model.
    ax : matplotlib Axes, optional
        Axes to draw on; by default a new figure (Agg canvas, fixed layout).
    logo : str or path, optional
        StatsBomb logo PNG, read when the frame has StatsBomb rows; default
        ``DEFAULT_LOGO``.
    area_per_xg : float, default AREA_PER_XG
        Marker area in pt² at xG = 1.

    Returns
    -------
    matplotlib Axes
        Goals are the collection with gid ``GOALS_GID``, other shots ``SHOTS_GID``.

    Raises
    ------
    TypeError
        ``shots`` is not a polars DataFrame.
    ValueError
        A missing column, a wrong dtype, a null flag, a bad xG or location, a bad
        ``area_per_xg``, or an unreadable logo.
    FileNotFoundError
        StatsBomb rows and no logo file; raised before anything is drawn.

    Examples
    --------
    >>> shots = pl.DataFrame({"provider": ["whoscored"], "is_goal": [True],
    ...                       "is_own_goal": [False], "period": [1], "x": [94.0], "y": [34.0],
    ...                       "p": [0.75]}).cast({"provider": schema.PROVIDER, "period": pl.Int8})
    >>> ax = shot_map(shots, xg="p")
    >>> [c.get_sizes().tolist() for c in ax.collections if c.get_gid() == GOALS_GID]
    [[450.0]]
    """
    img = _check(shots, xg, ("x", "y"), area_per_xg, logo)
    ax = draw_half_pitch(ax)
    keep, parts = _hidden(shots)
    s = shots.filter(pl.Series(keep))
    x, y = s["x"].to_numpy(), s["y"].to_numpy()
    view = (x >= HALFWAY_X) & (x <= pitch.PITCH_LENGTH) & (y >= 0) & (y <= pitch.PITCH_WIDTH)
    if n := int((~view).sum()):
        parts.append(f"{n} outside the view")
    p, goal = s[xg].to_numpy(), s["is_goal"].to_numpy()
    _scatter(ax, y[view], x[view], p[view], goal[view], area_per_xg)
    _finish(ax, shots, parts, img, area_per_xg)
    return ax


def goal_mouth(
    shots: pl.DataFrame,
    *,
    xg: str,
    ax: Axes | None = None,
    logo: StrPath | None = None,
    area_per_xg: float = AREA_PER_XG,
) -> Axes:
    """Front-on goal-mouth plot: shots at ``(goal_y, goal_z)``, area proportional to xG.

    The view is the 7.32 x 2.44 m frame padded by ``VIEW_PAD_Y`` either side and
    ``VIEW_PAD_Z`` above the bar; points are never clipped to the frame, so a save wide of a
    post is drawn where it was recorded. For a StatsBomb save that point is where the keeper
    stopped the ball, not a goal-line projection, so saves can sit outside the frame. Rows
    without a goal-mouth point (blocked shots, most misses), shots outside the view, own goals
    and shoot-out kicks are not drawn and are counted in the note. In-match penalties are
    drawn; markers for xG near 0 are drawn but invisible (area stays exactly proportional).

    Parameters
    ----------
    shots : polars.DataFrame
        As :func:`shot_map`, with ``goal_y`` and ``goal_z`` (metres; null together) instead
        of ``x`` and ``y``.
    xg : str
        As :func:`shot_map`. Pre-shot xG, not xGOT.
    ax, logo, area_per_xg
        As :func:`shot_map`.

    Returns
    -------
    matplotlib Axes

    Raises
    ------
    TypeError, ValueError, FileNotFoundError
        As :func:`shot_map`; also ``goal_y``/``goal_z`` NaN or null on their own.

    Examples
    --------
    >>> shots = pl.DataFrame({"provider": ["whoscored"], "is_goal": [True],
    ...                       "is_own_goal": [False], "period": [1], "goal_y": [31.0],
    ...                       "goal_z": [0.5], "p": [0.25]}).cast(
    ...     {"provider": schema.PROVIDER, "period": pl.Int8})
    >>> ax = goal_mouth(shots, xg="p")
    >>> [c.get_sizes().tolist() for c in ax.collections if c.get_gid() == GOALS_GID]
    [[150.0]]
    """
    img = _check(shots, xg, ("goal_y", "goal_z"), area_per_xg, logo)
    ax = draw_goal_mouth(ax)
    keep, parts = _hidden(shots)
    s = shots.filter(pl.Series(keep))
    has_point = s["goal_y"].is_not_null()
    if n := _count(~has_point):
        parts.append(f"{n} without a goal-mouth point")
    s = s.filter(has_point)
    gy, gz = s["goal_y"].to_numpy(), s["goal_z"].to_numpy()
    view = (
        (gy >= pitch.RIGHT_POST_Y - VIEW_PAD_Y)
        & (gy <= pitch.LEFT_POST_Y + VIEW_PAD_Y)
        & (gz >= 0.0)
        & (gz <= pitch.GOAL_HEIGHT + VIEW_PAD_Z)
    )
    if n := int((~view).sum()):
        parts.append(f"{n} outside the view")
    p, goal = s[xg].to_numpy(), s["is_goal"].to_numpy()
    _scatter(ax, gy[view], gz[view], p[view], goal[view], area_per_xg)
    _finish(ax, shots, parts, img, area_per_xg)
    return ax


# --- Theme layer: fonts, tokens, layout (CH1) -------------------------------------------------

FONT_DIR: Final = Path(__file__).parent / "fonts"
FONT_FAMILY: Final[str] = "Inter"
FALLBACK_FAMILY: Final[str] = "DejaVu Sans"  # ships with matplotlib; has ş and −
FONT_FILES: Final[Mapping[int, str]] = MappingProxyType(
    {400: "Inter-Regular.ttf", 500: "Inter-Medium.ttf", 600: "Inter-SemiBold.ttf",
     700: "Inter-Bold.ttf"}
)  # fmt: skip
FONT_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Inter-Regular.ttf": "64f8be6e55c37e32ef03da99714bf3aa58b8f2099bfe4f759a7578e3b8291123",
        "Inter-Medium.ttf": "1bda81124d6ae26ed16a7201e2bd93766af5a3b14faf79eea14d191ebbd41146",
        "Inter-SemiBold.ttf": "0dc98e8aa59585394880f25ab89e6d915ad5134522e961b046ca51fad3a18255",
        "Inter-Bold.ttf": "0cb1bc1335372d9e3a0cf6f5311c7cce87af90d2a777fdeec18be605a2a70bc1",
        "LICENSE.txt": "262481e844521b326f5ecd053e59b98c8b2da78c8ee1bdbb6e8174305e54935a",
    }
)


def register_fonts(directory: StrPath = FONT_DIR) -> str:
    """Register the four Inter weights with matplotlib; the family the themed charts use.

    Each file is added to matplotlib's process-wide font manager, then every weight must
    resolve to its own file (or a byte-identical copy already registered): a missing file, an
    unreadable one, or a file of another font under an Inter name would otherwise draw in a
    silently substituted face. ``rcParams`` is never touched, so charts that do not ask for the
    family (the MS2 charts, the keeper SVG) keep matplotlib's default font, byte for byte.

    Parameters
    ----------
    directory : str or path, default FONT_DIR
        Holds ``FONT_FILES`` (the vendored copies, SIL OFL, licence in ``LICENSE.txt``).

    Returns
    -------
    str
        ``FONT_FAMILY`` when all four weights resolve; otherwise ``FALLBACK_FAMILY``, after
        one ``UserWarning`` naming the first file that failed.

    Examples
    --------
    >>> register_fonts()
    'Inter'
    """
    for weight, name in FONT_FILES.items():
        path = Path(directory) / name
        try:
            font_manager.fontManager.addfont(str(path))
            found = font_manager.findfont(
                FontProperties(family=FONT_FAMILY, weight=weight), fallback_to_default=False
            )
            ok = Path(found) == path or filecmp.cmp(found, path, shallow=False)
        except (OSError, RuntimeError, ValueError) as e:  # missing, unreadable, unresolved
            warnings.warn(f"font {path} not usable ({e}); using {FALLBACK_FAMILY}", stacklevel=2)
            return FALLBACK_FAMILY
        if not ok:
            warnings.warn(
                f"font {path}: weight {weight} resolves to {found}; using {FALLBACK_FAMILY}",
                stacklevel=2,
            )
            return FALLBACK_FAMILY
    return FONT_FAMILY


FAMILY: Final[str] = register_fonts()


@dataclass(frozen=True)
class Theme:
    """One theme's named tokens: colours ``#rrggbb`` (WCAG-checked pairs), one alpha, a family.

    Text tokens (``title``, ``ink``, ``muted``) meet 4.5:1 on ``background`` and ``panel``;
    mark tokens (``accent``, ``mark`` at ``mark_alpha``, ``neutral``, ``ink``) meet 3:1 on
    every surface they sit on, the ``band`` tint included. ``line``, ``box_line`` and ``band``
    are decoration (no contrast floor). ``logo_plate`` is the light plate behind the
    colour-positive StatsBomb logo, or ``None`` when the background is light already.
    """

    name: str
    background: str
    panel: str
    title: str
    ink: str
    muted: str
    line: str
    box_line: str
    band: str
    accent: str
    mark: str
    mark_alpha: float
    neutral: str
    logo_plate: str | None
    family: str


THEMES: Final[Mapping[str, Theme]] = MappingProxyType(
    {
        "light": Theme("light", background="#f8fafc", panel="#eef2f6", title="#475569",
                       ink="#334155", muted="#5b6b80", line="#cbd5e1", box_line="#a3afc0",
                       band="#d8e0e9", accent="#e4002b", mark="#475569", mark_alpha=0.7,
                       neutral="#64748b", logo_plate=None, family=FAMILY),
        "dark": Theme("dark", background="#0f172a", panel="#1a2436", title="#f1f5f9",
                      ink="#e2e8f0", muted="#a3b1c4", line="#334155", box_line="#475569",
                      band="#2b3a52", accent="#ff4d6a", mark="#94a3b8", mark_alpha=0.7,
                      neutral="#94a3b8", logo_plate="#f8fafc", family=FAMILY),
    }
)  # fmt: skip
# (size pt, weight, colour token): one scale for both themes and sizes; data text >= 11.5 pt
TEXT_STYLES: Final[Mapping[str, tuple[float, str, str]]] = MappingProxyType(
    {
        "title": (21.0, "bold", "title"),
        "subtitle": (13.0, "regular", "muted"),
        "stat_value": (20.0, "semibold", "ink"),
        "stat_label": (11.5, "medium", "muted"),
        "stat_note": (11.5, "regular", "muted"),
        "heading": (13.0, "semibold", "ink"),
        "label": (12.0, "medium", "ink"),
        "caption": (11.5, "regular", "muted"),
        "legend": (11.5, "regular", "ink"),
        "footer": (9.5, "regular", "muted"),
        "footer_strong": (9.5, "semibold", "muted"),
    }
)
TITLE_STEP_PT: Final[float] = 19.0  # the first step down for a long title
TITLE_MIN_PT: Final[float] = 17.0  # the last step (a box-share title with its league rank)
TITLE_FALLBACK_PT: Final[float] = 15.0  # fallback family only: DejaVu Sans is ~16% wider
SIZES: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {"portrait": (1080, 1350), "square": (1080, 1080)}
)
DPI: Final[int] = 150
MARGIN_PX: Final[int] = 64  # sides; every width rule is against W - 2 * MARGIN_PX
TOP_PX: Final[int] = 48
BOTTOM_PX: Final[int] = 32
GAP_PX: Final[int] = 20
LINE_GAP_PX: Final[int] = 4  # between stacked text blocks of one band
STAT_PAD_PX: Final[int] = 12
MIN_BODY_PX: Final[int] = 380
LOGO_PX: Final[int] = 170
LOGO_PLATE_PAD_PX: Final[int] = 6
PANEL_RADIUS_PX: Final[float] = 12.0
KEY_GAP_PX: Final[int] = 8  # marker to label
KEY_SPACING_PX: Final[int] = 16  # between keys
MAX_STATS: Final[int] = 5
PREPARED_BY: Final[str] = "Prepared by R. Berk Karataş"
CROP_PAD: Final[float] = 2.0  # m around the shots and the box
CROP_MAX_DEPTH: Final[float] = 35.0  # m from the goal line (PM E4)
INK_SHARE: Final[float] = 0.12  # share of the pitch area the markers cover (p6, chosen on dev)
AREA_MAX: Final[float] = 900.0  # pt² of s per xG
AREA_PER_XG_MIN: Final[float] = 190.0  # keeps the 0.1 xG key >= MIN_KEY_PX across
GOAL_MIN_S: Final[float] = 36.0  # pt²: a goal is >= 6 pt across, whatever its xG
MIN_KEY_PX: Final[float] = 9.0
MARK_LW: Final[float] = 0.9
LEGEND_XG: Final[tuple[float, ...]] = (0.1, 0.3, 0.6)
MARK_SHOTS: Final[str] = "mark-shots"
MARK_GOALS: Final[str] = "mark-goals"
MARK_KEY: Final[str] = "mark-key"
PANEL_PREFIX: Final[str] = "panel-"
LARGE_PT: Final[tuple[float, float, int]] = (18.0, 14.0, 600)  # WCAG large: 18 pt, 14 pt bold
TEXT_CONTRAST: Final[tuple[float, float]] = (4.5, 3.0)  # body text, large text
MARK_CONTRAST: Final[float] = 3.0
EPS_PX: Final[float] = 1e-6  # float noise of boxes laid edge to edge


@dataclass(frozen=True)
class Stat:
    """One stat-strip column: the value, its label and an optional note (``"SD 6.5"``)."""

    value: str
    label: str
    note: str | None = None


@dataclass(frozen=True)
class Key:
    """One legend key: a ``"goal"`` (filled accent) or ``"shot"`` (ring) marker of area
    ``size`` (matplotlib's ``s`` in pt², ``√size`` pt across) and its label."""

    kind: Literal["goal", "shot"]
    label: str
    size: float


@dataclass(frozen=True)
class Footer:
    """What the footer credits: providers (``schema.PROVIDERS``), the model, a note."""

    providers: tuple[str, ...]
    model: str | None = None
    note: str | None = None


def _theme(theme: str | Theme) -> Theme:
    if isinstance(theme, Theme):
        return theme
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}; expected one of {list(THEMES)}")
    return THEMES[theme]


def _text_kwargs(theme: Theme, style: str) -> dict[str, Any]:
    if style not in TEXT_STYLES:
        raise ValueError(f"unknown text style {style!r}; expected one of {list(TEXT_STYLES)}")
    size, weight, token = TEXT_STYLES[style]
    colour = getattr(theme, token)
    return {"fontsize": size, "fontweight": weight, "color": colour, "family": theme.family}


@dataclass(frozen=True, eq=False)
class Canvas:
    """A themed figure of fixed pixel size: header, optional stat strip, body, footer.

    Attributes
    ----------
    fig : matplotlib Figure
        On the Agg canvas, ``SIZES[size]`` pixels at ``DPI``.
    theme : Theme
    size : str
        A key of ``SIZES``.
    body : tuple of float
        The body's ``(left, bottom, width, height)`` in figure fractions; :meth:`axes` places
        rects inside it.
    """

    fig: Figure
    theme: Theme
    size: str
    body: tuple[float, float, float, float]

    @property
    def px(self) -> Affine2D:
        """Transform from canvas pixels (origin bottom-left) to display."""
        w, h = SIZES[self.size]
        return Affine2D().scale(1 / w, 1 / h) + self.fig.transFigure  # type: ignore[return-value]

    def measure(self, text: str, style: str) -> tuple[float, float]:
        """Rendered ``(width, height)`` in px of ``text`` (newlines allowed) in a style.

        Examples
        --------
        >>> c = canvas("light", "square", title="T", subtitle="S", footer=Footer(("whoscored",)))
        >>> w, h = c.measure("Goals", "label"); 30 < w < 80 and 20 < h < 40
        True
        """
        return _measure(self, text, style)

    def text(
        self, x: float, y: float, text: str, style: str, *, ax: Axes | None = None, **kw: Any
    ) -> Text:
        """A themed ``Text`` at ``(x, y)`` in ``ax``'s data coordinates, or in canvas pixels
        (origin bottom-left) without ``ax``; ``kw`` go to matplotlib (``ha``, ``va``, ``gid``).

        Examples
        --------
        >>> c = canvas("dark", "square", title="T", subtitle="S", footer=Footer(("whoscored",)))
        >>> c.text(100, 100, "Hi", "label").get_color()
        '#e2e8f0'
        """
        style_kw = _text_kwargs(self.theme, style) | kw
        if ax is None:
            return self.fig.text(x, y, text, transform=self.px, **style_kw)
        return ax.text(x, y, text, **style_kw)

    def _rounded(self, x: float, y: float, w: float, h: float, colour: str, gid: str) -> None:
        self.fig.add_artist(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={PANEL_RADIUS_PX}",
            transform=self.px, fc=colour, ec="none", zorder=-1, gid=gid,
        ))  # fmt: skip

    def axes(
        self, rect: tuple[float, float, float, float], *, gid: str, panel: bool = False
    ) -> Axes:
        """An axis-off axes at ``rect`` (fractions of the body: left, bottom, width, height).

        ``panel=True`` draws the rounded ``panel`` patch behind it and prefixes the gid with
        ``PANEL_PREFIX`` (:func:`check` keeps its texts inside it).

        Examples
        --------
        >>> c = canvas("light", "square", title="T", subtitle="S", footer=Footer(("whoscored",)))
        >>> c.axes((0.0, 0.0, 0.5, 1.0), gid="zones", panel=True).get_gid()
        'panel-zones'
        """
        left, bottom, width, height = self.body
        r = (left + rect[0] * width, bottom + rect[1] * height, rect[2] * width, rect[3] * height)
        ax = self.fig.add_axes(r)
        ax.set_axis_off()
        ax.set_gid(PANEL_PREFIX + gid if panel else gid)
        if panel:
            w, h = SIZES[self.size]
            self._rounded(r[0] * w, r[1] * h, r[2] * w, r[3] * h, self.theme.panel, "panel")
        return ax

    def _legend_rows(
        self, keys: Sequence[Key], width: float
    ) -> tuple[list[list[int]], list[float], list[float], list[str]]:
        """Greedy rows of keys; while they need more than 2 rows, the widest one-line label of
        several words is split into two balanced lines and the rows are tried again."""
        diam = [math.sqrt(k.size) * DPI / 72 for k in keys]
        labels = [k.label for k in keys]
        while True:
            widths = [
                d + KEY_GAP_PX + self.measure(label, "legend")[0]
                for label, d in zip(labels, diam, strict=True)
            ]
            rows: list[list[int]] = [[]]
            for i, kw in enumerate(widths):
                used = sum(widths[j] + KEY_SPACING_PX for j in rows[-1])
                if rows[-1] and used + kw > width:
                    rows.append([])
                rows[-1].append(i)
            if len(rows) <= 2 and max(widths, default=0.0) <= width:
                return rows, widths, diam, labels
            split = [i for i, label in enumerate(labels) if " " in label and "\n" not in label]
            if not split:
                raise ValueError(f"legend: {len(keys)} keys do not fit in {width:.0f} px in 2 rows")
            i = max(split, key=lambda j: widths[j])
            labels[i] = _halves(self, labels[i])

    def _row_heights(
        self, rows: list[list[int]], diam: list[float], labels: list[str]
    ) -> list[float]:
        return [
            max(max(self.measure(labels[i], "legend")[1], diam[i]) for i in row) for row in rows
        ]

    def legend_height(self, keys: Sequence[Key], width: float) -> float:
        """Height in px that :meth:`legend` takes in ``width`` px (:func:`legend_width` of the
        axes), the gap above it included, so a chart can reserve it before drawing.

        Raises
        ------
        ValueError
            As :meth:`legend`.

        Examples
        --------
        >>> c = canvas("light", "square", title="T", subtitle="S", footer=Footer(("whoscored",)))
        >>> 30 < c.legend_height((Key("goal", "Goal", 36.0),), 500.0) < 50
        True
        """
        rows, _, diam, labels = self._legend_rows(keys, width)
        heights = self._row_heights(rows, diam, labels)
        return KEY_GAP_PX + sum(heights) + LINE_GAP_PX * (len(rows) - 1)

    def legend_width(self, ax: Axes) -> float:
        """The legend's width under ``ax``: from the drawn extent's left edge to the right edge
        of the box the axes was given (an equal-aspect map may be drawn narrower than its box).

        Examples
        --------
        >>> c = canvas("light", "square", title="T", subtitle="S", footer=Footer(("whoscored",)))
        >>> round(c.legend_width(c.axes((0.0, 0.0, 0.5, 1.0), gid="map")))
        476
        """
        ax.apply_aspect()
        w = SIZES[self.size][0]
        return float(ax.get_position(original=True).x1 - ax.get_position().x0) * w

    def legend(self, keys: Sequence[Key], *, ax: Axes) -> None:
        """Lay out ``keys`` under ``ax``'s drawn extent, left-aligned with it, in <= 2 rows.

        The legend belongs to the map it explains, not to the canvas. Each key is a marker of
        the theme's goal or shot style (gid ``MARK_KEY``) and its label; while one-line labels
        would need a third row, the widest is split into two lines.

        Raises
        ------
        ValueError
            The keys do not fit in :meth:`legend_width` in 2 rows.

        Examples
        --------
        >>> c = canvas("light", "square", title="T", subtitle="S", footer=Footer(("whoscored",)))
        >>> ax = c.axes((0.0, 0.3, 1.0, 0.7), gid="map")
        >>> c.legend((Key("goal", "Goal", 36.0), Key("shot", "No goal", 36.0)), ax=ax)
        >>> [a.get_gid() for a in c.fig.artists if a.get_gid() == MARK_KEY]
        ['mark-key', 'mark-key']
        """
        w, h = SIZES[self.size]
        pos = ax.get_position()
        x0, top = pos.x0 * w, pos.y0 * h - KEY_GAP_PX
        rows, widths, diam, labels = self._legend_rows(keys, self.legend_width(ax))
        for row, row_h in zip(rows, self._row_heights(rows, diam, labels), strict=True):
            x, cy = x0, top - row_h / 2
            for i in row:
                self._key(keys[i], x + diam[i] / 2, cy)
                self.text(x + diam[i] + KEY_GAP_PX, cy, labels[i], "legend", va="center")
                x += widths[i] + KEY_SPACING_PX
            top -= row_h + LINE_GAP_PX

    def _key(self, key: Key, x: float, y: float) -> None:
        t = self.theme
        goal = key.kind == "goal"
        marker = MarkerStyle("o")  # as scatter: the unit circle scaled to diameter 1
        path = marker.get_path().transformed(marker.get_transform())
        coll = PathCollection(
            (path,), sizes=[key.size], offsets=[(x, y)],
            offset_transform=self.px, facecolors=t.accent if goal else "none",
            edgecolors=t.background if goal else t.mark, linewidths=MARK_LW,
            alpha=None if goal else t.mark_alpha, gid=MARK_KEY,
        )  # fmt: skip
        coll.set_transform(IdentityTransform())
        self.fig.add_artist(coll)


def _halves(c: Canvas, text: str) -> str:
    """``text`` in two lines of balanced measured width (one word stays one line)."""
    words = text.split(" ")
    if len(words) < 2:
        return text
    cuts = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    a, b = min(
        cuts, key=lambda ab: max(c.measure(ab[0], "legend")[0], c.measure(ab[1], "legend")[0])
    )
    return f"{a}\n{b}"


def _measure(c: Canvas, text: str, style: str, size: float | None = None) -> tuple[float, float]:
    kw = _text_kwargs(c.theme, style) | ({} if size is None else {"fontsize": size})
    t = Text(0.0, 0.0, text, **kw)
    t.set_figure(c.fig)
    box = t.get_window_extent(c.fig.canvas.get_renderer())  # type: ignore[attr-defined]
    return float(box.width), float(box.height)


def _wrap(
    c: Canvas, text: str, style: str, width: float, lines: int, size: float | None = None
) -> str | None:
    """Greedy word wrap by measured width; ``None`` when it needs more than ``lines`` lines."""
    out: list[str] = []
    for word in text.split(" "):
        trial = f"{out[-1]} {word}" if out else word
        if out and _measure(c, trial, style, size)[0] <= width:
            out[-1] = trial
        else:
            out.append(word)
    if len(out) > lines or any(_measure(c, line, style, size)[0] > width for line in out):
        return None
    return "\n".join(out)


def _wrap_or_raise(c: Canvas, text: str, style: str, width: float, lines: int, what: str) -> str:
    wrapped = _wrap(c, text, style, width, lines)
    if wrapped is None:
        raise ValueError(f"{what} needs more than {lines} line(s) at {width:.0f} px: {text!r}")
    return wrapped


def canvas(
    theme: str | Theme,
    size: str,
    *,
    title: str,
    subtitle: str,
    footer: Footer,
    stats: Sequence[Stat] = (),
    logo: StrPath | None = None,
) -> Canvas:
    """A themed chart canvas: statement title, subtitle, stat strip, body and credited footer.

    Bands top to bottom: the title (<= 2 lines at 21 pt, else at ``TITLE_STEP_PT``, else at
    ``TITLE_MIN_PT``, and with the fallback family only at ``TITLE_FALLBACK_PT``), the
    subtitle (<= 2 lines), the optional stat strip on a rounded panel (column widths from the
    measured content, the slack shared equally), the body (the rest, at least
    ``MIN_BODY_PX`` tall), and the footer: a rule, then the note, the attribution with the
    model, and ``PREPARED_BY``, with the StatsBomb logo right-aligned (on the theme's
    ``logo_plate`` when it has one). Every width is measured against the content width
    ``W - 2 * MARGIN_PX``.

    Parameters
    ----------
    theme : str or Theme
        A key of ``THEMES`` or a Theme.
    size : str
        A key of ``SIZES``.
    title, subtitle : str
    footer : Footer
    stats : sequence of Stat, default ()
        0 to ``MAX_STATS`` columns.
    logo : str or path, optional
        The StatsBomb logo PNG (default ``DEFAULT_LOGO``), read when the footer credits
        StatsBomb.

    Returns
    -------
    Canvas

    Raises
    ------
    ValueError
        An unknown theme, size or provider, no provider, more than ``MAX_STATS`` stats; a
        title, subtitle or footer part over its line limit, stats wider than the content, a
        body under ``MIN_BODY_PX`` (each message names the part); an unreadable logo.
    FileNotFoundError
        StatsBomb credited and no logo file; raised before anything is drawn.

    Examples
    --------
    >>> c = canvas("light", "portrait", title="Most goals came from inside the box",
    ...            subtitle="Team · League 2015/2016", stats=[Stat("512", "Shots")],
    ...            footer=Footer(("whoscored",), model="xG: own model"))
    >>> c.fig.canvas.get_width_height(), check(c)
    ((1080, 1350), [])
    """
    t = _theme(theme)
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; expected one of {list(SIZES)}")
    bad = [p for p in footer.providers if p not in schema.PROVIDERS]
    if bad or not footer.providers:
        raise ValueError(
            f"footer providers {list(footer.providers)}: need some of {list(schema.PROVIDERS)}"
        )
    if len(stats) > MAX_STATS:
        raise ValueError(f"{len(stats)} stats; at most {MAX_STATS} fit the strip")
    img = _read_logo(logo) if "statsbomb" in footer.providers else None
    w, h = SIZES[size]
    fig = Figure(figsize=(w / DPI, h / DPI), dpi=DPI, facecolor=t.background)
    FigureCanvasAgg(fig)
    c = Canvas(fig, t, size, (0.0, 0.0, 1.0, 1.0))
    content = w - 2 * MARGIN_PX
    head, size_pt = None, TEXT_STYLES["title"][0]
    steps: tuple[float, ...] = (TEXT_STYLES["title"][0], TITLE_STEP_PT, TITLE_MIN_PT)
    if t.family != FONT_FAMILY:  # the wider fallback draws every title branch Inter draws
        steps += (TITLE_FALLBACK_PT,)
    for size_pt in steps:
        if (head := _wrap(c, title, "title", content, 2, size_pt)) is not None:
            break
    if head is None:
        raise ValueError(f"title needs more than 2 lines at {steps[-1]} pt: {title!r}")
    sub = _wrap_or_raise(c, subtitle, "subtitle", content, 2, "subtitle")
    y = float(h - TOP_PX)
    c.text(MARGIN_PX, y, head, "title", va="top", gid="title", fontsize=size_pt)
    y -= _measure(c, head, "title", size_pt)[1] + LINE_GAP_PX
    c.text(MARGIN_PX, y, sub, "subtitle", va="top", gid="subtitle")
    y -= c.measure(sub, "subtitle")[1] + GAP_PX
    if stats:
        y = _strip(c, stats, y, content) - GAP_PX
    bottom = _footer(c, footer, img, content) + GAP_PX
    if y - bottom < MIN_BODY_PX:
        raise ValueError(f"body is {y - bottom:.0f} px tall; at least {MIN_BODY_PX} px needed")
    body = (MARGIN_PX / w, bottom / h, content / w, (y - bottom) / h)
    return Canvas(fig, t, size, body)


def _strip(c: Canvas, stats: Sequence[Stat], top: float, content: float) -> float:
    """Draw the stat strip from ``top`` (px from the bottom); return its bottom."""
    cols = [[(s.value, "stat_value"), (s.label, "stat_label")] for s in stats]
    for col, s in zip(cols, stats, strict=True):
        if s.note is not None:
            col.append((s.note, "stat_note"))
    mins = [max(c.measure(v, st)[0] for v, st in col) + 2 * STAT_PAD_PX for col in cols]
    if sum(mins) > content:
        widest = stats[int(np.argmax(mins))]
        raise ValueError(
            f"stats need {sum(mins):.0f} px, over the {content:.0f} px content width; widest: "
            f"{widest.label!r} ({max(mins):.0f} px)"
        )
    slack = (content - sum(mins)) / len(stats)
    # each row as tall as its tallest text: extents differ by glyphs (DejaVu digits)
    depth = max(len(col) for col in cols)
    heights = [max(c.measure(*col[i])[1] for col in cols if i < len(col)) for i in range(depth)]
    height = sum(heights) + 2 * STAT_PAD_PX
    c._rounded(MARGIN_PX, top - height, content, height, c.theme.panel, "panel-stats")
    x = float(MARGIN_PX)
    for col, m in zip(cols, mins, strict=True):
        y = top - STAT_PAD_PX
        for (value, style), row_h in zip(col, heights, strict=False):
            c.text(x + STAT_PAD_PX, y, value, style, va="top", gid=style.replace("_", "-"))
            y -= row_h
        x += m + slack
    return top - height


def _footer(c: Canvas, footer: Footer, img: Image | None, content: float) -> float:
    """Draw the footer up from ``BOTTOM_PX``; return the y of its rule."""
    w, _ = SIZES[c.size]
    width = content - (LOGO_PX + GAP_PX if img is not None else 0)
    credit = " · ".join(ATTRIBUTION[p] for p in schema.PROVIDERS if p in footer.providers)
    credit += f" · {footer.model}" if footer.model else ""
    parts = [(PREPARED_BY, "footer_strong", 1, "prepared-by"), (credit, "footer", 2, TEXT_GID)]
    if footer.note:
        parts.append((footer.note, "caption", 2, NOTE_GID))
    y = float(BOTTOM_PX)
    for text, style, lines, gid in parts:
        wrapped = _wrap_or_raise(c, text, style, width, lines, f"footer {gid}")
        c.text(MARGIN_PX, y, wrapped, style, va="bottom", gid=gid)
        y += c.measure(wrapped, style)[1] + LINE_GAP_PX
    if img is not None:
        zoom = LOGO_PX * 72 / DPI / img.shape[1]  # OffsetImage scales by dpi / 72
        logo_h = img.shape[0] * LOGO_PX / img.shape[1]
        x1, y0 = w - MARGIN_PX, float(BOTTOM_PX)
        if c.theme.logo_plate is not None:
            pad = LOGO_PLATE_PAD_PX
            c.fig.add_artist(FancyBboxPatch(
                (x1 - LOGO_PX - pad, y0 - pad), LOGO_PX + 2 * pad, logo_h + 2 * pad,
                boxstyle="round,pad=0,rounding_size=4", transform=c.px, fc=c.theme.logo_plate,
                ec="none", zorder=1, gid="logo-plate",
            ))  # fmt: skip
        box = AnnotationBbox(
            OffsetImage(img, zoom=zoom), (x1, y0), xycoords=c.px, box_alignment=(1.0, 0.0),
            frameon=False, pad=0.0, annotation_clip=False, zorder=2,
        )  # fmt: skip
        box.set_gid(LOGO_GID)
        c.fig.add_artist(box)
        y = max(y, y0 + logo_h + LINE_GAP_PX)
    y += 12.0
    c.fig.add_artist(Line2D([MARGIN_PX, w - MARGIN_PX], [y, y], transform=c.px,
                            color=c.theme.line, lw=0.8, gid="footer-rule"))  # fmt: skip
    return y


def save(c: Canvas, path: StrPath) -> Path:
    """Write the canvas as PNG at ``DPI``: exactly ``SIZES[c.size]`` pixels, no timestamp.

    Examples
    --------
    >>> import tempfile
    >>> c = canvas("dark", "square", title="T", subtitle="S", footer=Footer(("whoscored",)))
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     mpimg.imread(save(c, Path(tmp) / "c.png")).shape
    (1080, 1080, 4)
    """
    c.fig.savefig(path, dpi=DPI, facecolor=c.theme.background, metadata={"Software": None})
    return Path(path)


# --- Pitch crop, marker scale, shots ----------------------------------------------------------


def crop_pitch(
    ax: Axes,
    x: FloatArray,
    y: FloatArray,
    *,
    pad: float = CROP_PAD,
    x_min: float | None = pitch.PITCH_LENGTH - CROP_MAX_DEPTH,
) -> BoolArray:
    """Crop a vertical pitch (goal at the top) to the shots and the box, filling the axes.

    The extent runs from ``max(x_min, min x - pad)`` (``x_min=None``: no floor beyond -1 m)
    to 1 m behind the goal, and across the shots and the penalty box, ``pad`` either side.
    It then widens to fill the axes' aspect: sideways about its centre (kept inside
    [-1, 69] m), or deeper (down to -1 m). Equal aspect, anchored at the top.

    Returns
    -------
    numpy.ndarray of bool
        Points inside the final view; the caller counts the rest in its note.

    Raises
    ------
    ValueError
        ``x`` and ``y`` differ in length.

    Examples
    --------
    >>> ax = draw_half_pitch()
    >>> crop_pitch(ax, np.array([95.0, 40.0]), np.array([34.0, 30.0])).tolist()
    [True, False]
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y differ in shape: {x.shape} and {y.shape}")
    box_x, box_y0, box_y1 = pitch.METRE_X[5], pitch.METRE_Y[1], pitch.METRE_Y[6]
    lo_x = float(x.min()) if x.size else box_x
    floor = -1.0 if x_min is None else max(x_min, -1.0)
    x0, x1 = max(floor, lo_x - pad), pitch.PITCH_LENGTH + GOAL_DEPTH + 1.0
    y0 = min([box_y0, *([float(y.min())] if y.size else [])]) - pad
    y1 = max([box_y1, *([float(y.max())] if y.size else [])]) + pad
    bbox = ax.get_position(original=True)  # the box allotted, not an earlier crop's drawn one
    fig = ax.figure.bbox
    ratio = (bbox.height * fig.height) / (bbox.width * fig.width)  # axes height / width
    lo_y, hi_y = -1.0, pitch.PITCH_WIDTH + 1.0
    if (x1 - x0) / (y1 - y0) > ratio:  # too deep for the box: widen sideways
        half = min((x1 - x0) / ratio, hi_y - lo_y) / 2
        mid = min(max((y0 + y1) / 2, lo_y + half), hi_y - half)
        y0, y1 = mid - half, mid + half
    else:  # too wide: go deeper
        x0 = max(-1.0, x1 - (y1 - y0) * ratio)
    ax.set_xlim(y1, y0)
    ax.set_ylim(x0, x1)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    inside: BoolArray = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return inside


def ink_scale(xg_sum: float, ax: Axes) -> float:
    """Marker area per unit xG (pt²) so the markers cover about ``INK_SHARE`` of the axes.

    ``clip(INK_SHARE * A / xg_sum, AREA_PER_XG_MIN, AREA_MAX)`` with ``A`` the axes' drawn
    area in pt² (after the crop has filled it); ``AREA_MAX`` when ``xg_sum`` is 0.

    Raises
    ------
    ValueError
        ``xg_sum`` negative or not finite.

    Examples
    --------
    >>> ink_scale(0.0, draw_half_pitch())
    900.0
    """
    if not (math.isfinite(xg_sum) and xg_sum >= 0):
        raise ValueError(f"xg_sum must be finite and >= 0, not {xg_sum!r}")
    if xg_sum == 0:
        return AREA_MAX
    ax.apply_aspect()
    pos = ax.get_position()
    fig, pt = ax.figure.bbox, 72 / ax.figure.dpi
    area = pos.width * fig.width * pt * pos.height * fig.height * pt
    return float(np.clip(INK_SHARE * area / xg_sum, AREA_PER_XG_MIN, AREA_MAX))


def draw_shots(
    ax: Axes,
    x: FloatArray,
    y: FloatArray,
    xg: FloatArray,
    goal: BoolArray,
    theme: str | Theme,
    *,
    area_per_xg: float,
) -> None:
    """Shots on a vertical pitch: misses as rings, goals filled above them.

    Misses are ``mark`` rings at ``mark_alpha`` with area ``area_per_xg * xg`` exactly; goals
    are ``accent`` discs with a ``background`` halo, at least ``GOAL_MIN_S`` (a goal at 0.01 xG
    stays visible; the legend says so). Within each group the larger xG is drawn first, so
    small markers sit on top. Markers are not clipped at the axes edge (:func:`check` catches a
    collision instead).

    Examples
    --------
    >>> ax = draw_half_pitch()
    >>> draw_shots(ax, np.array([100.0, 90.0]), np.array([34.0, 30.0]), np.array([0.01, 0.2]),
    ...            np.array([True, False]), "light", area_per_xg=300.0)
    >>> [(c.get_gid(), c.get_sizes().tolist()) for c in ax.collections]
    [('mark-shots', [60.0]), ('mark-goals', [36.0])]
    """
    t = _theme(theme)
    x, y, p = (np.asarray(v, dtype=np.float64) for v in (x, y, xg))
    goal = np.asarray(goal, dtype=np.bool_)
    g = np.flatnonzero(goal)[np.argsort(-p[goal], kind="stable")]
    n = np.flatnonzero(~goal)[np.argsort(-p[~goal], kind="stable")]
    # fmt: off
    ax.scatter(y[n], x[n], s=area_per_xg * p[n], facecolors="none", edgecolors=t.mark,
               alpha=t.mark_alpha, linewidths=MARK_LW, zorder=3, gid=MARK_SHOTS, clip_on=False)
    ax.scatter(y[g], x[g], s=np.maximum(area_per_xg * p[g], GOAL_MIN_S), c=t.accent,
               edgecolors=t.background, linewidths=MARK_LW, zorder=4, gid=MARK_GOALS,
               clip_on=False)
    # fmt: on


def shot_keys(area_per_xg: float, theme: str | Theme) -> tuple[Key, ...]:
    """The shot-map legend: goal (at its floor size), no goal, and one ring per ``LEGEND_XG``.

    ``theme`` is validated only: the colours come from the canvas when the keys are drawn.

    Examples
    --------
    >>> [(k.label, k.size) for k in shot_keys(300.0, "light")][2:]
    [('0.1 xG', 30.0), ('0.3 xG', 90.0), ('0.6 xG', 180.0)]
    """
    _theme(theme)
    keys = [Key("goal", "Goal (small xG drawn at this size)", GOAL_MIN_S),
            Key("shot", "No goal", GOAL_MIN_S)]  # fmt: skip
    keys += [Key("shot", f"{v:g} xG", area_per_xg * v) for v in LEGEND_XG]
    return tuple(keys)


# --- Machine checks ---------------------------------------------------------------------------


def _luminance(rgb: Sequence[float]) -> float:
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_ratio(a: str, b: str | tuple[float, float, float]) -> float:
    """WCAG 2.x contrast ratio of two colours (``b`` may be an RGB tuple in [0, 1]).

    Examples
    --------
    >>> round(contrast_ratio("#000000", "#ffffff"), 1), contrast_ratio("#334155", "#334155")
    (21.0, 1.0)
    """
    la, lb = _luminance(to_rgba(a)), _luminance(to_rgba(b))
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in display px


def _box(b: Bbox) -> Box:
    return (float(b.x0), float(b.y0), float(b.x1), float(b.y1))


def _intersects(a: Box, b: Box) -> bool:
    """Positive-area overlap; boxes laid edge to edge touch up to float noise (``EPS_PX``)."""
    return min(a[2], b[2]) - max(a[0], b[0]) > EPS_PX and min(a[3], b[3]) - max(a[1], b[1]) > EPS_PX


def _mark_boxes(artist: Artist, renderer: Any) -> list[Box]:
    if isinstance(artist, PathCollection):
        offsets = np.asarray(artist.get_offsets(), dtype=np.float64).reshape(-1, 2)
        if not offsets.size:
            return []
        pts = artist.get_offset_transform().transform(offsets)
        sizes = np.broadcast_to(np.asarray(artist.get_sizes(), dtype=np.float64), len(pts))
        r = np.sqrt(sizes) / 2 * DPI / 72
        return [(px - ri, py - ri, px + ri, py + ri) for (px, py), ri in zip(pts, r, strict=True)]
    b = _box(artist.get_window_extent(renderer))
    if isinstance(artist, Line2D):  # a tick is a zero-width line: its stroke is the mark
        half = artist.get_linewidth() * DPI / 72 / 2
        b = (b[0] - half, b[1] - half, b[2] + half, b[3] + half)
    return [b]


def _mark_colours(artist: Artist) -> list[tuple[float, ...]]:
    if isinstance(artist, Collection):
        faces, edges = artist.get_facecolor(), artist.get_edgecolor()
        filled = len(faces) and float(np.max(np.asarray(faces)[:, 3])) > 0
        return [tuple(c) for c in np.unique(np.asarray(faces if filled else edges), axis=0)]
    if isinstance(artist, Line2D):
        rgba = to_rgba(artist.get_color(), artist.get_alpha())
    else:
        rgba = to_rgba(artist.get_facecolor(), artist.get_alpha())  # type: ignore[attr-defined]
    return [rgba]


def _under(surface: npt.NDArray[np.float64], b: Box) -> tuple[float, float, float]:
    h, w = surface.shape[:2]
    x0, x1 = max(0, math.floor(b[0])), min(w, math.ceil(b[2]))
    r0, r1 = max(0, math.floor(h - b[3])), min(h, math.ceil(h - b[1]))
    if x1 <= x0 or r1 <= r0:
        return (1.0, 1.0, 1.0)
    m = np.median(surface[r0:r1, x0:x1, :3].reshape(-1, 3), axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def _over(rgba: Sequence[float], under: tuple[float, float, float]) -> tuple[float, float, float]:
    a = float(rgba[3])
    r, g, b = (a * float(rgba[i]) + (1 - a) * under[i] for i in range(3))
    return (r, g, b)


def _large(t: Text) -> bool:
    size, (big, bold_size, bold_weight) = float(t.get_fontsize()), LARGE_PT
    weight = t.get_fontweight()
    w = font_manager.weight_dict.get(weight, 400) if isinstance(weight, str) else int(weight)
    return size >= big or (size >= bold_size and w >= bold_weight)


def check(c: Canvas) -> list[str]:
    """Machine checks of a themed chart; problems as sentences, ``[]`` when clean.

    The figure is drawn, then drawn once more with every ``Text`` and every ``mark-*``
    artist hidden: that *surface render* is what text and marks are judged against (an
    axis-off axes paints no patch, so its facecolour property says nothing). Items are every
    visible non-blank ``Text``, the logo, and every visible ``mark-*`` artist (one box per
    marker of a ``PathCollection``, a line's box padded by its stroke).

    1. text overlap: two text or logo boxes intersect with positive area (touching is fine);
    2. text on mark: a text box intersects a mark box;
    3. clipping: a text, logo or mark box leaves the canvas;
    4. panel containment: a text in a ``panel-`` axes leaves that axes;
    5. text contrast: text colour (alpha composited) against the median surface under it,
       ``TEXT_CONTRAST`` (4.5, or 3.0 for large text: >= 18 pt, or >= 14 pt and weight >= 600);
    6. mark contrast: each mark colour (face if filled, else edge) against the median surface
       under the artist, >= ``MARK_CONTRAST``;
    7. key size: every ``mark-key`` marker is >= ``MIN_KEY_PX`` across.

    Examples
    --------
    >>> c = canvas("dark", "square", title="T", subtitle="S", footer=Footer(("whoscored",)))
    >>> c.text(5, 5, "clipped", "caption")
    Text(5, 5, 'clipped')
    >>> check(c)
    ["clipped: 'clipped' leaves the canvas"]
    """
    fig = c.fig
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    w, h = fig.canvas.get_width_height()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    marks = [
        a for a in fig.findobj(lambda a: str(a.get_gid() or "").startswith("mark-"))
        if a.get_visible()
    ]  # fmt: skip
    logos = [a for a in fig.findobj(lambda a: a.get_gid() == LOGO_GID) if a.get_visible()]
    boxed = [(repr(t.get_text()), _box(t.get_window_extent(renderer)), t) for t in texts]
    named: list[tuple[str, Box]] = [(a, b) for a, b, _ in boxed]
    named += [("the logo", _box(a.get_window_extent(renderer))) for a in logos]
    mark_boxes = [(m, _mark_boxes(m, renderer)) for m in marks]
    problems: list[str] = []
    for i, (a, ba) in enumerate(named):
        problems += [f"text overlap: {a} and {b}" for b, bb in named[i + 1:]
                     if _intersects(ba, bb)]  # fmt: skip
    for a, ba, t in boxed:
        hit = sorted({str(m.get_gid()) for m, boxes in mark_boxes
                      if any(_intersects(ba, mb) for mb in boxes)})  # fmt: skip
        problems += [f"text on mark: {a} on {g}" for g in hit]
        if t.axes is not None and str(t.axes.get_gid() or "").startswith(PANEL_PREFIX):
            ax_box = _box(t.axes.bbox)
            if not (ba[0] >= ax_box[0] and ba[1] >= ax_box[1] and ba[2] <= ax_box[2]
                    and ba[3] <= ax_box[3]):  # fmt: skip
                problems.append(f"panel containment: {a} leaves {t.axes.get_gid()}")
    outside = [a for a, b in named if b[0] < 0 or b[1] < 0 or b[2] > w or b[3] > h]
    outside += sorted(
        {
            str(m.get_gid())
            for m, boxes in mark_boxes
            if any(b[0] < 0 or b[1] < 0 or b[2] > w or b[3] > h for b in boxes)
        }
    )
    problems += [f"clipped: {a} leaves the canvas" for a in outside]
    hidden: list[Artist] = [*texts, *marks]
    for artist in hidden:
        artist.set_visible(False)
    try:
        fig.canvas.draw()
        surface = np.asarray(fig.canvas.buffer_rgba(), dtype=np.float64) / 255.0  # type: ignore[attr-defined]
    finally:
        for artist in hidden:
            artist.set_visible(True)
    for a, b, t in boxed:
        under = _under(surface, b)
        ratio = contrast_ratio(to_hex(_over(to_rgba(t.get_color(), t.get_alpha()), under)), under)
        need = TEXT_CONTRAST[1] if _large(t) else TEXT_CONTRAST[0]
        if ratio < need:
            problems.append(f"text contrast: {a} at {ratio:.2f}, needs {need}")
    for m, boxes in mark_boxes:
        if not boxes:
            continue
        union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                 max(b[2] for b in boxes), max(b[3] for b in boxes))  # fmt: skip
        under = _under(surface, union)
        for rgba in _mark_colours(m):
            ratio = contrast_ratio(to_hex(_over(rgba, under)), under)
            if ratio < MARK_CONTRAST:
                problems.append(f"mark contrast: {m.get_gid()} at {ratio:.2f}, needs 3.0")
        if m.get_gid() == MARK_KEY:
            small = [b[2] - b[0] for b in boxes if b[2] - b[0] < MIN_KEY_PX]
            problems += [f"key size: a key marker is {d:.1f} px, needs {MIN_KEY_PX}" for d in small]
    return problems
