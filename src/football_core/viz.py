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

import math
from pathlib import Path
from types import MappingProxyType
from typing import Final

import matplotlib.image as mpimg
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import ScaledTranslation

from football_core import pitch, schema
from football_core.io import statsbomb
from football_core.io.snapshot import StrPath

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Image = npt.NDArray[np.generic]

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


def _line(ax: Axes, gid: str, h: npt.ArrayLike, v: npt.ArrayLike) -> None:
    ax.plot(h, v, color=LINE_COLOUR, lw=PITCH_LW, zorder=1, gid=gid)


def _arc(
    cy: float, cx: float, r: float, a0: float, a1: float, n: int
) -> tuple[FloatArray, FloatArray]:
    """Points ``(y, x) = (cy + r sin a, cx + r cos a)`` for ``a`` from ``a0`` to ``a1``."""
    a = np.linspace(a0, a1, n)
    y: FloatArray = cy + r * np.sin(a)
    x: FloatArray = cx + r * np.cos(a)
    return y, x


def draw_half_pitch(ax: Axes | None = None) -> Axes:
    """Draw the attacking half, goal at the top, as the shooter sees it.

    Every line is built from the ``football_core.pitch`` landmarks, so the markings sit on
    the same IFAB points the provider maps use. Data coordinates are ``(y, x)`` in canonical
    metres; the horizontal axis is inverted so that ``y = 0`` (the attacker's right) is on the
    right.

    Parameters
    ----------
    ax : matplotlib Axes, optional
        Axes to draw on; by default a new 6.0 x 5.9 in figure on the Agg canvas.

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
    # fmt: off
    _line(ax, "pitch-outline", [0.0, 0.0, width, width, 0.0],
          [HALFWAY_X, length, length, HALFWAY_X, HALFWAY_X])
    _line(ax, "pitch-box", [box_y0, box_y0, box_y1, box_y1], [length, box_x, box_x, length])
    _line(ax, "pitch-six-yard-box", [six_y0, six_y0, six_y1, six_y1],
          [length, six_x, six_x, length])
    _line(ax, "pitch-goal", [post_r, post_r, post_l, post_l], [length, back, back, length])
    ax.plot([cy], [spot_x], ls="none", marker="o", ms=2.5, lw=PITCH_LW, color=LINE_COLOUR,
            zorder=1, gid="pitch-spot")
    _line(ax, "pitch-arc", *_arc(cy, spot_x, r, pi - d, pi + d, 61))
    _line(ax, "pitch-centre-circle", *_arc(cy, HALFWAY_X, r, -pi / 2, pi / 2, 91))
    _line(ax, "pitch-corner-right", *_arc(0.0, length, CORNER_RADIUS, pi / 2, pi, 16))
    _line(ax, "pitch-corner-left", *_arc(width, length, CORNER_RADIUS, -pi, -pi / 2, 16))
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
    path = statsbomb.logo_file() if logo is None else Path(logo)
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
        The StatsBomb logo PNG; default ``statsbomb.logo_file()``, relative to the working
        directory (fetch it once with ``statsbomb.fetch_logo()``).

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
        ``statsbomb.logo_file()``.
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
