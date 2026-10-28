"""Pitch coordinates and shot geometry on the canonical 105 x 68 m frame.

Provider grids are mapped to metres piecewise-linearly between IFAB pitch markings, so every
landmark (posts, penalty spot, box lines) lands exactly where it belongs. See
``docs/conventions.md`` for the frame definition and landmark tables.

**Goal mouth.** The goal-mouth point (``shots.goal_y`` / ``goal_z``) is stored on the pitch
width axis, so the posts sit at 30.34 / 37.66 m like every other y. The goal-mouth map adds two
landmarks per post and one above the bar: the outer faces of the woodwork, which is
``POST_WIDTH`` thick. Opta draws its posts and bar as bands (a post hit lies in goalMouthY
[44.2, 45.2]), and those bands map onto the 12 cm of woodwork instead of up to 0.76 m wide of
it. Checks and xGOT features read :func:`to_goal_frame`: metres from the goal centre, y
positive to the shooter's right, z up from the ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Provider = Literal["opta", "statsbomb"]

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44
GOAL_CENTRE_Y = PITCH_WIDTH / 2
RIGHT_POST_Y = GOAL_CENTRE_Y - GOAL_WIDTH / 2  # shooter's right
LEFT_POST_Y = GOAL_CENTRE_Y + GOAL_WIDTH / 2

POST_WIDTH = 0.12  # IFAB Law 1: posts and crossbar have the same width, at most 12 cm
YARD = 0.9144

# Landmarks in metres. Length: goal line, six-yard line, penalty spot, box line, halfway, ...
METRE_X = (0.0, 5.5, 11.0, 16.5, 52.5, 88.5, 94.0, 99.5, 105.0)
# Width, from the attacking team's right touchline: box side, six-yard side, posts, ...
METRE_Y = (0.0, 13.84, 24.84, 30.34, 37.66, 43.16, 54.16, 68.0)
# Goal mouth: METRE_Y plus each post's outer face; height: ground, bar underside, bar top.
METRE_MOUTH_Y = (
    *METRE_Y[:3],
    RIGHT_POST_Y - POST_WIDTH,
    *METRE_Y[3:5],
    LEFT_POST_Y + POST_WIDTH,
    *METRE_Y[5:],
)
METRE_MOUTH_Z = (0.0, GOAL_HEIGHT, GOAL_HEIGHT + POST_WIDTH)
_SB_FACE = POST_WIDTH * 6.0 / 5.5  # POST_WIDTH in StatsBomb yards by the 6 yd = 5.5 m segment


@dataclass(frozen=True)
class ProviderFrame:
    """A provider's pitch grid, described by the landmarks that pair with the metre tables.

    Attributes
    ----------
    name : str
        Provider identifier.
    x : tuple of float
        Length landmarks, paired element-wise with ``METRE_X``.
    y : tuple of float
        Width landmarks, paired element-wise with ``METRE_Y`` (right touchline first). A
        decreasing sequence means the provider's y axis points down.
    mouth_y : tuple of float
        Goal-mouth width landmarks, paired element-wise with ``METRE_MOUTH_Y``.
    mouth_z : tuple of float
        Goal-mouth heights of the ground, the bar's underside and its top, paired with
        ``METRE_MOUTH_Z``.
    """

    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    mouth_y: tuple[float, ...]
    mouth_z: tuple[float, ...]


OPTA = ProviderFrame(
    name="opta",
    x=(0.0, 5.8, 11.5, 17.0, 50.0, 83.0, 88.5, 94.2, 100.0),
    y=(0.0, 21.1, 36.8, 45.2, 54.8, 63.2, 78.9, 100.0),
    # the drawn posts [44.2, 45.2] and [54.8, 55.8] and the bar (38, 42] are the woodwork
    mouth_y=(0.0, 21.1, 36.8, 44.2, 45.2, 54.8, 55.8, 63.2, 78.9, 100.0),
    mouth_z=(0.0, 38.0, 42.0),
)
STATSBOMB = ProviderFrame(
    name="statsbomb",
    x=(0.0, 6.0, 12.0, 18.0, 60.0, 102.0, 108.0, 114.0, 120.0),
    y=(80.0, 62.0, 50.0, 44.0, 36.0, 30.0, 18.0, 0.0),
    # the faces are collinear with the pitch map, so StatsBomb's goal mouth maps as its pitch
    mouth_y=(80.0, 62.0, 50.0, 44.0 + _SB_FACE, 44.0, 36.0, 36.0 - _SB_FACE, 30.0, 18.0, 0.0),
    mouth_z=(0.0, GOAL_HEIGHT / YARD, (GOAL_HEIGHT + POST_WIDTH) / YARD),  # heights in yards
)
FRAMES: dict[str, ProviderFrame] = {"opta": OPTA, "statsbomb": STATSBOMB}


def _frame(provider: Provider) -> ProviderFrame:
    try:
        return FRAMES[provider]
    except KeyError:
        expected = ", ".join(sorted(FRAMES))
        raise ValueError(f"unknown provider {provider!r}; expected one of: {expected}") from None


def _interp(values: npt.ArrayLike, src: tuple[float, ...], dst: tuple[float, ...]) -> FloatArray:
    """Piecewise-linear map from ``src`` landmarks to ``dst``; clips outside the end landmarks."""
    xp, fp = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    if xp[0] > xp[-1]:  # np.interp needs increasing sample points
        xp, fp = xp[::-1], fp[::-1]
    return np.interp(np.asarray(values, dtype=np.float64), xp, fp)


def to_canonical(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    provider: Provider,
    *,
    cell_offset: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Convert provider pitch coordinates to the canonical 105 x 68 m frame.

    Parameters
    ----------
    x, y : array_like
        Coordinates in the provider's own grid, with the acting team attacking towards the
        provider's maximum ``x`` (true for both Opta and StatsBomb events).
    provider : {"opta", "statsbomb"}
        Source grid.
    cell_offset : float, default 0.0
        Shift subtracted before mapping, for cell-based grids (see the StatsBomb loader).

    Returns
    -------
    tuple of np.ndarray
        ``(x, y)`` in metres. Values outside the pitch are clipped to its edges.

    Examples
    --------
    >>> x, y = to_canonical([88.5, 100.0], [50.0, 45.2], "opta")
    >>> [float(v) for v in x], [round(float(v), 2) for v in y]
    ([94.0, 105.0], [34.0, 30.34])
    """
    frame = _frame(provider)
    xs = np.asarray(x, dtype=np.float64) - cell_offset
    ys = np.asarray(y, dtype=np.float64) - cell_offset
    return _interp(xs, frame.x, METRE_X), _interp(ys, frame.y, METRE_Y)


def from_canonical(
    x: npt.ArrayLike, y: npt.ArrayLike, provider: Provider
) -> tuple[FloatArray, FloatArray]:
    """Convert canonical metres back to a provider's grid (inverse of :func:`to_canonical`).

    Parameters
    ----------
    x, y : array_like
        Coordinates in metres on the 105 x 68 m pitch.
    provider : {"opta", "statsbomb"}
        Target grid.

    Returns
    -------
    tuple of np.ndarray
        ``(x, y)`` in the provider's grid.
    """
    frame = _frame(provider)
    return _interp(x, METRE_X, frame.x), _interp(y, METRE_Y, frame.y)


def goal_mouth_to_canonical(
    goal_y: npt.ArrayLike, goal_z: npt.ArrayLike, provider: Provider
) -> tuple[FloatArray, FloatArray]:
    """Convert where a shot crossed (or would cross) the goal line to metres.

    The goal-mouth ``y`` shares the pitch width axis, so the posts land on 30.34 / 37.66 m and
    their outer faces on 30.22 / 37.78 m. Heights map ground, bar underside (2.44 m) and bar
    top (2.56 m); above the bar top the below-bar scale continues, so a high miss keeps its
    height. A negative height clips to 0.

    Parameters
    ----------
    goal_y : array_like
        Provider goal-mouth width coordinate (Opta ``goalMouthY`` / StatsBomb ``end_location[1]``).
    goal_z : array_like
        Provider goal-mouth height (Opta ``goalMouthZ`` / StatsBomb ``end_location[2]``).
    provider : {"opta", "statsbomb"}
        Source grid.

    Returns
    -------
    tuple of np.ndarray
        ``(goal_y, goal_z)`` in metres; NaN stays NaN.

    Examples
    --------
    >>> gy, gz = goal_mouth_to_canonical([45.2, 44.2], [38.0, 42.0], "opta")
    >>> [round(float(v), 2) for v in gy], [round(float(v), 2) for v in gz]
    ([30.34, 30.22], [2.44, 2.56])
    """
    frame = _frame(provider)
    z = np.asarray(goal_z, dtype=np.float64)
    scale = METRE_MOUTH_Z[1] / frame.mouth_z[1]
    # np.maximum (not fmax) so a NaN height stays NaN
    gz = _interp(z, frame.mouth_z, METRE_MOUTH_Z) + np.maximum(z - frame.mouth_z[-1], 0.0) * scale
    return _interp(goal_y, frame.mouth_y, METRE_MOUTH_Y), gz


def goal_mouth_y_from_canonical(goal_y: npt.ArrayLike, provider: Provider) -> FloatArray:
    """Convert a stored goal-mouth y back to the provider's units (inverse of the y part of
    :func:`goal_mouth_to_canonical`).

    Values beyond the end landmarks clip to them, as every map here does; NaN stays NaN.

    Raises
    ------
    ValueError
        An unknown provider.

    Examples
    --------
    >>> [round(float(v), 6) for v in goal_mouth_y_from_canonical([30.34, 34.0], "statsbomb")]
    [44.0, 40.0]
    """
    return _interp(goal_y, METRE_MOUTH_Y, _frame(provider).mouth_y)


def to_goal_frame(goal_y: npt.ArrayLike, goal_z: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """The goal frame: a stored goal-mouth point as metres from the goal centre.

    Every goal-mouth check and every xGOT feature reads this frame, so the stored columns keep
    the pitch width axis (as charts use it) and nothing re-derives the sign.

    Parameters
    ----------
    goal_y, goal_z : array_like
        ``shots.goal_y`` / ``goal_z``: pitch-width metres and height in metres.

    Returns
    -------
    tuple of np.ndarray
        ``(y, z)``: ``y`` positive towards the shooter's right post (``GOAL_CENTRE_Y − goal_y``),
        ``z`` up from the ground. The posts are at ``y = ±3.66``.

    Examples
    --------
    >>> gy, gz = to_goal_frame([30.34, 37.66], [0.0, 2.44])
    >>> [round(float(v), 2) for v in gy], [float(v) for v in gz]
    ([3.66, -3.66], [0.0, 2.44])
    """
    return GOAL_CENTRE_Y - np.asarray(goal_y, dtype=np.float64), np.asarray(goal_z, np.float64)


def frame_distance(gy: npt.ArrayLike, gz: npt.ArrayLike) -> FloatArray:
    """Distance in metres from a goal-frame point to the goal opening (0 inside it).

    The opening is ``|y| ≤ 3.66, 0 ≤ z ≤ 2.44`` in :func:`to_goal_frame` coordinates.

    Examples
    --------
    >>> [round(float(d), 2) for d in frame_distance([0.0, 3.81, 4.66], [1.0, 1.0, 3.44])]
    [0.0, 0.15, 1.41]
    """
    y, z = np.asarray(gy, dtype=np.float64), np.asarray(gz, dtype=np.float64)
    dy = np.maximum(np.abs(y) - GOAL_WIDTH / 2, 0.0)
    dz = np.maximum(np.maximum(z - GOAL_HEIGHT, -z), 0.0)
    out: FloatArray = np.hypot(dy, dz)
    return out


def woodwork_distance(gy: npt.ArrayLike, gz: npt.ArrayLike) -> FloatArray:
    """Distance in metres from a goal-frame point to the woodwork (0 on or inside it).

    The posts are solids ``3.66 ≤ |y| ≤ 3.78, 0 ≤ z ≤ 2.56`` and the bar ``|y| ≤ 3.78,
    2.44 ≤ z ≤ 2.56``: ``POST_WIDTH`` thick, outside the opening's edges.

    Examples
    --------
    >>> [round(float(d), 2) for d in woodwork_distance([3.7, 0.0, 0.0], [1.0, 2.86, 1.22])]
    [0.0, 0.3, 1.22]
    """
    ay, z = np.abs(np.asarray(gy, dtype=np.float64)), np.asarray(gz, dtype=np.float64)
    half, top = GOAL_WIDTH / 2, GOAL_HEIGHT + POST_WIDTH
    post = np.hypot(
        np.maximum(np.maximum(half - ay, ay - half - POST_WIDTH), 0.0),
        np.maximum(np.maximum(-z, z - top), 0.0),
    )
    bar = np.hypot(
        np.maximum(ay - half - POST_WIDTH, 0.0),
        np.maximum(np.maximum(GOAL_HEIGHT - z, z - top), 0.0),
    )
    out: FloatArray = np.minimum(post, bar)
    return out


def project_to_goal_line(
    x: npt.ArrayLike, y: npt.ArrayLike, end_x: npt.ArrayLike, end_y: npt.ArrayLike
) -> FloatArray:
    """The y at which the straight line from a shot through its stop point meets the goal line.

    Canonical metres in and out. The piecewise pitch map bends straight lines drawn in provider
    units, so project after mapping. This does not decide *whether* a shot is projected; the
    loader does.

    Parameters
    ----------
    x, y : array_like
        Shot location.
    end_x, end_y : array_like
        Where the ball was stopped.

    Returns
    -------
    np.ndarray
        The projected y where ``x < end_x < 105``, else ``end_y``; clipped to [0, 68]. NaN in
        any input (or ``x = −inf``) gives NaN, with no warning.

    Examples
    --------
    >>> [float(v) for v in project_to_goal_line([95.0, 95.0], [34.0, 34.0],
    ...                                         [100.0, 105.0], [32.0, 30.0])]
    [30.0, 30.0]
    """
    xs, ys, ex, ey = (np.asarray(v, dtype=np.float64) for v in (x, y, end_x, end_y))
    ok = (ex > xs) & (ex < PITCH_LENGTH)
    with np.errstate(invalid="ignore", divide="ignore"):  # x = -inf: inf / inf is NaN
        k = np.where(ok, (PITCH_LENGTH - xs) / np.where(ok, ex - xs, 1.0), 1.0)
        out: FloatArray = np.clip(np.where(ok, ys + (ey - ys) * k, ey), 0.0, PITCH_WIDTH)
    missing = np.isnan(xs) | np.isnan(ys) | np.isnan(ex) | np.isnan(ey)
    return np.where(missing, np.nan, out)


def flip(x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Mirror canonical coordinates to the other team's attacking direction.

    Used for own goals, which providers record from the scoring side's perspective.
    """
    return PITCH_LENGTH - np.asarray(x, dtype=np.float64), PITCH_WIDTH - np.asarray(
        y, dtype=np.float64
    )


def shot_distance(x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
    """Distance in metres from a canonical location to the centre of the opponent goal.

    Examples
    --------
    >>> float(shot_distance(94.0, 34.0))
    11.0
    """
    dx = PITCH_LENGTH - np.asarray(x, dtype=np.float64)
    return np.hypot(dx, np.asarray(y, dtype=np.float64) - GOAL_CENTRE_Y)


def opening_angle(x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
    """Angle in radians subtended by the two goal posts at a canonical location.

    Computed as ``atan2(w·X, X² + C² − (w/2)²)`` with ``X`` the distance to the goal line,
    ``C`` the offset from the centre line and ``w`` the goal width; the result is in
    ``[0, π]`` and equals π on the goal line between the posts.

    Examples
    --------
    >>> round(float(opening_angle(94.0, 34.0)), 5)
    0.64241
    """
    dx = PITCH_LENGTH - np.asarray(x, dtype=np.float64)
    c = np.abs(np.asarray(y, dtype=np.float64) - GOAL_CENTRE_Y)
    half = GOAL_WIDTH / 2
    angle: FloatArray = np.arctan2(GOAL_WIDTH * dx, dx**2 + c**2 - half**2)
    return angle
