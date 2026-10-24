"""Pitch coordinates and shot geometry on the canonical 105 x 68 m frame.

Provider grids are mapped to metres piecewise-linearly between IFAB pitch markings, so every
landmark (posts, penalty spot, box lines) lands exactly where it belongs. See
``docs/conventions.md`` for the frame definition and landmark tables.
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

# Landmarks in metres. Length: goal line, six-yard line, penalty spot, box line, halfway, ...
METRE_X = (0.0, 5.5, 11.0, 16.5, 52.5, 88.5, 94.0, 99.5, 105.0)
# Width, from the attacking team's right touchline: box side, six-yard side, posts, ...
METRE_Y = (0.0, 13.84, 24.84, 30.34, 37.66, 43.16, 54.16, 68.0)


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
    z_to_metres : float
        Factor converting the provider's goal-mouth height to metres.
    """

    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    z_to_metres: float


OPTA = ProviderFrame(
    name="opta",
    x=(0.0, 5.8, 11.5, 17.0, 50.0, 83.0, 88.5, 94.2, 100.0),
    y=(0.0, 21.1, 36.8, 45.2, 54.8, 63.2, 78.9, 100.0),
    z_to_metres=GOAL_HEIGHT / 38.0,  # crossbar at goalMouthZ = 38 (to verify in P2.1)
)
STATSBOMB = ProviderFrame(
    name="statsbomb",
    x=(0.0, 6.0, 12.0, 18.0, 60.0, 102.0, 108.0, 114.0, 120.0),
    y=(80.0, 62.0, 50.0, 44.0, 36.0, 30.0, 18.0, 0.0),
    z_to_metres=0.9144,  # end_location z is in yards
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

    The goal-mouth ``y`` shares the pitch width axis, so the posts land on 30.34 / 37.66 m.

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
        ``(goal_y, goal_z)`` in metres; the crossbar is at ``z = 2.44``.

    Examples
    --------
    >>> gy, gz = goal_mouth_to_canonical([45.2], [38.0], "opta")
    >>> round(float(gy[0]), 2), round(float(gz[0]), 2)
    (30.34, 2.44)
    """
    frame = _frame(provider)
    z = np.asarray(goal_z, dtype=np.float64) * frame.z_to_metres
    return _interp(goal_y, frame.y, METRE_Y), z


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
