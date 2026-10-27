"""P2.1 goal-mouth harmonisation in ``pitch`` (spec ``runs/MS5/goalmouth`` §2, §10.1-3, §10.5).

The landmark tables are typed here from the spec (IFAB 12 cm posts and bar, the Opta band
[44.2, 45.2] / [54.8, 55.8] and bar zone (38, 42], StatsBomb yards), never read back from
``pitch``. Expected distances are hand arithmetic on goal-frame coordinates (metres from the goal
centre, y positive = shooter's right, z up).
"""

from __future__ import annotations

import inspect
import math
import warnings

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from football_core import pitch, schema

YARD = 0.9144
NAN, INF = float("nan"), float("inf")
SB_FACE = 0.12 * 6.0 / 5.5  # 12 cm on the StatsBomb six-yard-side-to-post segment, in yards
TOL = 1e-9
IN = 0.15 + schema.DISTANCE_TOLERANCE  # the census rule: a point at exactly 0.15 m counts


def _one(values: object) -> float:
    return float(np.asarray(values, dtype=np.float64).reshape(-1)[0])


# --- Landmarks (§2, §10.1) --------------------------------------------------------------------


def test_mouth_landmark_tables() -> None:
    assert pitch.POST_WIDTH == 0.12
    assert pitch.YARD == YARD
    metre_y, metre_z = pitch.METRE_MOUTH_Y, pitch.METRE_MOUTH_Z
    assert metre_y == pytest.approx(
        (0.0, 13.84, 24.84, 30.22, 30.34, 37.66, 37.78, 43.16, 54.16, 68.0), abs=TOL
    )
    assert metre_z == pytest.approx((0.0, 2.44, 2.56), abs=TOL)
    assert pitch.OPTA.mouth_y == (0.0, 21.1, 36.8, 44.2, 45.2, 54.8, 55.8, 63.2, 78.9, 100.0)
    assert pitch.OPTA.mouth_z == (0.0, 38.0, 42.0)
    sb_y = (80.0, 62.0, 50.0, 44.0 + SB_FACE, 44.0, 36.0, 36.0 - SB_FACE, 30.0, 18.0, 0.0)
    assert pitch.STATSBOMB.mouth_y == pytest.approx(sb_y, abs=TOL)
    assert pitch.STATSBOMB.mouth_z == pytest.approx((0.0, 2.44 / YARD, 2.56 / YARD), abs=TOL)
    for frame in (pitch.OPTA, pitch.STATSBOMB):  # §2: z_to_metres is replaced, not kept
        assert not hasattr(frame, "z_to_metres")
        assert len(frame.mouth_y) == len(pitch.METRE_MOUTH_Y)
        assert len(frame.mouth_z) == len(pitch.METRE_MOUTH_Z)


# provider, raw y, raw z, expected metres y, expected metres z
# fmt: off
LANDMARKS = [
    ("opta", 44.2, 0.0, 30.22, 0.0),                     # right post, outer face
    ("opta", 44.7, 40.0, 30.28, 2.50),                   # mid post band; mid bar zone
    ("opta", 45.2, 38.0, 30.34, 2.44),                   # right post inner face; bar underside
    ("opta", 54.8, 42.0, 37.66, 2.56),                   # left post inner face; bar top
    ("opta", 55.8, 100.0, 37.78, 2.56 + 58 * 2.44 / 38),  # above the bar top: below-bar scale
    ("opta", 50.0, 19.0, 34.0, 1.22),                    # unchanged inside the frame
    ("opta", 36.8, 19.0, 24.84, 1.22),                   # six-yard side, unchanged knot
    ("opta", 63.2, -1.0, 43.16, 0.0),                    # a negative height clips to 0 (§2)
    ("opta", 0.0, 5.0, 0.0, 5 * 2.44 / 38),
    ("opta", 100.0, 5.0, 68.0, 5 * 2.44 / 38),
    ("statsbomb", 44.0, 0.0, 30.34, 0.0),                # 44 yd is the RIGHT post (30.34 m)
    ("statsbomb", 36.0, 2.44 / YARD, 37.66, 2.44),
    ("statsbomb", 44.0 + SB_FACE, 2.56 / YARD, 30.22, 2.56),
    ("statsbomb", 36.0 - SB_FACE, 5.0, 37.78, 5.0 * YARD),  # heights never clipped above
    ("statsbomb", 40.0, -0.5, 34.0, 0.0),
]
# fmt: on


@pytest.mark.parametrize(
    ("provider", "raw_y", "raw_z", "y", "z"),
    LANDMARKS,
    ids=[f"{p}-{ry:g}-{rz:g}" for p, ry, rz, _, _ in LANDMARKS],
)
def test_goal_mouth_landmarks(
    provider: str, raw_y: float, raw_z: float, y: float, z: float
) -> None:
    gy, gz = pitch.goal_mouth_to_canonical([raw_y], [raw_z], provider)  # type: ignore[arg-type]
    assert (_one(gy), _one(gz)) == pytest.approx((y, z), abs=TOL)


@settings(max_examples=200, deadline=None)
@given(
    y=st.floats(0.0, 80.0, allow_nan=False),
    z=st.floats(0.0, 20.0, allow_nan=False),
)
def test_statsbomb_map_unchanged(y: float, z: float) -> None:
    """§2: StatsBomb's extra knots are collinear, so goal-mouth y is the pitch map's y and z is
    yards times 0.9144 (z >= 0)."""
    gy, gz = pitch.goal_mouth_to_canonical([y], [z], "statsbomb")
    _, py = pitch.to_canonical([60.0], [y], "statsbomb")
    assert _one(gy) == pytest.approx(_one(py), abs=TOL, rel=0)
    assert _one(gz) == pytest.approx(z * YARD, abs=TOL, rel=0)


@settings(max_examples=200, deadline=None)
@given(
    y=st.floats(45.2, 54.8, allow_nan=False),
    z=st.floats(0.0, 38.0, allow_nan=False),
)
def test_opta_inside_frame_unchanged(y: float, z: float) -> None:
    """§2: Opta values inside [45.2, 54.8] x [0, 38] map as before (pitch y, z * 2.44 / 38)."""
    gy, gz = pitch.goal_mouth_to_canonical([y], [z], "opta")
    _, py = pitch.to_canonical([50.0], [y], "opta")
    assert _one(gy) == pytest.approx(_one(py), abs=TOL, rel=0)
    assert _one(gz) == pytest.approx(z * 2.44 / 38.0, abs=TOL, rel=0)


@settings(max_examples=150, deadline=None)
@given(
    provider=st.sampled_from(["opta", "statsbomb"]),
    a=st.floats(-10.0, 110.0, allow_nan=False),
    b=st.floats(-10.0, 110.0, allow_nan=False),
)
def test_goal_mouth_monotone(provider: str, a: float, b: float) -> None:
    """Property: heights keep their order on both grids; Opta y keeps its order, StatsBomb y
    (axis pointing down) reverses it."""
    lo, hi = min(a, b), max(a, b)
    gy, gz = pitch.goal_mouth_to_canonical([lo, hi], [lo, hi], provider)  # type: ignore[arg-type]
    assert gz[0] <= gz[1] + TOL
    if provider == "opta":
        assert gy[0] <= gy[1] + TOL
    else:
        assert gy[0] >= gy[1] - TOL


@pytest.mark.parametrize("provider", ["opta", "statsbomb"])
def test_goal_mouth_nan(provider: str) -> None:
    """§2 totality: NaN in -> NaN out per element; +inf height stays inf (never clipped above)."""
    gy, gz = pitch.goal_mouth_to_canonical(  # type: ignore[arg-type]
        [NAN, 40.0, 40.0], [1.0, NAN, INF], provider
    )
    assert math.isnan(gy[0]) and not math.isnan(gz[0])
    assert not math.isnan(gy[1]) and math.isnan(gz[1])
    assert gz[2] == INF


def test_to_goal_frame_sides() -> None:
    """§2: metres from the goal centre, y positive = shooter's right (the right post, 30.34)."""
    gy, gz = pitch.to_goal_frame([30.34, 37.66, 34.0, 20.0, NAN], [0.0, 2.44, 1.0, 0.5, 1.0])
    np.testing.assert_allclose(gy[:4], [3.66, -3.66, 0.0, 14.0], atol=TOL, rtol=0)
    np.testing.assert_allclose(gz, [0.0, 2.44, 1.0, 0.5, 1.0], atol=TOL, rtol=0)
    assert math.isnan(gy[4])


# --- Distances (§2, §10.2-3) ------------------------------------------------------------------

# goal-frame y, z, expected distance to the opening, inside the 0.15 m census tolerance
# fmt: off
FRAME_CASES = [
    ("centre", 0.0, 1.22, 0.0, True),
    ("post_line", 3.66, 0.0, 0.0, True),
    ("y_3.81_exactly_in", 3.81, 1.0, 0.15, True),
    ("y_3.8101_out", 3.8101, 1.0, 0.1501, False),
    ("left_3.81_in", -3.81, 1.0, 0.15, True),
    ("corner_hypot", 3.76, 2.54, math.hypot(0.1, 0.1), True),
    ("above_bar_2.59_in", 0.0, 2.59, 0.15, True),
    ("above_bar_2.5901_out", 0.0, 2.5901, 0.1501, False),
    ("below_ground", 1.0, -0.1, 0.1, True),
    ("wide", -6.0, 0.5, 2.34, False),
]
# fmt: on


@pytest.mark.parametrize(
    ("gy", "gz", "dist", "inside"), [c[1:] for c in FRAME_CASES], ids=[c[0] for c in FRAME_CASES]
)
def test_frame_distance_cases(gy: float, gz: float, dist: float, inside: bool) -> None:
    d = _one(pitch.frame_distance([gy], [gz]))
    assert d == pytest.approx(dist, abs=1e-12, rel=0)
    assert (d <= IN) is inside


# fmt: off
WOODWORK_CASES = [
    ("post_face", 3.66, 1.0, 0.0),
    ("inside_post", 3.72, 1.0, 0.0),
    ("inside_left_post", -3.72, 1.0, 0.0),
    ("post_top_outer_corner", -3.78, 2.56, 0.0),
    ("inside_bar", 0.0, 2.5, 0.0),
    ("bar_top_plus_0.3", 0.0, 2.86, 0.3),
    ("outer_face_plus_0.3", 4.08, 1.0, 0.3),
    ("in_the_opening", 3.5, 1.0, 0.16),
    ("goal_centre_on_the_ground", 0.0, 0.0, 2.44),
    ("mid_opening", 0.0, 1.22, 1.22),
    ("beyond_the_corner", 3.88, 2.66, math.hypot(0.1, 0.1)),
    ("under_the_post", 3.72, -0.1, 0.1),
]
# fmt: on


@pytest.mark.parametrize(
    ("gy", "gz", "dist"), [c[1:] for c in WOODWORK_CASES], ids=[c[0] for c in WOODWORK_CASES]
)
def test_woodwork_distance_cases(gy: float, gz: float, dist: float) -> None:
    assert _one(pitch.woodwork_distance([gy], [gz])) == pytest.approx(dist, abs=1e-12, rel=0)


def test_distances_totality() -> None:
    """§2: NaN in -> NaN out; an infinite height gives an infinite distance; nothing raises."""
    for fn in (pitch.frame_distance, pitch.woodwork_distance):
        d = np.asarray(fn([NAN, 0.0, 0.0, 1.0], [1.0, NAN, INF, 1.0]), dtype=np.float64)
        assert math.isnan(d[0]) and math.isnan(d[1]), fn.__name__
        assert d[2] == INF, fn.__name__
        assert math.isfinite(d[3]), fn.__name__


@settings(max_examples=200, deadline=None)
@given(
    y=st.floats(-10.0, 10.0, allow_nan=False),
    z=st.floats(-1.0, 6.0, allow_nan=False),
)
def test_distances_symmetric_and_zero_inside(y: float, z: float) -> None:
    """Property: both distances are mirror-symmetric in y; the frame distance is 0 exactly on
    the opening and positive off it."""
    for fn in (pitch.frame_distance, pitch.woodwork_distance):
        assert _one(fn([y], [z])) == pytest.approx(_one(fn([-y], [z])), abs=1e-12)
    d = _one(pitch.frame_distance([y], [z]))
    opening = abs(y) <= 3.66 and 0.0 <= z <= 2.44
    assert (d == 0.0) is opening


# --- Projection (§2, §10.5) -------------------------------------------------------------------

# x, y, end_x, end_y (canonical metres) -> projected goal-line y
# fmt: off
PROJECTION_CASES = [
    ("line", 90.0, 34.0, 100.0, 36.0, 37.0),       # 34 + 2 * 15 / 10
    ("on_the_line", 90.0, 34.0, 105.0, 30.0, 30.0),  # end_x = 105: end_y unchanged
    ("backwards", 100.0, 34.0, 95.0, 40.0, 40.0),  # end_x < x: end_y unchanged
    ("same_x", 100.0, 34.0, 100.0, 40.0, 40.0),    # end_x = x: end_y unchanged
    ("clip_at_0", 100.0, 10.0, 102.0, 4.0, 0.0),    # 10 - 6 * 5 / 2 = -5
    ("clip_at_68", 100.0, 58.0, 102.0, 64.0, 68.0),  # 58 + 6 * 5 / 2 = 73
]
# fmt: on


@pytest.mark.parametrize(
    ("x", "y", "ex", "ey", "expected"),
    [c[1:] for c in PROJECTION_CASES],
    ids=[c[0] for c in PROJECTION_CASES],
)
def test_project_to_goal_line_cases(
    x: float, y: float, ex: float, ey: float, expected: float
) -> None:
    out = pitch.project_to_goal_line([x], [y], [ex], [ey])
    assert _one(out) == pytest.approx(expected, abs=TOL, rel=0)


def test_project_to_goal_line_nan_and_inf() -> None:
    """§2: NaN in any of the four inputs -> NaN; x = -inf -> NaN, with no numpy warning."""
    good = [90.0, 34.0, 100.0, 36.0]
    for i in range(4):
        args = [[v] for v in good]
        args[i] = [NAN]
        assert math.isnan(_one(pitch.project_to_goal_line(*args))), i
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = pitch.project_to_goal_line([-INF, 90.0], [34.0, 34.0], [100.0, 100.0], [30.0, 36.0])
    assert math.isnan(out[0])
    assert out[1] == pytest.approx(37.0, abs=TOL)  # vectorised: the other row is unaffected


@settings(max_examples=200, deadline=None)
@given(
    x=st.floats(50.0, 104.0, allow_nan=False),
    y=st.floats(0.0, 68.0, allow_nan=False),
    gap=st.floats(0.5, 20.0, allow_nan=False),
    ey=st.floats(0.0, 68.0, allow_nan=False),
)
def test_projection_on_the_line(x: float, y: float, gap: float, ey: float) -> None:
    """Property: the result is in [0, 68]; inside it, (105, result) is collinear with the shot
    location and the stop point; a stop point at or past the line is returned as recorded."""
    ex = x + gap
    out = _one(pitch.project_to_goal_line([x], [y], [ex], [ey]))
    assert 0.0 <= out <= 68.0
    if ex >= 105.0:
        assert out == pytest.approx(ey, abs=TOL)
    elif 1e-6 < out < 68.0 - 1e-6:
        assert (out - y) * (ex - x) == pytest.approx((ey - y) * (105.0 - x), abs=1e-6)


def test_new_functions_have_doctests() -> None:
    """§2 'Doctest each': every new public pitch function carries an Examples doctest."""
    for fn in (
        pitch.to_goal_frame,
        pitch.frame_distance,
        pitch.woodwork_distance,
        pitch.project_to_goal_line,
    ):
        assert ">>>" in (inspect.getdoc(fn) or ""), fn.__name__
