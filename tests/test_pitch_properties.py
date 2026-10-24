"""Properties that must hold for every point on the pitch, not just the landmarks."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from football_core.pitch import (
    FRAMES,
    flip,
    from_canonical,
    opening_angle,
    shot_distance,
    to_canonical,
)

providers = st.sampled_from(sorted(FRAMES))
unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
canon_x = st.floats(min_value=0.0, max_value=105.0, allow_nan=False)
canon_y = st.floats(min_value=0.0, max_value=68.0, allow_nan=False)


def _provider_point(provider: str, u: float, v: float) -> tuple[float, float]:
    frame = FRAMES[provider]
    return (
        min(frame.x) + u * (max(frame.x) - min(frame.x)),
        min(frame.y) + v * (max(frame.y) - min(frame.y)),
    )


@given(providers, unit, unit)
def test_round_trip(provider, u, v):
    px, py = _provider_point(provider, u, v)
    x, y = to_canonical(px, py, provider)
    bx, by = from_canonical(x, y, provider)
    assert np.isclose(bx, px, atol=1e-9) and np.isclose(by, py, atol=1e-9)


@given(providers, unit, unit)
def test_in_bounds_points_stay_on_the_pitch(provider, u, v):
    x, y = to_canonical(*_provider_point(provider, u, v), provider)
    assert 0.0 <= x <= 105.0 and 0.0 <= y <= 68.0


@given(providers, st.lists(unit, min_size=2, max_size=20))
def test_mapping_is_monotonic(provider, us):
    frame = FRAMES[provider]
    px = np.sort(np.array([_provider_point(provider, u, 0.5)[0] for u in us]))
    x, _ = to_canonical(px, np.full_like(px, 50.0), provider)
    assert np.all(np.diff(x) >= 0)
    py = np.sort(np.array([_provider_point(provider, 0.5, u)[1] for u in us]))
    _, y = to_canonical(np.full_like(py, 50.0), py, provider)
    y_increases = frame.y[0] < frame.y[-1]
    assert np.all(np.diff(y) >= 0) if y_increases else np.all(np.diff(y) <= 0)


@given(canon_x, canon_y)
def test_geometry_is_symmetric_about_the_centre_line(x, y):
    mirrored = 68.0 - y
    assert np.isclose(shot_distance(x, y), shot_distance(x, mirrored))
    assert np.isclose(opening_angle(x, y), opening_angle(x, mirrored))


@given(canon_x, canon_y)
def test_angle_is_within_zero_and_pi(x, y):
    theta = opening_angle(x, y)
    assert 0.0 <= theta <= np.pi


@given(st.floats(min_value=0.0, max_value=104.0), st.floats(min_value=0.01, max_value=1.0))
def test_angle_grows_towards_goal_on_the_centre_line(x, step):
    assert opening_angle(min(x + step, 105.0), 34.0) >= opening_angle(x, 34.0)


@given(canon_x, canon_y)
def test_flip_is_an_involution(x, y):
    fx, fy = flip(*flip(x, y))
    assert np.isclose(fx, x) and np.isclose(fy, y)
