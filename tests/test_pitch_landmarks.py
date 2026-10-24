"""Known pitch markings must land exactly on their metre values."""

import math

import numpy as np
import pytest

from football_core.pitch import (
    flip,
    from_canonical,
    goal_mouth_to_canonical,
    opening_angle,
    shot_distance,
    to_canonical,
)

# (provider, provider_xy, canonical_xy)
LANDMARKS = [
    ("opta", (100.0, 50.0), (105.0, 34.0)),  # goal centre
    ("opta", (88.5, 50.0), (94.0, 34.0)),  # penalty spot
    ("opta", (100.0, 45.2), (105.0, 30.34)),  # right post
    ("opta", (100.0, 54.8), (105.0, 37.66)),  # left post
    ("opta", (83.0, 21.1), (88.5, 13.84)),  # penalty-area corner, right side
    ("opta", (94.2, 36.8), (99.5, 24.84)),  # six-yard corner
    ("opta", (50.0, 50.0), (52.5, 34.0)),  # centre spot
    ("opta", (0.0, 0.0), (0.0, 0.0)),
    ("opta", (100.0, 100.0), (105.0, 68.0)),
    ("statsbomb", (120.0, 40.0), (105.0, 34.0)),
    ("statsbomb", (108.0, 40.0), (94.0, 34.0)),
    ("statsbomb", (120.0, 36.0), (105.0, 37.66)),  # y points down: 36 is the left post
    ("statsbomb", (120.0, 44.0), (105.0, 30.34)),
    ("statsbomb", (102.0, 62.0), (88.5, 13.84)),
    ("statsbomb", (0.0, 80.0), (0.0, 0.0)),
    ("statsbomb", (0.0, 0.0), (0.0, 68.0)),
]


@pytest.mark.parametrize(("provider", "src", "expected"), LANDMARKS)
def test_landmarks_map_to_metres(provider, src, expected):
    x, y = to_canonical(*src, provider)
    assert float(x) == pytest.approx(expected[0], abs=1e-9)
    assert float(y) == pytest.approx(expected[1], abs=1e-9)


@pytest.mark.parametrize(("provider", "src", "expected"), LANDMARKS)
def test_landmarks_map_back(provider, src, expected):
    x, y = from_canonical(*expected, provider)
    assert (float(x), float(y)) == pytest.approx(src, abs=1e-9)


def test_linear_scaling_would_be_wrong():
    # The reason this module exists: linear Opta scaling puts the spot 12.08 m out.
    x, _ = to_canonical(88.5, 50.0, "opta")
    assert float(shot_distance(x, 34.0)) == pytest.approx(11.0)
    linear_distance = (100 - 88.5) * 1.05
    assert linear_distance == pytest.approx(12.075)


def test_distance_and_angle_reference_values():
    assert float(shot_distance(94.0, 34.0)) == pytest.approx(11.0)
    assert float(opening_angle(94.0, 34.0)) == pytest.approx(2 * math.atan(3.66 / 11.0))
    assert float(opening_angle(88.5, 34.0)) == pytest.approx(2 * math.atan(3.66 / 16.5))


def test_angle_on_goal_line():
    assert float(opening_angle(105.0, 34.0)) == pytest.approx(math.pi)  # between the posts
    assert float(opening_angle(105.0, 20.0)) == pytest.approx(0.0)  # wide of the posts


def test_goal_mouth_frame_corners():
    gy, gz = goal_mouth_to_canonical([45.2, 54.8, 50.0], [0.0, 38.0, 19.0], "opta")
    np.testing.assert_allclose(gy, [30.34, 37.66, 34.0])
    np.testing.assert_allclose(gz, [0.0, 2.44, 1.22])
    gy, gz = goal_mouth_to_canonical([44.0, 36.0], [0.0, 2.67], "statsbomb")
    np.testing.assert_allclose(gy, [30.34, 37.66])
    np.testing.assert_allclose(gz, [0.0, 2.67 * 0.9144])


def test_cell_offset_shifts_before_mapping():
    x, y = to_canonical(120.5, 40.5, "statsbomb", cell_offset=0.5)
    assert (float(x), float(y)) == pytest.approx((105.0, 34.0))


def test_out_of_bounds_values_are_clipped():
    x, y = to_canonical([-3.0, 104.0], [-1.0, 101.0], "opta")
    np.testing.assert_allclose(x, [0.0, 105.0])
    np.testing.assert_allclose(y, [0.0, 68.0])


def test_flip_mirrors_through_centre_spot():
    x, y = flip(94.0, 30.34)
    assert (float(x), float(y)) == pytest.approx((11.0, 37.66))


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        to_canonical(1.0, 1.0, "wyscout")  # type: ignore[arg-type]
