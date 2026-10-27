"""Opta woodwork hits coded in the drawn post band and bar zone (spec ``runs/MS5/goalmouth`` §2,
§3.2, §10.3): through the loader, a ShotOnPost at goalMouthY 44.5 lands on the right post's 12 cm
and one at goalMouthZ 39.9 on the crossbar's, both 0 m from the woodwork."""

from __future__ import annotations

from pathlib import Path

import pytest
import whoscored_cases as wsc

from football_core import pitch

TOL = 1e-9


def _post_hit(tmp_path: Path, mouth: tuple[str, str]) -> tuple[float, float, float, float]:
    """Load MAIN with its ShotOnPost moved to ``mouth``: stored goal_y / goal_z and the
    goal-frame distances to the woodwork and to the opening."""
    fx = wsc.mini()
    post = fx.match("main").event(wsc.POST)
    assert post.type == "ShotOnPost"  # precondition: a woodwork hit
    post.mouth = mouth
    shots = wsc.load_fixture(tmp_path, fx).archive.shots
    (row,) = shots.filter(shots["shot_id"] == f"ws:{wsc.POST}").to_dicts()
    assert row["outcome"] == "post"
    gy, gz = pitch.to_goal_frame([row["goal_y"]], [row["goal_z"]])
    wood, frame = pitch.woodwork_distance(gy, gz), pitch.frame_distance(gy, gz)
    return row["goal_y"], row["goal_z"], float(wood[0]), float(frame[0])


def test_band_post_hit(tmp_path: Path) -> None:
    """goalMouthY 44.5 is in the drawn right post [44.2, 45.2]: 30.22 + 0.3 x 12 cm."""
    goal_y, goal_z, wood, frame = _post_hit(tmp_path, ("44.5", "10.0"))
    assert (goal_y, goal_z) == pytest.approx((30.22 + 0.3 * 0.12, 10.0 * 2.44 / 38), abs=TOL)
    assert wood == pytest.approx(0.0, abs=TOL)
    assert frame > 0.0  # outside the opening: on the post, not in the goal


def test_band_bar_hit(tmp_path: Path) -> None:
    """goalMouthZ 39.9 is in the drawn bar zone (38, 42]: 2.44 + 1.9 / 4 x 12 cm."""
    goal_y, goal_z, wood, frame = _post_hit(tmp_path, ("50.0", "39.9"))
    assert (goal_y, goal_z) == pytest.approx((34.0, 2.44 + 1.9 / 4.0 * 0.12), abs=TOL)
    assert wood == pytest.approx(0.0, abs=TOL)
    assert frame > 0.0
