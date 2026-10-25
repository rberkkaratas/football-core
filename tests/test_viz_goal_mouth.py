"""``viz.goal_mouth`` and ``viz.draw_goal_mouth``: the front-on 7.32 x 2.44 m frame, the
shooter's view, shots placed by goal_y/goal_z, hidden rows and the inclusive view (C7 spec §2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import viz_cases as vc
from matplotlib.colors import to_rgba

from football_core import pitch, viz

Y0 = pitch.RIGHT_POST_Y - viz.VIEW_PAD_Y  # the spec's view bounds, as written in §2
Y1 = pitch.LEFT_POST_Y + viz.VIEW_PAD_Y
Z1 = pitch.GOAL_HEIGHT + viz.VIEW_PAD_Z


def test_goal_frame_is_7_32_by_2_44() -> None:
    ax = viz.draw_goal_mouth()
    drawn = vc.lines(ax)
    frame = drawn[viz.FRAME_GID]
    ys, zs = np.asarray(frame.get_xdata(), float), np.asarray(frame.get_ydata(), float)
    np.testing.assert_allclose(
        ys, [pitch.RIGHT_POST_Y, pitch.RIGHT_POST_Y, pitch.LEFT_POST_Y, pitch.LEFT_POST_Y]
    )
    np.testing.assert_allclose(zs, [0.0, pitch.GOAL_HEIGHT, pitch.GOAL_HEIGHT, 0.0])
    assert ys.max() - ys.min() == pytest.approx(pitch.GOAL_WIDTH)  # 7.32
    assert zs.max() - zs.min() == pytest.approx(pitch.GOAL_HEIGHT)  # 2.44
    assert to_rgba(frame.get_color()) == pytest.approx(to_rgba(viz.INK))
    assert frame.get_linewidth() == viz.FRAME_LW
    assert frame.get_zorder() == 2
    ground = drawn["ground"]
    np.testing.assert_allclose(np.asarray(ground.get_xdata(), float), [Y0, Y1])
    np.testing.assert_allclose(np.asarray(ground.get_ydata(), float), [0.0, 0.0])
    assert ax.get_xlim() == pytest.approx((Y1, Y0))
    assert ax.get_ylim() == pytest.approx((-0.3, Z1))
    assert ax.get_aspect() == 1.0
    assert not ax.axison
    # front-on and to scale: on screen the mouth is 7.32 / 2.44 = 3 times wider than high
    ax.figure.canvas.draw()
    (ra, ga), (lb, cb) = ax.transData.transform(
        [(pitch.RIGHT_POST_Y, 0.0), (pitch.LEFT_POST_Y, pitch.GOAL_HEIGHT)]
    )
    assert abs(ra - lb) / abs(cb - ga) == pytest.approx(3.0, rel=1e-6)


def test_shooters_view() -> None:
    ax = viz.draw_goal_mouth()
    ax.figure.canvas.draw()
    (right_px, _), (left_px, _) = ax.transData.transform(
        [(pitch.RIGHT_POST_Y, 0.0), (pitch.LEFT_POST_Y, 0.0)]
    )
    assert right_px > left_px  # the right post (y = 30.34) is on the right
    (_, bar_px), (_, ground_px) = ax.transData.transform([(34.0, pitch.GOAL_HEIGHT), (34.0, 0.0)])
    assert bar_px > ground_px


def test_shots_placed_by_goal_y_goal_z(synthetic_logo: Path) -> None:
    shots = vc.edge_frame()
    ax = vc.render("goal_mouth", shots, logo=synthetic_logo)
    goal = shots["is_goal"].to_numpy()
    drawn = vc.EDGE_GOAL_MOUTH_DRAWN
    goals = [i for i in drawn if goal[i]]
    others = [i for i in drawn if not goal[i]]
    assert vc.same_points(
        vc.offsets(ax, viz.GOALS_GID), vc.expected_points(shots, "goal_mouth", goals)
    )
    assert vc.same_points(
        vc.offsets(ax, viz.SHOTS_GID), vc.expected_points(shots, "goal_mouth", others)
    )
    xg = shots["xg"].to_numpy()
    np.testing.assert_allclose(
        np.sort(vc.sizes(ax, viz.SHOTS_GID)), np.sort(viz.AREA_PER_XG * xg[others])
    )
    # the save wide of the right post (29.5, 0.0) is drawn where it is, not clipped to the frame
    assert [29.5, 0.0] in vc.offsets(ax, viz.SHOTS_GID).tolist()
    assert [31.0, 0.5] in vc.offsets(ax, viz.GOALS_GID).tolist()
    assert vc.n_drawn(ax) + vc.note_count(vc.note(ax)) == shots.height  # drawn rows not counted


def test_hidden_shots_counted(synthetic_logo: Path) -> None:
    ax = vc.render("goal_mouth", vc.edge_frame(), logo=synthetic_logo)
    assert vc.note(ax) == vc.EDGE_NOTES["goal_mouth"]
    assert vc.n_drawn(ax) == len(vc.EDGE_GOAL_MOUTH_DRAWN)


def test_view_bounds_inclusive() -> None:
    # on each view edge (drawn) and 1 cm beyond three of them (hidden)
    on_edge = [(Y0, 1.0), (Y1, 1.0), (34.0, 0.0), (34.0, Z1), (Y0, Z1)]
    beyond = [(Y0 - 0.01, 1.0), (Y1 + 0.01, 1.0), (34.0, Z1 + 0.01)]
    points = on_edge + beyond
    n = len(points)
    shots = vc.frame(
        x=[100.0] * n,
        y=[34.0] * n,
        goal_y=[p[0] for p in points],
        goal_z=[p[1] for p in points],
        xg=[0.1] * n,
    )
    ax = vc.render("goal_mouth", shots)
    assert vc.same_points(vc.offsets(ax, viz.SHOTS_GID), np.array(on_edge))
    assert vc.note(ax) == "not shown: 3 outside the view"
