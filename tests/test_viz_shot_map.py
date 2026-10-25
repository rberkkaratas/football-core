"""``viz.shot_map`` and ``viz.draw_half_pitch``: marker area proportional to xG, goals
highlighted, the shooter's orientation, hidden rows and the drawing contract (C7 spec §2, §7)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import viz_cases as vc
from matplotlib.colors import to_rgba

from football_core import pitch, viz

CHARTS = ("shot_map", "goal_mouth")


@pytest.mark.parametrize("chart", CHARTS)
@pytest.mark.parametrize("area", [viz.AREA_PER_XG, 150.0])
def test_marker_area_proportional_to_xg(chart: str, area: float, synthetic_logo: Path) -> None:
    # xG 0 and 2e-4 included: area stays exactly proportional, no floor
    shots = vc.frame(
        provider="statsbomb",
        x=[94.0, 99.0, 88.0, 80.0, 102.5, 95.0, 96.0],
        y=[34.0, 30.0, 45.0, 20.0, 36.0, 33.0, 35.0],
        is_goal=[True, False, False, False, True, False, True],
        goal_y=[31.0, 33.0, 36.0, 35.0, 34.0, 32.0, 36.5],
        goal_z=[0.5, 1.0, 2.0, 0.3, 2.3, 0.1, 1.9],
        xg=[0.76, 0.12, 0.0, 0.0002, 1.0, 0.33, 0.07],
    )
    kwargs = {} if area == viz.AREA_PER_XG else {"area_per_xg": area}
    ax = vc.render(chart, shots, logo=synthetic_logo, **kwargs)
    xg = shots["xg"].to_numpy()
    goal = shots["is_goal"].to_numpy()
    for gid, rows in ((viz.GOALS_GID, goal), (viz.SHOTS_GID, ~goal)):
        expected = np.sort(area * xg[rows])[::-1]
        got = np.sort(vc.sizes(ax, gid))[::-1]
        assert len(vc.offsets(ax, gid)) == int(rows.sum()), gid
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0, err_msg=gid)


def test_goals_highlighted(synthetic_logo: Path) -> None:
    ax = vc.render("shot_map", vc.edge_frame(), logo=synthetic_logo)
    goals, shots = vc.collection(ax, viz.GOALS_GID), vc.collection(ax, viz.SHOTS_GID)
    assert len(shots.get_facecolors()) == 0  # hollow, not a transparent fill
    faces = goals.get_facecolors()
    assert len(faces) >= 1
    for face in faces:
        assert tuple(face) == pytest.approx(to_rgba(viz.GOAL_COLOUR))
    for edge in shots.get_edgecolors():
        assert tuple(edge[:3]) == pytest.approx(to_rgba(viz.SHOT_COLOUR)[:3])
    assert goals.get_zorder() == 3
    assert shots.get_zorder() == 4  # hollow non-goals on top: no marker hides another
    # the split follows is_goal: two goals drawn (the spot and under the bar), four non-goals
    assert len(vc.offsets(ax, viz.GOALS_GID)) == 2
    assert len(vc.offsets(ax, viz.SHOTS_GID)) == 4


def test_orientation() -> None:
    shots = vc.frame(x=[100.0, 60.0], y=[20.0, 50.0], xg=[0.1, 0.02])
    ax = vc.render("shot_map", shots)
    assert vc.same_points(vc.offsets(ax, viz.SHOTS_GID), np.array([[20.0, 100.0], [50.0, 60.0]]))
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert xlim == pytest.approx((pitch.PITCH_WIDTH + 1, -1.0))  # inverted
    assert ylim == pytest.approx((viz.HALFWAY_X - 1, pitch.PITCH_LENGTH + viz.GOAL_DEPTH + 1))
    assert ylim[0] < pitch.PITCH_LENGTH < ylim[1]
    assert ax.get_aspect() == 1.0
    assert not ax.axison
    # as the shooter sees it: the attacker's right (y = 0) on the right, the goal at the top
    (right_x, _), (left_x, _) = ax.transData.transform([(0.0, 100.0), (68.0, 100.0)])
    assert right_x > left_x
    (_, goal_px), (_, halfway_px) = ax.transData.transform([(34.0, 105.0), (34.0, 52.5)])
    assert goal_px > halfway_px


def test_own_half_not_shown_and_counted() -> None:
    # x = HALFWAY_X is in view (inclusive); 52.4 and 30 are not
    shots = vc.frame(x=[95.0, viz.HALFWAY_X, 52.4], y=[34.0, 20.0, 34.0], xg=[0.1, 0.01, 0.01])
    ax = vc.render("shot_map", shots)
    assert vc.same_points(
        vc.offsets(ax, viz.SHOTS_GID), np.array([[34.0, 95.0], [20.0, viz.HALFWAY_X]])
    )
    assert vc.note(ax) == "not shown: 1 outside the view"
    assert vc.note(vc.render("shot_map", shots[:2])) is None  # nothing hidden: no note


@pytest.mark.parametrize("chart", CHARTS)
def test_own_goals_and_shootout_not_shown(chart: str) -> None:
    # An own goal at x = 50 (own goals can sit in the own half, x 47.8..105) with no xg and no
    # goal-mouth point: counted once, as an own goal, never as "outside the view" or "without a
    # goal-mouth point". A period-5 kick scored at the spot: hidden although in view.
    shots = vc.frame(
        x=[50.0, 94.0, 96.0],
        y=[33.0, 34.0, 30.0],
        is_goal=[True, True, False],
        is_own_goal=[True, False, False],
        period=[2, 5, 1],
        goal_y=[None, 35.0, 33.0],
        goal_z=[None, 0.5, 1.0],
        xg=[None, 0.78, 0.2],
    )
    ax = vc.render(chart, shots)
    assert vc.note(ax) == "not shown: 1 own goal(s), 1 shoot-out kick(s)"
    assert len(vc.offsets(ax, viz.GOALS_GID)) == 0
    assert vc.same_points(vc.offsets(ax, viz.SHOTS_GID), vc.expected_points(shots, chart, [2]))


@pytest.mark.parametrize("chart", CHARTS)
@pytest.mark.parametrize("is_goal", [True, False])
def test_draw_order_by_xg_descending(chart: str, is_goal: bool) -> None:
    # small markers on top of large ones; ties keep input order (stable)
    xg = [0.1, 0.5, 0.3, 0.5, 0.05]
    shots = vc.frame(
        x=[90.0, 91.0, 92.0, 93.0, 94.0],
        y=[30.0, 31.0, 32.0, 33.0, 34.0],
        is_goal=[is_goal] * 5,
        goal_y=[31.0, 32.0, 33.0, 34.0, 35.0],
        goal_z=[0.2, 0.4, 0.6, 0.8, 1.0],
        xg=xg,
    )
    ax = vc.render(chart, shots)
    gid = viz.GOALS_GID if is_goal else viz.SHOTS_GID
    order = [1, 3, 2, 0, 4]
    np.testing.assert_array_equal(vc.offsets(ax, gid), vc.expected_points(shots, chart, order))
    np.testing.assert_allclose(vc.sizes(ax, gid), [viz.AREA_PER_XG * xg[i] for i in order])


L, W, MX, MY = pitch.PITCH_LENGTH, pitch.PITCH_WIDTH, pitch.METRE_X, pitch.METRE_Y
POLYLINES = {
    "pitch-outline": ([0.0, 0.0, W, W, 0.0], [L / 2, L, L, L / 2, L / 2]),
    "pitch-box": ([MY[1], MY[1], MY[6], MY[6]], [L, MX[5], MX[5], L]),
    "pitch-six-yard-box": ([MY[2], MY[2], MY[5], MY[5]], [L, MX[7], MX[7], L]),
    "pitch-goal": (
        [pitch.RIGHT_POST_Y, pitch.RIGHT_POST_Y, pitch.LEFT_POST_Y, pitch.LEFT_POST_Y],
        [L, L + 2.0, L + 2.0, L],
    ),
    "pitch-spot": ([pitch.GOAL_CENTRE_Y], [MX[6]]),
}
# gid -> (centre (y, x), radius, number of points, the two end points)
ARCS = {
    "pitch-arc": (
        (34.0, MX[6]),
        9.15,
        61,
        {(34.0 - 9.15 * math.sin(math.acos(5.5 / 9.15)), MX[5]),
         (34.0 + 9.15 * math.sin(math.acos(5.5 / 9.15)), MX[5])},
    ),
    "pitch-centre-circle": ((34.0, L / 2), 9.15, 91, {(24.85, L / 2), (43.15, L / 2)}),
    "pitch-corner-right": ((0.0, L), 1.0, 16, {(1.0, L), (0.0, L - 1.0)}),
    "pitch-corner-left": ((W, L), 1.0, 16, {(W - 1.0, L), (W, L - 1.0)}),
}  # fmt: skip


@pytest.mark.parametrize("gid", [*POLYLINES, *ARCS])
def test_half_pitch_landmarks(gid: str) -> None:
    drawn = vc.lines(viz.draw_half_pitch())
    assert {g for g in drawn if g.startswith("pitch-")} == {*POLYLINES, *ARCS}
    line = drawn[gid]
    ys, xs = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
    assert to_rgba(line.get_color()) == pytest.approx(to_rgba(viz.LINE_COLOUR))
    assert line.get_linewidth() == viz.PITCH_LW
    assert line.get_zorder() == 1
    if gid in POLYLINES:
        exp_y, exp_x = POLYLINES[gid]
        np.testing.assert_allclose(ys, exp_y, atol=1e-12)
        np.testing.assert_allclose(xs, exp_x, atol=1e-12)
        if gid == "pitch-spot":
            assert line.get_linestyle() == "None"
            assert line.get_marker() == "o"
            assert line.get_markersize() == 2.5
        return
    (cy, cx), radius, n, ends = ARCS[gid]
    assert len(ys) == n
    np.testing.assert_allclose(np.hypot(ys - cy, xs - cx), radius, atol=1e-9)
    got_ends = {(round(ys[i], 6), round(xs[i], 6)) for i in (0, -1)}
    assert got_ends == {(round(a, 6), round(b, 6)) for a, b in ends}
    # every arc stays on the pitch side of its line: inside the attacking half, outside the box
    assert np.all(xs >= L / 2 - 1e-9)
    assert np.all(xs <= L + 1e-9)
    assert np.all((ys >= -1e-9) & (ys <= W + 1e-9))
    if gid == "pitch-arc":
        assert np.all(xs <= MX[5] + 1e-9)
