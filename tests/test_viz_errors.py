"""Input checks before anything is drawn (C7 spec §4): one table of bad and accepted inputs, and
the empty frame.

The xg column is called ``p_goal`` here so that "the message names the column" means something.
Bad values sit on row 5 (outside both views) or row 1 (no goal-mouth point) on purpose: the
checks cover every row that is neither an own goal nor a shoot-out kick, drawn or not.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import viz_cases as vc

from football_core import schema, viz

XG = "p_goal"
OWN_GOAL, SHOOTOUT, FAR, BLOCKED = 6, 7, 5, 1  # rows of the edge fixture


def base() -> pl.DataFrame:
    return vc.edge_frame("whoscored").rename({"xg": XG})


Mutation = Callable[[pl.DataFrame], Any]


def cell(col: str, row: int, value: Any) -> Mutation:
    return lambda df: vc.set_cell(df, col, row, value)


def drop(*cols: str) -> Mutation:
    return lambda df: df.drop(*cols)


def with_notes(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.Series("notes", [[1], None] * 4 + [[]], dtype=pl.List(pl.Int64())))


def cast(col: str, dtype: pl.DataType) -> Mutation:
    return lambda df: df.with_columns(pl.col(col).cast(dtype))


BOTH = ("shot_map", "goal_mouth")
NAN, INF = math.nan, math.inf

# (id, charts, mutation, extra kwargs, expected error or None, words the message must name)
CASES: list[tuple[str, tuple[str, ...], Mutation, dict[str, Any], type[Exception] | None,
                  tuple[str, ...]]] = [
    ("lazy-frame", BOTH, lambda df: df.lazy(), {}, TypeError, ()),
    ("dict", BOTH, lambda df: df.to_dict(as_series=False), {}, TypeError, ()),
    ("missing-provider", BOTH, drop("provider"), {}, ValueError, ("provider",)),
    ("missing-is_goal", BOTH, drop("is_goal"), {}, ValueError, ("is_goal",)),
    ("missing-own-goal-and-period", BOTH, drop("is_own_goal", "period"), {}, ValueError,
     ("is_own_goal", "period")),
    ("missing-xg", BOTH, drop(XG), {}, ValueError, (XG,)),
    ("missing-x", ("shot_map",), drop("x"), {}, ValueError, ("x",)),
    ("missing-y", ("shot_map",), drop("y"), {}, ValueError, ("y",)),
    ("missing-goal_y", ("goal_mouth",), drop("goal_y"), {}, ValueError, ("goal_y",)),
    ("missing-goal_z", ("goal_mouth",), drop("goal_z"), {}, ValueError, ("goal_z",)),
    ("provider-string", BOTH, cast("provider", pl.String()), {}, ValueError,
     ("provider", "String")),
    ("provider-other-enum", BOTH, cast("provider", pl.Enum(["statsbomb", "whoscored", "x"])), {},
     ValueError, ("provider",)),
    ("is_goal-int8", BOTH, cast("is_goal", pl.Int8()), {}, ValueError, ("is_goal", "Boolean")),
    ("is_own_goal-int8", BOTH, cast("is_own_goal", pl.Int8()), {}, ValueError, ("is_own_goal",)),
    ("period-int64", BOTH, cast("period", pl.Int64()), {}, ValueError,
     ("period", "Int64", "Int8")),
    ("x-float32", ("shot_map",), cast("x", pl.Float32()), {}, ValueError, ("x", "Float32")),
    ("goal_z-float32", ("goal_mouth",), cast("goal_z", pl.Float32()), {}, ValueError,
     ("goal_z",)),
    ("xg-float32", BOTH, cast(XG, pl.Float32()), {}, ValueError, (XG,)),
    ("xg-null", BOTH, cell(XG, FAR, None), {}, ValueError, (XG,)),
    ("xg-null-no-point", BOTH, cell(XG, BLOCKED, None), {}, ValueError, (XG,)),
    ("xg-nan", BOTH, cell(XG, FAR, NAN), {}, ValueError, (XG,)),
    ("xg-inf", BOTH, cell(XG, FAR, INF), {}, ValueError, (XG,)),
    ("xg-minus-inf", BOTH, cell(XG, FAR, -INF), {}, ValueError, (XG,)),
    ("xg-negative", BOTH, cell(XG, FAR, -0.1), {}, ValueError, (XG,)),
    ("xg-above-one", BOTH, cell(XG, FAR, 1.1), {}, ValueError, (XG,)),
    ("x-null", ("shot_map",), cell("x", 0, None), {}, ValueError, ("x",)),
    ("x-nan", ("shot_map",), cell("x", 0, NAN), {}, ValueError, ("x",)),
    ("y-inf", ("shot_map",), cell("y", 0, INF), {}, ValueError, ("y",)),
    ("goal_y-nan", ("goal_mouth",), cell("goal_y", 0, NAN), {}, ValueError, ("goal_y",)),
    ("goal_z-inf", ("goal_mouth",), cell("goal_z", 0, INF), {}, ValueError, ("goal_z",)),
    ("goal_y-only-null", ("goal_mouth",), cell("goal_y", 0, None), {}, ValueError, ("goal_y",)),
    ("goal_z-only-null", ("goal_mouth",), cell("goal_z", 0, None), {}, ValueError, ("goal_z",)),
    ("provider-null", BOTH, cell("provider", 0, None), {}, ValueError, ("provider",)),
    ("is_goal-null", BOTH, cell("is_goal", 1, None), {}, ValueError, ("is_goal",)),
    ("is_own_goal-null", BOTH, cell("is_own_goal", 1, None), {}, ValueError, ("is_own_goal",)),
    ("period-null", BOTH, cell("period", 1, None), {}, ValueError, ("period",)),
    ("area-zero", BOTH, lambda df: df, {"area_per_xg": 0.0}, ValueError, ("0.0",)),
    ("area-negative", BOTH, lambda df: df, {"area_per_xg": -1.0}, ValueError, ("-1.0",)),
    ("area-nan", BOTH, lambda df: df, {"area_per_xg": NAN}, ValueError, ("nan",)),
    ("area-inf", BOTH, lambda df: df, {"area_per_xg": INF}, ValueError, ("inf",)),
    # accepted: only the required columns are read, and xg only off own goals / shoot-outs
    ("ok-goal_y-string", ("shot_map",), cast("goal_y", pl.String()), {}, None, ()),
    ("ok-no-goal-mouth", ("shot_map",), drop("goal_y", "goal_z"), {}, None, ()),
    ("ok-no-x-y", ("goal_mouth",), drop("x", "y"), {}, None, ()),
    ("ok-own-goal-null-xg", BOTH, lambda df: df, {}, None, ()),
    ("ok-own-goal-nan-xg", BOTH, cell(XG, OWN_GOAL, NAN), {}, None, ()),
    ("ok-shootout-null-xg", BOTH, cell(XG, SHOOTOUT, None), {}, None, ()),
    ("ok-xg-bounds", BOTH, lambda df: vc.set_cell(vc.set_cell(df, XG, 0, 1.0), XG, 1, 0.0), {},
     None, ()),
    ("ok-extra-time", BOTH, cell("period", 0, 4), {}, None, ()),
    ("ok-extra-list-column", BOTH,
     with_notes, {}, None, ()),
    ("ok-own-goal-inf-xg", BOTH, cell(XG, OWN_GOAL, INF), {}, None, ()),
]  # fmt: skip

PARAMS = [
    pytest.param(chart, mutate, kwargs, error, words, id=f"{chart}-{case_id}")
    for case_id, charts, mutate, kwargs, error, words in CASES
    for chart in charts
]


def _named(word: str, message: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", message) is not None


@pytest.mark.parametrize(("chart", "mutate", "kwargs", "error", "words"), PARAMS)
def test_bad_input_raises(
    chart: str,
    mutate: Mutation,
    kwargs: dict[str, Any],
    error: type[Exception] | None,
    words: tuple[str, ...],
) -> None:
    shots = mutate(base())
    ax = vc.blank_axes()
    call = {"xg": XG, "ax": ax, **kwargs}
    if error is None:
        assert vc.render(chart, shots, **call) is ax
        return
    with pytest.raises(error) as info:
        vc.render(chart, shots, **call)
    assert type(info.value) is error  # not a subclass leaking from polars or numpy
    for word in words:
        assert _named(word, str(info.value)), (word, str(info.value))
    assert vc.is_blank(ax), "checks run before anything is drawn"


@pytest.mark.parametrize("chart", BOTH)
def test_empty_frame(chart: str, tmp_path: Path) -> None:
    # no provider value: no attribution, so even a missing logo path is never read
    shots = vc.edge_frame("statsbomb").head(0)
    assert shots.schema["provider"] == schema.PROVIDER
    ax = vc.render(chart, shots, logo=tmp_path / "missing.png")
    gids = set(vc.lines(ax))
    assert ("pitch-outline" if chart == "shot_map" else viz.FRAME_GID) in gids
    assert len(ax.get_legend().legend_handles) == 5
    assert vc.n_drawn(ax) == 0
    assert vc.note(ax) is None
    assert vc.texts(ax, viz.TEXT_GID) == []
    assert vc.logos(ax) == []
