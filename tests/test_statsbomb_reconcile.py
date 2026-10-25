"""``statsbomb.reconcile_goals``: goals per match against the matches-file scores, shoot-outs
excluded and own goals credited to the other side (spec ``runs/MS2/c4/spec.md`` §1, §5)."""

from __future__ import annotations

from typing import Any

import polars as pl

from football_core.io import statsbomb

COLUMNS = ["match_id", "home_score", "away_score", "home_goals", "away_goals"]


def test_reconcile_ok(sb_standard: Any) -> None:
    shots, matches = sb_standard.shots, sb_standard.matches
    # the fixture really exercises the three rules:
    assert shots.filter(pl.col("is_goal") & (pl.col("period") == 5)).height == 1  # shoot-out goal
    b = shots.filter(
        (pl.col("match_id") == "sb:900002") & pl.col("is_goal") & (pl.col("period") < 5)
    )
    assert b["is_own_goal"].to_list() == [True] and b["is_home"].to_list() == [True]  # B is 0-1
    assert shots.filter(pl.col("match_id") == "sb:900003").height == 0  # C: 0-0 without shots
    out = statsbomb.reconcile_goals(shots, matches)
    assert out.columns == COLUMNS
    assert out.height == 0


def test_reconcile_reports_mismatch(sb_standard: Any) -> None:
    mid = pl.col("match_id")
    matches = sb_standard.matches.with_columns(
        pl.when(mid == "sb:900001")
        .then(4)
        .otherwise(pl.col("home_score"))
        .cast(pl.Int16)
        .alias("home_score"),
        pl.when(mid == "sb:900003")
        .then(2)
        .otherwise(pl.col("away_score"))
        .cast(pl.Int16)
        .alias("away_score"),
    )
    out = statsbomb.reconcile_goals(sb_standard.shots, matches)
    assert out.columns == COLUMNS
    assert out["home_goals"].dtype == pl.Int64 and out["away_goals"].dtype == pl.Int64
    assert out.rows() == [("sb:900001", 4, 1, 3, 1), ("sb:900003", 0, 2, 0, 0)]
