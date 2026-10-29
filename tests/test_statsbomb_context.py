"""StatsBomb does not derive the schema 1.3 context columns (spec ``runs/PR1/xg-context`` §2.2:
``io/statsbomb.py`` writes null for both)."""

from __future__ import annotations

from typing import Any

import polars as pl

from football_core import schema


def test_statsbomb_context_is_null(sb_standard: Any) -> None:
    """Every StatsBomb shot has null ``prev_action`` and ``rebound`` (not derived), in the
    canonical dtypes."""
    shots = sb_standard.shots
    assert shots.height > 0  # precondition: the fixture has shots
    assert shots.schema["prev_action"] == schema.PREV_ACTION
    assert shots.schema["rebound"] == pl.Boolean()
    assert shots["prev_action"].null_count() == shots.height
    assert shots["rebound"].null_count() == shots.height
