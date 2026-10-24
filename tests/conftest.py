"""Shared fixtures for the schema tests (C3). Builders live in ``schema_cases``.

Imports are inside the fixtures so the pitch tests still collect while the C3 dependencies
(polars, pandera) are not installed yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import polars as pl


@pytest.fixture
def valid_shots() -> pl.DataFrame:
    """Five synthetic shots that follow spec §2.0 / §7 (see ``schema_cases.base_shot_rows``)."""
    from schema_cases import shots_frame

    return shots_frame()


@pytest.fixture
def valid_freeze_frames() -> pl.DataFrame:
    """Four players visible at shot 1: keeper, two opponents, a teammate with no id."""
    from schema_cases import freeze_frames_frame

    return freeze_frames_frame()
