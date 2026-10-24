"""Shared fixtures for the schema (C3) and snapshot (C6) tests.

Builders live in ``schema_cases`` and ``snapshot_cases``.

Imports are inside the fixtures so the pitch tests still collect while the C3 dependencies
(polars, pandera) are not installed yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

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


@pytest.fixture(autouse=True)
def _snapshot_isolation(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """C6 spec §5: a fresh builder registry and an empty environment cache for every test.

    Scoped to the ``test_snapshot_*`` files so the pitch and schema tests still run when
    ``football_core.io.snapshot`` cannot be imported. The import sits inside the body, like the
    C3 fixtures, so the pitch tests still collect without polars.
    """
    if not request.node.path.name.startswith("test_snapshot_"):
        yield
        return
    from football_core.io import snapshot

    monkeypatch.setattr(snapshot, "_REGISTRY", {})
    snapshot._environment_git.cache_clear()
    yield
    snapshot._environment_git.cache_clear()
