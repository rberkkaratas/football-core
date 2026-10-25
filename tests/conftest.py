"""Shared fixtures for the schema (C3), snapshot (C6), StatsBomb loader (C4), WhoScored loader
(C5) and viz (C7) tests.

Builders live in ``schema_cases``, ``snapshot_cases``, ``statsbomb_cases``, ``whoscored_cases``
and ``viz_cases``.

Imports are inside the fixtures so the pitch tests still collect while the C3 dependencies
(polars, pandera) are not installed yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

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


@pytest.fixture(scope="module")
def synthetic_logo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """C7 spec §7: a 10 x 60 red/blue RGBA PNG standing in for the StatsBomb logo."""
    from viz_cases import write_logo

    return write_logo(tmp_path_factory.mktemp("logo") / "logo.png", 10, 60)


@pytest.fixture(scope="session")
def sb_standard(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The standard StatsBomb fixture (matches A, B, C of ``statsbomb_cases``), loaded once.

    The result is a frozen ``OpenData`` of immutable polars frames, so sharing it is safe.
    """
    import statsbomb_cases as sbc

    return sbc.load_matches(tmp_path_factory.mktemp("sb_standard"), sbc.standard_matches())


@pytest.fixture(scope="session")
def ws_standard(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The standard WhoScored archive (``whoscored_cases.standard``), written and loaded once.

    Returns ``whoscored_cases.Loaded(root, archive)``; the archive is a frozen dataclass of
    polars frames, so sharing it is safe.
    """
    import whoscored_cases as wsc

    return wsc.load_fixture(tmp_path_factory.mktemp("ws_standard"), wsc.standard())


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
