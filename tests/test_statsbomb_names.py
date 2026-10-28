"""``statsbomb.player_names``: keeper display names from the lineup files (spec
``runs/MS6/gsax`` §2; for the StatsBomb keeper table and chart of football-shot-models).

The checkout is ``statsbomb_cases.write_cache`` of the standard matches (A = 900001, B = 900002,
both with players 1001-1011 and 2001-2011), with lineups edited per test. Every name is
invented. The display name is ``player_nickname`` when not null, else ``player_name``; with
several spellings, the one from the last listed match wins and ``names`` counts them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
import statsbomb_cases as sbc

from football_core.io import statsbomb

A, B = f"sb:{sbc.A}", f"sb:{sbc.B}"


def _set(files: dict[str, bytes], match: int, player: int, name: str, nick: str | None) -> None:
    def edit(teams: list[dict[str, Any]]) -> None:
        for team in teams:
            for p in team["lineup"]:
                if p["player_id"] == player:
                    p["player_name"], p["player_nickname"] = name, nick

    sbc.edit_json(files, f"data/lineups/{match}.json", edit)


def _checkout(tmp_path: Path, edits: list[tuple[int, int, str, str | None]]) -> Path:
    files = sbc.repo_files(sbc.standard_matches())
    for e in edits:
        _set(files, *e)
    return sbc.write_cache(tmp_path / "cache", files)


def test_player_names(tmp_path: Path) -> None:
    """The nickname when present, else the name; only the requested ids, one row each, sorted by
    ``player_id`` (String, String, Int64); an empty ``player_ids`` reads no file; a bare ``str``
    raises ``TypeError``, another prefix ``ValueError``."""
    checkout = _checkout(tmp_path, [
        (sbc.A, 2001, "Invented Legal Name", "Nicky"),
        (sbc.B, 2001, "Invented Legal Name", "Nicky"),
        (sbc.A, 1001, "Plain Keeper", None), (sbc.B, 1001, "Plain Keeper", None),
    ])  # fmt: skip
    out = statsbomb.player_names(checkout, [A, B], player_ids=["sb:2001", "sb:1001"])
    assert out.schema == pl.Schema(
        {"player_id": pl.String(), "player_name": pl.String(), "names": pl.Int64()}
    )
    assert out.rows() == [("sb:1001", "Plain Keeper", 1), ("sb:2001", "Nicky", 1)]
    empty = statsbomb.player_names(tmp_path / "nowhere", [A], player_ids=[])
    assert empty.height == 0 and empty.schema == out.schema
    with pytest.raises(TypeError):
        statsbomb.player_names(checkout, A, player_ids=["sb:1001"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        statsbomb.player_names(checkout, [A], player_ids="sb:1001")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ws:1001"):
        statsbomb.player_names(checkout, [A], player_ids=["ws:1001"])
    with pytest.raises(ValueError):
        statsbomb.player_names(checkout, ["ws:900001"], player_ids=["sb:1001"])


def test_player_names_conflict(tmp_path: Path) -> None:
    """A requested id with two spellings takes the last listed match's (``match_ids`` order, not
    id order) and ``names`` 2; an unrequested id with two names is ignored; a requested id in no
    lineup, a blank display name and a lineup missing from the tree raise."""
    checkout = _checkout(tmp_path, [
        (sbc.A, 1001, "First Spelling", None), (sbc.B, 1001, "Second Spelling", None),
        (sbc.A, 1005, "Outfield One", None), (sbc.B, 1005, "Outfield Two", None),
    ])  # fmt: skip
    out = statsbomb.player_names(checkout, [A, B], player_ids=["sb:1001"])
    assert out.rows() == [("sb:1001", "Second Spelling", 2)]
    out = statsbomb.player_names(checkout, [B, A], player_ids=["sb:1001"])
    assert out.rows() == [("sb:1001", "First Spelling", 2)]
    with pytest.raises(ValueError, match="sb:999999"):
        statsbomb.player_names(checkout, [A, B], player_ids=["sb:1001", "sb:999999"])
    blank = _checkout(tmp_path / "blank", [(sbc.A, 2001, "Some Name", "  ")])
    with pytest.raises(ValueError):
        statsbomb.player_names(blank, [A], player_ids=["sb:2001"])
    with pytest.raises(FileNotFoundError):
        statsbomb.player_names(checkout, [A, "sb:123456"], player_ids=["sb:1001"])
