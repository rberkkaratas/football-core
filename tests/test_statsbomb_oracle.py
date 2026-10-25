"""kloppy as a dev-only coordinate oracle (spec ``runs/MS2/c4/spec.md`` §2.3, §7).

kloppy shifts StatsBomb values by half a cell as the loader does, but maps x with two extra
landmarks (penalty-arc tip, centre-circle edge), so x agrees to 1e-9 m only where the shifted
raw x is in [0, 18] or [102, 120]; elsewhere the tolerance is 0.6 m. y agrees everywhere.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import statsbomb_cases as sbc

from football_core.io import statsbomb

kloppy = pytest.importorskip("kloppy")
from kloppy import statsbomb as kloppy_statsbomb  # noqa: E402
from kloppy.domain import (  # noqa: E402
    CustomCoordinateSystem,
    Dimension,
    MetricPitchDimensions,
    Origin,
    VerticalOrientation,
)

METRIC = CustomCoordinateSystem(
    origin=Origin.BOTTOM_LEFT,
    vertical_orientation=VerticalOrientation.BOTTOM_TO_TOP,
    pitch_dimensions=MetricPitchDimensions(
        x_dim=Dimension(0, 105),
        y_dim=Dimension(0, 68),
        pitch_length=105,
        pitch_width=68,
        standardized=True,
    ),
)
BOX_TOL, X_TOL = 1e-9, 0.6
SAMPLE = (18235, 18236, 18237, 15946, 15956, 15973, 3788741, 3845506, 3869685)
OPEN_DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "statsbomb" / "open-data"


def _offset(events: list[dict[str, Any]]) -> float:
    """kloppy's rule: 0.5 when every shot location of the match is an integer, else 0.05."""
    values = [float(v) for e in events if e["type"]["name"] == "Shot" for v in e["location"][:2]]
    return sbc.COARSE if all(v.is_integer() for v in values) else sbc.FINE


def _x_tol(raw_x: float, offset: float) -> float:
    shifted = raw_x - offset
    return BOX_TOL if shifted <= 18 or shifted >= 102 else X_TOL


def _compare(events_json: bytes, lineups_json: bytes, data: Any, tmp: Path) -> int:
    """Compare every Shot and its freeze-frame players by id; returns the points compared."""
    (tmp / "e.json").write_bytes(events_json)
    (tmp / "l.json").write_bytes(lineups_json)
    raw = kloppy_statsbomb.load(
        event_data=str(tmp / "e.json"), lineup_data=str(tmp / "l.json"), coordinates="statsbomb"
    )
    metric = raw.transform(to_coordinate_system=METRIC)
    events = json.loads(events_json)
    offset = _offset(events)
    shots = {e["id"]: e for e in events if e["type"]["name"] == "Shot"}  # kloppy adds own goals
    ours = data.shots.filter(pl.col("raw_type") == "Shot")
    ff = data.freeze_frames
    compared = 0
    for k in metric.events:
        if k.event_id not in shots:
            continue
        ev = shots[k.event_id]
        r = sbc.row(ours, k.event_id)
        tol = _x_tol(ev["location"][0], offset)
        assert abs(r["x"] - k.coordinates.x) <= tol, (k.event_id, r["x"], k.coordinates.x)
        assert abs(r["y"] - k.coordinates.y) <= BOX_TOL, (k.event_id, r["y"], k.coordinates.y)
        compared += 1
        frame = {str(p["player"]["id"]): p["location"] for p in ev["shot"].get("freeze_frame", [])}
        rows = ff.filter(pl.col("shot_id") == f"sb:{k.event_id}")
        if k.freeze_frame is None or rows.height == 0:  # penalties have no rows by design
            continue
        mine = {pid: (x, y) for pid, x, y in rows.select("player_id", "x", "y").iter_rows()}
        for player, coords in k.freeze_frame.players_data.items():
            pid = str(player.player_id)
            if pid not in frame or coords.coordinates is None:
                continue  # kloppy also lists the shooter
            x, y = mine[f"sb:{pid}"]
            assert abs(x - coords.coordinates.x) <= _x_tol(frame[pid][0], offset), (k.event_id, pid)
            assert abs(y - coords.coordinates.y) <= BOX_TOL, (k.event_id, pid)
            compared += 1
    return compared


def test_fixture_coordinates_match_kloppy(sb_standard: Any, tmp_path: Path) -> None:
    files = sbc.repo_files(sbc.standard_matches())
    compared = 0
    for mid in (sbc.A, sbc.B):
        compared += _compare(
            files[f"data/events/{mid}.json"],
            files[f"data/lineups/{mid}.json"],
            sb_standard,
            tmp_path,
        )
    assert compared >= 15 + 30  # 15 Shot events, 37 frame rows


@pytest.mark.skipif(
    not (OPEN_DATA / statsbomb.OPEN_DATA_COMMIT / "tree.json").is_file(),
    reason="needs the local open-data cache at the pinned commit (data/ is git-ignored)",
)
def test_open_data_sample_matches_kloppy(tmp_path: Path) -> None:
    checkout = OPEN_DATA / statsbomb.OPEN_DATA_COMMIT
    data = statsbomb.load(checkout, match_ids=SAMPLE)
    assert set(data.matches["match_id"]) == {f"sb:{m}" for m in SAMPLE}
    for mid in SAMPLE:
        events = gzip.decompress((checkout / f"data/events/{mid}.json.gz").read_bytes())
        lineups = gzip.decompress((checkout / f"data/lineups/{mid}.json.gz").read_bytes())
        assert _compare(events, lineups, data, tmp_path) > 0, mid
