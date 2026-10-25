"""``write_snapshot`` and the manifest reader with the canonical matches table (MS3.matches §4.2).

``write_snapshot`` validates ``matches`` first, then ``shots`` against it, then freeze frames
against the shots; a snapshot may hold matches or shots without the other.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest
import schema_cases
import snapshot_cases as sc
from polars.testing import assert_frame_equal

from football_core import schema
from football_core.io import snapshot


def _tables(**changes: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {
        "shots": schema_cases.shots_frame(),
        "freeze_frames": schema_cases.freeze_frames_frame(),
        "matches": schema_cases.matches_frame(),
        **changes,
    }


def test_write_validates_matches(tmp_path: Path) -> None:
    bad = schema_cases.with_value(schema_cases.matches_frame(), "gender", 0, None)
    bad_shots = schema_cases.with_value(schema_cases.shots_frame(), "is_goal", 0, False)
    for tables in ({"matches": bad}, _tables(matches=bad, shots=bad_shots)):
        with pytest.raises(schema.SchemaValidationError) as info:
            sc.write(tmp_path, tables)
        assert info.value.table == "matches"  # matches is validated before shots
        assert {f.check for f in info.value.failures} == {"nullable"}
        sc.assert_nothing_written(tmp_path, before=set())
    days = pl.Series("match_date", [-800_000] * 4, dtype=pl.Int32).cast(pl.Date)
    far = schema_cases.replace_column(schema_cases.matches_frame(), days)
    with pytest.raises(ValueError, match="(?=.*matches)(?=.*'match_date')"):
        sc.write(tmp_path, {"matches": far})  # groups-eligible: C6 refuses the date
    sc.assert_nothing_written(tmp_path, before=set())

    result = sc.write(tmp_path, _tables())
    assert result.status == "created"
    entry = sc.table_entry(sc.manifest_json(tmp_path), "matches")
    assert entry["schema"] == "matches"
    assert entry["rows"] == 4 and len(entry["columns"]) == 14
    loaded = snapshot.load_snapshot(result.manifest_path, tables=("matches",))["matches"]
    assert_frame_equal(loaded.sort("match_id"), schema_cases.matches_frame().sort("match_id"))


def test_write_checks_shots_against_matches(tmp_path: Path) -> None:
    matches = schema_cases.matches_frame()
    orphan = _tables(matches=matches.filter(pl.col("match_id") != "sb:9000002"))
    other_date = schema_cases.with_value(matches, "match_date", 0, matches["match_date"][2])
    for tables, check, rows in (
        (orphan, "match_exists", (4,)),
        (_tables(matches=other_date), "shot_agrees_with_match", (0, 1)),
    ):
        with pytest.raises(schema.SchemaValidationError) as info:
            sc.write(tmp_path, tables)
        assert info.value.table == "shots"
        assert [(f.check, f.rows) for f in info.value.failures] == [(check, rows)]
        sc.assert_nothing_written(tmp_path, before=set())
    # either table without the other; freeze frames still checked against shots (keyed rules)
    without_matches = {k: v for k, v in _tables().items() if k != "matches"}
    for name, tables in (
        ("shots-only", {"shots": schema_cases.shots_frame()}),
        ("matches-only", {"matches": matches}),
        ("shots-frames", without_matches),
        ("all", _tables()),
    ):
        assert sc.write(tmp_path, tables, name=name).status == "created", name
    shooter = schema_cases.with_value(
        schema_cases.freeze_frames_frame(), "player_id", 1, schema_cases.SHOOTER_1
    )
    with pytest.raises(schema.SchemaValidationError) as info:
        sc.write(tmp_path, _tables(freeze_frames=shooter), name="shooter")
    assert info.value.table == "freeze_frames"
    assert {f.check for f in info.value.failures} == {"shooter_not_in_frame"}


def test_reader_refuses_schemaless_matches() -> None:
    """A 1.0 snapshot's matches table (``schema: null``) is rebuilt, never read (spec §5)."""
    old = sc.add_table(sc.base_manifest(), "matches", schema=None)
    with pytest.raises(snapshot.ManifestError) as info:
        snapshot.Manifest.from_json(sc.dumps(old))
    assert re.search(r"tables\[\d+\]\.schema", str(info.value)), str(info.value)
    assert "'matches'" in str(info.value)
    new = sc.add_table(sc.base_manifest(), "matches", schema="matches")
    parsed = snapshot.Manifest.from_json(sc.dumps(new))
    assert {t.name: t.schema for t in parsed.tables}["matches"] == "matches"
