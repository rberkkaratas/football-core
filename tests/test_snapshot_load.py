"""``load_snapshot``: verify first, then read the requested tables (C6 spec §1.2, §5)."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import polars as pl
import pytest
import snapshot_cases as sc
from polars.testing import assert_frame_equal

from football_core import schema
from football_core.io import snapshot


@pytest.fixture
def snap(tmp_path: Path) -> Path:
    sc.write(tmp_path)
    return sc.manifest_path(tmp_path)


def _expected() -> dict[str, pl.DataFrame]:
    tables = sc.all_tables()
    return {
        "events": sc.sort_all(tables["events"]),
        "freeze_frames": sc.sort_all(schema.validate_freeze_frames(tables["freeze_frames"])),
        "shots": sc.sort_all(schema.validate_shots(tables["shots"])),
    }


def _set_schema_version(manifest: Path, version: str) -> None:
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m["schema_version"] = version
    manifest.write_text(sc.dumps(m), encoding="utf-8")


def test_load_returns_written_frames(snap: Path) -> None:
    loaded = snapshot.load_snapshot(snap)
    assert list(loaded) == ["events", "freeze_frames", "shots"]
    for name, expected in _expected().items():
        assert loaded[name].schema == expected.schema
        assert_frame_equal(loaded[name], expected)
    assert loaded["shots"].schema == schema.SHOT_SCHEMA
    assert loaded["freeze_frames"].schema == schema.FREEZE_FRAME_SCHEMA


def test_load_accepts_str(snap: Path) -> None:
    assert list(snapshot.load_snapshot(str(snap))) == ["events", "freeze_frames", "shots"]


def test_load_subset(snap: Path) -> None:
    subset = snapshot.load_snapshot(snap, tables=["shots", "events"])
    assert list(subset) == ["shots", "events"]
    assert snapshot.load_snapshot(snap, tables=[]) == {}
    assert snapshot.load_snapshot(snap, tables=("events",)).keys() == {"events"}
    _set_schema_version(snap, "9.9")
    only_generic = snapshot.load_snapshot(snap, tables=["events"])
    assert list(only_generic) == ["events"]


@pytest.mark.parametrize("tables", [["nope"], ["events", "events"], ["shots", "nope"]])
def test_load_unknown_table(snap: Path, tables: list[str]) -> None:
    with pytest.raises(ValueError):
        snapshot.load_snapshot(snap, tables=tables)


def test_load_unknown_table_checked_before_verify(snap: Path) -> None:
    """``tables`` is checked first (spec §1.2): a bad name is a ValueError even when the data
    would also fail verification."""
    sc.flip_byte(sc.data_dir(snap.parent.parent) / "shots.parquet", 0)
    with pytest.raises(ValueError) as info:
        snapshot.load_snapshot(snap, tables=["nope"])
    assert not isinstance(info.value, snapshot.SnapshotVerificationError)
    assert "nope" in str(info.value)


def test_load_raises_on_changed_file(snap: Path) -> None:
    sc.flip_byte(sc.data_dir(snap.parent.parent) / "shots.parquet", -1)
    with pytest.raises(snapshot.SnapshotVerificationError) as info:
        snapshot.load_snapshot(snap)
    assert isinstance(info.value, snapshot.SnapshotError)
    report = info.value.report
    assert isinstance(report, snapshot.VerifyReport)
    assert [(p.kind, p.path) for p in report.problems] == [("changed", "shots.parquet")]


def test_load_raises_on_problem_in_unrequested_table(snap: Path) -> None:
    """A full verify runs, whatever subset is requested."""
    (sc.data_dir(snap.parent.parent) / "shots.parquet").unlink()
    with pytest.raises(snapshot.SnapshotVerificationError):
        snapshot.load_snapshot(snap, tables=["events"])


def test_load_raises_on_extra_file(snap: Path) -> None:
    (sc.data_dir(snap.parent.parent) / "extra.parquet").write_bytes(b"x")
    with pytest.raises(snapshot.SnapshotVerificationError):
        snapshot.load_snapshot(snap)


def test_load_verify_false_skips(snap: Path) -> None:
    replacement = pl.DataFrame({"other": [1, 2, 3]})
    replacement.write_parquet(sc.data_dir(snap.parent.parent) / "events.parquet")
    with pytest.raises(snapshot.SnapshotVerificationError):
        snapshot.load_snapshot(snap)
    loaded = snapshot.load_snapshot(snap, tables=["events"], verify=False)
    assert_frame_equal(loaded["events"], replacement)


def test_load_verify_false_missing_file_is_oserror(snap: Path) -> None:
    (sc.data_dir(snap.parent.parent) / "events.parquet").unlink()
    with pytest.raises(OSError):
        snapshot.load_snapshot(snap, tables=["events"], verify=False)


def test_load_schema_version_mismatch(snap: Path) -> None:
    _set_schema_version(snap, "9.9")
    for tables in (None, ["shots"], ["freeze_frames"]):
        with pytest.raises(ValueError) as info:
            snapshot.load_snapshot(snap, tables=tables)
        assert "9.9" in str(info.value)
        assert schema.SCHEMA_VERSION in str(info.value)
        assert not isinstance(info.value, snapshot.ManifestError)


def test_load_data_dir_override(snap: Path, tmp_path: Path) -> None:
    moved = tmp_path / "elsewhere"
    shutil.move(sc.data_dir(snap.parent.parent), moved)
    with pytest.raises(snapshot.SnapshotVerificationError):
        snapshot.load_snapshot(snap)
    loaded = snapshot.load_snapshot(snap, data_dir=moved)
    assert list(loaded) == ["events", "freeze_frames", "shots"]


def test_load_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(snapshot.ManifestError):
        snapshot.load_snapshot(tmp_path / "manifests" / "nope.json")


def test_errors_pickle(snap: Path) -> None:
    sc.flip_byte(sc.data_dir(snap.parent.parent) / "events.parquet", 0)
    report = snapshot.verify_snapshot(snap)
    err = snapshot.SnapshotVerificationError(report)
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is snapshot.SnapshotVerificationError
    assert back.report == report
    assert str(back) == str(err)
    for cls in (snapshot.SnapshotError, snapshot.ManifestError, snapshot.SnapshotExistsError):
        e = cls("something went wrong: tables[1].rows")
        again = pickle.loads(pickle.dumps(e))
        assert type(again) is cls
        assert str(again) == "something went wrong: tables[1].rows"


def test_verification_error_message_names_problem(snap: Path) -> None:
    (sc.data_dir(snap.parent.parent) / "freeze_frames.parquet").unlink()
    with pytest.raises(snapshot.SnapshotVerificationError) as info:
        snapshot.load_snapshot(snap)
    assert "freeze_frames.parquet" in str(info.value)
