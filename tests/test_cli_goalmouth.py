"""``fc goalmouth MANIFEST [MANIFEST ...]`` (spec ``runs/MS5/goalmouth`` §5.2, §7 Rule 1, §10.9):
counts to stdout, nothing written, exit 2 for a manifest that cannot be read, exit 1 for a
snapshot of another schema version, never a traceback."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import schema_cases
import snapshot_cases as sc

from football_core import cli

LOW_LEFT = json.dumps([{"type": {"value": 76, "displayName": "LowLeft"}}], separators=(",", ":"))


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    rc = cli.main(list(argv))
    out, err = capsys.readouterr()
    return rc, out, err


@pytest.fixture
def manifests(tmp_path: Path) -> tuple[Path, Path]:
    """One StatsBomb-only and one WhoScored-only snapshot of the schema fixture's shots; the
    WhoScored own goal (goal-frame y -1.0) carries a LowLeft zone in the stored shape."""
    shots = schema_cases.shots_frame()
    sb = shots.filter(pl.col("provider") == "statsbomb")
    ws = shots.filter(pl.col("provider") == "whoscored").with_columns(
        pl.when(pl.col("goal_y").is_not_null())
        .then(pl.lit(LOW_LEFT))
        .otherwise(pl.col("raw_qualifiers"))
        .alias("raw_qualifiers")
    )
    sc.write(tmp_path, {"shots": sb}, name="gm-sb")
    sc.write(tmp_path, {"shots": ws}, name="gm-ws")
    return sc.manifest_path(tmp_path, "gm-sb"), sc.manifest_path(tmp_path, "gm-ws")


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_goalmouth_output(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, manifests: tuple[Path, Path]
) -> None:
    sb, ws = manifests
    before = _files(tmp_path)
    rc, out, err = _run(capsys, "goalmouth", str(sb), str(ws))
    assert (rc, err) == (0, "")
    for word in ("statsbomb", "whoscored", "goal", "saved", "blocked", "own_goal", "in_frame",
                 "near_woodwork", "named_side", "shooter_left"):  # fmt: skip
        assert word in out, word
    # one left zone, no right zone: the criterion's check cannot confirm the direction
    assert "direction confirmed (zones 76-87): no" in out.splitlines()
    assert _files(tmp_path) == before  # Rule 1: prints counts, writes nothing
    rc, out, _ = _run(capsys, "goalmouth", str(sb))
    assert rc == 0 and "statsbomb" in out
    assert "direction confirmed" not in out and "named_side" not in out  # no whoscored rows


def test_goalmouth_bad_manifest(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    rc, out, err = _run(capsys, "goalmouth", str(bad))
    assert rc == 2
    assert "Traceback" not in out + err
    assert "bad.json" in err  # the ManifestError names the manifest (not an argparse error)


def test_goalmouth_schema_mismatch(
    capsys: pytest.CaptureFixture[str], manifests: tuple[Path, Path]
) -> None:
    sb, ws = manifests
    m = json.loads(sb.read_text(encoding="utf-8"))
    m["schema_version"] = "1.1"  # a snapshot from before the bump
    sb.write_text(sc.dumps(m), encoding="utf-8")
    rc, out, err = _run(capsys, "goalmouth", str(ws), str(sb))
    assert rc == 1
    assert "Traceback" not in out + err
    assert "1.1" in err
