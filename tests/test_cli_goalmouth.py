"""``fc goalmouth MANIFEST [MANIFEST ...]`` (spec ``runs/MS5/goalmouth`` §5.2, §7 Rule 1, §10.9;
``runs/MS6/xgot-ready`` §2.5): counts to stdout, nothing written, exit 2 for a manifest that cannot
be read, exit 1 for a snapshot of another schema version, never a traceback. MS6: with StatsBomb
rows it also prints the fidelity mapping, the grid census and the lattice line, and ``--check``
fails on a ``goal_y`` off the 0.1 yd lattice.

The StatsBomb rows of the schema fixture get invented raw events (``location`` and a 3-value
``shot.end_location``) and a ``goal_y`` on the lattice, as the MS6 loader writes them: match
9000001 is fine (tenths in a location), 9000002 coarse (whole yards)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import schema_cases
import snapshot_cases as sc
import statsbomb_cases as sbc

from football_core import cli

LOW_LEFT = json.dumps([{"type": {"value": 76, "displayName": "LowLeft"}}], separators=(",", ":"))
MAPPING = (
    'fidelity: fine = shot_fidelity_version "2" (or inferred tenths); '
    'coarse = "1" (or inferred whole yards)'
)
# StatsBomb fixture rows in order: raw location, raw end_location (y names its 0.1 yd cell)
SB_RAW = (
    ((108.1, 40.0), (120.0, 38.7, 0.66)),  # goal on the line: cell centre 38.65 yd
    ((108.1, 40.0), (119.9, 41.1, 0.55)),  # saved within one fine cell: 41.05 yd
    ((108.0, 44.0), (120.0, 34.8, 1.3)),  # coarse match, off target past the left post
)


def _raw(location: tuple[float, ...], end: tuple[float, ...]) -> str:
    event = {"location": list(location), "shot": {"end_location": list(end)}}
    return json.dumps(event, sort_keys=True, separators=(",", ":"))


def _statsbomb(shots: pl.DataFrame) -> pl.DataFrame:
    """The StatsBomb rows with raw events and a lattice ``goal_y`` (cell centre, in metres)."""
    sb = shots.filter(pl.col("provider") == "statsbomb")
    assert sb.height == len(SB_RAW)  # precondition: the schema fixture's three StatsBomb rows
    goal_y = [sbc.mouth_metres(end[1] - 0.05) for _, end in SB_RAW]
    return sb.with_columns(
        raw_qualifiers=pl.Series([_raw(loc, end) for loc, end in SB_RAW]),
        goal_y=pl.Series(goal_y, dtype=pl.Float64()),
    )


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    rc = cli.main(list(argv))
    out, err = capsys.readouterr()
    return rc, out, err


@pytest.fixture
def manifests(tmp_path: Path) -> tuple[Path, Path]:
    """One StatsBomb-only and one WhoScored-only snapshot of the schema fixture's shots; the
    WhoScored own goal (goal-frame y -1.0) carries a LowLeft zone in the stored shape."""
    shots = schema_cases.shots_frame()
    sb = _statsbomb(shots)
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


def test_goalmouth_check(capsys: pytest.CaptureFixture[str], manifests: tuple[Path, Path]) -> None:
    """``--check`` exits 1, naming the failure on stderr, when the zone qualifiers do not confirm
    the Opta direction (the fixture has a left zone only), and prints the goals-in-frame share
    per provider; without ``--check`` the same snapshots exit 0 (run5-ms5 B6, Q9)."""
    sb, ws = manifests
    rc, out, err = _run(capsys, "goalmouth", "--check", str(sb), str(ws))
    assert rc == 1
    assert "fc: check failed: whoscored: the zone qualifiers 76-87 do not confirm" in err
    assert "Traceback" not in out + err
    assert "goals inside the frame: statsbomb" in out and "goals inside the frame: whoscored" in out
    rc, _, err = _run(capsys, "goalmouth", str(sb), str(ws))
    assert (rc, err) == (0, "")


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


# ---- MS6 xgot-ready §2.5 -----------------------------------------------------------------------


def test_census_header_mapping(
    capsys: pytest.CaptureFixture[str], manifests: tuple[Path, Path]
) -> None:
    """With StatsBomb rows: the fidelity mapping line, then the grid census (one fine match, one
    coarse), then ``goal_y on the 0.1 yd lattice: statsbomb X of M``; ``--check`` passes on the
    StatsBomb snapshot alone (every goal_y is a cell centre). Without StatsBomb rows none of it
    is printed."""
    sb, ws = manifests
    rc, out, err = _run(capsys, "goalmouth", str(sb))
    assert (rc, err) == (0, "")
    lines = out.splitlines()
    assert MAPPING in lines
    for word in ("fidelity", "stop", "x_whole", "y_whole", "y_odd_tenth", "with_z", "z_whole",
                 "coarse", "fine", "near line"):  # fmt: skip
        assert word in out, word
    assert out.index(MAPPING) < out.index("y_odd_tenth")
    assert "goal_y on the 0.1 yd lattice: statsbomb 3 of 3" in lines
    rc, out, err = _run(capsys, "goalmouth", "--check", str(sb))
    assert (rc, err) == (0, "")
    rc, out, _ = _run(capsys, "goalmouth", str(ws))
    assert rc == 0
    assert MAPPING not in out and "y_odd_tenth" not in out and "lattice" not in out


def test_grid_lines_order(capsys: pytest.CaptureFixture[str], manifests: tuple[Path, Path]) -> None:
    """Ratify 3: ``cli.GRID_HEADER`` is the mapping line and is printed right before the grid
    census (after the census tables); the lattice line, built from ``cli.MOUTH_CELL`` (the
    loader's 0.1 yd cell), comes after the goal-share lines and is the last line."""
    from football_core.io import statsbomb

    assert cli.GRID_HEADER == MAPPING and cli.MOUTH_CELL == statsbomb.MOUTH_CELL == 0.1
    rc, out, err = _run(capsys, "goalmouth", str(manifests[0]))
    assert (rc, err) == (0, "")
    lines = out.splitlines()
    header = lines.index(cli.GRID_HEADER)
    census = next(i for i, x in enumerate(lines) if "y_odd_tenth" in x)
    shares = [i for i, x in enumerate(lines) if x.startswith("goals inside the frame:")]
    lattice = lines.index("goal_y on the 0.1 yd lattice: statsbomb 3 of 3")
    assert shares and header < census < min(shares) and max(shares) < lattice
    assert lattice == len(lines) - 1


def _rewrite(shots: pl.DataFrame, tmp_path: Path, name: str) -> Path:
    sc.write(tmp_path, {"shots": shots}, name=name)
    return sc.manifest_path(tmp_path, name)


def test_check_fails_off_lattice(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """``--check`` exits 1, the lattice problem on stderr and no traceback, when one StatsBomb
    ``goal_y`` is 1e-6 m off its cell centre; without ``--check`` it exits 0 and prints the
    lattice line with 2 of 3."""
    sb = _statsbomb(schema_cases.shots_frame())
    off = sb.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(pl.col("goal_y") + 1e-6)
        .otherwise(pl.col("goal_y")).alias("goal_y")
    )  # fmt: skip
    manifest = _rewrite(off, tmp_path, "gm-sb-off")
    rc, out, err = _run(capsys, "goalmouth", "--check", str(manifest))
    assert rc == 1
    assert "Traceback" not in out + err
    assert "statsbomb: 1 of 3 goal_y values are off the 0.1 yd goal-mouth lattice" in err
    rc, out, err = _run(capsys, "goalmouth", str(manifest))
    assert (rc, err) == (0, "")
    assert "goal_y on the 0.1 yd lattice: statsbomb 2 of 3" in out.splitlines()


def test_bad_raw_qualifiers_exit_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """A StatsBomb row whose ``raw_qualifiers`` has no ``location`` exits 1 with the census's
    ``ValueError`` message on stderr (``fc: error:``) and no traceback."""
    sb = _statsbomb(schema_cases.shots_frame())
    broken = sb.with_columns(
        pl.when(pl.int_range(pl.len()) == 1)
        .then(pl.lit(json.dumps({"shot": {"end_location": [119.9, 41.1, 0.55]}})))
        .otherwise(pl.col("raw_qualifiers")).alias("raw_qualifiers")
    )  # fmt: skip
    manifest = _rewrite(broken, tmp_path, "gm-sb-bad")
    rc, out, err = _run(capsys, "goalmouth", str(manifest))
    assert rc == 1
    assert "Traceback" not in out + err
    assert err.startswith("fc: error: ") and "location" in err
