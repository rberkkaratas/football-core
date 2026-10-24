"""Acceptance, part 1: a rerun on unchanged inputs gives byte-identical data files and identical
sha256 values in the manifest (C6 spec §0.3, §2.1, §5 criterion table)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl
import pytest
import schema_cases
import snapshot_cases as sc
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from football_core.io import snapshot

TESTS_DIR = Path(__file__).resolve().parent


# --- Data files -------------------------------------------------------------------------------


def test_rerun_same_root_files_byte_identical(tmp_path: Path) -> None:
    sc.write(tmp_path)
    before = sc.files_bytes(tmp_path)
    assert set(before) == {"events.parquet", "freeze_frames.parquet", "shots.parquet"}
    sc.write(tmp_path)
    assert sc.files_bytes(tmp_path) == before


def test_rerun_fresh_root_files_byte_identical(tmp_path: Path) -> None:
    sc.write(tmp_path / "a")
    sc.write(tmp_path / "b")
    assert sc.files_bytes(tmp_path / "a") == sc.files_bytes(tmp_path / "b")


_CHILD = """
import sys
from pathlib import Path
import snapshot_cases as sc
result = sc.write(Path(sys.argv[1]))
print(result.status)
"""


def _run_child(root: Path, threads: str | None, *, code: str = _CHILD) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(TESTS_DIR), env.get("PYTHONPATH", "")) if p)
    env.pop("POLARS_MAX_THREADS", None)
    if threads is not None:
        env["POLARS_MAX_THREADS"] = threads
    out = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "created"


def test_rerun_in_fresh_process_byte_identical(tmp_path: Path) -> None:
    sc.write(tmp_path / "here")
    _run_child(tmp_path / "one_thread", "1")
    _run_child(tmp_path / "default_threads", None)
    expected = sc.files_bytes(tmp_path / "here")
    assert sc.files_bytes(tmp_path / "one_thread") == expected
    assert sc.files_bytes(tmp_path / "default_threads") == expected
    hashes = {
        name: [f["sha256"] for f in sc.manifest_json(tmp_path / name)["files"]]
        for name in ("here", "one_thread", "default_threads")
    }
    assert hashes["here"] == hashes["one_thread"] == hashes["default_threads"]


def test_row_order_of_input_does_not_change_bytes(tmp_path: Path) -> None:
    sc.write(tmp_path / "base")
    tables = sc.all_tables()
    reversed_tables = {name: frame.reverse() for name, frame in tables.items()}
    shuffled_tables = {
        name: frame.sample(fraction=1.0, shuffle=True, seed=7) for name, frame in tables.items()
    }
    assert not shuffled_tables["shots"].equals(tables["shots"])
    sc.write(tmp_path / "reversed", reversed_tables)
    sc.write(tmp_path / "shuffled", shuffled_tables)
    expected = sc.files_bytes(tmp_path / "base")
    assert sc.files_bytes(tmp_path / "reversed") == expected
    assert sc.files_bytes(tmp_path / "shuffled") == expected


def _chunked(frame: pl.DataFrame) -> pl.DataFrame:
    pieces = [frame.slice(i, 1) for i in range(frame.height)]
    out = pl.concat(pieces, rechunk=False)
    assert out.n_chunks() > 1
    return out


def test_chunked_input_does_not_change_bytes(tmp_path: Path) -> None:
    sc.write(tmp_path / "contiguous")
    chunked = {name: _chunked(frame) for name, frame in sc.all_tables().items()}
    sc.write(tmp_path / "chunked", chunked)
    assert sc.files_bytes(tmp_path / "chunked") == sc.files_bytes(tmp_path / "contiguous")


def test_chunked_nested_frame_above_row_group_size_byte_identical(tmp_path: Path) -> None:
    """A nested ``List(Array)`` column in a table longer than ``row_group_size``, given in
    1000-row chunks, writes the same bytes as the contiguous frame (spec §0.3, p1 case 2)."""
    frame = sc.nested_frame()
    sc.write(tmp_path / "contiguous", {"areas": frame})
    pieces = pl.concat([frame.slice(i, 1000) for i in range(0, frame.height, 1000)], rechunk=False)
    assert pieces.n_chunks() > 1
    sc.write(tmp_path / "chunked", {"areas": pieces})
    assert sc.files_bytes(tmp_path / "chunked") == sc.files_bytes(tmp_path / "contiguous")


_NESTED_CHILD = """
import sys
from pathlib import Path
import snapshot_cases as sc
print(sc.write(Path(sys.argv[1]), {"areas": sc.nested_frame()}).status)
"""


def test_nested_frame_thread_count_byte_identical(tmp_path: Path) -> None:
    """With 1 thread the sort returns 1 chunk, with the default thread count several (QA
    probe: 8 on this machine), and without ``.rechunk()`` the two files differ. The writer's
    rechunk makes them equal (spec §0.3, p2)."""
    sc.write(tmp_path / "here", {"areas": sc.nested_frame()})
    _run_child(tmp_path / "one_thread", "1", code=_NESTED_CHILD)
    assert sc.files_bytes(tmp_path / "one_thread") == sc.files_bytes(tmp_path / "here")


def test_canonical_column_order_normalised(tmp_path: Path) -> None:
    sc.write(tmp_path / "base")
    tables = sc.all_tables()
    shuffled = {
        "events": tables["events"],
        "shots": tables["shots"].select(list(reversed(tables["shots"].columns))),
        "freeze_frames": tables["freeze_frames"].select(
            list(reversed(tables["freeze_frames"].columns))
        ),
    }
    sc.write(tmp_path / "reordered", shuffled)
    assert sc.files_bytes(tmp_path / "reordered") == sc.files_bytes(tmp_path / "base")
    shots = sc.table_entry(sc.manifest_json(tmp_path / "reordered"), "shots")
    assert tuple(c["name"] for c in shots["columns"]) == schema_cases.SHOT_NAMES


def test_generic_column_order_kept(tmp_path: Path) -> None:
    """Generic tables keep their column order (spec §2.1), so a reordered one is a new file."""
    frame = sc.generic_frame()
    sc.write(tmp_path, {"events": frame.select(list(reversed(frame.columns)))})
    entry = sc.table_entry(sc.manifest_json(tmp_path), "events")
    assert [c["name"] for c in entry["columns"]] == list(reversed(frame.columns))
    back = pl.read_parquet(sc.data_dir(tmp_path) / "events.parquet")
    assert back.columns == list(reversed(frame.columns))


def test_data_file_equals_independent_write(tmp_path: Path) -> None:
    """The file is exactly sort(all columns, nulls last) + rechunk + the fixed settings."""
    frame = sc.generic_frame()
    sc.write(tmp_path, {"events": frame.reverse()})
    expected = tmp_path / "expected.parquet"
    sc.sort_all(frame).rechunk().write_parquet(
        expected,
        compression="zstd",
        compression_level=3,
        statistics=True,
        row_group_size=100_000,
    )
    assert (sc.data_dir(tmp_path) / "events.parquet").read_bytes() == expected.read_bytes()


# --- Manifest ---------------------------------------------------------------------------------


def test_rerun_manifest_sha256_identical(tmp_path: Path) -> None:
    sc.write(tmp_path)
    first = sc.manifest_json(tmp_path)
    sc.write(tmp_path)
    second = sc.manifest_json(tmp_path)
    assert first["files"] == second["files"]
    assert first["data_digest"] == second["data_digest"]
    on_disk = sc.files_sha(tmp_path)
    assert {f["path"]: f["sha256"] for f in second["files"]} == on_disk


def test_rerun_manifest_bytes_identical(tmp_path: Path) -> None:
    sc.write(tmp_path / "a")
    first = sc.manifest_path(tmp_path / "a").read_bytes()
    sc.write(tmp_path / "a")
    assert sc.manifest_path(tmp_path / "a").read_bytes() == first
    sc.write(tmp_path / "b")
    assert sc.manifest_path(tmp_path / "b").read_bytes() == first


def test_rerun_status_unchanged(tmp_path: Path) -> None:
    first = sc.write(tmp_path)
    assert first.status == "created"
    assert first.manifest_written is True
    second = sc.write(tmp_path)
    assert second.status == "unchanged"
    assert second.manifest_written is False
    assert second.manifest == first.manifest


# --- Property ---------------------------------------------------------------------------------

# Floats: allow_nan=False and ``x + 0.0`` (turns -0.0 into 0.0); NaN only as float("nan"), so
# no two rows differ only in the sign of a zero or a NaN payload (spec §2.1 known limit).
_FLOATS = st.one_of(
    st.none(),
    st.just(float("nan")),
    st.floats(allow_nan=False, allow_infinity=True).map(lambda x: x + 0.0),
)
_ROW = st.tuples(
    st.one_of(st.none(), st.sampled_from(["a", "b", "ä", ""])),
    st.one_of(st.none(), st.integers(min_value=-(2**63), max_value=2**63 - 1)),
    _FLOATS,
)


@settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(rows=st.lists(_ROW, max_size=8), data=st.data())
def test_property_rerun_and_permutation_identical(
    rows: list[tuple[str | None, int | None, float | None]], data: st.DataObject
) -> None:
    order = data.draw(st.permutations(range(len(rows))))

    def frame(ordered: list[tuple[str | None, int | None, float | None]]) -> pl.DataFrame:
        return pl.DataFrame(
            [
                pl.Series("s", [r[0] for r in ordered], dtype=pl.String),
                pl.Series("i", [r[1] for r in ordered], dtype=pl.Int64),
                pl.Series("f", [r[2] for r in ordered], dtype=pl.Float64),
            ]
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = sc.write(root / "a", {"t": frame(rows)})
        permuted = sc.write(root / "b", {"t": frame([rows[i] for i in order])})
        assert sc.files_bytes(root / "a") == sc.files_bytes(root / "b")
        assert first.manifest.files == permuted.manifest.files
        rerun = sc.write(root / "a", {"t": frame(rows)})
        assert rerun.status == "unchanged"
        assert (
            sc.manifest_path(root / "a").read_bytes() == sc.manifest_path(root / "b").read_bytes()
        )


@pytest.mark.parametrize("threads", ["1", "4"])
def test_manifest_has_no_thread_dependent_field(tmp_path: Path, threads: str) -> None:
    """Nothing in the manifest text depends on the thread count of the writer."""
    _run_child(tmp_path / "child", threads)
    sc.write(tmp_path / "here")
    child = json.loads(sc.manifest_path(tmp_path / "child").read_text(encoding="utf-8"))
    here = json.loads(sc.manifest_path(tmp_path / "here").read_text(encoding="utf-8"))
    assert child == here


def test_result_manifest_equals_file(tmp_path: Path) -> None:
    result = sc.write(tmp_path)
    assert snapshot.read_manifest(result.manifest_path) == result.manifest
    assert result.manifest_path.read_text(encoding="utf-8") == result.manifest.to_json()
