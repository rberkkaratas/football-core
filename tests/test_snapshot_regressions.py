"""Regression tests for the C6 carry-over fixes (C6 gate open problems 1, 2, 6; MS2.c4 spec
§4): a symlink loop and a non-parquet ``.parquet`` input end in ``fc: error:`` and exit 1, and
``load_snapshot(tables=<str>)`` is a ``TypeError``."""

from __future__ import annotations

import errno
import io
import os
from pathlib import Path

import polars as pl
import pytest
import snapshot_cases as sc

from football_core import cli
from football_core.io import snapshot

needs_symlinks = pytest.mark.skipif(os.name == "nt", reason="symlinks")


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    rc = cli.main(list(argv))
    out, err = capsys.readouterr()
    return rc, out, err


def _loop(tmp_path: Path) -> Path:
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    return loop


def _good_input(tmp_path: Path) -> Path:
    folder = tmp_path / "in"
    folder.mkdir()
    sc.generic_frame().write_parquet(folder / "events.parquet")
    return folder


@needs_symlinks
@pytest.mark.parametrize("case", ["--input", "--root", "statsbomb_cache"])
def test_symlink_loop_clean_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, case: str
) -> None:
    loop = _loop(tmp_path)
    if case == "--input":
        args = ["snapshot", "demo", "--input", str(loop), "--root", str(tmp_path / "out")]
    elif case == "--root":
        args = ["snapshot", "demo", "--input", str(_good_input(tmp_path)), "--root", str(loop)]
    else:
        args = [
            "snapshot",
            "demo",
            "--builder",
            "statsbomb",
            "--input",
            str(loop),
            "--root",
            str(tmp_path / "out"),
            "--param",
            "commit=" + "f00d" * 10,
        ]
    rc, out, err = _run(capsys, *args)
    assert rc == 1
    assert err.startswith("fc: error:"), err
    assert "symlink loop" in err
    assert "Traceback" not in err and "RuntimeError" not in err


@needs_symlinks
def test_source_from_files_symlink_loop_is_oserror(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    good = _good_input(tmp_path) / "events.parquet"
    for paths, root in (([loop], tmp_path), ([good], loop)):
        with pytest.raises(OSError) as info:
            snapshot.source_from_files(paths, root=root)
        assert info.value.errno == errno.ELOOP
        assert str(loop) in str(info.value)


def _parquet_bytes() -> bytes:
    out = io.BytesIO()
    sc.generic_frame().write_parquet(out)
    return out.getvalue()


@pytest.mark.parametrize(
    "content",
    [b"not parquet at all", b"", _parquet_bytes()[: len(_parquet_bytes()) // 2], b"PAR1"],
    ids=["garbage", "empty", "truncated", "magic_only"],
)
def test_non_parquet_input_clean_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, content: bytes
) -> None:
    bad = tmp_path / "in" / "events.parquet"
    bad.parent.mkdir()
    bad.write_bytes(content)
    rc, out, err = _run(
        capsys, "snapshot", "demo", "--input", str(bad), "--root", str(tmp_path / "out")
    )
    assert rc == 1
    assert err.startswith(f"fc: error: input '{bad}' is not a readable parquet file"), err
    assert "Traceback" not in err
    assert not sc.manifest_path(tmp_path / "out").exists()
    with pytest.raises(ValueError, match="is not a readable parquet file"):
        snapshot.build_from_parquet(snapshot.BuildRequest((bad,)))


def test_load_snapshot_rejects_str_tables(tmp_path: Path) -> None:
    written = sc.write(tmp_path, {"ev": pl.DataFrame({"a": [1]})})
    with pytest.raises(
        TypeError, match="tables must be a sequence of table names, not a single str"
    ):
        snapshot.load_snapshot(written.manifest_path, tables="ev")
    assert sorted(snapshot.load_snapshot(written.manifest_path, tables=["ev"])) == ["ev"]
