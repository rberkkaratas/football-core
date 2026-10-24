"""Acceptance, part 2: ``verify_snapshot`` names every changed, missing and extra file, and is
total on filesystem states (C6 spec §3.2, §3.3, §5)."""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import snapshot_cases as sc
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from football_core.io import snapshot

TABLES = ("events", "freeze_frames", "shots")
FILES = tuple(f"{t}.parquet" for t in TABLES)


@pytest.fixture
def snap(tmp_path: Path) -> tuple[Path, Path]:
    """A clean snapshot ``demo`` with a generic and both canonical tables."""
    root = tmp_path / "repo"
    sc.write(root)
    return sc.manifest_path(root), sc.data_dir(root)


def _kinds(report: snapshot.VerifyReport) -> list[tuple[str, str]]:
    return [(p.kind, p.path) for p in report.problems]


# --- Clean ------------------------------------------------------------------------------------


def test_clean_snapshot_ok(snap: tuple[Path, Path]) -> None:
    manifest, data = snap
    report = snapshot.verify_snapshot(manifest)
    assert report.ok is True
    assert report.problems == ()
    assert report.checked == 3
    assert report.name == "demo"
    assert Path(report.manifest_path) == manifest
    assert os.path.samefile(report.data_dir, data)


def test_clean_snapshot_ok_with_str_path(snap: tuple[Path, Path]) -> None:
    manifest, _ = snap
    assert snapshot.verify_snapshot(str(manifest)).ok


# --- One byte changed -------------------------------------------------------------------------


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_one_byte_flip_reported_changed(snap: tuple[Path, Path], table: str, where: str) -> None:
    manifest, data = snap
    path = data / f"{table}.parquet"
    size = path.stat().st_size
    expected_sha = sc.sha256_file(path)
    offset = {"first": 0, "middle": size // 2, "last": -1}[where]
    sc.flip_byte(path, offset)
    assert path.stat().st_size == size
    report = snapshot.verify_snapshot(manifest)
    assert not report.ok
    assert _kinds(report) == [("changed", f"{table}.parquet")]
    problem = report.problems[0]
    got_sha = sc.sha256_file(path)
    assert problem.detail == f"sha256 {got_sha} != manifest {expected_sha}"
    assert str(problem) == f"changed: {table}.parquet ({problem.detail})"


def test_one_byte_appended_reported_changed(snap: tuple[Path, Path]) -> None:
    manifest, data = snap
    path = data / "shots.parquet"
    size = path.stat().st_size
    with path.open("ab") as f:
        f.write(b"\x00")
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [("changed", "shots.parquet")]
    assert report.problems[0].detail == f"{size + 1} bytes, manifest says {size}"


def test_one_byte_truncated_reported_changed(snap: tuple[Path, Path]) -> None:
    manifest, data = snap
    path = data / "freeze_frames.parquet"
    size = path.stat().st_size
    path.write_bytes(path.read_bytes()[:-1])
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [("changed", "freeze_frames.parquet")]
    assert report.problems[0].detail == f"{size - 1} bytes, manifest says {size}"


def test_same_size_rewrite_reported_changed(snap: tuple[Path, Path]) -> None:
    """A file replaced by other bytes of the same size (not a flip) is still caught."""
    manifest, data = snap
    path = data / "events.parquet"
    path.write_bytes(b"\x00" * path.stat().st_size)
    assert _kinds(snapshot.verify_snapshot(manifest)) == [("changed", "events.parquet")]


@pytest.fixture(scope="module")
def pristine(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("pristine")
    sc.write(root)
    return sc.manifest_path(root), sc.data_dir(root)


@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_property_any_byte_flip_reported(pristine: tuple[Path, Path], data: st.DataObject) -> None:
    manifest, folder = pristine
    name = data.draw(st.sampled_from(FILES))
    path = folder / name
    offset = data.draw(st.integers(min_value=0, max_value=path.stat().st_size - 1))
    sc.flip_byte(path, offset)
    try:
        report = snapshot.verify_snapshot(manifest)
        assert _kinds(report) == [("changed", name)]
        assert str(report.problems[0]).startswith(f"changed: {name} (sha256 ")
    finally:
        sc.flip_byte(path, offset)
    assert snapshot.verify_snapshot(manifest).ok


# --- Missing ----------------------------------------------------------------------------------


def test_missing_file_reported(snap: tuple[Path, Path]) -> None:
    manifest, data = snap
    (data / "shots.parquet").unlink()
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [("missing", "shots.parquet")]
    assert report.problems[0].detail == ""
    assert str(report.problems[0]) == "missing: shots.parquet"
    assert report.checked == 2


def test_missing_data_dir_all_missing(snap: tuple[Path, Path]) -> None:
    manifest, data = snap
    shutil.rmtree(data)
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [("missing", f) for f in FILES]
    assert report.checked == 0


def test_listed_name_is_a_directory_reported_missing(snap: tuple[Path, Path]) -> None:
    manifest, data = snap
    (data / "shots.parquet").unlink()
    (data / "shots.parquet").mkdir()
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [("missing", "shots.parquet")]
    assert report.problems[0].detail == "not a regular file"
    assert str(report.problems[0]) == "missing: shots.parquet (not a regular file)"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_listed_name_is_a_fifo_reported_missing(snap: tuple[Path, Path]) -> None:
    """Hashing a FIFO would block forever: it must be rejected by ``S_ISREG`` first."""
    manifest, data = snap
    (data / "events.parquet").unlink()
    os.mkfifo(data / "events.parquet")
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [("missing", "events.parquet")]
    assert report.problems[0].detail == "not a regular file"


def test_symlinked_file_with_same_bytes_is_ok(snap: tuple[Path, Path], tmp_path: Path) -> None:
    """Symlinks are followed (spec §3.2 step 2): a link to identical bytes verifies."""
    manifest, data = snap
    target = tmp_path / "elsewhere.parquet"
    shutil.move(data / "shots.parquet", target)
    (data / "shots.parquet").symlink_to(target)
    assert snapshot.verify_snapshot(manifest).ok


def test_dangling_symlink_reported_missing(snap: tuple[Path, Path], tmp_path: Path) -> None:
    manifest, data = snap
    (data / "shots.parquet").unlink()
    (data / "shots.parquet").symlink_to(tmp_path / "nowhere.parquet")
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [("missing", "shots.parquet")]


# --- Extra ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "shown", "is_dir"),
    [
        ("extra.parquet", "extra.parquet", False),
        ("notes.txt", "notes.txt", False),
        (".hidden.parquet", ".hidden.parquet", False),
        ("sub", "sub/", True),
        (".demo.staging", ".demo.staging/", True),
        ("._shots.parquet", "._shots.parquet", False),
    ],
)
def test_extra_file_reported(snap: tuple[Path, Path], entry: str, shown: str, is_dir: bool) -> None:
    manifest, data = snap
    if is_dir:
        (data / entry).mkdir()
        (data / entry / "x.parquet").write_bytes(b"x")
    else:
        (data / entry).write_bytes(b"x")
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [("extra", shown)]
    assert str(report.problems[0]) == f"extra: {shown}"
    assert report.checked == 3


def test_ds_store_ignored(snap: tuple[Path, Path]) -> None:
    manifest, data = snap
    (data / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    report = snapshot.verify_snapshot(manifest)
    assert report.ok
    assert frozenset({".DS_Store"}) == snapshot.IGNORED_NAMES


def test_ds_store_is_case_sensitive_name(snap: tuple[Path, Path]) -> None:
    """Only the exact name is ignored; a look-alike is reported (spec §3.2 step 3)."""
    manifest, data = snap
    (data / ".ds_store_backup").write_bytes(b"x")
    assert _kinds(snapshot.verify_snapshot(manifest)) == [("extra", ".ds_store_backup")]


# --- Several problems, overrides --------------------------------------------------------------


def test_all_problems_reported_sorted(snap: tuple[Path, Path]) -> None:
    manifest, data = snap
    sc.flip_byte(data / "shots.parquet", 10)
    (data / "freeze_frames.parquet").unlink()
    (data / "extra.parquet").write_bytes(b"x")
    (data / "aaa.txt").write_bytes(b"x")
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [
        ("extra", "aaa.txt"),
        ("extra", "extra.parquet"),
        ("missing", "freeze_frames.parquet"),
        ("changed", "shots.parquet"),
    ]
    assert list(report.problems) == sorted(report.problems, key=lambda p: (p.path, p.kind))


def test_data_dir_override(snap: tuple[Path, Path], tmp_path: Path) -> None:
    manifest, data = snap
    moved = tmp_path / "moved" / "demo"
    moved.parent.mkdir()
    shutil.move(data, moved)
    assert _kinds(snapshot.verify_snapshot(manifest)) == [("missing", f) for f in FILES]
    report = snapshot.verify_snapshot(manifest, data_dir=moved)
    assert report.ok
    assert Path(report.data_dir) == moved
    assert snapshot.verify_snapshot(str(manifest), data_dir=str(moved)).ok


def test_default_data_dir_resolves_against_manifest(snap: tuple[Path, Path]) -> None:
    manifest, data = snap
    report = snapshot.verify_snapshot(manifest)
    assert os.path.normpath(report.data_dir) == os.path.normpath(
        manifest.parent / "../data/snapshots/demo"
    )


def test_size_checked_before_hash(snap: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, data = snap
    real = snapshot._file_sha256

    def guarded(path: Path) -> str:
        if Path(path).name == "shots.parquet":
            raise AssertionError("hashed a file whose size already differs")
        return real(path)

    monkeypatch.setattr(snapshot, "_file_sha256", guarded)
    with (data / "shots.parquet").open("ab") as f:
        f.write(b"\x00")
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [("changed", "shots.parquet")]


def test_missing_manifest_raises_manifest_error(tmp_path: Path) -> None:
    with pytest.raises(snapshot.ManifestError, match="nope.json"):
        snapshot.verify_snapshot(tmp_path / "nope.json")


def test_invalid_manifest_raises_manifest_error(snap: tuple[Path, Path]) -> None:
    manifest, _ = snap
    manifest.write_text("{ not json", encoding="utf-8")
    with pytest.raises(snapshot.ManifestError):
        snapshot.verify_snapshot(manifest)


# --- Unreadable states (spec §3.2; p9) --------------------------------------------------------


@pytest.fixture
def restore_modes() -> Iterator[Callable[[Path, int], None]]:
    """``chmod`` that is undone in a ``finally``, so tmp_path cleanup still works."""
    changed: list[tuple[Path, int]] = []

    def chmod(path: Path, mode: int) -> None:
        changed.append((path, stat.S_IMODE(path.stat().st_mode)))
        path.chmod(mode)

    try:
        yield chmod
    finally:
        for path, mode in reversed(changed):
            path.chmod(mode)


@sc.needs_posix_perms
def test_unreadable_file_reported(
    snap: tuple[Path, Path], restore_modes: Callable[[Path, int], None]
) -> None:
    manifest, data = snap
    restore_modes(data / "shots.parquet", 0o000)
    report = snapshot.verify_snapshot(manifest)
    assert _kinds(report) == [("unreadable", "shots.parquet")]
    assert report.problems[0].detail == "PermissionError: Permission denied"
    assert (
        str(report.problems[0]) == "unreadable: shots.parquet (PermissionError: Permission denied)"
    )


def test_data_dir_is_a_file(snap: tuple[Path, Path]) -> None:
    manifest, data = snap
    shutil.rmtree(data)
    data.write_bytes(b"not a directory")
    report = snapshot.verify_snapshot(manifest)
    shown = str(report.data_dir) + "/"
    assert sorted(_kinds(report)) == sorted(
        [("unreadable", shown), *[("missing", f) for f in FILES]]
    )
    (dir_problem,) = [p for p in report.problems if p.kind == "unreadable"]
    assert dir_problem.detail == "not a directory"
    assert dir_problem.path.endswith("data/snapshots/demo/")
    assert str(dir_problem) == f"unreadable: {shown} (not a directory)"


@sc.needs_posix_perms
def test_unreadable_data_dir_reported(
    snap: tuple[Path, Path], restore_modes: Callable[[Path, int], None]
) -> None:
    """Mode 0o300: no listing, but each file can still be stat'ed and hashed (p9 case 9)."""
    manifest, data = snap
    (data / "extra.parquet").write_bytes(b"x")  # cannot be seen without a listing
    restore_modes(data, 0o300)
    report = snapshot.verify_snapshot(manifest)
    shown = str(report.data_dir) + "/"
    assert _kinds(report) == [("unreadable", shown)]
    assert report.problems[0].detail.startswith("cannot list: PermissionError")
    assert report.checked == 3


@sc.needs_posix_perms
def test_unreadable_data_dir_still_finds_changed_file(
    snap: tuple[Path, Path], restore_modes: Callable[[Path, int], None]
) -> None:
    manifest, data = snap
    sc.flip_byte(data / "shots.parquet", 0)
    restore_modes(data, 0o300)
    report = snapshot.verify_snapshot(manifest)
    kinds = _kinds(report)
    assert ("changed", "shots.parquet") in kinds
    assert ("unreadable", str(report.data_dir) + "/") in kinds


@sc.needs_posix_perms
@pytest.mark.parametrize("with_extra", [False, True])
def test_unsearchable_data_dir_reported(
    snap: tuple[Path, Path], restore_modes: Callable[[Path, int], None], with_extra: bool
) -> None:
    """Mode 0o600: the listing works, every ``stat`` of a child fails (p9 case 9)."""
    manifest, data = snap
    if with_extra:
        (data / "notes.txt").write_bytes(b"x")
    restore_modes(data, 0o600)
    report = snapshot.verify_snapshot(manifest)
    expected = [("unreadable", f) for f in FILES]
    if with_extra:
        expected = sorted([*expected, ("extra", "notes.txt")], key=lambda kp: (kp[1], kp[0]))
    assert _kinds(report) == expected
    for problem in report.problems:
        if problem.kind == "unreadable":
            assert problem.detail == "PermissionError: Permission denied"
    assert report.checked == 0


@sc.needs_posix_perms
def test_unsearchable_parent_reported(
    snap: tuple[Path, Path], restore_modes: Callable[[Path, int], None]
) -> None:
    """``stat(data_dir)`` itself fails: one ``cannot stat`` problem, every file unreadable."""
    manifest, data = snap
    restore_modes(data.parent, 0o600)
    report = snapshot.verify_snapshot(manifest)
    shown = str(report.data_dir) + "/"
    kinds = _kinds(report)
    assert ("unreadable", shown) in kinds
    (dir_problem,) = [p for p in report.problems if p.path == shown]
    assert dir_problem.detail.startswith("cannot stat: PermissionError")
    assert sorted(k for k in kinds if k[1] != shown) == [("unreadable", f) for f in FILES]


def _state_flip(data: Path) -> None:
    sc.flip_byte(data / "shots.parquet", 5)


def _state_missing(data: Path) -> None:
    (data / "shots.parquet").unlink()


def _state_extra(data: Path) -> None:
    (data / "x.bin").write_bytes(b"x")


def _state_dir_as_file(data: Path) -> None:
    (data / "shots.parquet").unlink()
    (data / "shots.parquet").mkdir()


def _state_data_dir_file(data: Path) -> None:
    shutil.rmtree(data)
    data.write_bytes(b"x")


def _state_data_dir_missing(data: Path) -> None:
    shutil.rmtree(data)


def _state_data_dir_dangling_link(data: Path) -> None:
    shutil.rmtree(data)
    data.symlink_to(data.parent / "nowhere")


def _state_empty_file(data: Path) -> None:
    (data / "shots.parquet").write_bytes(b"")


STATES: dict[str, Callable[[Path], None]] = {
    "flip": _state_flip,
    "missing": _state_missing,
    "extra": _state_extra,
    "dir_as_file": _state_dir_as_file,
    "data_dir_file": _state_data_dir_file,
    "data_dir_missing": _state_data_dir_missing,
    "data_dir_dangling_link": _state_data_dir_dangling_link,
    "empty_file": _state_empty_file,
}
PERM_STATES = {"unreadable_file": 0o000, "unreadable_dir": 0o300, "unsearchable_dir": 0o600}


@pytest.mark.parametrize("state", sorted(STATES))
def test_verify_never_raises_on_data_states(snap: tuple[Path, Path], state: str) -> None:
    manifest, data = snap
    STATES[state](data)
    report = snapshot.verify_snapshot(manifest)
    assert isinstance(report, snapshot.VerifyReport)
    assert not report.ok


@sc.needs_posix_perms
@pytest.mark.parametrize("state", sorted(PERM_STATES))
def test_verify_never_raises_on_permission_states(
    snap: tuple[Path, Path], restore_modes: Callable[[Path, int], None], state: str
) -> None:
    manifest, data = snap
    target = data / "shots.parquet" if state == "unreadable_file" else data
    restore_modes(target, PERM_STATES[state])
    report = snapshot.verify_snapshot(manifest)
    assert isinstance(report, snapshot.VerifyReport)
    assert not report.ok


# --- Problem / report values ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("problem", "text"),
    [
        (snapshot.Problem("missing", "a.parquet", ""), "missing: a.parquet"),
        (snapshot.Problem("extra", "sub/", ""), "extra: sub/"),
        (
            snapshot.Problem("changed", "a.parquet", "1 bytes, manifest says 2"),
            "changed: a.parquet (1 bytes, manifest says 2)",
        ),  # noqa: E501
    ],
)
def test_problem_str(problem: snapshot.Problem, text: str) -> None:
    assert str(problem) == text


def test_report_ok_is_not_problems(tmp_path: Path) -> None:
    clean = snapshot.VerifyReport("demo", tmp_path / "m.json", tmp_path, (), 0)
    dirty = snapshot.VerifyReport(
        "demo", tmp_path / "m.json", tmp_path, (snapshot.Problem("extra", "x", ""),), 0
    )
    assert clean.ok is True
    assert dirty.ok is False
