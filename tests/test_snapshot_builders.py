"""Builders (the C4/C5 extension point), the ``parquet`` builder and the ``Source``
constructors (C6 spec §1.2, §2.3, §4, §5)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

import polars as pl
import pytest
import snapshot_cases as sc
from polars.testing import assert_frame_equal

from football_core.io import snapshot


def _toy(request: snapshot.BuildRequest) -> snapshot.BuildResult:
    return snapshot.BuildResult({"events": sc.generic_frame()}, sc.git_source())


def _toy2(request: snapshot.BuildRequest) -> snapshot.BuildResult:
    return snapshot.BuildResult({"lookup": sc.lookup_frame()}, sc.git_source())


# --- Registry ---------------------------------------------------------------------------------


def test_builtin_parquet_listed() -> None:
    assert snapshot.BUILTIN_BUILDERS["parquet"] == "football_core.io.snapshot:build_from_parquet"
    assert "parquet" in snapshot.builder_names()
    assert snapshot.get_builder("parquet") is snapshot.build_from_parquet
    with pytest.raises(TypeError):
        snapshot.BUILTIN_BUILDERS["x"] = "a:b"  # type: ignore[index]


def test_register_and_get() -> None:
    snapshot.register_builder("toy", _toy)
    assert snapshot.get_builder("toy") is _toy
    assert snapshot.builder_names() == ("parquet", "statsbomb", "toy")
    snapshot.register_builder("a-first", _toy2)
    assert snapshot.builder_names() == ("a-first", "parquet", "statsbomb", "toy")


def test_registry_is_isolated_between_tests() -> None:
    """The autouse fixture (conftest) gives every test a fresh registry."""
    assert snapshot.builder_names() == ("parquet", "statsbomb")


def test_register_duplicate_rejected() -> None:
    snapshot.register_builder("toy", _toy)
    with pytest.raises(ValueError, match="toy"):
        snapshot.register_builder("toy", _toy2)
    assert snapshot.get_builder("toy") is _toy


def test_register_replace() -> None:
    snapshot.register_builder("toy", _toy)
    snapshot.register_builder("toy", _toy2, replace=True)
    assert snapshot.get_builder("toy") is _toy2


def test_register_replace_builtin() -> None:
    snapshot.register_builder("parquet", _toy, replace=True)
    assert snapshot.get_builder("parquet") is _toy
    assert snapshot.builder_names() == ("parquet", "statsbomb")


@pytest.mark.parametrize("name", ["Toy", "", "a b", "toy\n", "-toy", "a" * 65, "tö"])
def test_register_rejects_bad_name(name: str) -> None:
    with pytest.raises(ValueError):
        snapshot.register_builder(name, _toy)
    assert snapshot.builder_names() == ("parquet", "statsbomb")


def test_register_rejects() -> None:
    with pytest.raises(TypeError):
        snapshot.register_builder("toy", 3)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        snapshot.register_builder("toy", "football_core.io.snapshot:build_from_parquet")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parquet"):
        snapshot.register_builder("parquet", _toy)
    assert snapshot.get_builder("parquet") is snapshot.build_from_parquet
    assert snapshot.builder_names() == ("parquet", "statsbomb")


def test_get_builder_non_callable_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        snapshot,
        "BUILTIN_BUILDERS",
        MappingProxyType(
            {
                "parquet": "football_core.io.snapshot:build_from_parquet",
                "bad": "football_core.io.snapshot:NAME_PATTERN",
            }
        ),
    )
    with pytest.raises(TypeError, match="bad|NAME_PATTERN"):
        snapshot.get_builder("bad")


def test_get_unknown_lists_names() -> None:
    snapshot.register_builder("toy", _toy)
    with pytest.raises(ValueError) as info:
        snapshot.get_builder("nope")
    assert "nope" in str(info.value)
    assert "parquet" in str(info.value)
    assert "toy" in str(info.value)


def test_builder_wrong_return_type(tmp_path: Path) -> None:
    snapshot.register_builder("toy", lambda request: {"events": sc.generic_frame()})  # type: ignore[arg-type, return-value]
    with pytest.raises(TypeError, match="toy"):
        snapshot.snapshot_from_builder("demo", "toy", root=tmp_path)
    assert sc.tree(tmp_path) == set()


def test_snapshot_from_builder_passes_request_and_sort_by(tmp_path: Path) -> None:
    seen: list[snapshot.BuildRequest] = []

    def builder(request: snapshot.BuildRequest) -> snapshot.BuildResult:
        seen.append(request)
        return snapshot.BuildResult(
            {"events": sc.generic_frame()}, sc.git_source(), sort_by={"events": ("minute",)}
        )

    snapshot.register_builder("toy", builder)
    result = snapshot.snapshot_from_builder(
        "demo",
        "toy",
        inputs=["a/b.json", tmp_path / "c"],
        params={"season": "2015/2016"},
        root=tmp_path,
    )
    (request,) = seen
    assert request.inputs == (Path("a/b.json"), tmp_path / "c")
    assert all(isinstance(p, Path) for p in request.inputs)
    assert dict(request.params) == {"season": "2015/2016"}
    assert result.status == "created"
    assert result.manifest.builder == "toy"
    (entry,) = result.manifest.tables
    assert entry.sort_by == ("minute",)
    m = sc.manifest_json(tmp_path)
    assert "2015/2016" not in json.dumps(m)  # parameter values never reach the manifest


def test_snapshot_from_builder_overwrite(tmp_path: Path) -> None:
    snapshot.register_builder("toy", _toy)
    snapshot.snapshot_from_builder("demo", "toy", root=tmp_path)
    snapshot.register_builder("toy", _toy2, replace=True)
    with pytest.raises(snapshot.SnapshotExistsError):
        snapshot.snapshot_from_builder("demo", "toy", root=tmp_path)
    result = snapshot.snapshot_from_builder("demo", "toy", root=tmp_path, overwrite=True)
    assert result.status == "replaced"


def test_build_request_defaults() -> None:
    request = snapshot.BuildRequest()
    assert request.inputs == ()
    assert dict(request.params) == {}
    result = snapshot.BuildResult({"t": sc.lookup_frame()}, sc.git_source())
    assert dict(result.sort_by) == {}


# --- The parquet builder ----------------------------------------------------------------------


def _parquet(path: Path, frame: pl.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def test_parquet_builder_tables_from_stems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _parquet(tmp_path / "in" / "events.parquet", sc.generic_frame())
    b = _parquet(tmp_path / "in" / "sub" / "lookup.parquet", sc.lookup_frame())
    monkeypatch.chdir(tmp_path / "in")
    result = snapshot.build_from_parquet(
        snapshot.BuildRequest((Path("events.parquet"), b))  # relative + absolute
    )
    assert set(result.tables) == {"events", "lookup"}
    assert_frame_equal(result.tables["events"], sc.generic_frame())
    assert_frame_equal(result.tables["lookup"], sc.lookup_frame())
    source = result.source
    assert source.kind == "files"
    assert [f.path for f in source.files] == ["events.parquet", "sub/lookup.parquet"]
    assert source.files[0].sha256 == sc.sha256_file(a)
    assert source.files[0].bytes == a.stat().st_size
    assert source.revision == sc.sha256sum_digest(
        [("events.parquet", sc.sha256_file(a)), ("sub/lookup.parquet", sc.sha256_file(b))]
    )
    assert dict(result.sort_by) == {}


def test_parquet_builder_directory_input(tmp_path: Path) -> None:
    folder = tmp_path / "exports"
    _parquet(folder / "lookup.parquet", sc.lookup_frame())
    _parquet(folder / "events.parquet", sc.generic_frame())
    (folder / "notes.txt").write_text("not a table")
    _parquet(folder / "deeper" / "hidden.parquet", sc.lookup_frame())  # not recursive
    result = snapshot.build_from_parquet(snapshot.BuildRequest((folder,)))
    assert sorted(result.tables) == ["events", "lookup"]
    assert [f.path for f in result.source.files] == ["events.parquet", "lookup.parquet"]
    assert result.source.uri == "exports"


def test_parquet_builder_end_to_end(tmp_path: Path) -> None:
    _parquet(tmp_path / "in" / "events.parquet", sc.generic_frame())
    result = snapshot.snapshot_from_builder(
        "demo", "parquet", inputs=[tmp_path / "in"], root=tmp_path / "out"
    )
    assert result.status == "created"
    assert result.manifest.builder == "parquet"
    assert result.manifest.source.kind == "files"
    assert snapshot.verify_snapshot(result.manifest_path).ok
    again = snapshot.snapshot_from_builder(
        "demo", "parquet", inputs=[tmp_path / "in"], root=tmp_path / "out"
    )
    assert again.status == "unchanged"


def test_parquet_builder_canonical_inputs_validated(tmp_path: Path) -> None:
    import schema_cases

    from football_core import schema

    bad = schema_cases.with_value(schema_cases.shots_frame(), "is_goal", 0, False)
    _parquet(tmp_path / "in" / "shots.parquet", bad)
    with pytest.raises(schema.SchemaValidationError):
        snapshot.snapshot_from_builder(
            "demo", "parquet", inputs=[tmp_path / "in"], root=tmp_path / "out"
        )
    assert sc.tree(tmp_path / "out") == set()


def _rejects_duplicate_stem(tmp_path: Path) -> snapshot.BuildRequest:
    a = _parquet(tmp_path / "x" / "events.parquet", sc.generic_frame())
    b = _parquet(tmp_path / "y" / "events.parquet", sc.generic_frame())
    return snapshot.BuildRequest((a, b))


def _rejects_same_file_twice(tmp_path: Path) -> snapshot.BuildRequest:
    a = _parquet(tmp_path / "x" / "events.parquet", sc.generic_frame())
    return snapshot.BuildRequest((a, tmp_path / "x"))


def _rejects_csv(tmp_path: Path) -> snapshot.BuildRequest:
    (tmp_path / "events.csv").write_text("a\n1\n")
    return snapshot.BuildRequest((tmp_path / "events.csv",))


def _rejects_empty_dir(tmp_path: Path) -> snapshot.BuildRequest:
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("x")
    return snapshot.BuildRequest((tmp_path / "empty",))


def _rejects_param(tmp_path: Path) -> snapshot.BuildRequest:
    a = _parquet(tmp_path / "events.parquet", sc.generic_frame())
    return snapshot.BuildRequest((a,), {"season": "2015"})


def _rejects_bad_stem(tmp_path: Path) -> snapshot.BuildRequest:
    a = _parquet(tmp_path / "Events.parquet", sc.generic_frame())
    return snapshot.BuildRequest((a,))


def _rejects_dash_stem(tmp_path: Path) -> snapshot.BuildRequest:
    a = _parquet(tmp_path / "match-events.parquet", sc.generic_frame())
    return snapshot.BuildRequest((a,))


def _rejects_no_inputs(tmp_path: Path) -> snapshot.BuildRequest:
    return snapshot.BuildRequest(())


PARQUET_REJECTS = {
    "duplicate_stem": _rejects_duplicate_stem,
    "same_file_twice": _rejects_same_file_twice,
    "csv": _rejects_csv,
    "empty_dir": _rejects_empty_dir,
    "param": _rejects_param,
    "bad_stem": _rejects_bad_stem,
    "dash_stem": _rejects_dash_stem,
    "no_inputs": _rejects_no_inputs,
}


@pytest.mark.parametrize("case", sorted(PARQUET_REJECTS))
def test_parquet_builder_rejects(tmp_path: Path, case: str) -> None:
    request = PARQUET_REJECTS[case](tmp_path)
    with pytest.raises(ValueError):
        snapshot.build_from_parquet(request)


def test_parquet_builder_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        snapshot.build_from_parquet(snapshot.BuildRequest((tmp_path / "nope.parquet",)))


# --- source_from_files ------------------------------------------------------------------------


def _files(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "exports"
    (root / "b").mkdir(parents=True)
    a = root / "a.json"
    a.write_bytes(b'{"a": 1}')
    b = root / "b" / "c.json"
    b.write_bytes(b"[]")
    return root, a, b


def test_source_from_files_digest(tmp_path: Path) -> None:
    root, a, b = _files(tmp_path)
    source = snapshot.source_from_files([b, a], root=root)
    sha_a = hashlib.sha256(b'{"a": 1}').hexdigest()
    sha_b = hashlib.sha256(b"[]").hexdigest()
    assert source.kind == "files"
    assert source.uri == "exports"
    assert source.files == (
        snapshot.InputFile("a.json", sha_a, 8),
        snapshot.InputFile("b/c.json", sha_b, 2),
    )
    lines = f"{sha_a}  a.json\n{sha_b}  b/c.json\n"
    assert source.revision == hashlib.sha256(lines.encode()).hexdigest()
    assert source.content_sha256 is None
    explicit = snapshot.source_from_files([a, b], root=str(root), uri="https://example.invalid/x")
    assert explicit.uri == "https://example.invalid/x"
    assert explicit.revision == source.revision


def test_source_from_files_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, a, b = _files(tmp_path)
    monkeypatch.chdir(tmp_path)
    relative = snapshot.source_from_files(["exports/a.json", "exports/b/c.json"], root="exports")
    assert relative == snapshot.source_from_files([a, b], root=root)


def _bad_paths(tmp_path: Path) -> dict[str, tuple[list[Path], Path]]:
    root, a, b = _files(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"x")
    newline = root / "a\nb.json"
    newline.write_bytes(b"x")
    backslash = root / "a\\b.json"
    backslash.write_bytes(b"x")
    return {
        "no_paths": ([], root),
        "outside_root": ([a, outside], root),
        "directory": ([root / "b"], root),
        "duplicate": ([a, a], root),
        "duplicate_spelling": ([a, root / "b" / ".." / "a.json"], root),
        "newline_name": ([newline], root),
        "backslash_name": ([backslash], root),
        "root_itself_is_file": ([a], a),
    }


@pytest.mark.parametrize(
    "case",
    [
        "no_paths",
        "outside_root",
        "directory",
        "duplicate",
        "duplicate_spelling",
        "newline_name",
        "backslash_name",
        "root_itself_is_file",
    ],
)
def test_source_from_files_rejects(tmp_path: Path, case: str) -> None:
    paths, root = _bad_paths(tmp_path)[case]
    with pytest.raises(ValueError):
        snapshot.source_from_files(paths, root=root)


def test_source_from_files_missing_path(tmp_path: Path) -> None:
    root, a, _ = _files(tmp_path)
    with pytest.raises(FileNotFoundError):
        snapshot.source_from_files([a, root / "missing.json"], root=root)


@pytest.mark.parametrize(
    ("path", "sha", "size"),
    [
        ("/a.json", "a" * 64, 1),
        ("a/./b", "a" * 64, 1),
        ("a/../b", "a" * 64, 1),
        ("a//b", "a" * 64, 1),
        ("a/", "a" * 64, 1),
        (".", "a" * 64, 1),
        ("..", "a" * 64, 1),
        ("", "a" * 64, 1),
        ("a\rb", "a" * 64, 1),
        ("a\nb", "a" * 64, 1),
        ("a\\b", "a" * 64, 1),
        ("a.json", "A" * 64, 1),
        ("a.json", "a" * 63, 1),
        ("a.json", "a" * 65, 1),
        ("a.json", "a" * 64, True),
        ("a.json", "a" * 64, -1),
        ("a.json", "a" * 64, 1.0),
    ],
    ids=[
        "leading_slash",
        "dot_segment",
        "dotdot_segment",
        "empty_segment",
        "trailing_slash",
        "dot",
        "dotdot",
        "empty",
        "cr",
        "lf",
        "backslash",
        "upper_sha",
        "short_sha",
        "long_sha",
        "bool_bytes",
        "negative_bytes",
        "float_bytes",
    ],
)
def test_input_file_rejects(path: str, sha: str, size: object) -> None:
    with pytest.raises(ValueError):
        snapshot.InputFile(path, sha, size)  # type: ignore[arg-type]


def test_input_file_accepts() -> None:
    entry = snapshot.InputFile("data/2024 season/a.b.json", "0" * 64, 0)
    assert entry.bytes == 0


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlinks")
def test_source_from_files_symlinked_root(tmp_path: Path) -> None:
    root, a, b = _files(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    plain = snapshot.source_from_files([a, b], root=root, uri="u")
    through_root = snapshot.source_from_files([a, b], root=link, uri="u")
    through_files = snapshot.source_from_files(
        [link / "a.json", link / "b" / "c.json"], root=root, uri="u"
    )  # noqa: E501
    assert plain == through_root == through_files


def test_list_files_false_keeps_revision(tmp_path: Path) -> None:
    root, a, b = _files(tmp_path)
    listed = snapshot.source_from_files([a, b], root=root)
    digest_only = snapshot.source_from_files([a, b], root=root, list_files=False)
    assert digest_only.files == ()
    assert digest_only.revision == listed.revision
    assert digest_only.kind == "files"


# --- Source rules -----------------------------------------------------------------------------


def _digest_of(*files: snapshot.InputFile) -> str:
    return sc.sha256sum_digest((f.path, f.sha256) for f in files)


def _source_cases() -> dict[str, dict[str, object]]:
    f1 = snapshot.InputFile("a.json", "1" * 64, 1)
    f2 = snapshot.InputFile("b.json", "2" * 64, 2)
    return {
        "git_short_sha": {"kind": "git", "uri": "u", "revision": "abc"},
        "git_upper_sha": {"kind": "git", "uri": "u", "revision": "A" * 40},
        "git_41_chars": {"kind": "git", "uri": "u", "revision": "a" * 41},
        "git_sha_newline": {"kind": "git", "uri": "u", "revision": "a" * 40 + "\n"},
        "files_40_chars": {"kind": "files", "uri": "u", "revision": "a" * 40},
        "sql_40_chars": {"kind": "sql", "uri": "u", "revision": "a" * 40},
        "unknown_kind": {"kind": "svn", "uri": "u", "revision": "a" * 40},
        "files_on_git": {"kind": "git", "uri": "u", "revision": "a" * 40, "files": (f1,)},
        "files_on_sql": {"kind": "sql", "uri": "u", "revision": "a" * 64, "files": (f1,)},
        "content_on_git": {
            "kind": "git", "uri": "u", "revision": "a" * 40, "content_sha256": "b" * 64,
        },
        "content_on_files": {
            "kind": "files", "uri": "u", "revision": "a" * 64, "content_sha256": "b" * 64,
        },
        "content_upper": {
            "kind": "sql", "uri": "u", "revision": "a" * 64, "content_sha256": "B" * 64,
        },
        "content_short": {
            "kind": "sql", "uri": "u", "revision": "a" * 64, "content_sha256": "b" * 63,
        },
        "uri_empty": {"kind": "git", "uri": "", "revision": "a" * 40},
        "uri_newline": {"kind": "git", "uri": "a\nb", "revision": "a" * 40},
        "uri_cr": {"kind": "git", "uri": "a\rb", "revision": "a" * 40},
        "revision_not_digest": {
            "kind": "files", "uri": "u", "revision": "a" * 64, "files": (f1, f2),
        },
        "files_unsorted": {
            "kind": "files", "uri": "u", "revision": _digest_of(f1, f2), "files": (f2, f1),
        },
        "files_duplicate": {
            "kind": "files", "uri": "u", "revision": _digest_of(f1, f1), "files": (f1, f1),
        },
    }  # fmt: skip


@pytest.mark.parametrize(
    "case",
    [
        "git_short_sha",
        "git_upper_sha",
        "git_41_chars",
        "git_sha_newline",
        "files_40_chars",
        "sql_40_chars",
        "unknown_kind",
        "files_on_git",
        "files_on_sql",
        "content_on_git",
        "content_on_files",
        "content_upper",
        "content_short",
        "uri_empty",
        "uri_newline",
        "uri_cr",
        "revision_not_digest",
        "files_unsorted",
        "files_duplicate",
    ],
)
def test_source_validation(case: str) -> None:
    with pytest.raises(ValueError):
        snapshot.Source(**_source_cases()[case])  # type: ignore[arg-type]


def test_source_accepts() -> None:
    f1 = snapshot.InputFile("a.json", "1" * 64, 1)
    snapshot.Source("git", "u", "a" * 40)
    snapshot.Source("git", "u", "a" * 64)
    snapshot.Source("files", "u", "a" * 64)  # list_files=False: nothing to recompute
    snapshot.Source("files", "u", _digest_of(f1), files=(f1,))
    snapshot.Source("sql", "u", "a" * 64, content_sha256="b" * 64)
    snapshot.Source("sql", "u", "a" * 64)


def test_source_from_git() -> None:
    source = snapshot.source_from_git("https://example.invalid/open-data", sc.GIT_SHA)
    assert source == snapshot.Source("git", "https://example.invalid/open-data", sc.GIT_SHA)
    assert source.files == ()
    assert source.content_sha256 is None
    with pytest.raises(ValueError):
        snapshot.source_from_git("u", "not-a-sha")


def test_source_from_sql() -> None:
    sql = "select * from raw.raw_match_json where season = '2024'"
    source = snapshot.source_from_sql(sql, uri="db:raw.raw_match_json")
    assert source.kind == "sql"
    assert source.uri == "db:raw.raw_match_json"
    assert source.revision == hashlib.sha256(sql.encode("utf-8")).hexdigest()
    assert source.content_sha256 is None
    assert source.files == ()
    spaced = snapshot.source_from_sql(sql + " ", uri="db:raw.raw_match_json")
    assert spaced.revision != source.revision  # no normalisation
    pinned = snapshot.source_from_sql(sql, uri="db:x", content_sha256="c" * 64)
    assert pinned.content_sha256 == "c" * 64
    unicode_sql = snapshot.source_from_sql("select 'Beşiktaş'", uri="db:x")
    assert unicode_sql.revision == hashlib.sha256("select 'Beşiktaş'".encode()).hexdigest()
    with pytest.raises(ValueError):
        snapshot.source_from_sql("", uri="db:x")
    with pytest.raises(ValueError):
        snapshot.source_from_sql(sql, uri="db:x", content_sha256="nothex")


def test_sql_source_content_sha256_round_trip(tmp_path: Path) -> None:
    content = snapshot.content_digest({"ws:1": b'{"a": 1}', "ws:2": b"{}"})
    source = snapshot.source_from_sql("select 1", uri="db:raw", content_sha256=content)
    result = sc.write(tmp_path, {"events": sc.generic_frame()}, source=source)
    m = sc.manifest_json(tmp_path)
    assert m["source"] == {
        "kind": "sql",
        "uri": "db:raw",
        "revision": hashlib.sha256(b"select 1").hexdigest(),
        "content_sha256": content,
        "files": [],
    }
    parsed = snapshot.Manifest.from_json(result.manifest.to_json())
    assert parsed.source == source


def test_files_source_round_trip(tmp_path: Path) -> None:
    root, a, b = _files(tmp_path)
    source = snapshot.source_from_files([a, b], root=root)
    result = sc.write(tmp_path / "out", {"events": sc.generic_frame()}, source=source)
    assert snapshot.read_manifest(result.manifest_path).source == source
    assert sc.manifest_json(tmp_path / "out")["source"]["files"] == [
        {"path": "a.json", "sha256": hashlib.sha256(b'{"a": 1}').hexdigest(), "bytes": 8},
        {"path": "b/c.json", "sha256": hashlib.sha256(b"[]").hexdigest(), "bytes": 2},
    ]


def test_content_digest() -> None:
    items = {"ws:2": b"{}", "ws:1": b'{"a": 1}', "sb:9": b""}
    expected_lines = "".join(
        f"{hashlib.sha256(items[key]).hexdigest()}  {key}\n" for key in sorted(items)
    )
    expected = hashlib.sha256(expected_lines.encode("utf-8")).hexdigest()
    assert snapshot.content_digest(items) == expected
    assert snapshot.content_digest(dict(reversed(list(items.items())))) == expected
    assert snapshot.content_digest({}) == hashlib.sha256(b"").hexdigest()
    assert snapshot.content_digest({"a": b"1"}) != snapshot.content_digest({"a": b"2"})
    for bad in ["", "a\nb", "a\rb", "\n"]:
        with pytest.raises(ValueError):
            snapshot.content_digest({bad: b"x"})


def test_digest_helper_matches_sha256sum_format() -> None:
    pairs = [("b.parquet", "2" * 64), ("a.parquet", "1" * 64), ("shots_extra.parquet", "3" * 64)]
    assert snapshot._digest(pairs) == sc.sha256sum_digest(pairs)
    assert snapshot._digest([]) == hashlib.sha256(b"").hexdigest()


def test_file_sha256_helper(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"\x00" * 100_000 + b"tail")
    assert snapshot._file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


# --- Build decisions (build.md "Architect: ratify" items 11-14) ------------------------------


@pytest.mark.parametrize("single", ["str", "path"])
def test_source_from_files_single_path_rejected(tmp_path: Path, single: str) -> None:
    """Item 11: a single path is not split into one-character paths."""
    root, a, _ = _files(tmp_path)
    paths = str(a) if single == "str" else a
    with pytest.raises(TypeError, match="single path"):
        snapshot.source_from_files(paths, root=root)  # type: ignore[arg-type]


@pytest.mark.parametrize("single", ["str", "path"])
def test_snapshot_from_builder_single_input_rejected(tmp_path: Path, single: str) -> None:
    calls: list[snapshot.BuildRequest] = []

    def recording(request: snapshot.BuildRequest) -> snapshot.BuildResult:
        calls.append(request)
        return _toy(request)

    snapshot.register_builder("toy", recording)
    one = _parquet(tmp_path / "in" / "events.parquet", sc.generic_frame())
    inputs = str(one) if single == "str" else one
    with pytest.raises(TypeError, match="single path"):
        snapshot.snapshot_from_builder(
            "demo",
            "toy",
            inputs=inputs,  # type: ignore[arg-type]
            root=tmp_path / "out",
        )
    assert calls == []
    assert sc.tree(tmp_path / "out") == set()


@pytest.mark.parametrize("name", ["Bad Name", "", "a/b"])
def test_snapshot_from_builder_checks_name_before_building(tmp_path: Path, name: str) -> None:
    """Item 12: a bad snapshot name fails fast, before the (possibly slow) builder runs."""
    calls: list[snapshot.BuildRequest] = []

    def recording(request: snapshot.BuildRequest) -> snapshot.BuildResult:
        calls.append(request)
        return _toy(request)

    snapshot.register_builder("toy", recording)
    with pytest.raises(ValueError, match="snapshot name"):
        snapshot.snapshot_from_builder(name, "toy", root=tmp_path / "out")
    assert calls == []
    assert sc.tree(tmp_path / "out") == set()


def test_source_from_files_missing_root(tmp_path: Path) -> None:
    """Item 13: a missing root is ``FileNotFoundError``; a file as root is ``ValueError``."""
    _, a, _ = _files(tmp_path)
    with pytest.raises(FileNotFoundError, match="source root"):
        snapshot.source_from_files([a], root=tmp_path / "no_such_root")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs os.mkfifo")
def test_parquet_builder_rejects_input_neither_file_nor_directory(tmp_path: Path) -> None:
    """Item 14: an input that exists but is neither a file nor a directory (a FIFO)."""
    fifo = tmp_path / "events.parquet"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="neither a file nor a directory"):
        snapshot.build_from_parquet(snapshot.BuildRequest((fifo,)))
