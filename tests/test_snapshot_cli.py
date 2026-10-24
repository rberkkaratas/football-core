"""The ``fc`` CLI: commands, messages and exit codes (C6 spec §1.3, §3.3, §5)."""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
import schema_cases
import snapshot_cases as sc

from football_core import cli
from football_core.io import snapshot


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    rc = cli.main(list(argv))
    out, err = capsys.readouterr()
    assert isinstance(rc, int)
    return rc, out, err


def _inputs(tmp_path: Path) -> Path:
    folder = tmp_path / "in"
    folder.mkdir(parents=True, exist_ok=True)
    sc.generic_frame().write_parquet(folder / "events.parquet")
    schema_cases.shots_frame().write_parquet(folder / "shots.parquet")
    return folder


def _snapshot(capsys: pytest.CaptureFixture[str], tmp_path: Path, *extra: str) -> int:
    rc, _, err = _run(
        capsys,
        "snapshot",
        "demo",
        "--input",
        str(_inputs(tmp_path)),
        "--root",
        str(tmp_path / "out"),
        *extra,
    )
    return rc


@pytest.fixture
def made(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> Path:
    """A clean snapshot ``demo`` of the ``parquet`` builder under ``tmp_path/out``."""
    assert _snapshot(capsys, tmp_path) == 0
    return sc.manifest_path(tmp_path / "out")


def _data(made: Path) -> Path:
    return sc.data_dir(made.parent.parent)


# --- snapshot ---------------------------------------------------------------------------------

_OK_LINE = re.compile(
    r"^(?P<status>created|unchanged|unchanged \(manifest updated\)|replaced) snapshot 'demo': "
    r"2 table\(s\), 11 row\(s\); data (?P<data>.+); manifest (?P<manifest>.+)$"
)


def test_snapshot_parquet_end_to_end(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    root = tmp_path / "out"
    rc, out, err = _run(
        capsys, "snapshot", "demo", "--input", str(_inputs(tmp_path)), "--root", str(root)
    )
    assert rc == 0, err
    assert err == ""
    (line,) = out.splitlines()
    match = _OK_LINE.match(line)
    assert match, line
    assert match["status"] == "created"
    assert os.path.samefile(match["data"], sc.data_dir(root))
    assert os.path.samefile(match["manifest"], sc.manifest_path(root))
    m = sc.manifest_json(root)
    assert m["builder"] == "parquet"
    assert m["source"]["kind"] == "files"
    rc, out, err = _run(capsys, "verify", str(sc.manifest_path(root)))
    assert rc == 0, err
    assert re.fullmatch(r"ok: snapshot 'demo' \(.+demo\.json\): 2 file\(s\) match\n", out)


def test_snapshot_explicit_builder(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    rc, _, err = _run(
        capsys,
        "snapshot",
        "demo",
        "--builder",
        "parquet",
        "--input",
        str(_inputs(tmp_path) / "events.parquet"),
        "--input",
        str(tmp_path / "in" / "shots.parquet"),
        "--root",
        str(tmp_path / "out"),
    )
    assert rc == 0, err
    assert [t["name"] for t in sc.manifest_json(tmp_path / "out")["tables"]] == ["events", "shots"]


def test_snapshot_default_root_is_cwd(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    (tmp_path / "consumer").mkdir()
    monkeypatch.chdir(tmp_path / "consumer")
    rc, _, err = _run(capsys, "snapshot", "demo", "--input", str(inputs))
    assert rc == 0, err
    assert sc.manifest_path(tmp_path / "consumer").is_file()


def test_snapshot_rerun_prints_unchanged(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _snapshot(capsys, tmp_path) == 0
    rc, out, _ = _run(
        capsys, "snapshot", "demo", "--input", str(tmp_path / "in"), "--root", str(tmp_path / "out")
    )
    assert rc == 0
    match = _OK_LINE.match(out.strip())
    assert match and match["status"] == "unchanged"
    real = snapshot.environment()
    monkeypatch.setattr(snapshot, "environment", lambda: real | {"polars": "9.9.9"})
    rc, out, _ = _run(
        capsys, "snapshot", "demo", "--input", str(tmp_path / "in"), "--root", str(tmp_path / "out")
    )
    assert rc == 0
    match = _OK_LINE.match(out.strip())
    assert match and match["status"] == "unchanged (manifest updated)"


def test_snapshot_conflict_exits_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert _snapshot(capsys, tmp_path) == 0
    sc.generic_frame().head(2).write_parquet(tmp_path / "in" / "events.parquet")
    rc, out, err = _run(
        capsys, "snapshot", "demo", "--input", str(tmp_path / "in"), "--root", str(tmp_path / "out")
    )
    assert rc == 1
    assert out == ""
    assert err.startswith("fc: error: ")
    assert "changed events.parquet" in err
    assert "--overwrite" in err
    assert "Traceback" not in err


def test_snapshot_overwrite_exits_0_replaced(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert _snapshot(capsys, tmp_path) == 0
    sc.generic_frame().head(2).write_parquet(tmp_path / "in" / "events.parquet")
    rc, out, err = _run(
        capsys,
        "snapshot",
        "demo",
        "--input",
        str(tmp_path / "in"),
        "--root",
        str(tmp_path / "out"),
        "--overwrite",
    )
    assert rc == 0, err
    assert out.startswith("replaced snapshot 'demo': 2 table(s), 7 row(s); data ")


def test_snapshot_invalid_shots_exits_1_names_column(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    folder = tmp_path / "in"
    folder.mkdir()
    schema_cases.shots_frame().drop("distance_m").write_parquet(folder / "shots.parquet")
    rc, out, err = _run(
        capsys, "snapshot", "demo", "--input", str(folder), "--root", str(tmp_path / "out")
    )
    assert rc == 1
    assert err.startswith("fc: error: ")
    assert "'distance_m'" in err
    assert sc.tree(tmp_path / "out") == set()


def test_snapshot_bad_name_exits_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    rc, _, err = _run(
        capsys,
        "snapshot",
        "Bad Name",
        "--input",
        str(_inputs(tmp_path)),
        "--root",
        str(tmp_path / "o"),
    )
    assert rc == 1
    assert err.startswith("fc: error: ")


def test_snapshot_unreadable_manifest_exits_1(
    capsys: pytest.CaptureFixture[str], made: Path, tmp_path: Path
) -> None:
    made.write_text("{ broken", encoding="utf-8")
    rc, _, err = _run(
        capsys, "snapshot", "demo", "--input", str(tmp_path / "in"), "--root", str(tmp_path / "out")
    )
    assert rc == 1
    assert err.startswith("fc: error: ")


def test_read_manifest_non_utf8(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = tmp_path / "manifests" / "latin.json"
    path.parent.mkdir()
    path.write_bytes(b'{"name": "\xff\xfe"}')
    with pytest.raises(snapshot.ManifestError, match="latin.json"):
        snapshot.read_manifest(path)
    rc, _, err = _run(capsys, "verify", str(path))
    assert rc == 2
    assert err.startswith("fc: error: ")
    assert "latin.json" in err


def test_snapshot_missing_input_exits_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    rc, _, err = _run(
        capsys,
        "snapshot",
        "demo",
        "--input",
        str(tmp_path / "missing.parquet"),
        "--root",
        str(tmp_path / "out"),
    )
    assert rc == 1
    assert err.startswith("fc: error: ")
    assert "missing.parquet" in err


def test_snapshot_without_input_exits_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """The parquet builder needs inputs: its ValueError is exit 1, not a usage error."""
    rc, _, err = _run(capsys, "snapshot", "demo", "--root", str(tmp_path / "out"))
    assert rc == 1
    assert err.startswith("fc: error: ")


@sc.needs_git
def test_snapshot_unignored_root_exits_1(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    repo = sc.git_repo(tmp_path / "repo", ignore="")
    rc, _, err = _run(
        capsys, "snapshot", "demo", "--input", str(_inputs(tmp_path)), "--root", str(repo)
    )
    assert rc == 1
    assert "would not be git-ignored" in err
    assert not sc.manifest_path(repo).exists()


def test_unknown_builder_exits_2(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    rc, _, err = _run(
        capsys, "snapshot", "demo", "--builder", "nope", "--root", str(tmp_path / "out")
    )
    assert rc == 2
    assert err.startswith("fc: error: ")
    assert "parquet" in err
    assert sc.tree(tmp_path / "out") == set()


def test_builder_value_error_exits_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    def refusing(request: snapshot.BuildRequest) -> snapshot.BuildResult:
        raise ValueError("toy says no to season 1888")

    snapshot.register_builder("toy", refusing)
    rc, _, err = _run(capsys, "snapshot", "demo", "--builder", "toy", "--root", str(tmp_path))
    assert rc == 1
    assert err == "fc: error: toy says no to season 1888\n"


def test_builder_oserror_exits_1(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    def failing(request: snapshot.BuildRequest) -> snapshot.BuildResult:
        raise PermissionError(13, "Permission denied", "/x/raw.json")

    snapshot.register_builder("toy", failing)
    rc, _, err = _run(capsys, "snapshot", "demo", "--builder", "toy", "--root", str(tmp_path))
    assert rc == 1
    assert err.startswith("fc: error: ")


def test_params_passed_to_builder(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    seen: list[dict[str, str]] = []

    def builder(request: snapshot.BuildRequest) -> snapshot.BuildResult:
        seen.append(dict(request.params))
        return snapshot.BuildResult({"events": sc.generic_frame()}, sc.git_source())

    snapshot.register_builder("toy", builder)
    rc, _, err = _run(
        capsys,
        "snapshot",
        "demo",
        "--builder",
        "toy",
        "--param",
        "season=2015/2016",
        "--param",
        "only-360=a=b",
        "--param",
        "empty=",
        "--root",
        str(tmp_path),
    )
    assert rc == 0, err
    assert seen == [{"season": "2015/2016", "only-360": "a=b", "empty": ""}]
    assert "2015/2016" not in sc.manifest_path(tmp_path).read_text(encoding="utf-8")


def test_param_given_to_parquet_builder_exits_1(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rc, _, _ = _run(
        capsys,
        "snapshot",
        "demo",
        "--input",
        str(_inputs(tmp_path)),
        "--param",
        "season=2015",
        "--root",
        str(tmp_path / "out"),
    )
    assert rc == 1


@pytest.mark.parametrize(
    "params",
    [["x"], ["a=1", "a=2"], ["A=1"], ["=1"], ["1a=2"], ["a b=1"], ["_a=1"]],
    ids=[
        "no_equals",
        "duplicate",
        "upper",
        "empty_key",
        "digit_first",
        "space",
        "underscore_first",
    ],
)
def test_bad_param_exits_2(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, params: list[str]
) -> None:
    calls: list[object] = []
    snapshot.register_builder("toy", lambda request: calls.append(request))  # type: ignore[arg-type, func-returns-value]
    argv = ["snapshot", "demo", "--builder", "toy", "--root", str(tmp_path)]
    for p in params:
        argv += ["--param", p]
    rc, _, err = _run(capsys, *argv)
    assert rc == 2
    assert calls == []
    assert err.strip()


def test_no_command_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc, _, err = _run(capsys)
    assert rc == 2
    assert "usage" in err.lower()


def test_unknown_command_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc, _, _ = _run(capsys, "frobnicate")
    assert rc == 2


@pytest.mark.parametrize("argv", [["--help"], ["snapshot", "--help"], ["verify", "-h"]])
def test_help_exits_0(capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    rc, out, _ = _run(capsys, *argv)
    assert rc == 0
    assert "usage: fc" in out


def test_main_never_raises_system_exit(capsys: pytest.CaptureFixture[str]) -> None:
    rc, _, _ = _run(capsys, "verify")  # missing MANIFEST
    assert rc == 2
    rc, _, _ = _run(capsys, "snapshot")  # missing NAME
    assert rc == 2


# --- verify (the acceptance criterion through the CLI) ----------------------------------------


def test_verify_clean_exits_0(capsys: pytest.CaptureFixture[str], made: Path) -> None:
    rc, out, err = _run(capsys, "verify", str(made))
    assert rc == 0
    assert err == ""
    assert out == f"ok: snapshot 'demo' ({made}): 2 file(s) match\n"


def test_verify_one_byte_change_exits_1_names_file(
    capsys: pytest.CaptureFixture[str], made: Path
) -> None:
    sc.flip_byte(_data(made) / "shots.parquet", 100)
    rc, out, err = _run(capsys, "verify", str(made))
    assert rc == 1
    assert out == ""
    lines = err.splitlines()
    assert any(line.startswith("changed: shots.parquet (sha256 ") for line in lines)
    assert "changed: shots.parquet" in err
    assert f"verify failed: snapshot 'demo' ({made}): 1 problem(s)" in lines
    assert "events.parquet" not in err


def test_verify_missing_file_exits_1_names_file(
    capsys: pytest.CaptureFixture[str], made: Path
) -> None:
    (_data(made) / "shots.parquet").unlink()
    rc, _, err = _run(capsys, "verify", str(made))
    assert rc == 1
    assert "missing: shots.parquet" in err.splitlines()
    assert f"verify failed: snapshot 'demo' ({made}): 1 problem(s)" in err


def test_verify_extra_file_exits_1_names_file(
    capsys: pytest.CaptureFixture[str], made: Path
) -> None:
    (_data(made) / "extra.parquet").write_bytes(b"x")
    rc, _, err = _run(capsys, "verify", str(made))
    assert rc == 1
    assert "extra: extra.parquet" in err.splitlines()


def test_verify_lists_every_problem(capsys: pytest.CaptureFixture[str], made: Path) -> None:
    sc.flip_byte(_data(made) / "shots.parquet", 0)
    (_data(made) / "events.parquet").unlink()
    (_data(made) / "notes.txt").write_text("x")
    rc, _, err = _run(capsys, "verify", str(made))
    assert rc == 1
    lines = err.splitlines()
    assert "missing: events.parquet" in lines
    assert "extra: notes.txt" in lines
    assert any(line.startswith("changed: shots.parquet") for line in lines)
    assert f"verify failed: snapshot 'demo' ({made}): 3 problem(s)" in lines


def test_verify_from_other_cwd(
    capsys: pytest.CaptureFixture[str], made: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    relative = os.path.relpath(made, elsewhere)
    rc, out, err = _run(capsys, "verify", relative)
    assert rc == 0, err
    assert out.startswith("ok: snapshot 'demo' (")


def test_verify_data_dir_option(
    capsys: pytest.CaptureFixture[str], made: Path, tmp_path: Path
) -> None:
    moved = tmp_path / "moved"
    _data(made).rename(moved)
    rc, _, _ = _run(capsys, "verify", str(made))
    assert rc == 1
    rc, out, err = _run(capsys, "verify", str(made), "--data-dir", str(moved))
    assert rc == 0, err


@sc.needs_posix_perms
def test_verify_unreadable_file_exits_1(capsys: pytest.CaptureFixture[str], made: Path) -> None:
    path = _data(made) / "shots.parquet"
    path.chmod(0o000)
    try:
        rc, _, err = _run(capsys, "verify", str(made))
    finally:
        path.chmod(0o644)
    assert rc == 1
    assert "unreadable: shots.parquet (PermissionError: Permission denied)" in err.splitlines()


@pytest.mark.parametrize("content", [None, "{ not json", '{"manifest_version": 2}', "[]"])
def test_verify_unreadable_manifest_exits_2(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, content: str | None
) -> None:
    path = tmp_path / "manifests" / "bad.json"
    if content is not None:
        path.parent.mkdir()
        path.write_text(content, encoding="utf-8")
    rc, out, err = _run(capsys, "verify", str(path))
    assert rc == 2
    assert out == ""
    assert err.startswith("fc: error: ")


def test_verify_many_manifests_max_code(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    clean = sc.write(tmp_path / "a").manifest_path
    changed = sc.write(tmp_path / "b").manifest_path
    sc.flip_byte(sc.data_dir(tmp_path / "b") / "shots.parquet", 0)
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    cases = [
        ([clean, clean], 0),
        ([clean, changed], 1),
        ([changed, clean], 1),
        ([clean, broken], 2),
        ([broken, changed], 2),
        ([changed, broken, clean], 2),
    ]
    for manifests, expected in cases:
        rc, out, err = _run(capsys, "verify", *map(str, manifests))
        assert rc == expected, (manifests, out, err)
        if clean in manifests:
            assert f"ok: snapshot 'demo' ({clean})" in out
        if changed in manifests:
            assert f"verify failed: snapshot 'demo' ({changed}): 1 problem(s)" in err


def test_data_dir_with_many_manifests_exits_2(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    a = sc.write(tmp_path / "a").manifest_path
    b = sc.write(tmp_path / "b").manifest_path
    rc, out, _ = _run(capsys, "verify", str(a), str(b), "--data-dir", str(tmp_path))
    assert rc == 2
    assert out == ""


# --- builders, version, entry points ----------------------------------------------------------


def test_builders_lists_parquet(capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = _run(capsys, "builders")
    assert rc == 0
    assert out == "parquet\n"
    snapshot.register_builder("toy", lambda request: None)  # type: ignore[arg-type, return-value]
    snapshot.register_builder("alpha", lambda request: None)  # type: ignore[arg-type, return-value]
    rc, out, _ = _run(capsys, "builders")
    assert out.splitlines() == ["alpha", "parquet", "toy"]


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = _run(capsys, "--version")
    assert rc == 0
    assert out == f"fc (football-core) {importlib.metadata.version('football-core')}\n"


def test_console_scripts_declared() -> None:
    scripts = {
        ep.name: ep.value
        for ep in importlib.metadata.entry_points(group="console_scripts")
        if ep.dist is not None and ep.dist.name == "football-core"
    }
    assert scripts.get("fc") == "football_core.cli:main"
    assert scripts.get("football-core") == "football_core.cli:main"


def test_pyproject_declares_scripts() -> None:
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts["fc"] == "football_core.cli:main"
    assert scripts["football-core"] == "football_core.cli:main"


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("POLARS_MAX_THREADS", None)
    return env


def test_module_entry_exit_codes(made: Path) -> None:
    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "football_core.cli", "verify", str(made)],
            capture_output=True,
            text=True,
            env=_child_env(),
            check=False,
            timeout=120,
        )

    clean = run()
    assert clean.returncode == 0, clean.stderr
    sc.flip_byte(_data(made) / "shots.parquet", 7)
    flipped = run()
    assert flipped.returncode == 1
    assert "changed: shots.parquet" in flipped.stderr
    usage = subprocess.run(
        [sys.executable, "-m", "football_core.cli"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert usage.returncode == 2


@pytest.mark.parametrize("script", ["fc", "football-core"])
def test_console_script_exit_codes(made: Path, script: str) -> None:
    exe = Path(sys.executable).parent / script
    if not exe.exists():
        pytest.skip(f"{script} is not installed next to {sys.executable}")
    clean = subprocess.run(
        [str(exe), "verify", str(made)], capture_output=True, text=True, check=False, timeout=120
    )
    assert clean.returncode == 0, clean.stderr
    (_data(made) / "shots.parquet").unlink()
    missing = subprocess.run(
        [str(exe), "verify", str(made)], capture_output=True, text=True, check=False, timeout=120
    )
    assert missing.returncode == 1
    assert "missing: shots.parquet" in missing.stderr


# --- Bugs propagate (spec §1.3; open issue 2) -------------------------------------------------


def test_internal_error_propagates(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raising(cls: type[snapshot.Manifest], text: str) -> snapshot.Manifest:
        raise snapshot.ManifestError("injected reader drift")

    inputs = _inputs(tmp_path)
    monkeypatch.setattr(snapshot.Manifest, "from_json", classmethod(raising))
    with pytest.raises(RuntimeError, match="^internal error"):
        cli.main(["snapshot", "demo", "--input", str(inputs), "--root", str(tmp_path / "out")])
    assert not sc.manifest_path(tmp_path / "out").exists()


def test_unexpected_builder_exception_propagates(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def buggy(request: snapshot.BuildRequest) -> snapshot.BuildResult:
        raise KeyError("a bug in the builder")

    snapshot.register_builder("toy", buggy)
    with pytest.raises(KeyError):
        cli.main(["snapshot", "demo", "--builder", "toy", "--root", str(tmp_path)])


def test_builder_wrong_return_type_propagates(tmp_path: Path) -> None:
    """``TypeError`` is not in the exit-code table, so it is a bug and propagates."""
    snapshot.register_builder("toy", lambda request: {"events": sc.generic_frame()})  # type: ignore[arg-type, return-value]
    with pytest.raises(TypeError, match="toy"):
        cli.main(["snapshot", "demo", "--builder", "toy", "--root", str(tmp_path)])


def test_main_returns_int_for_every_command(capsys: pytest.CaptureFixture[str], made: Path) -> None:
    for argv in (["builders"], ["verify", str(made)], ["--version"], ["--help"]):
        assert type(cli.main(argv)) is int
    capsys.readouterr()


def test_snapshot_input_without_parquet_rows(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    folder = tmp_path / "in"
    folder.mkdir()
    pl.DataFrame({"k": pl.Series([], dtype=pl.String)}).write_parquet(folder / "empty.parquet")
    rc, out, err = _run(
        capsys, "snapshot", "demo", "--input", str(folder), "--root", str(tmp_path / "out")
    )
    assert rc == 0, err
    assert "1 table(s), 0 row(s)" in out


def test_verify_many_manifests_names_rule_breaking_file(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """build.md ratify item 15: with several manifests, the error line names the bad one."""
    clean = sc.write(tmp_path / "a").manifest_path
    m = sc.base_manifest()
    m["tables"][1]["rows"] = -1
    bad = tmp_path / "manifests" / "rule.json"
    bad.parent.mkdir()
    bad.write_text(sc.dumps(m), encoding="utf-8")
    rc, out, err = _run(capsys, "verify", str(clean), str(bad))
    assert rc == 2
    assert f"ok: snapshot 'demo' ({clean})" in out
    assert f"fc: error: manifest '{bad}': tables[1].rows" in err
