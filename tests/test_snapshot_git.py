"""Git provenance: ``_run_git``, ``git_revision`` and the environment's git fields (C6 spec
§3.4). Tests that build real repos are skipped without git; the no-git and timeout paths are
not (open issue 2)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import snapshot_cases as sc

from football_core.io import snapshot

# --- _run_git without a real git (never skipped) ----------------------------------------------


def test_run_git_none_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert snapshot._run_git(tmp_path, "--version") is None
    assert snapshot._run_git(tmp_path, "rev-parse", "--is-inside-work-tree") is None
    assert snapshot.git_revision(tmp_path) is None
    snapshot._environment_git.cache_clear()
    env = snapshot.environment()
    assert env["football_core_git"] is None
    assert env["football_core_dirty"] is None
    result = sc.write(tmp_path / "root")
    assert result.status == "created"
    assert result.manifest.environment["football_core_git"] is None


def test_run_git_timeout_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def slow(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs)
        raise subprocess.TimeoutExpired(args[0] if args else "git", kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", slow)
    assert snapshot._run_git(tmp_path, "status", "--porcelain") is None
    assert seen and seen[0]["timeout"] == snapshot.GIT_TIMEOUT == 60.0
    assert snapshot.git_revision(tmp_path) is None


@pytest.mark.parametrize(
    "error", [FileNotFoundError("git"), PermissionError("git"), OSError(8, "Exec format error")]
)
def test_run_git_oserror_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    def broken(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise error

    monkeypatch.setattr(subprocess, "run", broken)
    assert snapshot._run_git(tmp_path, "rev-parse", "HEAD") is None


def test_run_git_call_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """argv, decoding, timeout, check=False and an environment without ``GIT_*``."""
    seen: list[tuple[Any, dict[str, Any]]] = []

    def fake(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append((args[0] if args else kwargs["args"], kwargs))
        return subprocess.CompletedProcess(["git"], 0, "ok\n", "")

    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere" / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'status.showuntrackedfiles=no'")
    monkeypatch.setenv("FC_TEST_MARKER", "kept")
    monkeypatch.setattr(subprocess, "run", fake)
    out = snapshot._run_git(tmp_path, "rev-parse", "HEAD")
    assert out is not None and out.stdout == "ok\n"
    ((argv, kwargs),) = seen
    assert list(argv) == ["git", "-C", str(tmp_path), "rev-parse", "HEAD"]
    assert kwargs["capture_output"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "surrogateescape"
    assert kwargs["timeout"] == snapshot.GIT_TIMEOUT
    assert kwargs.get("check", False) is False
    env = kwargs["env"]
    assert not [k for k in env if k.startswith("GIT_") and k != "GIT_OPTIONAL_LOCKS"]
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["FC_TEST_MARKER"] == "kept"


def _fake_revision(toplevel: str | None, head: str | None, status: str | None, rc: int = 0) -> Any:
    def fake(cwd: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
        if args == ("rev-parse", "--show-toplevel"):
            out = toplevel
        elif args == ("rev-parse", "HEAD"):
            out = head
        elif args[:1] == ("status",):
            out = status
        else:
            return None
        return None if out is None else subprocess.CompletedProcess(["git"], rc, out, "")

    return fake


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (sc.GIT_SHA + "\n", sc.GIT_SHA),
        ("a" * 64 + "\n", "a" * 64),
        (sc.GIT_SHA.upper() + "\n", None),
        ("abc\n", None),
        ("", None),
        (sc.GIT_SHA + "x\n", None),
    ],
    ids=["sha1", "sha256", "upper", "short", "empty", "trailing_junk"],
)
def test_git_revision_head_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head: str, expected: str | None
) -> None:
    monkeypatch.setattr(
        snapshot, "_run_git", _fake_revision(str(tmp_path.resolve()) + "\n", head, "")
    )
    rev = snapshot.git_revision(tmp_path)
    if expected is None:
        assert rev is None
    else:
        assert rev == snapshot.GitRevision(expected, False)


@pytest.mark.parametrize(
    ("status", "dirty"),
    [("", False), ("\n", False), ("?? new.json\n", True), (" M README.md\n", True)],
)
def test_git_revision_dirty_from_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, dirty: bool
) -> None:
    monkeypatch.setattr(
        snapshot,
        "_run_git",
        _fake_revision(str(tmp_path.resolve()) + "\n", sc.GIT_SHA + "\n", status),
    )
    assert snapshot.git_revision(tmp_path) == snapshot.GitRevision(sc.GIT_SHA, dirty)


def test_git_revision_none_when_status_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(cwd: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
        if args == ("rev-parse", "--show-toplevel"):
            return subprocess.CompletedProcess(["git"], 0, str(tmp_path.resolve()) + "\n", "")
        if args == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(["git"], 0, sc.GIT_SHA + "\n", "")
        return subprocess.CompletedProcess(["git"], 128, "", "fatal")

    monkeypatch.setattr(snapshot, "_run_git", fake)
    assert snapshot.git_revision(tmp_path) is None


def test_git_revision_none_when_toplevel_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        snapshot,
        "_run_git",
        _fake_revision(str(tmp_path.parent.resolve()) + "\n", sc.GIT_SHA + "\n", ""),
    )
    assert snapshot.git_revision(tmp_path) is None


def test_git_revision_status_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []
    inner = _fake_revision(str(tmp_path.resolve()) + "\n", sc.GIT_SHA + "\n", "")

    def fake(cwd: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
        calls.append(args)
        return inner(cwd, *args)

    monkeypatch.setattr(snapshot, "_run_git", fake)
    snapshot.git_revision(tmp_path)
    (status,) = [c for c in calls if c[:1] == ("status",)]
    assert "--porcelain" in status
    assert "--untracked-files=normal" in status
    assert "--ignore-submodules=none" in status


# --- Real repos -------------------------------------------------------------------------------


@sc.needs_git
def test_git_revision_none_outside_repo(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert snapshot.git_revision(plain) is None
    assert snapshot.git_revision(tmp_path / "does_not_exist") is None


@sc.needs_git
def test_git_revision_in_repo(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    rev = snapshot.git_revision(repo)
    assert rev == snapshot.GitRevision(sc.head_sha(repo), False)
    assert rev is not None and "\n" not in rev.sha
    assert snapshot.git_revision(str(repo)) == rev


@sc.needs_git
def test_git_revision_none_in_ignored_subdir_of_repo(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    downloaded = repo / "data" / "raw" / "sb"
    downloaded.mkdir(parents=True)
    (downloaded / "events.json").write_text("[]")
    assert snapshot.git_revision(downloaded) is None
    (repo / "src").mkdir()
    assert snapshot.git_revision(repo / "src") is None  # a tracked subdir is not a top level


@sc.needs_git
def test_git_revision_none_without_commit(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo", commit=False)
    assert snapshot.git_revision(repo) is None


@sc.needs_git
def test_git_revision_dirty_on_untracked_file(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    (repo / "new.json").write_text("{}")
    assert snapshot.git_revision(repo) == snapshot.GitRevision(sc.head_sha(repo), True)


@sc.needs_git
def test_git_revision_dirty_on_untracked_file_in_new_dir(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    (repo / "season").mkdir()
    (repo / "season" / "m1.json").write_text("{}")
    rev = snapshot.git_revision(repo)
    assert rev is not None and rev.dirty is True


@sc.needs_git
def test_git_revision_dirty_despite_show_untracked_config(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    sc.git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "new.json").write_text("{}")
    rev = snapshot.git_revision(repo)
    assert rev is not None and rev.dirty is True


@sc.needs_git
def test_git_revision_non_ascii_path(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "Masaüstü" / "Beşiktaş repo")
    assert snapshot.git_revision(repo) == snapshot.GitRevision(sc.head_sha(repo), False)


@sc.needs_git
def test_git_revision_dir_name_ending_in_space(tmp_path: Path) -> None:
    """``rstrip("\\n")``, not ``strip()``: a directory name may end in a space."""
    repo = sc.git_repo(tmp_path / "repo ")
    assert snapshot.git_revision(repo) == snapshot.GitRevision(sc.head_sha(repo), False)


@sc.needs_git
def test_git_revision_through_symlink(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    link = tmp_path / "link"
    link.symlink_to(repo, target_is_directory=True)
    assert snapshot.git_revision(link) == snapshot.GitRevision(sc.head_sha(repo), False)


@sc.needs_git
def test_git_revision_clean_with_ignored_file(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    (repo / "data").mkdir()
    (repo / "data" / "big.parquet").write_bytes(b"x")
    assert snapshot.git_revision(repo) == snapshot.GitRevision(sc.head_sha(repo), False)


@sc.needs_git
def test_git_revision_dirty_on_modified_file(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    (repo / "README.md").write_text("changed\n")
    rev = snapshot.git_revision(repo)
    assert rev is not None and rev.dirty is True


@sc.needs_git
def test_git_revision_dirty_on_staged_file(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    (repo / "staged.txt").write_text("x\n")
    sc.git(repo, "add", "staged.txt")
    rev = snapshot.git_revision(repo)
    assert rev is not None and rev.dirty is True


@sc.needs_git
def test_git_revision_ignores_git_dir_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    other = sc.git_repo(tmp_path / "other", ignore="*.log\n")
    assert sc.head_sha(repo) != sc.head_sha(other)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert snapshot.git_revision(repo) == snapshot.GitRevision(sc.head_sha(repo), False)


@sc.needs_git
def test_git_revision_never_raises_on_odd_paths(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    assert snapshot.git_revision(file_path) is None
    assert snapshot.git_revision(tmp_path / "new\nline") is None


# --- environment(): football-core's own checkout only -----------------------------------------


def _checkout(root: Path, *, tracked: bool) -> Path:
    """``<root>/pyproject.toml`` + ``<root>/src/football_core/__init__.py`` in its own repo."""
    sc.git_repo(root, ignore="data/*\n" if tracked else "data/*\nsrc/\n")
    (root / "pyproject.toml").write_text('[project]\nname = "football-core"\n')
    init = root / "src" / "football_core" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text('"""stub"""\n')
    sc.git(root, "add", "pyproject.toml")
    if tracked:
        sc.git(root, "add", "src/football_core/__init__.py")
    sc.git(root, "commit", "-q", "-m", "layout")
    return init


@sc.needs_git
def test_environment_git_none_when_package_inside_foreign_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumer = sc.git_repo(tmp_path / "consumer", ignore="data/*\n.venv/\n")
    init = consumer / ".venv/lib/python3.12/site-packages/football_core/__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text("")
    monkeypatch.setattr(snapshot, "_package_init", lambda: init)
    snapshot._environment_git.cache_clear()
    env = snapshot.environment()
    assert env["football_core_git"] is None
    assert env["football_core_dirty"] is None


@sc.needs_git
def test_environment_git_none_when_not_own_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init = _checkout(tmp_path / "fc", tracked=False)
    monkeypatch.setattr(snapshot, "_package_init", lambda: init)
    snapshot._environment_git.cache_clear()
    env = snapshot.environment()
    assert env["football_core_git"] is None
    assert env["football_core_dirty"] is None


@sc.needs_git
def test_environment_git_set_for_own_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init = _checkout(tmp_path / "fc", tracked=True)
    monkeypatch.setattr(snapshot, "_package_init", lambda: init)
    snapshot._environment_git.cache_clear()
    env = snapshot.environment()
    assert env["football_core_git"] == sc.head_sha(tmp_path / "fc")
    assert env["football_core_dirty"] is False


@sc.needs_git
def test_environment_git_dirty_own_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init = _checkout(tmp_path / "fc", tracked=True)
    (tmp_path / "fc" / "scratch.py").write_text("x = 1\n")
    monkeypatch.setattr(snapshot, "_package_init", lambda: init)
    snapshot._environment_git.cache_clear()
    assert snapshot.environment()["football_core_dirty"] is True


@sc.needs_git
def test_environment_git_none_without_src_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = sc.git_repo(tmp_path / "fc")
    (root / "pyproject.toml").write_text("[project]\n")
    init = root / "lib" / "football_core" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text("")
    sc.git(root, "add", ".")
    sc.git(root, "commit", "-q", "-m", "flat")
    monkeypatch.setattr(snapshot, "_package_init", lambda: init)
    snapshot._environment_git.cache_clear()
    assert snapshot.environment()["football_core_git"] is None


@sc.needs_git
def test_environment_git_none_without_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = sc.git_repo(tmp_path / "fc")
    init = root / "src" / "football_core" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text("")
    sc.git(root, "add", ".")
    sc.git(root, "commit", "-q", "-m", "no pyproject")
    monkeypatch.setattr(snapshot, "_package_init", lambda: init)
    snapshot._environment_git.cache_clear()
    assert snapshot.environment()["football_core_git"] is None


def test_environment_git_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_environment_git`` is ``functools.cache``d: a second ``environment()`` neither looks
    at the package path again nor runs git (spec §1.2, §3.4)."""
    inits: list[int] = []
    gits: list[tuple[str, ...]] = []
    real_init = snapshot._package_init
    real_git = snapshot._run_git

    def counting_init() -> Path:
        inits.append(1)
        return real_init()

    def counting_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
        gits.append(args)
        return real_git(cwd, *args)

    monkeypatch.setattr(snapshot, "_package_init", counting_init)
    monkeypatch.setattr(snapshot, "_run_git", counting_git)
    snapshot._environment_git.cache_clear()
    first = snapshot.environment()
    assert len(inits) == 1
    n_git = len(gits)
    for _ in range(3):
        assert snapshot.environment() == first
    assert len(inits) == 1
    assert len(gits) == n_git
    snapshot._environment_git.cache_clear()
    snapshot.environment()
    assert len(inits) == 2


def test_package_init_points_at_installed_package() -> None:
    import football_core

    assert snapshot._package_init() == Path(football_core.__file__).resolve()


def test_environment_never_raises_with_hostile_package_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot, "_package_init", lambda: tmp_path / "x")
    snapshot._environment_git.cache_clear()
    env = snapshot.environment()
    assert env["football_core_git"] is None
    assert env["football_core_dirty"] is None


def test_run_git_nul_in_argument_returns_none(tmp_path: Path) -> None:
    """build.md ratify item 4: ``subprocess.run`` raises ``ValueError`` on an embedded NUL;
    ``_run_git`` returns None, so ``git_revision`` never raises."""
    assert snapshot._run_git(tmp_path, "ls-files", "--", "a\0b") is None
    assert snapshot._run_git(tmp_path / "a\0b", "status") is None
    assert snapshot.git_revision(str(tmp_path / "a\0b")) is None
