"""Immutable parquet snapshots with a committed JSON manifest, and their verification.

A model is only reproducible if its training data is. A snapshot freezes a set of tables as
zstd-compressed parquet under ``data/snapshots/<name>/`` (git-ignored, never committed) and
describes them in ``manifests/<name>.json`` (committed): the sha256 of every file, row and
match counts, a per-competition-season summary, the schema version, the source it came from
and the library versions. ``verify_snapshot`` re-hashes the files against the manifest and
names every file that changed, went missing or appeared.

**Determinism.** Every table is sorted by its sort keys and then by all columns, rechunked
and written with fixed settings, and the manifest holds no wall-clock time. A rerun on the
same inputs in the same environment therefore writes the same bytes and the same manifest,
so it produces no git diff. The manifest's commit time in the consuming repo is its
creation time.

**Safety.** Shot, freeze-frame and match tables pass the C3 validators before anything
touches the disk. The writer refuses to put data where git would not ignore it, stages every
file and renames it into place with the manifest last, and checks that its reader accepts the
manifest it is about to commit.

**Extension point.** A builder turns inputs into tables plus a ``Source``. Provider loaders
add one ``"module:function"`` line to ``BUILTIN_BUILDERS``; ``register_builder`` adds one at
runtime. Built in: ``parquet`` (plain parquet files), ``statsbomb`` (StatsBomb open data,
``football_core.io.statsbomb``) and ``whoscored`` (the private WhoScored CSV archive,
``football_core.io.whoscored``). The two provider builders share ``C3_SORT_BY``.
"""

from __future__ import annotations

import errno
import functools
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypedDict, cast, get_args

import polars as pl

import football_core
from football_core import schema

StrPath = str | os.PathLike[str]
SourceKind = Literal["git", "files", "sql"]
Status = Literal["created", "unchanged", "replaced"]
ProblemKind = Literal["missing", "changed", "extra", "unreadable"]
Builder = Callable[["BuildRequest"], "BuildResult"]


class ParquetSettings(TypedDict):
    """The fixed ``write_parquet`` keyword arguments (a TypedDict, so ``**`` type-checks)."""

    compression: Literal["zstd"]
    compression_level: int
    statistics: bool
    row_group_size: int


MANIFEST_VERSION: Final[int] = 1  # manifest format; independent of schema.SCHEMA_VERSION
DATA_SUBDIR: Final[str] = "data/snapshots"  # POSIX, relative to the root
MANIFEST_SUBDIR: Final[str] = "manifests"
PARQUET_SETTINGS: Final[ParquetSettings] = {
    "compression": "zstd",
    "compression_level": 3,
    "statistics": True,
    "row_group_size": 100_000,
}
NAME_PATTERN: Final[str] = r"^[a-z0-9][a-z0-9._-]{0,63}$"  # snapshot and builder names
TABLE_PATTERN: Final[str] = r"^[a-z][a-z0-9_]{0,63}$"  # table names = file stems
IGNORED_NAMES: Final[frozenset[str]] = frozenset({".DS_Store"})  # verify never reports these
SORT_ROLES: Final[tuple[schema.Role, ...]] = ("identity", "pre_shot")  # allowed sort_by roles
BUILTIN_BUILDERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "parquet": "football_core.io.snapshot:build_from_parquet",
        "statsbomb": "football_core.io.statsbomb:build_snapshot",
        "whoscored": "football_core.io.whoscored:build_snapshot",
    }
)
# The sort keys of the C3 provider builders (statsbomb, whoscored): one definition, same bytes.
C3_SORT_BY: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "shots": (
            *("match_date", "competition_id", "season_id", "match_id", "period"),
            *("seconds_in_period", "shot_id"),
        ),
        "freeze_frames": ("shot_id",),
        "matches": ("match_date", "match_id"),
    }
)
DATE_RANGE: Final[tuple[str, str]] = ("0001-01-01", "9999-12-31")  # Python date's range
GIT_TIMEOUT: Final[float] = 60.0  # seconds, per git call
# Days since 1970-01-01 of DATE_RANGE; the epoch is ordinal 719_163.
_DAY_MIN: Final[int] = -719_162
_DAY_MAX: Final[int] = 2_932_896
_EPOCH_ORDINAL: Final[int] = 719_163

_SOURCE_KINDS: Final[tuple[SourceKind, ...]] = get_args(SourceKind)
_CANONICAL: Final[tuple[schema.Table, ...]] = schema.TABLES
_HEX64: Final = re.compile(r"[0-9a-f]{64}")
_HEX40: Final = re.compile(r"[0-9a-f]{40}")
_SCHEMA_VERSION_PATTERN: Final[str] = r"^[0-9]+\.[0-9]+$"
_FILE_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
_DATE_PATTERN: Final[str] = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
_GROUP_KEYS: Final[tuple[str, str]] = ("competition_id", "season_id")
_OVERWRITE_HINT: Final[str] = "pass overwrite=True (fc snapshot --overwrite) to replace it"


def _full(pattern: str, value: object) -> bool:
    """Whole-string match: ``$`` alone would also accept a trailing newline."""
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _is_hex(value: object, *lengths: int) -> bool:
    if not isinstance(value, str):
        return False
    return any((_HEX64 if n == 64 else _HEX40).fullmatch(value) for n in lengths)


def _encodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


# --- Value objects ----------------------------------------------------------------------------


@dataclass(frozen=True)
class InputFile:
    """One input file of a ``files`` source.

    Attributes
    ----------
    path : str
        POSIX path relative to the source root: no leading ``/``, no ``.``, ``..`` or empty
        segment, and no ``\\n``, ``\\r`` or backslash (one entry must never read as two).
    sha256 : str
        64 lowercase hex characters.
    bytes : int
        File size, ``>= 0``.
    """

    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        path = self.path
        if (
            not isinstance(path, str)
            or path == ""
            or path.startswith("/")
            or any(c in path for c in "\n\r\\")
            or any(seg in ("", ".", "..") for seg in path.split("/"))
            or not _encodable(path)
        ):
            raise ValueError(
                f"bad input file path {path!r}: expected a relative POSIX path without '.', "
                "'..' or empty segments, newlines or backslashes"
            )
        if not _is_hex(self.sha256, 64):
            raise ValueError(f"bad sha256 {self.sha256!r} for {path!r}: expected 64 lowercase hex")
        if not _is_count(self.bytes):
            raise ValueError(f"bad size {self.bytes!r} for {path!r}: expected an int >= 0")


@dataclass(frozen=True)
class Source:
    """Where a snapshot's data came from.

    Attributes
    ----------
    kind : {"git", "files", "sql"}
        ``git``: a commit pins the content. ``files``: a digest of the input files.
        ``sql``: a hash of the query text, plus ``content_sha256`` for the fetched rows.
    uri : str
        Human-readable location (a URL, a folder name, a table). No newline.
    revision : str
        The commit SHA (40 or 64 hex) for ``git``; a sha256 for ``files`` and ``sql``.
    files : tuple of InputFile
        The listed input files (``files`` only; may be empty when not listed).
    content_sha256 : str or None
        Fingerprint of the fetched rows (``sql`` only).
    """

    kind: SourceKind
    uri: str
    revision: str
    files: tuple[InputFile, ...] = ()
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _SOURCE_KINDS:
            raise ValueError(
                f"unknown source kind {self.kind!r}; expected one of: {', '.join(_SOURCE_KINDS)}"
            )
        uri = self.uri
        if not isinstance(uri, str) or uri == "" or "\n" in uri or "\r" in uri:
            raise ValueError(f"bad source uri {uri!r}: expected a non-empty str without newlines")
        if not _encodable(uri):
            raise ValueError(f"bad source uri {uri!r}: it cannot be encoded as UTF-8")
        lengths = (40, 64) if self.kind == "git" else (64,)
        if not _is_hex(self.revision, *lengths):
            expected = " or ".join(str(n) for n in lengths)
            raise ValueError(
                f"bad {self.kind} revision {self.revision!r}: expected {expected} lowercase hex"
            )
        files = self.files
        if not isinstance(files, tuple) or not all(isinstance(f, InputFile) for f in files):
            raise ValueError("source files must be a tuple of InputFile")
        if files and self.kind != "files":
            raise ValueError(f"source files are allowed only for kind 'files', not {self.kind!r}")
        paths = [f.path for f in files]
        if any(a >= b for a, b in zip(paths, paths[1:], strict=False)):
            raise ValueError("source files must be sorted by path, without duplicates")
        if files and self.revision != _digest((f.path, f.sha256) for f in files):
            raise ValueError(
                f"source revision {self.revision} is not the digest of its {len(files)} file(s)"
            )
        content = self.content_sha256
        if content is not None:
            if not _is_hex(content, 64):
                raise ValueError(f"bad content_sha256 {content!r}: expected 64 lowercase hex")
            if self.kind != "sql":
                raise ValueError(
                    f"content_sha256 is allowed only for kind 'sql', not {self.kind!r}"
                )


@dataclass(frozen=True)
class FileEntry:
    """One data file of a snapshot: a bare file name in the data directory, its hash and size."""

    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class GroupEntry:
    """Rows, matches and the ISO date range of one (competition, season) group of a table."""

    competition_id: str | None
    season_id: str | None
    rows: int
    matches: int | None
    first_date: str | None
    last_date: str | None


@dataclass(frozen=True)
class TableEntry:
    """One table of a snapshot, as recorded in the manifest (spec ``tables[]``)."""

    name: str
    file: str
    rows: int
    matches: int | None
    schema: schema.Table | None
    sort_by: tuple[str, ...]
    columns: tuple[tuple[str, str], ...]
    groups: tuple[GroupEntry, ...] | None


@dataclass(frozen=True)
class Manifest:
    """A snapshot manifest; ``to_json`` / ``from_json`` are exact inverses."""

    name: str
    builder: str | None
    schema_version: str
    data_dir: str
    data_digest: str
    source: Source
    tables: tuple[TableEntry, ...]
    files: tuple[FileEntry, ...]
    parquet: Mapping[str, str | int | bool]
    environment: Mapping[str, str | bool | None]

    def to_json(self) -> str:
        """The canonical manifest text: 2-space JSON, keys in spec order, trailing newline.

        Returns
        -------
        str
            The text the writer commits; ``Manifest.from_json`` of it equals ``self``.
        """
        obj: dict[str, object] = {
            "manifest_version": MANIFEST_VERSION,
            "name": self.name,
            "builder": self.builder,
            "schema_version": self.schema_version,
            "data_dir": self.data_dir,
            "data_digest": self.data_digest,
            "source": {
                "kind": self.source.kind,
                "uri": self.source.uri,
                "revision": self.source.revision,
                "content_sha256": self.source.content_sha256,
                "files": [_file_obj(f.path, f.sha256, f.bytes) for f in self.source.files],
            },
            "tables": [_table_obj(t) for t in self.tables],
            "files": [_file_obj(f.path, f.sha256, f.bytes) for f in self.files],
            "parquet": dict(self.parquet),
            "environment": dict(self.environment),
        }
        return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        """Parse and check manifest text; every rule violation is a ``ManifestError``.

        Parameters
        ----------
        text : str
            The manifest JSON.

        Returns
        -------
        Manifest
            The parsed manifest. Unknown keys are ignored; nothing is normalised.

        Raises
        ------
        ManifestError
            On invalid JSON or any broken rule, naming the key path (``tables[1].rows``).
        """
        return _parse_manifest(text)


@dataclass(frozen=True)
class SnapshotResult:
    """What ``write_snapshot`` did.

    Attributes
    ----------
    manifest : Manifest
        The manifest that describes the data now on disk.
    manifest_path, data_dir : pathlib.Path
        Where the manifest and the data files are.
    status : {"created", "unchanged", "replaced"}
        ``unchanged`` means the new files hash exactly like the existing manifest's.
    manifest_written : bool
        False only for ``unchanged`` when the manifest bytes were already identical.
    """

    manifest: Manifest
    manifest_path: Path
    data_dir: Path
    status: Status
    manifest_written: bool


@dataclass(frozen=True)
class Problem:
    """One verification problem: a file (or the data directory) and what is wrong with it."""

    kind: ProblemKind
    path: str
    detail: str

    def __str__(self) -> str:
        text = f"{self.kind}: {self.path}"
        return f"{text} ({self.detail})" if self.detail else text


@dataclass(frozen=True)
class VerifyReport:
    """The result of ``verify_snapshot``; ``ok`` when there is no problem.

    Attributes
    ----------
    name : str
        Snapshot name from the manifest.
    manifest_path, data_dir : pathlib.Path
        The manifest checked and the data directory hashed.
    problems : tuple of Problem
        Sorted by (path, kind).
    checked : int
        Number of files that reached the size check.
    """

    name: str
    manifest_path: Path
    data_dir: Path
    problems: tuple[Problem, ...]
    checked: int

    @property
    def ok(self) -> bool:
        """True when no problem was found."""
        return not self.problems


@dataclass(frozen=True)
class GitRevision:
    """The HEAD commit of a work tree and whether it has uncommitted or untracked changes."""

    sha: str
    dirty: bool


@dataclass(frozen=True)
class BuildRequest:
    """What a builder receives: input paths and string parameters (never recorded)."""

    inputs: tuple[Path, ...] = ()
    params: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildResult:
    """What a builder returns: the tables, their source and optional leading sort keys.

    ``private`` marks data whose manifest must stay out of git too (its per competition-season
    counts and dates describe private data): ``write_snapshot`` then refuses a manifest path
    that git would not ignore, as it does for the table files.
    """

    tables: Mapping[str, pl.DataFrame]
    source: Source
    sort_by: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    private: bool = False


# --- Errors -----------------------------------------------------------------------------------


class SnapshotError(Exception):
    """Base class of the snapshot errors."""


class ManifestError(SnapshotError, ValueError):
    """A manifest that cannot be read, or that breaks a manifest rule."""


class SnapshotExistsError(SnapshotError):
    """A different snapshot (or orphan data) already holds the name."""


class SnapshotVerificationError(SnapshotError):
    """Raised by ``load_snapshot`` when verification finds a problem.

    Attributes
    ----------
    report : VerifyReport
        The full report.
    """

    report: VerifyReport

    def __init__(self, report: VerifyReport) -> None:
        self.report = report
        problems = "; ".join(str(p) for p in report.problems)
        super().__init__(
            f"snapshot '{report.name}' ({report.manifest_path}) failed verification: {problems}"
        )

    def __reduce__(self) -> tuple[type[SnapshotVerificationError], tuple[VerifyReport]]:
        return type(self), (self.report,)


# --- Hashing ----------------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _digest(lines: Iterable[tuple[str, str]]) -> str:
    """sha256 of the ``sha256sum`` lines ``"<sha>  <key>\\n"``, sorted by key."""
    text = "".join(f"{sha}  {key}\n" for key, sha in sorted(lines))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parquet_record(s: ParquetSettings) -> dict[str, str | int | bool]:
    # Key by key: every shorter TypedDict -> Mapping conversion fails mypy strict.
    return {
        "compression": s["compression"],
        "compression_level": s["compression_level"],
        "statistics": s["statistics"],
        "row_group_size": s["row_group_size"],
    }


def content_digest(items: Mapping[str, bytes]) -> str:
    """Digest of keyed byte strings, for a ``sql`` source's ``content_sha256``.

    The sha256 of the ``sha256sum``-format lines ``"<sha256(value)>  <key>\\n"`` sorted by
    key, so the order of the mapping does not matter.

    Parameters
    ----------
    items : mapping of str to bytes
        For example ``{match_id: raw_json_bytes}``.

    Returns
    -------
    str
        64 lowercase hex characters; ``sha256(b"")`` for an empty mapping.

    Raises
    ------
    ValueError
        If a key is empty or contains ``\\n`` or ``\\r``.

    Examples
    --------
    >>> content_digest({"ws:2": b"{}", "ws:1": b"[]"}) == content_digest(
    ...     {"ws:1": b"[]", "ws:2": b"{}"}
    ... )
    True
    >>> content_digest({})
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    pairs: list[tuple[str, str]] = []
    for key, value in items.items():
        if not isinstance(key, str) or key == "" or "\n" in key or "\r" in key:
            raise ValueError(f"bad content key {key!r}: expected a non-empty str without newlines")
        if not _encodable(key):
            raise ValueError(f"bad content key {key!r}: it cannot be encoded as UTF-8")
        pairs.append((key, hashlib.sha256(value).hexdigest()))
    return _digest(pairs)


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except RuntimeError:  # Python 3.12 raises RuntimeError on a symlink loop (3.13+: OSError)
        raise OSError(errno.ELOOP, "symlink loop", str(path)) from None


# --- Sources ----------------------------------------------------------------------------------


def source_from_files(
    paths: Sequence[StrPath],
    *,
    root: StrPath,
    uri: str | None = None,
    list_files: bool = True,
) -> Source:
    """A ``files`` source: every input file hashed, relative to ``root``.

    ``root`` and the paths are resolved first, so a symlinked root and resolved paths (or the
    reverse) give the same source.

    Parameters
    ----------
    paths : sequence of str or path
        Regular files under ``root``.
    root : str or path
        Directory the recorded paths are relative to.
    uri : str, optional
        Defaults to the name of ``root``.
    list_files : bool, default True
        False records no file list (for thousands of files, or a private source); the
        revision still digests every file.

    Returns
    -------
    Source
        ``kind="files"``; ``revision`` is the digest of the (path, sha256) pairs.

    Raises
    ------
    TypeError
        ``paths`` is a single path instead of a sequence.
    ValueError
        No paths, a path outside ``root``, a directory, the same file twice, or a name that
        breaks the ``InputFile`` rules.
    FileNotFoundError
        A path or the root does not exist.
    OSError
        ``ELOOP`` for a symlink loop.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     (Path(tmp) / "a.json").write_bytes(b"[]")
    ...     source = source_from_files([Path(tmp) / "a.json"], root=tmp, uri="exports")
    2
    >>> source.kind, [f.path for f in source.files], source.files[0].bytes
    ('files', ['a.json'], 2)
    """
    if isinstance(paths, str | os.PathLike):
        raise TypeError("paths must be a sequence of paths, not a single path")
    if not paths:
        raise ValueError("source_from_files needs at least one path")
    r = _resolve(Path(root))
    if not os.path.isdir(r):
        if not os.path.exists(r):
            raise FileNotFoundError(f"source root '{root}' does not exist")
        raise ValueError(f"source root '{root}' is not a directory")
    entries: dict[str, InputFile] = {}
    for raw in paths:
        p = _resolve(Path(raw))
        try:
            st = os.stat(p)
        except FileNotFoundError:
            raise FileNotFoundError(f"input file '{raw}' does not exist") from None
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"input '{raw}' is not a regular file")
        try:
            rel = p.relative_to(r).as_posix()
        except ValueError:
            raise ValueError(f"input file '{raw}' is not under the root '{r}'") from None
        if rel in entries:
            raise ValueError(f"input file '{raw}' is given twice")
        entries[rel] = InputFile(rel, _file_sha256(p), st.st_size)
    files = tuple(entries[k] for k in sorted(entries))
    revision = _digest((f.path, f.sha256) for f in files)
    listed = files if list_files else ()
    return Source("files", uri if uri is not None else (r.name or "."), revision, listed)


def source_from_git(uri: str, commit: str) -> Source:
    """A ``git`` source pinned by a commit SHA.

    Parameters
    ----------
    uri : str
        The repository, for example its URL.
    commit : str
        40 or 64 lowercase hex characters.

    Returns
    -------
    Source
        ``kind="git"``, ``revision=commit``.

    Raises
    ------
    ValueError
        If the SHA or the uri is malformed.

    Examples
    --------
    >>> source_from_git("https://github.com/statsbomb/open-data", "a" * 40).kind
    'git'
    """
    return Source("git", uri, commit)


def source_from_sql(sql: str, *, uri: str, content_sha256: str | None = None) -> Source:
    """A ``sql`` source: the query text hashed as given, plus a digest of the fetched rows.

    The same query on a live table returns different rows over time, so pass
    ``content_sha256`` (normally from ``content_digest``) to pin the rows themselves.

    Parameters
    ----------
    sql : str
        The export query; hashed as UTF-8 with no normalisation.
    uri : str
        The database and table, for example ``"opposition-engine:raw.raw_match_json"``.
    content_sha256 : str, optional
        64 lowercase hex characters.

    Returns
    -------
    Source
        ``kind="sql"``.

    Raises
    ------
    ValueError
        An empty query or a malformed ``content_sha256``.

    Examples
    --------
    >>> s = source_from_sql("select 1", uri="db:raw", content_sha256=content_digest({}))
    >>> s.kind, s.revision == source_from_sql("select 1", uri="db:x").revision
    ('sql', True)
    """
    if not isinstance(sql, str) or sql == "":
        raise ValueError("source_from_sql needs a non-empty query")
    revision = hashlib.sha256(sql.encode("utf-8")).hexdigest()
    return Source("sql", uri, revision, content_sha256=content_sha256)


# --- Git --------------------------------------------------------------------------------------


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    # A GIT_DIR inherited from a hook would redirect -C to another repo; GIT_OPTIONAL_LOCKS=0
    # stops status from refreshing the index of a checkout we only read.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")} | {
        "GIT_OPTIONAL_LOCKS": "0"
    }
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            encoding="utf-8",
            errors="surrogateescape",
            timeout=GIT_TIMEOUT,
            check=False,
            env=env,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def git_revision(path: StrPath) -> GitRevision | None:
    """HEAD and dirtiness of ``path``, only when ``path`` is the top level of its work tree.

    A folder inside another repo (a downloaded dataset in a consumer's ignored ``data/``)
    gets ``None``, never the enclosing repo's commit. Untracked files make the tree dirty
    whatever the git config says; ignored files do not.

    Parameters
    ----------
    path : str or path
        The directory to describe.

    Returns
    -------
    GitRevision or None
        ``None`` without git, outside a work tree, below its top level, or without a commit.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     git_revision(tmp) is None
    True
    """
    try:
        p = Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    top = _run_git(p, "rev-parse", "--show-toplevel")
    # rstrip("\n"), not strip(): a directory name may end in a space.
    if top is None or top.returncode != 0 or Path(top.stdout.rstrip("\n")).resolve() != p:
        return None
    head = _run_git(p, "rev-parse", "HEAD")
    if head is None or head.returncode != 0:
        return None
    sha = head.stdout.rstrip("\n")
    if not _is_hex(sha, 40, 64):
        return None
    status = _run_git(
        p, "status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=none"
    )
    if status is None or status.returncode != 0:
        return None
    return GitRevision(sha, status.stdout.strip() != "")


def _package_init() -> Path:
    return Path(football_core.__file__).resolve()


@functools.cache
def _environment_git() -> tuple[str | None, bool | None]:
    """football-core's own commit, only when it runs from its own git checkout."""
    init = _package_init()
    if len(init.parents) < 3 or init.parent.parent.name != "src":
        return None, None
    root = init.parents[2]
    if not os.path.isfile(root / "pyproject.toml"):
        return None, None
    rev = git_revision(root)
    if rev is None:
        return None, None
    tracked = _run_git(root, "ls-files", "--error-unmatch", "src/football_core/__init__.py")
    if tracked is None or tracked.returncode != 0:
        return None, None
    return rev.sha, rev.dirty


def environment() -> dict[str, str | bool | None]:
    """The library versions and football-core commit recorded in every manifest.

    Returns
    -------
    dict
        ``python``, ``polars``, ``pandera``, ``numpy``, ``football_core`` (``None`` for a
        missing distribution), then ``football_core_git`` and ``football_core_dirty``
        (``None`` unless football-core runs from its own git checkout).

    Examples
    --------
    >>> list(environment())  # doctest: +NORMALIZE_WHITESPACE
    ['python', 'polars', 'pandera', 'numpy', 'football_core', 'football_core_git',
     'football_core_dirty']
    """
    env: dict[str, str | bool | None] = {"python": platform.python_version()}
    for key, dist in (
        ("polars", "polars"),
        ("pandera", "pandera"),
        ("numpy", "numpy"),
        ("football_core", "football-core"),
    ):
        try:
            env[key] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            env[key] = None
    sha, dirty = _environment_git()
    env["football_core_git"] = sha
    env["football_core_dirty"] = dirty
    return env


# --- Writing ----------------------------------------------------------------------------------


def _key_dtype_ok(dtype: pl.DataType) -> bool:
    return dtype == pl.String or isinstance(dtype, (pl.Enum, pl.Categorical)) or dtype.is_integer()


def _groups_eligible(frame: pl.DataFrame) -> bool:
    return all(k in frame.columns and _key_dtype_ok(frame.schema[k]) for k in _GROUP_KEYS)


def _has_date(frame: pl.DataFrame) -> bool:
    return "match_date" in frame.columns and isinstance(frame.schema["match_date"], pl.Date)


def _matches(frame: pl.DataFrame) -> int | None:
    if "match_id" not in frame.columns:
        return None
    return frame["match_id"].drop_nulls().n_unique()


def _iso_day(day: int | None) -> str | None:
    return None if day is None else date.fromordinal(day + _EPOCH_ORDINAL).isoformat()


def _groups(frame: pl.DataFrame) -> tuple[GroupEntry, ...] | None:
    if not _groups_eligible(frame):
        return None
    has_matches, has_date = "match_id" in frame.columns, _has_date(frame)
    aggs = [pl.len().alias("rows")]
    if has_matches:
        aggs.append(pl.col("match_id").drop_nulls().n_unique().alias("matches"))
    if has_date:  # inside polars: step 1 bounds the days, so fromordinal cannot fail
        aggs.append(pl.col("match_date").to_physical().min().alias("lo"))
        aggs.append(pl.col("match_date").to_physical().max().alias("hi"))
    # Cast before grouping, so the order is lexical on the strings ("11" before "2").
    keyed = frame.with_columns(*(pl.col(k).cast(pl.String) for k in _GROUP_KEYS))
    grouped = (
        keyed.group_by(*_GROUP_KEYS).agg(aggs).sort(list(_GROUP_KEYS), nulls_last=True).to_dicts()
    )
    return tuple(
        GroupEntry(
            competition_id=g["competition_id"],
            season_id=g["season_id"],
            rows=g["rows"],
            matches=g["matches"] if has_matches else None,
            first_date=_iso_day(g["lo"]) if has_date else None,
            last_date=_iso_day(g["hi"]) if has_date else None,
        )
        for g in grouped
    )


def _check_name(value: object, what: str) -> None:
    if not _full(NAME_PATTERN, value):
        raise ValueError(f"bad {what} {value!r}: expected a match of {NAME_PATTERN}")


def _canonical_roles(table: schema.Table) -> dict[str, schema.Role]:
    return {c: role for role in schema.ROLES for c in schema.column_names(table, roles=role)}


def _check_sort_by(
    sort_by: Mapping[str, Sequence[str]] | None, tables: Mapping[str, pl.DataFrame]
) -> dict[str, tuple[str, ...]]:
    if sort_by is None:
        return {}
    if not isinstance(sort_by, Mapping):
        raise TypeError(f"sort_by must be a mapping of table name to columns, not {sort_by!r}")
    out: dict[str, tuple[str, ...]] = {}
    for table, raw in sort_by.items():
        if table not in tables:
            raise ValueError(f"sort_by names table {table!r}, which is not in the snapshot")
        if isinstance(raw, str):
            raise TypeError(
                f"sort_by[{table!r}] must be a sequence of column names, not a str ({raw!r})"
            )
        if not isinstance(raw, Sequence):
            raise TypeError(
                f"sort_by[{table!r}] must be a sequence of column names, not {type(raw).__name__}"
            )
        keys = tuple(raw)
        if not keys:
            raise ValueError(f"sort_by[{table!r}] is empty; leave the table out instead")
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(f"sort_by[{table!r}] repeats {', '.join(map(repr, dupes))}")
        columns = tables[table].columns
        unknown = [k for k in keys if k not in columns]
        if unknown:
            raise ValueError(
                f"sort_by[{table!r}] names unknown column(s) {', '.join(map(repr, unknown))}"
            )
        canonical = next((t for t in _CANONICAL if t == table), None)
        if canonical is not None:
            roles = _canonical_roles(canonical)
            for key in keys:
                role = roles.get(key)
                if role not in SORT_ROLES:
                    raise ValueError(
                        f"sort_by[{table!r}] key {key!r} has role {role!r}; a canonical table "
                        f"sorts only by roles {', '.join(SORT_ROLES)}, so no file is ordered "
                        "by a label"
                    )
        out[table] = keys
    return out


def _check_tables(tables: object) -> dict[str, pl.DataFrame]:
    if not isinstance(tables, Mapping):
        raise TypeError(f"tables must be a mapping of name to DataFrame, not {type(tables)}")
    if not tables:
        raise ValueError("a snapshot needs at least one table")
    out: dict[str, pl.DataFrame] = {}
    for name, frame in tables.items():
        if not _full(TABLE_PATTERN, name):
            raise ValueError(f"bad table name {name!r}: expected a match of {TABLE_PATTERN}")
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                f"table {name!r} must be a polars DataFrame, not {type(frame).__name__}"
            )
        if frame.width == 0:
            raise ValueError(f"table {name!r} has no columns")
        out[name] = frame
    if "freeze_frames" in out and "shots" not in out:
        raise ValueError(
            "freeze_frames requires a shots table in the same snapshot, so the cross-table "
            "rules run"
        )
    return out


def _check_values(tables: Mapping[str, pl.DataFrame]) -> None:
    for name, frame in tables.items():
        for column, dtype in frame.schema.items():
            if isinstance(dtype, pl.Object):
                raise ValueError(
                    f"table {name!r} column {column!r} has dtype Object, which parquet cannot store"
                )
        if _groups_eligible(frame) and _has_date(frame):
            # Physical Int32 days inside polars: an out-of-range Date never reaches Python.
            lo, hi = frame.select(
                pl.col("match_date").to_physical().min().alias("lo"),
                pl.col("match_date").to_physical().max().alias("hi"),
            ).row(0)
            if lo is not None and (lo < _DAY_MIN or hi > _DAY_MAX):
                raise ValueError(
                    f"table {name!r} column 'match_date': min {lo} / max {hi} (days since "
                    f"1970-01-01) lies outside {DATE_RANGE[0]}..{DATE_RANGE[1]} (days "
                    f"{_DAY_MIN}..{_DAY_MAX}); a wrong epoch unit in a loader gives such dates"
                )


def _existing_anchor(path: Path) -> Path:
    """The nearest existing directory among ``path`` and its parents."""
    for candidate in (path, *path.parents):
        if os.path.isdir(candidate):
            return candidate
    return Path(path.anchor or ".")


_DATA_HINT: Final = "add `data/*` to the repo's .gitignore or run from the repo root (--root)"


def check_ignored(path: Path, what: str = "data file", hint: str = _DATA_HINT) -> None:
    """Refuse a data path that git would not ignore (fail closed when git cannot tell).

    Outside a git work tree there is nothing to refuse. The model repos call it before writing
    a git-ignored file (the prediction cache, the StatsBomb keeper chart).

    Raises
    ------
    ValueError
        ``path`` (unresolved) is in a work tree and git would not ignore it, naming ``what``
        and ``hint``; git cannot answer.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     check_ignored(Path(tmp) / "shots.parquet") is None
    True
    """
    # git -C on a missing directory looks like "no work tree", and so does one reached through a
    # symlinked data/ that points outside the repo: anchor at the nearest real directory.
    anchor = _existing_anchor(path.parent)
    while os.path.realpath(anchor) != str(anchor) and anchor != anchor.parent:
        anchor = _existing_anchor(anchor.parent)
    inside = _run_git(anchor, "rev-parse", "--is-inside-work-tree")
    if inside is None or inside.returncode != 0 or inside.stdout != "true\n":
        return
    # Exit code only: git C-quotes non-ASCII paths in its output.
    ci = _run_git(anchor, "check-ignore", "-q", "--", str(path))
    if ci is None:
        raise ValueError(f"cannot check whether '{path}' is git-ignored (git did not answer)")
    if ci.returncode == 0:
        return
    if ci.returncode == 1:
        raise ValueError(f"{what} '{path}' would not be git-ignored; {hint}")
    first = next(iter(ci.stderr.splitlines()), "").strip() or "(no stderr)"
    raise ValueError(
        f"cannot check whether '{path}' is git-ignored (git exit {ci.returncode}): {first}"
    )


def _guard(r: Path, name: str, tables: Iterable[str], *, private: bool = False) -> None:
    """Refuse data files that git would not ignore, one ``check_ignored`` per table file, and
    for private data the manifest too."""
    for table in sorted(tables):
        check_ignored(r / DATA_SUBDIR / name / f"{table}.parquet")
    if private:
        hint = f"add `{MANIFEST_SUBDIR}/{name}.json` (or a glob) to the repo's .gitignore"
        check_ignored(r / MANIFEST_SUBDIR / f"{name}.json", "private manifest", hint)


def _remove(path: Path) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def _file_obj(path: str, sha256: str, size: int) -> dict[str, object]:
    return {"path": path, "sha256": sha256, "bytes": size}


def _table_obj(t: TableEntry) -> dict[str, object]:
    groups = None
    if t.groups is not None:
        groups = [
            {
                "competition_id": g.competition_id,
                "season_id": g.season_id,
                "rows": g.rows,
                "matches": g.matches,
                "first_date": g.first_date,
                "last_date": g.last_date,
            }
            for g in t.groups
        ]
    return {
        "name": t.name,
        "file": t.file,
        "rows": t.rows,
        "matches": t.matches,
        "schema": t.schema,
        "sort_by": list(t.sort_by),
        "columns": [{"name": n, "dtype": d} for n, d in t.columns],
        "groups": groups,
    }


def _conflict(name: str, mpath: Path, old: Manifest, new: Manifest) -> str:
    before = {f.path: f for f in old.files}
    after = {f.path: f for f in new.files}
    lines = [f"changed {p}" for p in sorted(before.keys() & after.keys()) if before[p] != after[p]]
    lines += [f"added {p}" for p in sorted(after.keys() - before.keys())]
    lines += [f"removed {p}" for p in sorted(before.keys() - after.keys())]
    body = "".join(f"\n  {line}" for line in lines)
    return f"snapshot '{name}' ({mpath}) already holds different data:{body}\n{_OVERWRITE_HINT}"


def _self_check(manifest: Manifest, text: str) -> None:
    """Refuse to commit a manifest that the reader rejects or would re-emit differently."""
    prefix = "internal error: the writer built a manifest that its reader rejects or changes"
    try:
        parsed = Manifest.from_json(text)
    except ManifestError as e:
        raise RuntimeError(f"{prefix}: {e}") from e
    if parsed != manifest:
        raise RuntimeError(f"{prefix}: the parsed manifest differs from the one built")
    if parsed.to_json() != text:
        raise RuntimeError(f"{prefix}: the parsed manifest serialises to different text")


def write_snapshot(
    name: str,
    tables: Mapping[str, pl.DataFrame],
    source: Source,
    *,
    root: StrPath = ".",
    sort_by: Mapping[str, Sequence[str]] | None = None,
    overwrite: bool = False,
    builder: str | None = None,
    private: bool = False,
) -> SnapshotResult:
    """Validate, sort and write tables as a snapshot plus its manifest.

    Data goes to ``<root>/data/snapshots/<name>/<table>.parquet`` and the manifest to
    ``<root>/manifests/<name>.json``. A table named ``matches`` passes
    ``schema.validate_matches``; ``shots`` passes ``schema.validate_shots``, against those
    matches when both are present; ``freeze_frames`` passes ``schema.validate_freeze_frames``
    against those shots (and needs them). Every check runs before anything is written.

    Parameters
    ----------
    name : str
        Snapshot name, matching ``NAME_PATTERN``.
    tables : mapping of str to polars.DataFrame
        Table name (a file stem matching ``TABLE_PATTERN``) to frame.
    source : Source
        Where the data came from.
    root : str or path, default "."
        The consuming repo. Its ``data/`` must be git-ignored when it is in a work tree.
    sort_by : mapping of str to sequence of str, optional
        Leading sort keys per table; ties are broken by every remaining column. Canonical
        tables accept only ``identity`` and ``pre_shot`` columns.
    overwrite : bool, default False
        Replace a snapshot whose files differ (otherwise ``SnapshotExistsError``).
    builder : str, optional
        Name of the builder that produced the tables, recorded in the manifest.
    private : bool, default False
        The manifest describes private data (``BuildResult.private``): it must be git-ignored
        too when ``root`` is in a work tree.

    Returns
    -------
    SnapshotResult
        ``created``, ``unchanged`` (the files hash like the existing manifest's) or
        ``replaced``.

    Raises
    ------
    TypeError
        A table that is not a DataFrame, a ``source`` that is not a ``Source``, or a ``str``
        given as a ``sort_by`` value.
    ValueError
        A bad name, builder or table name, a bad ``sort_by``, no tables, a 0-column table,
        an ``Object`` column, ``freeze_frames`` without ``shots``, a grouped ``match_date``
        outside ``DATE_RANGE``, or data (with ``private``, also a manifest) that git would not
        ignore.
    football_core.schema.SchemaValidationError
        A canonical table fails its schema.
    SnapshotExistsError
        Different data already holds the name and ``overwrite`` is False.
    ManifestError
        The existing manifest is unreadable and ``overwrite`` is False.
    OSError
        ``ELOOP`` for a symlink loop in ``root``, or any filesystem error while writing.

    Examples
    --------
    >>> import tempfile
    >>> import polars as pl
    >>> frame = pl.DataFrame({"code": ["b", "a"], "weight": [2.0, 1.0]})
    >>> source = source_from_git("https://example.invalid/x", "0" * 40)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     first = write_snapshot("demo", {"lookup": frame}, source, root=tmp)
    ...     again = write_snapshot("demo", {"lookup": frame.reverse()}, source, root=tmp)
    >>> first.status, again.status, first.manifest.files == again.manifest.files
    ('created', 'unchanged', True)
    """
    # 1. Arguments, before any IO.
    _check_name(name, "snapshot name")
    if builder is not None:
        _check_name(builder, "builder name")
    frames = _check_tables(tables)
    if not isinstance(source, Source):
        raise TypeError(f"source must be a Source, not {type(source).__name__}")
    keys = _check_sort_by(sort_by, frames)
    _check_values(frames)

    # 2. Validate the canonical tables, each before the one whose cross-table rules need it.
    validated = dict(frames)
    if "matches" in frames:
        validated["matches"] = schema.validate_matches(frames["matches"])
    if "shots" in frames:
        validated["shots"] = schema.validate_shots(
            frames["shots"], matches=validated.get("matches")
        )
    if "freeze_frames" in frames:
        validated["freeze_frames"] = schema.validate_freeze_frames(
            frames["freeze_frames"], shots=validated["shots"]
        )

    # 3. Sort by the keys, then every remaining column; rechunk, because the chunk layout of
    # the sort output depends on the thread count and can reach the file bytes.
    prepared: dict[str, pl.DataFrame] = {}
    for table in sorted(validated):
        frame = validated[table]
        lead = keys.get(table, ())
        order = [*lead, *(c for c in frame.columns if c not in lead)]
        prepared[table] = frame.sort(order, nulls_last=True, maintain_order=True).rechunk()

    # 4. The data must be git-ignored (still nothing written).
    base = Path(root)
    _guard(_resolve(base), name, prepared, private=private)

    snapdir = base / DATA_SUBDIR
    final = snapdir / name
    staging = snapdir / f".{name}.staging"
    old = snapdir / f".{name}.old"
    mpath = base / MANIFEST_SUBDIR / f"{name}.json"

    # 5. Recover from an interrupted run.
    if os.path.lexists(staging):
        _remove(staging)
    if os.path.lexists(old):
        if os.path.lexists(final):
            _remove(old)
        else:  # the crash fell between the two commit renames: old matches the manifest
            os.rename(old, final)

    # 6-7. Stage, self-check and decide; staging never outlives an error.
    staging.mkdir(parents=True)
    try:
        entries: list[FileEntry] = []
        table_entries: list[TableEntry] = []
        for table, frame in prepared.items():
            path = staging / f"{table}.parquet"
            frame.write_parquet(path, **PARQUET_SETTINGS)
            entries.append(FileEntry(path.name, _file_sha256(path), path.stat().st_size))
            canonical = next((t for t in _CANONICAL if t == table), None)
            table_entries.append(
                TableEntry(
                    name=table,
                    file=path.name,
                    rows=frame.height,
                    matches=_matches(frame),
                    schema=canonical,
                    sort_by=keys.get(table, ()),
                    columns=tuple((c, str(d)) for c, d in frame.schema.items()),
                    groups=_groups(frame),
                )
            )
        files = tuple(sorted(entries, key=lambda e: e.path))
        manifest = Manifest(
            name=name,
            builder=builder,
            schema_version=schema.SCHEMA_VERSION,
            data_dir=f"../{DATA_SUBDIR}/{name}",
            data_digest=_digest((f.path, f.sha256) for f in files),
            source=source,
            tables=tuple(table_entries),
            files=files,
            parquet=_parquet_record(PARQUET_SETTINGS),
            environment=environment(),
        )
        text = manifest.to_json()
        _self_check(manifest, text)
        payload = text.encode("utf-8")

        status: Status = "created"
        if os.path.lexists(mpath):
            try:
                existing = read_manifest(mpath)
            except ManifestError:
                if not overwrite:
                    raise
                if os.path.isdir(mpath):  # it could not be replaced after the data commit
                    raise ManifestError(
                        f"manifest path {mpath} is a directory; remove it to write the snapshot"
                    ) from None
                status = "replaced"
            else:
                if existing.files == manifest.files:
                    status = "unchanged"
                elif not overwrite:
                    raise SnapshotExistsError(_conflict(name, mpath, existing, manifest))
                else:
                    status = "replaced"
        elif os.path.lexists(final) and (
            not os.path.isdir(final) or any(n not in IGNORED_NAMES for n in os.listdir(final))
        ):
            if not overwrite:
                raise SnapshotExistsError(
                    f"data directory {final} exists but has no manifest {mpath}; {_OVERWRITE_HINT}"
                )
            status = "replaced"
    except BaseException:
        _remove(staging)
        raise

    # 8. Commit: data first, the manifest last (it is the commit marker).
    if os.path.lexists(final):
        os.rename(final, old)
    os.rename(staging, final)
    if os.path.lexists(old):
        _remove(old)
    written = True
    if status == "unchanged":
        try:
            written = mpath.read_bytes() != payload
        except OSError:
            written = True
    if written:
        mpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = mpath.with_suffix(".json.tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, mpath)
    return SnapshotResult(manifest, mpath, final, status, written)


# --- Reading and verifying --------------------------------------------------------------------


def read_manifest(path: StrPath) -> Manifest:
    """Read a manifest file (UTF-8) and parse it with ``Manifest.from_json``.

    Parameters
    ----------
    path : str or path
        The manifest file.

    Returns
    -------
    Manifest
        The parsed manifest.

    Raises
    ------
    ManifestError
        The file is missing, unreadable, not UTF-8 or breaks a manifest rule; the message
        names the path.

    Examples
    --------
    >>> try:
    ...     read_manifest("no/such/manifest.json")
    ... except ManifestError as e:
    ...     print("no/such/manifest.json" in str(e))
    True
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ManifestError(f"cannot read manifest '{path}': {e}") from None
    try:
        return Manifest.from_json(text)
    except ManifestError as e:
        raise ManifestError(f"manifest '{path}': {e}") from None


def _errno_text(e: OSError) -> str:
    return f"{type(e).__name__}: {e.strerror}"


def _verify(m: Manifest, mpath: Path, d: Path) -> VerifyReport:
    problems: list[Problem] = []
    checked = 0
    for entry in sorted(m.files, key=lambda f: f.path):
        p = d / entry.path
        # os.stat in try, never Path.is_file(): that raises under an unsearchable parent.
        try:
            st = os.stat(p)
        except (FileNotFoundError, NotADirectoryError):
            problems.append(Problem("missing", entry.path, ""))
            continue
        except OSError as e:
            problems.append(Problem("unreadable", entry.path, _errno_text(e)))
            continue
        if not stat.S_ISREG(st.st_mode):  # a directory or a FIFO; hashing a FIFO blocks
            problems.append(Problem("missing", entry.path, "not a regular file"))
            continue
        checked += 1
        if st.st_size != entry.bytes:
            detail = f"{st.st_size} bytes, manifest says {entry.bytes}"
            problems.append(Problem("changed", entry.path, detail))
            continue
        try:
            got = _file_sha256(p)
        except OSError as e:
            problems.append(Problem("unreadable", entry.path, _errno_text(e)))
            continue
        if got != entry.sha256:
            problems.append(
                Problem("changed", entry.path, f"sha256 {got} != manifest {entry.sha256}")
            )

    shown = f"{d}/"
    try:
        dst = os.stat(d)
    except (FileNotFoundError, NotADirectoryError):
        pass  # every file is already missing
    except OSError as e:
        problems.append(Problem("unreadable", shown, f"cannot stat: {_errno_text(e)}"))
    else:
        if not stat.S_ISDIR(dst.st_mode):
            problems.append(Problem("unreadable", shown, "not a directory"))
        else:
            names = {f.path for f in m.files}
            try:
                listing: list[tuple[str, bool]] = []
                with os.scandir(d) as it:
                    for item in it:
                        try:
                            is_dir = item.is_dir()
                        except OSError:
                            is_dir = False
                        listing.append((item.name, is_dir))
            except OSError as e:
                problems.append(Problem("unreadable", shown, f"cannot list: {_errno_text(e)}"))
            else:
                for entry_name, is_dir in listing:
                    if entry_name in names or entry_name in IGNORED_NAMES:
                        continue
                    problems.append(
                        Problem("extra", entry_name + "/" if is_dir else entry_name, "")
                    )
    problems.sort(key=lambda pr: (pr.path, pr.kind))
    return VerifyReport(m.name, mpath, d, tuple(problems), checked)


def verify_snapshot(manifest: StrPath, *, data_dir: StrPath | None = None) -> VerifyReport:
    """Re-hash a snapshot's files against its manifest.

    Every changed, missing, extra or unreadable file is a ``Problem`` in the report; no
    filesystem state of the data raises.

    Parameters
    ----------
    manifest : str or path
        The manifest file.
    data_dir : str or path, optional
        Where the data lives; defaults to the manifest's ``data_dir``, resolved against the
        manifest's directory (not the working directory).

    Returns
    -------
    VerifyReport
        ``report.ok`` is True when every file matches and nothing else is there.

    Raises
    ------
    ManifestError
        The manifest is missing, unreadable or invalid.

    Examples
    --------
    >>> import tempfile
    >>> import polars as pl
    >>> source = source_from_git("https://example.invalid/x", "0" * 40)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     result = write_snapshot("demo", {"t": pl.DataFrame({"a": [1]})}, source, root=tmp)
    ...     clean = verify_snapshot(result.manifest_path)
    ...     (result.data_dir / "t.parquet").write_bytes(b"x")
    ...     changed = verify_snapshot(result.manifest_path)
    1
    >>> clean.ok, [str(p).split(" (")[0] for p in changed.problems]
    (True, ['changed: t.parquet'])
    """
    mpath = Path(manifest)
    m = read_manifest(mpath)
    d = Path(data_dir) if data_dir is not None else mpath.parent / m.data_dir
    return _verify(m, mpath, d)


def load_snapshot(
    manifest: StrPath,
    *,
    tables: Sequence[str] | None = None,
    data_dir: StrPath | None = None,
    verify: bool = True,
) -> dict[str, pl.DataFrame]:
    """Verify a snapshot, then read its tables.

    No re-validation: the hash proves each file is the one validated at write time.

    Parameters
    ----------
    manifest : str or path
        The manifest file.
    tables : sequence of str, optional
        Tables to read, in this order; default all, in manifest order. ``[]`` gives ``{}``.
    data_dir : str or path, optional
        Overrides the manifest's ``data_dir``.
    verify : bool, default True
        Run a full ``verify_snapshot`` first.

    Returns
    -------
    dict of str to polars.DataFrame
        The requested tables.

    Raises
    ------
    TypeError
        ``tables`` is a single ``str``.
    ManifestError
        The manifest is missing, unreadable or invalid.
    SnapshotVerificationError
        Verification found a problem (``err.report`` has the details).
    ValueError
        An unknown or duplicate table name, or a requested canonical table written with a
        different ``schema_version``.

    Examples
    --------
    >>> import tempfile
    >>> import polars as pl
    >>> source = source_from_git("https://example.invalid/x", "0" * 40)
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     result = write_snapshot("demo", {"t": pl.DataFrame({"a": [2, 1]})}, source, root=tmp)
    ...     loaded = load_snapshot(result.manifest_path)
    >>> loaded["t"]["a"].to_list()
    [1, 2]
    """
    if isinstance(tables, str):
        raise TypeError("tables must be a sequence of table names, not a single str")
    mpath = Path(manifest)
    m = read_manifest(mpath)
    known = {t.name: t for t in m.tables}
    if tables is None:
        wanted = list(known)
    else:
        wanted = list(tables)
        unknown = [t for t in wanted if t not in known]
        if unknown:
            raise ValueError(
                f"snapshot '{m.name}' has no table(s) {', '.join(map(repr, unknown))}; "
                f"it has: {', '.join(known)}"
            )
        dupes = sorted({t for t in wanted if wanted.count(t) > 1})
        if dupes:
            raise ValueError(f"table(s) requested twice: {', '.join(map(repr, dupes))}")
    d = Path(data_dir) if data_dir is not None else mpath.parent / m.data_dir
    if verify:
        report = _verify(m, mpath, d)
        if not report.ok:
            raise SnapshotVerificationError(report)
    if any(known[t].schema is not None for t in wanted) and (
        m.schema_version != schema.SCHEMA_VERSION
    ):
        raise ValueError(
            f"snapshot '{m.name}' was written with schema_version {m.schema_version}; this "
            f"football-core reads schema_version {schema.SCHEMA_VERSION}"
        )
    return {t: pl.read_parquet(d / known[t].file) for t in wanted}


# --- Manifest parsing (spec §2.2 from_json rules) ---------------------------------------------


def _reject_constant(name: str) -> object:
    raise ValueError(f"invalid JSON constant {name}")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _k(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _obj(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where or 'manifest'}: expected a JSON object")
    return value


def _get(obj: Mapping[str, object], key: str, where: str) -> object:
    if key not in obj:
        raise ManifestError(f"{_k(where, key)}: missing required key")
    return obj[key]


def _str(obj: Mapping[str, object], key: str, where: str) -> str:
    value = _get(obj, key, where)
    if not isinstance(value, str):
        raise ManifestError(f"{_k(where, key)}: expected a string, got {value!r}")
    return value


def _opt_str(obj: Mapping[str, object], key: str, where: str) -> str | None:
    value = _get(obj, key, where)
    if value is not None and not isinstance(value, str):
        raise ManifestError(f"{_k(where, key)}: expected a string or null, got {value!r}")
    return value


def _int(obj: Mapping[str, object], key: str, where: str) -> int:
    value = _get(obj, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{_k(where, key)}: expected an integer, got {value!r}")
    return value


def _opt_int(obj: Mapping[str, object], key: str, where: str) -> int | None:
    if _get(obj, key, where) is None:
        return None
    return _int(obj, key, where)


def _list(obj: Mapping[str, object], key: str, where: str) -> list[object]:
    value = _get(obj, key, where)
    if not isinstance(value, list):
        raise ManifestError(f"{_k(where, key)}: expected a list, got {value!r}")
    return value


def _scalar(value: object, where: str) -> str | int | bool:
    if isinstance(value, str | bool | int):  # bool is kept as bool, never read as 0 / 1
        return value
    raise ManifestError(f"{where}: expected a string, integer or boolean, got {value!r}")


def _env_value(value: object, where: str) -> str | bool | None:
    if value is None or isinstance(value, str | bool):
        return value
    raise ManifestError(f"{where}: expected a string, boolean or null, got {value!r}")


def _source_kind(value: str, where: str) -> SourceKind:
    for kind in _SOURCE_KINDS:
        if value == kind:
            return kind
    raise ManifestError(
        f"{where}: unknown source kind {value!r}; expected one of: {', '.join(_SOURCE_KINDS)}"
    )


def _parse_source(obj: dict[str, object]) -> Source:
    where = "source"
    kind = _source_kind(_str(obj, "kind", where), f"{where}.kind")
    uri = _str(obj, "uri", where)
    revision = _str(obj, "revision", where)
    content = _opt_str(obj, "content_sha256", where)
    files: list[InputFile] = []
    for i, raw in enumerate(_list(obj, "files", where)):
        w = f"{where}.files[{i}]"
        f = _obj(raw, w)
        path, sha, size = _str(f, "path", w), _str(f, "sha256", w), _int(f, "bytes", w)
        try:
            files.append(InputFile(path, sha, size))
        except ValueError as e:
            raise ManifestError(f"{w}: {e}") from None
    try:
        return Source(kind, uri, revision, tuple(files), content)
    except ValueError as e:
        raise ManifestError(f"{where}: {e}") from None


def _check_date(value: str | None, where: str) -> date | None:
    if value is None:
        return None
    if not _full(_DATE_PATTERN, value):
        raise ManifestError(f"{where}: expected an ISO date YYYY-MM-DD, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ManifestError(f"{where}: not a valid date: {value!r}") from None


def _parse_group(obj: dict[str, object], where: str, table_matches: int | None) -> GroupEntry:
    comp = _opt_str(obj, "competition_id", where)
    season = _opt_str(obj, "season_id", where)
    rows = _int(obj, "rows", where)
    if rows < 1:
        raise ManifestError(f"{where}.rows: a group has at least 1 row, got {rows}")
    matches = _opt_int(obj, "matches", where)
    if (matches is None) != (table_matches is None):
        raise ManifestError(f"{where}.matches: null exactly when the table's matches is null")
    if matches is not None and not 0 <= matches <= rows:
        raise ManifestError(f"{where}.matches: {matches} is outside 0..{rows} (rows)")
    first_raw = _opt_str(obj, "first_date", where)
    last_raw = _opt_str(obj, "last_date", where)
    first = _check_date(first_raw, f"{where}.first_date")
    last = _check_date(last_raw, f"{where}.last_date")
    if (first is None) != (last is None):
        raise ManifestError(f"{where}: first_date and last_date are both null or both set")
    if first is not None and last is not None and first > last:
        raise ManifestError(f"{where}: first_date {first_raw} is after last_date {last_raw}")
    return GroupEntry(comp, season, rows, matches, first_raw, last_raw)


def _group_key(g: GroupEntry) -> tuple[bool, str, bool, str]:
    c, s = g.competition_id, g.season_id
    return (c is None, c or "", s is None, s or "")


def _parse_table(obj: dict[str, object], where: str) -> TableEntry:
    name = _str(obj, "name", where)
    if not _full(TABLE_PATTERN, name):
        raise ManifestError(f"{where}.name: {name!r} does not match {TABLE_PATTERN}")
    file = _str(obj, "file", where)
    if file != f"{name}.parquet":
        raise ManifestError(f"{where}.file: expected {name + '.parquet'!r}, got {file!r}")
    rows = _int(obj, "rows", where)
    if rows < 0:
        raise ManifestError(f"{where}.rows: {rows} is negative")
    matches = _opt_int(obj, "matches", where)
    raw_schema = _opt_str(obj, "schema", where)
    table_schema = next((t for t in _CANONICAL if t == raw_schema), None)
    if raw_schema is not None and table_schema is None:
        raise ManifestError(
            f"{where}.schema: {raw_schema!r} is not one of {', '.join(_CANONICAL)} or null"
        )
    expected_schema = name if name in _CANONICAL else None
    if raw_schema != expected_schema:
        raise ManifestError(
            f"{where}.schema: table {name!r} must have schema {expected_schema!r}, "
            f"got {raw_schema!r}"
        )
    columns: list[tuple[str, str]] = []
    for i, raw in enumerate(_list(obj, "columns", where)):
        w = f"{where}.columns[{i}]"
        c = _obj(raw, w)
        col, dtype = _str(c, "name", w), _str(c, "dtype", w)
        if dtype == "":
            raise ManifestError(f"{w}.dtype: empty")
        columns.append((col, dtype))
    if not columns:
        raise ManifestError(f"{where}.columns: a table has at least one column")
    names = [c for c, _ in columns]
    if len(set(names)) != len(names):
        raise ManifestError(f"{where}.columns: duplicate column name")
    sort_by: list[str] = []
    for i, raw in enumerate(_list(obj, "sort_by", where)):
        if not isinstance(raw, str):
            raise ManifestError(f"{where}.sort_by[{i}]: expected a string, got {raw!r}")
        sort_by.append(raw)
    if any(k not in names for k in sort_by) or len(set(sort_by)) != len(sort_by):
        raise ManifestError(f"{where}.sort_by: keys must be distinct columns, got {sort_by!r}")
    if matches is not None and not 0 <= matches <= rows:
        raise ManifestError(f"{where}.matches: {matches} is outside 0..{rows} (rows)")
    if (matches is None) == ("match_id" in names):
        raise ManifestError(f"{where}.matches: null exactly when there is no match_id column")
    groups: tuple[GroupEntry, ...] | None = None
    if _get(obj, "groups", where) is not None:
        if not all(k in names for k in _GROUP_KEYS):
            raise ManifestError(f"{where}.groups: set, but competition_id / season_id are absent")
        groups = tuple(
            _parse_group(_obj(raw, f"{where}.groups[{j}]"), f"{where}.groups[{j}]", matches)
            for j, raw in enumerate(_list(obj, "groups", where))
        )
        group_keys = [_group_key(g) for g in groups]
        if any(a >= b for a, b in zip(group_keys, group_keys[1:], strict=False)):
            raise ManifestError(
                f"{where}.groups: not strictly sorted by (competition_id, season_id), nulls last"
            )
        total = sum(g.rows for g in groups)
        if total != rows:
            raise ManifestError(f"{where}.groups: rows sum to {total}, the table has {rows}")
    return TableEntry(
        name, file, rows, matches, table_schema, tuple(sort_by), tuple(columns), groups
    )


def _parse_file(obj: dict[str, object], where: str) -> FileEntry:
    path = _str(obj, "path", where)
    if not _full(_FILE_PATTERN, path) or ".." in path:
        raise ManifestError(f"{where}.path: {path!r} is not a bare file name")
    sha = _str(obj, "sha256", where)
    if not _is_hex(sha, 64):
        raise ManifestError(f"{where}.sha256: expected 64 lowercase hex, got {sha!r}")
    size = _int(obj, "bytes", where)
    if size < 0:
        raise ManifestError(f"{where}.bytes: {size} is negative")
    return FileEntry(path, sha, size)


def _parse_manifest(text: str) -> Manifest:
    try:
        raw: object = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_pairs
        )
    except (ValueError, RecursionError) as e:
        raise ManifestError(f"not valid manifest JSON: {e}") from None
    top = _obj(raw, "")
    version = _get(top, "manifest_version", "")
    if isinstance(version, bool) or not isinstance(version, int) or version != MANIFEST_VERSION:
        raise ManifestError(
            f"unsupported manifest_version {version!r}; this football-core reads {MANIFEST_VERSION}"
        )
    name = _str(top, "name", "")
    if not _full(NAME_PATTERN, name):
        raise ManifestError(f"name: {name!r} does not match {NAME_PATTERN}")
    builder = _opt_str(top, "builder", "")
    if builder is not None and not _full(NAME_PATTERN, builder):
        raise ManifestError(f"builder: {builder!r} does not match {NAME_PATTERN}")
    schema_version = _str(top, "schema_version", "")
    if not _full(_SCHEMA_VERSION_PATTERN, schema_version):
        raise ManifestError(f"schema_version: {schema_version!r} is not MAJOR.MINOR")
    data_dir = _str(top, "data_dir", "")
    if data_dir != f"../{DATA_SUBDIR}/{name}":
        raise ManifestError(
            f"data_dir: expected '../{DATA_SUBDIR}/{name}', got {data_dir!r} (use --data-dir "
            "for data kept elsewhere)"
        )
    data_digest = _str(top, "data_digest", "")
    if not _is_hex(data_digest, 64):
        raise ManifestError(f"data_digest: expected 64 lowercase hex, got {data_digest!r}")
    source = _parse_source(_obj(_get(top, "source", ""), "source"))

    raw_tables = _list(top, "tables", "")
    tables = tuple(
        _parse_table(_obj(t, f"tables[{i}]"), f"tables[{i}]") for i, t in enumerate(raw_tables)
    )
    if not tables:
        raise ManifestError("tables: a snapshot has at least one table")
    table_names = [t.name for t in tables]
    if any(a >= b for a, b in zip(table_names, table_names[1:], strict=False)):
        raise ManifestError("tables: not strictly sorted by name")
    if "freeze_frames" in table_names and "shots" not in table_names:
        raise ManifestError("tables: a freeze_frames table needs a shots table")

    files = tuple(
        _parse_file(_obj(f, f"files[{i}]"), f"files[{i}]")
        for i, f in enumerate(_list(top, "files", ""))
    )
    paths = [f.path for f in files]
    if any(a >= b for a, b in zip(paths, paths[1:], strict=False)):
        raise ManifestError("files: not strictly sorted by path")
    if set(paths) != {t.file for t in tables}:
        raise ManifestError("files: the data files and the tables' files differ")
    if data_digest != _digest((f.path, f.sha256) for f in files):
        raise ManifestError("data_digest: does not match the digest of files")

    parquet_obj = _obj(_get(top, "parquet", ""), "parquet")
    parquet = {k: _scalar(v, f"parquet.{k}") for k, v in parquet_obj.items()}
    env_obj = _obj(_get(top, "environment", ""), "environment")
    env = {k: _env_value(v, f"environment.{k}") for k, v in env_obj.items()}
    return Manifest(
        name=name,
        builder=builder,
        schema_version=schema_version,
        data_dir=data_dir,
        data_digest=data_digest,
        source=source,
        tables=tables,
        files=files,
        parquet=parquet,
        environment=env,
    )


# --- Builders ---------------------------------------------------------------------------------

_REGISTRY: dict[str, Builder] = {}


def register_builder(name: str, builder: Builder, *, replace: bool = False) -> None:
    """Register a runtime builder (tests, notebooks, other repos).

    Parameters
    ----------
    name : str
        Builder name, matching ``NAME_PATTERN``.
    builder : callable
        ``builder(BuildRequest) -> BuildResult``.
    replace : bool, default False
        Allow replacing a built-in or registered builder of the same name.

    Raises
    ------
    TypeError
        ``builder`` is not callable.
    ValueError
        A bad name, or a taken one without ``replace``.

    Examples
    --------
    >>> register_builder("example-parquet", build_from_parquet)
    >>> get_builder("example-parquet") is build_from_parquet
    True
    >>> _REGISTRY.pop("example-parquet") is build_from_parquet  # tidy up after the example
    True
    """
    if not callable(builder):
        raise TypeError(f"builder {name!r} must be callable, not {type(builder).__name__}")
    _check_name(name, "builder name")
    if not replace and (name in _REGISTRY or name in BUILTIN_BUILDERS):
        raise ValueError(f"builder {name!r} is already registered; pass replace=True")
    _REGISTRY[name] = builder


def builder_names() -> tuple[str, ...]:
    """Built-in and registered builder names, sorted.

    Returns
    -------
    tuple of str
        The names ``fc snapshot --builder`` accepts.

    Examples
    --------
    >>> "parquet" in builder_names()
    True
    """
    return tuple(sorted(set(BUILTIN_BUILDERS) | set(_REGISTRY)))


def get_builder(name: str) -> Builder:
    """Look up a builder: registered ones first, then the lazily imported built-ins.

    Parameters
    ----------
    name : str
        Builder name.

    Returns
    -------
    callable
        The builder.

    Raises
    ------
    ValueError
        Unknown name; the message lists ``builder_names()``.
    TypeError
        The built-in target is not callable.

    Examples
    --------
    >>> get_builder("parquet") is build_from_parquet
    True
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    target = BUILTIN_BUILDERS.get(name)
    if target is None:
        raise ValueError(f"unknown builder {name!r}; known builders: {', '.join(builder_names())}")
    module_name, _, attr = target.partition(":")
    obj: object = getattr(importlib.import_module(module_name), attr)
    if not callable(obj):
        raise TypeError(f"builder {name!r}: target {target!r} is not callable")
    return cast(Builder, obj)


def snapshot_from_builder(
    name: str,
    builder: str,
    *,
    inputs: Sequence[StrPath] = (),
    params: Mapping[str, str] | None = None,
    root: StrPath = ".",
    overwrite: bool = False,
) -> SnapshotResult:
    """Run a named builder and write its tables as a snapshot.

    Parameters
    ----------
    name : str
        Snapshot name.
    builder : str
        Builder name (see ``builder_names``); recorded in the manifest.
    inputs : sequence of str or path, default ()
        Passed to the builder as ``BuildRequest.inputs``.
    params : mapping of str to str, optional
        Passed to the builder; never written to the manifest.
    root : str or path, default "."
        As in ``write_snapshot``.
    overwrite : bool, default False
        As in ``write_snapshot``.

    Returns
    -------
    SnapshotResult
        As ``write_snapshot``.

    Raises
    ------
    TypeError
        ``inputs`` is a single path, or the builder did not return a ``BuildResult``.
    ValueError
        As ``get_builder``, the builder and ``write_snapshot``.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import polars as pl
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     pl.DataFrame({"a": [1, 2]}).write_parquet(Path(tmp) / "events.parquet")
    ...     result = snapshot_from_builder(
    ...         "demo", "parquet", inputs=[Path(tmp) / "events.parquet"], root=Path(tmp) / "out"
    ...     )
    >>> result.status, result.manifest.builder, result.manifest.source.kind
    ('created', 'parquet', 'files')
    """
    _check_name(name, "snapshot name")
    if isinstance(inputs, str | os.PathLike):
        raise TypeError("inputs must be a sequence of paths, not a single path")
    fn = get_builder(builder)
    result: object = fn(BuildRequest(tuple(Path(p) for p in inputs), dict(params or {})))
    if not isinstance(result, BuildResult):
        raise TypeError(f"builder {builder!r} returned {type(result).__name__}, not a BuildResult")
    return write_snapshot(
        name,
        result.tables,
        result.source,
        root=root,
        sort_by=result.sort_by,
        overwrite=overwrite,
        builder=builder,
        private=result.private,
    )


def build_from_parquet(request: BuildRequest) -> BuildResult:
    """The built-in ``parquet`` builder: one table per input parquet file.

    Each input is a ``.parquet`` file or a directory, which expands to its ``*.parquet``
    files (not recursive). The table name is the file stem. The source is a ``files``
    source over every file, relative to their common parent directory.

    Parameters
    ----------
    request : BuildRequest
        At least one input; no params.

    Returns
    -------
    BuildResult
        The tables, read with ``pl.read_parquet``, and their source.

    Raises
    ------
    ValueError
        No inputs, any param, a non-parquet file (by suffix or content), a directory without
        parquet files, a stem that fails ``TABLE_PATTERN``, two inputs with the same stem.
    FileNotFoundError
        An input does not exist.
    OSError
        ``ELOOP`` for a symlink loop.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import polars as pl
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     pl.DataFrame({"a": [1]}).write_parquet(Path(tmp) / "events.parquet")
    ...     result = build_from_parquet(BuildRequest((Path(tmp),)))
    >>> list(result.tables), [f.path for f in result.source.files]
    (['events'], ['events.parquet'])
    """
    if not request.inputs:
        raise ValueError("the parquet builder needs at least one input (--input PATH)")
    if request.params:
        raise ValueError(
            f"the parquet builder takes no params, got: {', '.join(sorted(request.params))}"
        )
    files: list[Path] = []
    for raw in request.inputs:
        p = _resolve(Path(raw))
        if os.path.isdir(p):
            found = sorted(c for c in p.iterdir() if c.suffix == ".parquet" and c.is_file())
            if not found:
                raise ValueError(f"input directory '{raw}' holds no .parquet file")
            files.extend(found)
        elif os.path.isfile(p):
            if p.suffix != ".parquet":
                raise ValueError(f"input '{raw}' is not a .parquet file")
            files.append(p)
        elif not os.path.exists(p):
            raise FileNotFoundError(f"input '{raw}' does not exist")
        else:
            raise ValueError(f"input '{raw}' is neither a file nor a directory")
    stems: dict[str, Path] = {}
    for f in files:
        if f in stems.values():
            raise ValueError(f"input file '{f}' is given twice")
        if not _full(TABLE_PATTERN, f.stem):
            raise ValueError(f"table name {f.stem!r} (from '{f}') does not match {TABLE_PATTERN}")
        if f.stem in stems:
            raise ValueError(f"two inputs give table {f.stem!r}: '{stems[f.stem]}' and '{f}'")
        stems[f.stem] = f
    source = source_from_files(files, root=os.path.commonpath([str(f.parent) for f in files]))
    return BuildResult({stem: _read_parquet(path) for stem, path in stems.items()}, source)


def _read_parquet(path: Path) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except pl.exceptions.ComputeError as e:
        raise ValueError(f"input '{path}' is not a readable parquet file: {e}") from None
