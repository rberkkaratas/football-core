"""Hand-written synthetic cases for the snapshot tests (C6; spec ``runs/C6/spec.md`` §5).

Everything here follows the C6 spec, independently of ``io/snapshot.py``: the digest is
recomputed in the ``sha256sum`` line format, and the base manifest is typed by hand, so the
tests hold the module to the spec instead of to itself. No value is copied from a provider
file. Frames are built column by column (playbook test checklist).
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import schema_cases

from football_core.io import snapshot

NAN, INF = float("nan"), float("inf")
GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git is not installed")
needs_posix_perms = pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="permission tests need a POSIX non-root user",
)
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
EPOCH_ORDINAL = 719_163  # date(1970, 1, 1).toordinal()
DAY_MIN, DAY_MAX = -719_162, 2_932_896  # 0001-01-01 and 9999-12-31 as days since the epoch
OUT_OF_RANGE_DAYS = (-800_000, -719_163, 2_932_897, 100_000_000)
PARQUET_LITERAL = {
    "compression": "zstd",
    "compression_level": 3,
    "statistics": True,
    "row_group_size": 100_000,
}


# --- Paths ------------------------------------------------------------------------------------


def data_dir(root: Path, name: str = "demo") -> Path:
    return root / "data" / "snapshots" / name


def manifest_path(root: Path, name: str = "demo") -> Path:
    return root / "manifests" / f"{name}.json"


def staging_dir(root: Path, name: str = "demo") -> Path:
    return root / "data" / "snapshots" / f".{name}.staging"


def old_dir(root: Path, name: str = "demo") -> Path:
    return root / "data" / "snapshots" / f".{name}.old"


def tree(root: Path) -> set[str]:
    """Every path under ``root`` (relative, POSIX); empty when ``root`` does not exist."""
    if not root.exists():
        return set()
    return {p.relative_to(root).as_posix() for p in root.rglob("*")}


def assert_nothing_written(root: Path, name: str = "demo", before: set[str] | None = None) -> None:
    assert not data_dir(root, name).exists()
    assert not staging_dir(root, name).exists()
    assert not old_dir(root, name).exists()
    assert not manifest_path(root, name).exists()
    if before is not None:
        assert tree(root) == before


# --- Frames -----------------------------------------------------------------------------------


def generic_frame() -> pl.DataFrame:
    """Six rows: String, Int64, Float64 (NaN, null, ±inf), match / competition / season ids
    and a Date. Groups: (c1, s1) 2 rows, (c1, s2) 2 rows, (c2, s1) 1 row, (null, s1) 1 row."""
    return pl.DataFrame(
        [
            pl.Series("event_id", ["e1", "e2", "e3", "e4", "e5", "e6"], dtype=pl.String),
            pl.Series("match_id", ["m1", "m1", "m2", None, "m3", "m2"], dtype=pl.String),
            pl.Series("competition_id", ["c1", "c1", "c1", "c2", None, "c1"], dtype=pl.String),
            pl.Series("season_id", ["s1", "s1", "s2", "s1", "s1", "s2"], dtype=pl.String),
            pl.Series(
                "match_date",
                [
                    dt.date(2024, 1, 6),
                    dt.date(2024, 1, 6),
                    dt.date(2024, 2, 3),
                    None,
                    dt.date(2024, 3, 9),
                    dt.date(2024, 2, 10),
                ],
                dtype=pl.Date,
            ),
            pl.Series("minute", [1, 2, 3, None, 5, 6], dtype=pl.Int64),
            pl.Series("value", [0.5, NAN, None, INF, -INF, 1e308], dtype=pl.Float64),
        ]
    )


def nested_frame(n: int = 150_000) -> pl.DataFrame:
    """A ``List(Array)`` column (the ``visible_area`` dtype) longer than ``row_group_size``:
    the one shape where the chunk layout of the sorted frame reaches the file bytes."""
    return pl.DataFrame(
        [
            pl.Series("k", list(range(n)), dtype=pl.Int64),
            pl.Series(
                "poly",
                [[[1.0, 2.0], [3.0, float(i)]] for i in range(n)],
                dtype=pl.List(pl.Array(pl.Float64, 2)),
            ),
        ]
    )


def lookup_frame() -> pl.DataFrame:
    """A generic table without any id column: no matches, no groups."""
    return pl.DataFrame(
        [
            pl.Series("code", ["b", "a", "c"], dtype=pl.String),
            pl.Series("weight", [2.0, 1.0, 3.0], dtype=pl.Float64),
        ]
    )


def canonical_tables() -> dict[str, pl.DataFrame]:
    return {
        "shots": schema_cases.shots_frame(),
        "freeze_frames": schema_cases.freeze_frames_frame(),
    }


def all_tables() -> dict[str, pl.DataFrame]:
    return {"events": generic_frame(), **canonical_tables()}


def date_frame(days: Sequence[int | None], *, grouped: bool) -> pl.DataFrame:
    """A generic frame whose ``match_date`` is given as days since 1970-01-01."""
    n = len(days)
    columns = [
        pl.Series("event_id", [f"e{i}" for i in range(n)], dtype=pl.String),
        pl.Series("match_date", list(days), dtype=pl.Int32).cast(pl.Date),
    ]
    if grouped:
        columns += [
            pl.Series("competition_id", ["c1"] * n, dtype=pl.String),
            pl.Series("season_id", ["s1"] * n, dtype=pl.String),
        ]
    return pl.DataFrame(columns)


def iso_of_day(day: int) -> str:
    return dt.date.fromordinal(day + EPOCH_ORDINAL).isoformat()


def sort_all(frame: pl.DataFrame, keys: Sequence[str] = ()) -> pl.DataFrame:
    """The spec §2.1 order: the explicit keys, then every remaining column in column order."""
    rest = [c for c in frame.columns if c not in keys]
    return frame.sort([*keys, *rest], nulls_last=True, maintain_order=True)


# --- Sources ----------------------------------------------------------------------------------


def git_source(uri: str = "https://example.invalid/synthetic") -> snapshot.Source:
    return snapshot.source_from_git(uri, GIT_SHA)


def write(
    root: Path,
    tables: Mapping[str, pl.DataFrame] | None = None,
    *,
    name: str = "demo",
    source: snapshot.Source | None = None,
    **kwargs: Any,
) -> snapshot.SnapshotResult:
    return snapshot.write_snapshot(
        name,
        all_tables() if tables is None else tables,
        git_source() if source is None else source,
        root=root,
        **kwargs,
    )


# --- Bytes and digests ------------------------------------------------------------------------


def flip_byte(path: Path, offset: int) -> None:
    """XOR 0x01 into the byte at ``offset`` (negative offsets count from the end)."""
    data = bytearray(path.read_bytes())
    data[offset] ^= 0x01
    path.write_bytes(bytes(data))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_sha(root: Path, name: str = "demo") -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(data_dir(root, name).iterdir())}


def files_bytes(root: Path, name: str = "demo") -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(data_dir(root, name).iterdir())}


def sha256sum_digest(pairs: Iterable[tuple[str, str]]) -> str:
    """Spec §2.3, written independently: the ``sha256sum`` lines sorted by key, hashed."""
    text = "".join(f"{sha}  {key}\n" for key, sha in sorted(pairs))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_json(root: Path, name: str = "demo") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(manifest_path(root, name).read_text(encoding="utf-8"))
    return loaded


def table_entry(manifest: Mapping[str, Any], table: str) -> dict[str, Any]:
    (entry,) = [t for t in manifest["tables"] if t["name"] == table]
    return dict(entry)


# --- Git --------------------------------------------------------------------------------------


def git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return env | {"GIT_CONFIG_NOSYSTEM": "1", "HOME": str(Path.home())}


def git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        [
            "git",
            "-C",
            str(cwd),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
            *args,
        ],
        capture_output=True,
        encoding="utf-8",
        env=git_env(),
        check=True,
    )
    return out.stdout


def git_repo(path: Path, *, ignore: str = "data/*\n", commit: bool = True) -> Path:
    """``git init`` + one commit holding ``.gitignore`` (and a README)."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    (path / ".gitignore").write_text(ignore, encoding="utf-8")
    (path / "README.md").write_text("synthetic test repo\n", encoding="utf-8")
    if commit:
        git(path, "add", ".gitignore", "README.md")
        git(path, "commit", "-q", "-m", "init")
    return path


def head_sha(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").rstrip("\n")


# --- A hand-written, valid manifest (spec §2.2) -----------------------------------------------

EVENTS_SHA = "d" * 64
LOOKUP_SHA = "e" * 64
SOURCE_EVENTS_SHA = "1" * 64
SOURCE_LOOKUP_SHA = "2" * 64


def _events_columns() -> list[dict[str, str]]:
    return [
        {"name": "event_id", "dtype": "String"},
        {"name": "match_id", "dtype": "String"},
        {"name": "competition_id", "dtype": "String"},
        {"name": "season_id", "dtype": "String"},
        {"name": "match_date", "dtype": "Date"},
    ]


def base_manifest() -> dict[str, Any]:
    """A manifest that satisfies every ``from_json`` rule, keys in the §2.2 order."""
    source_files = [
        {"path": "exports/events.parquet", "sha256": SOURCE_EVENTS_SHA, "bytes": 1234},
        {"path": "exports/lookup.parquet", "sha256": SOURCE_LOOKUP_SHA, "bytes": 99},
    ]
    files = [
        {"path": "events.parquet", "sha256": EVENTS_SHA, "bytes": 2048},
        {"path": "lookup.parquet", "sha256": LOOKUP_SHA, "bytes": 512},
    ]
    return {
        "manifest_version": 1,
        "name": "demo",
        "builder": "parquet",
        "schema_version": "1.0",
        "data_dir": "../data/snapshots/demo",
        "data_digest": sha256sum_digest((f["path"], f["sha256"]) for f in files),
        "source": {
            "kind": "files",
            "uri": "inputs",
            "revision": sha256sum_digest((f["path"], f["sha256"]) for f in source_files),
            "content_sha256": None,
            "files": source_files,
        },
        "tables": [
            {
                "name": "events",
                "file": "events.parquet",
                "rows": 4,
                "matches": 2,
                "schema": None,
                "sort_by": ["event_id"],
                "columns": _events_columns(),
                "groups": [
                    {
                        "competition_id": "c1",
                        "season_id": "s1",
                        "rows": 2,
                        "matches": 1,
                        "first_date": "2024-01-06",
                        "last_date": "2024-01-13",
                    },
                    {
                        "competition_id": "c1",
                        "season_id": "s2",
                        "rows": 1,
                        "matches": 1,
                        "first_date": "2024-02-03",
                        "last_date": "2024-02-03",
                    },
                    {
                        "competition_id": None,
                        "season_id": "s1",
                        "rows": 1,
                        "matches": 0,
                        "first_date": None,
                        "last_date": None,
                    },
                ],
            },
            {
                "name": "lookup",
                "file": "lookup.parquet",
                "rows": 3,
                "matches": None,
                "schema": None,
                "sort_by": [],
                "columns": [{"name": "code", "dtype": "String"}],
                "groups": None,
            },
        ],
        "files": files,
        "parquet": dict(PARQUET_LITERAL),
        "environment": {
            "python": "3.12.13",
            "polars": "2.0.0",
            "pandera": "0.34.1",
            "numpy": "2.5.4",
            "football_core": "0.1.0",
            "football_core_git": None,
            "football_core_dirty": None,
        },
    }


def dumps(obj: object) -> str:
    """The writer's serialisation (spec §2.2)."""
    return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def refresh_digest(m: dict[str, Any]) -> dict[str, Any]:
    """Recompute ``data_digest`` after a mutation of ``files`` (keeps rule 6 satisfied)."""
    m["data_digest"] = sha256sum_digest((f["path"], f["sha256"]) for f in m["files"])
    return m


def add_table(
    m: dict[str, Any], name: str, *, schema: str | None = None, sha: str = "f" * 64
) -> dict[str, Any]:
    """Add a consistent table + file entry, keep both lists sorted, refresh the digest."""
    m = copy.deepcopy(m)
    m["tables"].append(
        {
            "name": name,
            "file": f"{name}.parquet",
            "rows": 1,
            "matches": None,
            "schema": schema,
            "sort_by": [],
            "columns": [{"name": "k", "dtype": "String"}],
            "groups": None,
        }
    )
    m["tables"].sort(key=lambda t: t["name"])
    m["files"].append({"path": f"{name}.parquet", "sha256": sha, "bytes": 10})
    m["files"].sort(key=lambda f: f["path"])
    return refresh_digest(m)
