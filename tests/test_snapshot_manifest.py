"""The manifest: fields, counts, groups, environment, JSON format, round trip and every
``from_json`` rule (C6 spec §2.2, §2.3, §3.4, §5)."""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import importlib.metadata
import json
import platform
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import schema_cases
import snapshot_cases as sc
from hypothesis import given, settings
from hypothesis import strategies as st

from football_core import schema
from football_core.io import snapshot

TOP_KEYS = [
    "manifest_version",
    "name",
    "builder",
    "schema_version",
    "data_dir",
    "data_digest",
    "source",
    "tables",
    "files",
    "parquet",
    "environment",
]
SOURCE_KEYS = ["kind", "uri", "revision", "content_sha256", "files"]
TABLE_KEYS = ["name", "file", "rows", "matches", "schema", "sort_by", "columns", "groups"]
GROUP_KEYS = ["competition_id", "season_id", "rows", "matches", "first_date", "last_date"]
FILE_KEYS = ["path", "sha256", "bytes"]
ENV_KEYS = [
    "python",
    "polars",
    "pandera",
    "numpy",
    "football_core",
    "football_core_git",
    "football_core_dirty",
]
HEX64 = re.compile(r"[0-9a-f]{64}")


# --- Fields -----------------------------------------------------------------------------------


def test_fields_and_key_order(tmp_path: Path) -> None:
    sc.write(tmp_path, builder="toy-builder")
    m = sc.manifest_json(tmp_path)
    assert list(m) == TOP_KEYS
    assert list(m["source"]) == SOURCE_KEYS
    assert m["manifest_version"] == 1 == snapshot.MANIFEST_VERSION
    assert m["name"] == "demo"
    assert m["builder"] == "toy-builder"
    assert m["schema_version"] == schema.SCHEMA_VERSION
    assert m["data_dir"] == "../data/snapshots/demo"
    assert m["source"] == {
        "kind": "git",
        "uri": "https://example.invalid/synthetic",
        "revision": sc.GIT_SHA,
        "content_sha256": None,
        "files": [],
    }
    assert [t["name"] for t in m["tables"]] == ["events", "freeze_frames", "shots"]
    for table in m["tables"]:
        assert list(table) == TABLE_KEYS
        assert table["file"] == f"{table['name']}.parquet"
        for column in table["columns"]:
            assert list(column) == ["name", "dtype"]
        for group in table["groups"] or []:
            assert list(group) == GROUP_KEYS
    on_disk = sc.data_dir(tmp_path)
    assert m["files"] == [
        {"path": p.name, "sha256": sc.sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(on_disk.iterdir())
    ]
    for entry in m["files"]:
        assert list(entry) == FILE_KEYS
    assert m["data_digest"] == sc.sha256sum_digest((f["path"], f["sha256"]) for f in m["files"])
    assert list(m["environment"]) == ENV_KEYS


def test_builder_null_when_written_directly(tmp_path: Path) -> None:
    sc.write(tmp_path)
    assert sc.manifest_json(tmp_path)["builder"] is None


def test_columns_and_dtypes_recorded(tmp_path: Path) -> None:
    sc.write(tmp_path)
    m = sc.manifest_json(tmp_path)
    events = sc.table_entry(m, "events")
    assert events["columns"] == [
        {"name": name, "dtype": str(dtype)} for name, dtype in sc.generic_frame().schema.items()
    ]
    shots = sc.table_entry(m, "shots")
    assert [(c["name"], c["dtype"]) for c in shots["columns"]] == [
        (name, str(dtype)) for name, dtype, _ in schema_cases.SHOT_COLUMNS
    ]
    assert events["sort_by"] == []


@pytest.mark.parametrize(
    ("match_ids", "expected"),
    [
        (["m1", "m1", None, "m2"], 2),
        ([None, None], 0),
        ([], 0),
        (["m1"], 1),
        (["", "m1"], 2),
    ],
    ids=["with_nulls", "all_null", "empty", "one", "empty_string_counts"],
)
def test_rows_and_matches(tmp_path: Path, match_ids: list[str | None], expected: int) -> None:
    frame = pl.DataFrame(
        [
            pl.Series("k", [f"r{i}" for i in range(len(match_ids))], dtype=pl.String),
            pl.Series("match_id", match_ids, dtype=pl.String),
        ]
    )
    sc.write(tmp_path, {"t": frame, "lookup": sc.lookup_frame()})
    m = sc.manifest_json(tmp_path)
    assert sc.table_entry(m, "t")["rows"] == len(match_ids)
    assert sc.table_entry(m, "t")["matches"] == expected
    assert sc.table_entry(m, "lookup")["rows"] == 3
    assert sc.table_entry(m, "lookup")["matches"] is None


def test_matches_with_integer_match_id(tmp_path: Path) -> None:
    frame = pl.DataFrame([pl.Series("match_id", [3, 3, 7, None], dtype=pl.Int64)])
    sc.write(tmp_path, {"t": frame})
    assert sc.table_entry(sc.manifest_json(tmp_path), "t")["matches"] == 2


# --- Groups -----------------------------------------------------------------------------------


def _group(
    comp: str | None,
    season: str | None,
    rows: int,
    matches: int | None,
    first: str | None,
    last: str | None,
) -> dict[str, Any]:
    return {
        "competition_id": comp,
        "season_id": season,
        "rows": rows,
        "matches": matches,
        "first_date": first,
        "last_date": last,
    }


def _frame_groups_basic() -> pl.DataFrame:
    return sc.generic_frame()


def _frame_null_dates() -> pl.DataFrame:
    return sc.generic_frame().with_columns(
        pl.when(pl.col("season_id") == "s2")
        .then(None)
        .otherwise(pl.col("match_date"))
        .alias("match_date")
    )


def _frame_string_dates() -> pl.DataFrame:
    return sc.generic_frame().with_columns(pl.col("match_date").cast(pl.String))


def _frame_datetime_dates() -> pl.DataFrame:
    return sc.generic_frame().with_columns(pl.col("match_date").cast(pl.Datetime("us")))


def _frame_no_match_id() -> pl.DataFrame:
    return sc.generic_frame().drop("match_id")


def _frame_int_keys() -> pl.DataFrame:
    return pl.DataFrame(
        [
            pl.Series("competition_id", [2, 11, 2], dtype=pl.Int64),
            pl.Series("season_id", [1, 1, 1], dtype=pl.Int16),
            pl.Series("match_date", [dt.date(2024, 1, 1)] * 3, dtype=pl.Date),
        ]
    )


def _frame_enum_keys() -> pl.DataFrame:
    return pl.DataFrame(
        [
            pl.Series("competition_id", ["z", "a", "z"], dtype=pl.Enum(["z", "a"])),
            pl.Series("season_id", ["s", "s", "s"], dtype=pl.Categorical),
        ]
    )


def _frame_null_season() -> pl.DataFrame:
    return pl.DataFrame(
        [
            pl.Series("competition_id", ["c", "c", None, None], dtype=pl.String),
            pl.Series("season_id", [None, "s", None, "s"], dtype=pl.String),
        ]
    )


GROUP_CASES: dict[str, tuple[Callable[[], pl.DataFrame], list[dict[str, Any]] | None]] = {
    "basic": (
        _frame_groups_basic,
        [
            _group("c1", "s1", 2, 1, "2024-01-06", "2024-01-06"),
            _group("c1", "s2", 2, 1, "2024-02-03", "2024-02-10"),
            _group("c2", "s1", 1, 0, None, None),
            _group(None, "s1", 1, 1, "2024-03-09", "2024-03-09"),
        ],
    ),
    "all_null_dates_in_one_group": (
        _frame_null_dates,
        [
            _group("c1", "s1", 2, 1, "2024-01-06", "2024-01-06"),
            _group("c1", "s2", 2, 1, None, None),
            _group("c2", "s1", 1, 0, None, None),
            _group(None, "s1", 1, 1, "2024-03-09", "2024-03-09"),
        ],
    ),
    "string_match_date": (
        _frame_string_dates,
        [
            _group("c1", "s1", 2, 1, None, None),
            _group("c1", "s2", 2, 1, None, None),
            _group("c2", "s1", 1, 0, None, None),
            _group(None, "s1", 1, 1, None, None),
        ],
    ),
    "datetime_match_date": (
        _frame_datetime_dates,
        [
            _group("c1", "s1", 2, 1, None, None),
            _group("c1", "s2", 2, 1, None, None),
            _group("c2", "s1", 1, 0, None, None),
            _group(None, "s1", 1, 1, None, None),
        ],
    ),
    "no_match_id": (
        _frame_no_match_id,
        [
            _group("c1", "s1", 2, None, "2024-01-06", "2024-01-06"),
            _group("c1", "s2", 2, None, "2024-02-03", "2024-02-10"),
            _group("c2", "s1", 1, None, None, None),
            _group(None, "s1", 1, None, "2024-03-09", "2024-03-09"),
        ],
    ),
    "int_keys_lexical": (
        _frame_int_keys,
        [
            _group("11", "1", 1, None, "2024-01-01", "2024-01-01"),
            _group("2", "1", 2, None, "2024-01-01", "2024-01-01"),
        ],
    ),
    "enum_label_order": (
        _frame_enum_keys,
        [_group("a", "s", 1, None, None, None), _group("z", "s", 2, None, None, None)],
    ),
    "nulls_last_in_both_keys": (
        _frame_null_season,
        [
            _group("c", "s", 1, None, None, None),
            _group("c", None, 1, None, None, None),
            _group(None, "s", 1, None, None, None),
            _group(None, None, 1, None, None, None),
        ],
    ),
    "unsupported_key_dtype": (
        lambda: sc.generic_frame().with_columns(pl.lit(1.5).alias("season_id")),
        None,
    ),
    "only_one_key": (lambda: sc.generic_frame().drop("season_id"), None),
}


@pytest.mark.parametrize("case", sorted(GROUP_CASES))
def test_groups(tmp_path: Path, case: str) -> None:
    build, expected = GROUP_CASES[case]
    result = sc.write(tmp_path, {"events": build()})
    entry = sc.table_entry(sc.manifest_json(tmp_path), "events")
    assert entry["groups"] == expected
    assert snapshot.read_manifest(result.manifest_path) == result.manifest


# --- Schema, environment, parquet settings ----------------------------------------------------


def test_schema_version_and_table_schema(tmp_path: Path) -> None:
    sc.write(tmp_path, {**sc.all_tables(), "shots_extra": sc.lookup_frame()})
    m = sc.manifest_json(tmp_path)
    assert m["schema_version"] == schema.SCHEMA_VERSION
    assert {t["name"]: t["schema"] for t in m["tables"]} == {
        "events": None,
        "freeze_frames": "freeze_frames",
        "shots": "shots",
        "shots_extra": None,
    }


def test_environment_matches_importlib_metadata() -> None:
    env = snapshot.environment()
    assert list(env) == ENV_KEYS
    assert env["python"] == platform.python_version()
    for key, dist in [
        ("polars", "polars"),
        ("pandera", "pandera"),
        ("numpy", "numpy"),
        ("football_core", "football-core"),
    ]:
        assert env[key] == importlib.metadata.version(dist)
    git_sha, dirty = env["football_core_git"], env["football_core_dirty"]
    assert (git_sha is None) == (dirty is None)
    if git_sha is not None:
        assert isinstance(git_sha, str)
        assert re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", git_sha)
        assert isinstance(dirty, bool)


def test_manifest_environment_is_environment(tmp_path: Path) -> None:
    sc.write(tmp_path)
    assert sc.manifest_json(tmp_path)["environment"] == snapshot.environment()


def test_environment_missing_distribution_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.metadata.version

    def fake(dist: str) -> str:
        if dist == "pandera":
            raise importlib.metadata.PackageNotFoundError(dist)
        return real(dist)

    monkeypatch.setattr(importlib.metadata, "version", fake)
    if getattr(snapshot, "version", None) is real:
        monkeypatch.setattr(snapshot, "version", fake)
    env = snapshot.environment()
    assert env["pandera"] is None
    assert env["polars"] == real("polars")


def test_parquet_settings_recorded(tmp_path: Path) -> None:
    assert dict(snapshot.PARQUET_SETTINGS) == sc.PARQUET_LITERAL
    sc.write(tmp_path)
    recorded = sc.manifest_json(tmp_path)["parquet"]
    assert recorded == sc.PARQUET_LITERAL
    assert list(recorded) == list(sc.PARQUET_LITERAL)
    assert type(recorded["statistics"]) is bool
    assert type(recorded["compression_level"]) is int
    assert snapshot._parquet_record(snapshot.PARQUET_SETTINGS) == sc.PARQUET_LITERAL


def test_data_dir_relative_and_no_absolute_path(tmp_path: Path) -> None:
    root = tmp_path / "consumer"
    sc.write(root)
    text = sc.manifest_path(root).read_text(encoding="utf-8")
    for absolute in {str(tmp_path), str(tmp_path.resolve()), str(Path.home())}:
        assert absolute not in text
    assert sc.manifest_json(root)["data_dir"] == "../data/snapshots/demo"


_TIME_KEY = re.compile(r"time|created|updated|_at$|stamp|when|date", re.IGNORECASE)


def _keys(obj: object) -> list[str]:
    if isinstance(obj, dict):
        return [k for k, v in obj.items() for k in [k, *_keys(v)]]
    if isinstance(obj, list):
        return [k for v in obj for k in _keys(v)]
    return []


def test_no_timestamp_fields(tmp_path: Path) -> None:
    sc.write(tmp_path)
    m = sc.manifest_json(tmp_path)
    timeish = {k for k in _keys(m) if _TIME_KEY.search(k)}
    assert timeish <= {"first_date", "last_date"}
    text = sc.manifest_path(tmp_path).read_text(encoding="utf-8")
    assert dt.date.today().isoformat() not in text


def test_json_format(tmp_path: Path) -> None:
    sc.write(tmp_path, source=sc.git_source("Masaüstü/exports"))
    raw = sc.manifest_path(tmp_path).read_bytes()
    text = raw.decode("utf-8")
    assert text.endswith("}\n")
    assert not text.endswith("\n\n")
    lines = text.splitlines()
    assert lines[0] == "{"
    assert lines[1] == '  "manifest_version": 1,'
    assert "NaN" not in text and "Infinity" not in text
    assert "Masaüstü/exports" in text  # ensure_ascii=False
    assert "\\u00fc" not in text
    assert "\r" not in text
    assert text == sc.dumps(json.loads(text))


# --- Round trip -------------------------------------------------------------------------------


def test_round_trip(tmp_path: Path) -> None:
    result = sc.write(tmp_path)
    text = result.manifest_path.read_text(encoding="utf-8")
    parsed = snapshot.Manifest.from_json(text)
    assert parsed == result.manifest
    assert parsed.to_json() == text
    assert snapshot.Manifest.from_json(parsed.to_json()) == parsed


def test_round_trip_hand_written_manifest() -> None:
    text = sc.dumps(sc.base_manifest())
    parsed = snapshot.Manifest.from_json(text)
    assert parsed.to_json() == text
    assert parsed.name == "demo"
    assert parsed.builder == "parquet"
    assert parsed.source.kind == "files"
    assert [f.path for f in parsed.source.files] == [
        "exports/events.parquet",
        "exports/lookup.parquet",
    ]
    assert [t.name for t in parsed.tables] == ["events", "lookup"]
    assert parsed.tables[0].groups is not None
    assert parsed.tables[0].groups[2].competition_id is None
    assert parsed.tables[1].groups is None
    assert parsed.tables[0].columns[0] == ("event_id", "String")
    assert parsed.tables[0].sort_by == ("event_id",)
    assert parsed.files[0] == snapshot.FileEntry("events.parquet", sc.EVENTS_SHA, 2048)


_KEY = st.text(alphabet="abc_", min_size=1, max_size=4)
_SCALAR = st.one_of(
    st.booleans(),
    st.integers(min_value=-(10**18), max_value=10**18),
    st.text(max_size=5),
)
_ENV_VALUE = st.one_of(st.none(), st.booleans(), st.text(max_size=5))


@settings(deadline=None, max_examples=60)
@given(
    parquet=st.dictionaries(_KEY, _SCALAR, max_size=5),
    env=st.dictionaries(_KEY, _ENV_VALUE, max_size=5),
    name=st.from_regex(r"[a-z0-9][a-z0-9._-]{0,10}", fullmatch=True),
    builder=st.one_of(st.none(), st.from_regex(r"[a-z0-9][a-z0-9._-]{0,10}", fullmatch=True)),
)
def test_round_trip_property(
    parquet: dict[str, bool | int | str],
    env: dict[str, bool | str | None],
    name: str,
    builder: str | None,
) -> None:
    base = sc.base_manifest()
    base["name"] = name
    base["builder"] = builder
    base["data_dir"] = f"../data/snapshots/{name}"
    base["parquet"] = parquet
    base["environment"] = env
    text = sc.dumps(base)
    m = snapshot.Manifest.from_json(text)
    assert m.to_json() == text
    again = snapshot.Manifest.from_json(m.to_json())
    assert again == m
    # bool / int confusion is invisible to ==, so compare the types too (QA r3 m1)
    assert {k: type(v) for k, v in again.parquet.items()} == {
        k: type(v) for k, v in parquet.items()
    }
    assert {k: type(v) for k, v in again.environment.items()} == {
        k: type(v) for k, v in env.items()
    }
    assert dict(again.parquet) == parquet


def test_round_trip_keeps_parquet_mapping_not_constant() -> None:
    base = sc.base_manifest()
    base["parquet"] = {"compression": "zstd", "compression_level": 9, "statistics": False}
    text = sc.dumps(base)
    assert snapshot.Manifest.from_json(text).to_json() == text


def test_unknown_keys_ignored() -> None:
    plain = snapshot.Manifest.from_json(sc.dumps(sc.base_manifest()))
    m = sc.base_manifest()
    m["future_field"] = {"x": 1}
    m["source"]["future"] = 1
    m["source"]["files"][0]["future"] = [1, 2]
    m["tables"][0]["future"] = None
    m["tables"][0]["columns"][0]["future"] = "x"
    m["tables"][0]["groups"][0]["future"] = 1.5
    m["files"][0]["future"] = True
    assert snapshot.Manifest.from_json(sc.dumps(m)) == plain


def test_key_order_not_checked_on_read() -> None:
    m = sc.base_manifest()
    reordered = dict(reversed(list(m.items())))
    reordered["source"] = dict(reversed(list(m["source"].items())))
    assert snapshot.Manifest.from_json(json.dumps(reordered)) == snapshot.Manifest.from_json(
        sc.dumps(m)
    )


# --- from_json rejects every rule violation (spec §2.2 rules 1-15) ----------------------------

Mutation = Callable[[dict[str, Any]], Any]


def _set(path: str, value: object) -> Mutation:
    """Set a value by a dotted path with list indices, e.g. ``tables.0.rows``."""

    def mutate(m: dict[str, Any]) -> dict[str, Any]:
        *parents, last = path.split(".")
        node: Any = m
        for part in parents:
            node = node[int(part)] if part.isdigit() else node[part]
        if last.isdigit():
            node[int(last)] = value
        else:
            node[last] = value
        return m

    return mutate


def _delete(path: str) -> Mutation:
    def mutate(m: dict[str, Any]) -> dict[str, Any]:
        *parents, last = path.split(".")
        node: Any = m
        for part in parents:
            node = node[int(part)] if part.isdigit() else node[part]
        del node[last]
        return m

    return mutate


def _text(raw: str) -> Mutation:
    return lambda m: raw


def _with_files_digest(mutate: Mutation) -> Mutation:
    def wrapped(m: dict[str, Any]) -> Any:
        return sc.refresh_digest(mutate(m))

    return wrapped


def _source_files(files: list[dict[str, Any]], *, fix_revision: bool = True) -> Mutation:
    def mutate(m: dict[str, Any]) -> dict[str, Any]:
        m["source"]["files"] = files
        if fix_revision:
            m["source"]["revision"] = sc.sha256sum_digest((f["path"], f["sha256"]) for f in files)
        return m

    return mutate


def _deep(m: dict[str, Any]) -> str:
    return '{"manifest_version": ' + "[" * 100_000 + "]" * 100_000 + "}"


def _duplicate_key(m: dict[str, Any]) -> str:
    text = sc.dumps(m)
    return text.replace('"name": "demo",', '"name": "demo",\n  "name": "demo",', 1)


def _nan(m: dict[str, Any]) -> str:
    m["tables"][0]["rows"] = "__NAN__"
    return sc.dumps(m).replace('"__NAN__"', "NaN")


def _infinity(m: dict[str, Any]) -> str:
    m["parquet"]["compression_level"] = "__INF__"
    return sc.dumps(m).replace('"__INF__"', "Infinity")


def _git_source_with_files(m: dict[str, Any]) -> dict[str, Any]:
    m["source"]["kind"] = "git"
    m["source"]["revision"] = sc.GIT_SHA
    return m


def _sql_source(content: str) -> Mutation:
    def mutate(m: dict[str, Any]) -> dict[str, Any]:
        m["source"] = {
            "kind": "sql",
            "uri": "db:raw.table",
            "revision": "a" * 64,
            "content_sha256": content,
            "files": [],
        }
        return m

    return mutate


def _git_source_with_content(m: dict[str, Any]) -> dict[str, Any]:
    m["source"] = {
        "kind": "git",
        "uri": "https://example.invalid/x",
        "revision": sc.GIT_SHA,
        "content_sha256": "b" * 64,
        "files": [],
    }
    return m


def _unsorted_tables(m: dict[str, Any]) -> dict[str, Any]:
    m["tables"].reverse()
    return m


def _duplicate_table(m: dict[str, Any]) -> dict[str, Any]:
    m["tables"] = [m["tables"][0], copy.deepcopy(m["tables"][0]), m["tables"][1]]
    return m


def _rename_table(new: str) -> Mutation:
    def mutate(m: dict[str, Any]) -> dict[str, Any]:
        m["tables"][1]["name"] = new
        m["tables"][1]["file"] = f"{new}.parquet"
        m["files"][1]["path"] = f"{new}.parquet"
        return sc.refresh_digest(m)

    return mutate


def _canonical_without_schema(m: dict[str, Any]) -> dict[str, Any]:
    return sc.add_table(m, "shots", schema=None)


def _freeze_frames_without_shots(m: dict[str, Any]) -> dict[str, Any]:
    return sc.add_table(m, "freeze_frames", schema="freeze_frames")


def _duplicate_column(m: dict[str, Any]) -> dict[str, Any]:
    m["tables"][1]["columns"].append({"name": "code", "dtype": "Int64"})
    return m


def _groups_without_keys(m: dict[str, Any]) -> dict[str, Any]:
    m["tables"][1]["groups"] = [
        {
            "competition_id": "c",
            "season_id": "s",
            "rows": 3,
            "matches": None,
            "first_date": None,
            "last_date": None,
        }
    ]
    return m


def _groups_unsorted(m: dict[str, Any]) -> dict[str, Any]:
    groups = m["tables"][0]["groups"]
    groups[0], groups[1] = groups[1], groups[0]
    return m


def _groups_null_first(m: dict[str, Any]) -> dict[str, Any]:
    groups = m["tables"][0]["groups"]
    m["tables"][0]["groups"] = [groups[2], groups[0], groups[1]]
    return m


def _group_duplicate(m: dict[str, Any]) -> dict[str, Any]:
    groups = m["tables"][0]["groups"]
    groups[1]["season_id"] = "s1"
    return m


def _group_rows_sum(m: dict[str, Any]) -> dict[str, Any]:
    m["tables"][0]["rows"] = 5
    return m


def _files_unsorted(m: dict[str, Any]) -> dict[str, Any]:
    m["files"].reverse()
    return sc.refresh_digest(m)


def _files_duplicate(m: dict[str, Any]) -> dict[str, Any]:
    m["files"].insert(1, copy.deepcopy(m["files"][0]))
    return sc.refresh_digest(m)


def _file_without_table(m: dict[str, Any]) -> dict[str, Any]:
    m["files"].append({"path": "zz.parquet", "sha256": "0" * 64, "bytes": 1})
    return sc.refresh_digest(m)


def _table_file_not_in_files(m: dict[str, Any]) -> dict[str, Any]:
    del m["files"][1]
    return sc.refresh_digest(m)


def _source_unsorted(m: dict[str, Any]) -> dict[str, Any]:
    files = list(reversed(m["source"]["files"]))
    return _source_files(files)(m)


def _source_duplicate(m: dict[str, Any]) -> dict[str, Any]:
    files = [m["source"]["files"][0], copy.deepcopy(m["source"]["files"][0])]
    return _source_files(files)(m)


def _source_path(path: str) -> Mutation:
    def mutate(m: dict[str, Any]) -> dict[str, Any]:
        files = copy.deepcopy(m["source"]["files"])
        files[0]["path"] = path
        return _source_files(files)(m)

    return mutate


# (case id, mutation, a substring the message must contain or None)
REJECT_CASES: list[tuple[str, Mutation, str | None]] = [
    # rule 1
    ("invalid_json", _text("{ not json"), None),
    ("empty_text", _text(""), None),
    ("nan", _nan, None),
    ("infinity", _infinity, None),
    ("deep_nesting", _deep, None),
    ("duplicate_key", _duplicate_key, "name"),
    # rule 2
    ("top_level_list", lambda m: [m], None),
    ("top_level_null", lambda m: None, None),
    ("missing_key", _delete("data_digest"), "data_digest"),
    ("missing_source_kind", _delete("source.kind"), "kind"),
    ("missing_table_rows", _delete("tables.0.rows"), "rows"),
    ("wrong_type", _set("tables.0.rows", "4"), "rows"),
    ("bool_as_int", _set("tables.0.rows", True), "rows"),
    ("float_as_int", _set("tables.0.rows", 4.0), "rows"),
    ("wrong_type_tables", _set("tables", {"events": {}}), "tables"),
    ("wrong_type_name", _set("name", 5), "name"),
    # rule 3
    ("manifest_version_2", _set("manifest_version", 2), "manifest_version"),
    ("manifest_version_bool", _set("manifest_version", True), "manifest_version"),
    ("manifest_version_str", _set("manifest_version", "1"), "manifest_version"),
    # rule 4
    ("bad_name", _set("name", "Demo"), "name"),
    ("bad_name_newline", _set("name", "demo\n"), "name"),
    ("bad_builder", _set("builder", "Parquet loader"), "builder"),
    ("bad_builder_empty", _set("builder", ""), "builder"),
    ("bad_schema_version", _set("schema_version", "1"), "schema_version"),
    ("bad_schema_version_v", _set("schema_version", "v1.0"), "schema_version"),
    ("bad_schema_version_newline", _set("schema_version", "1.0\n"), "schema_version"),
    # rule 5
    ("data_dir_absolute", _set("data_dir", "/etc"), "data_dir"),
    ("data_dir_escape", _set("data_dir", "../../x"), "data_dir"),
    ("data_dir_other_name", _set("data_dir", "../data/snapshots/other"), "data_dir"),
    ("data_dir_trailing_slash", _set("data_dir", "../data/snapshots/demo/"), "data_dir"),
    ("data_dir_dot_segment", _set("data_dir", "../data/./snapshots/demo"), "data_dir"),
    # rule 6
    ("bad_digest_hex", _set("data_digest", "Z" * 64), "data_digest"),
    ("bad_digest_upper", lambda m: _set("data_digest", m["data_digest"].upper())(m), "data_digest"),
    ("digest_mismatch", _set("data_digest", "0" * 64), "data_digest"),
    # rule 7
    ("source_bad_kind", _set("source.kind", "svn"), "source"),
    ("source_short_revision", _set("source.revision", "a" * 40), "source"),
    ("source_files_on_git", _git_source_with_files, "source"),
    ("source_files_unsorted", _source_unsorted, "source"),
    ("source_files_duplicate", _source_duplicate, "source"),
    ("source_revision_mismatch", _set("source.revision", "a" * 64), "source"),
    ("source_path_newline", _source_path("exports/a\nb.parquet"), "source"),
    ("source_path_absolute", _source_path("/exports/a.parquet"), "source"),
    ("source_path_dotdot", _source_path("../a.parquet"), "source"),
    ("source_path_backslash", _source_path("exports\\a.parquet"), "source"),
    ("source_bytes_bool", _set("source.files.0.bytes", True), "source"),
    ("source_uri_empty", _set("source.uri", ""), "source"),
    ("source_uri_newline", _set("source.uri", "a\nb"), "source"),
    ("content_sha256_bad_hex", _sql_source("g" * 64), "source"),
    ("content_sha256_upper", _sql_source("B" * 64), "source"),
    ("content_sha256_on_git", _git_source_with_content, "source"),
    ("content_sha256_on_files", _set("source.content_sha256", "a" * 64), "source"),
    # rule 8
    ("no_tables", _set("tables", []), "tables"),
    ("tables_unsorted", _unsorted_tables, "tables"),
    ("tables_duplicate", _duplicate_table, "tables"),
    ("bad_table_name", _rename_table("lookup-x"), "tables"),
    ("bad_table_name_newline", _rename_table("lookup\n"), "tables"),
    ("table_file_mismatch", _set("tables.1.file", "other.parquet"), "file"),
    # rule 9
    ("negative_rows", _set("tables.1.rows", -1), "rows"),
    ("negative_matches", _set("tables.0.matches", -1), "matches"),
    ("matches_gt_rows", _set("tables.0.matches", 5), "matches"),
    ("matches_null_with_match_id", _set("tables.0.matches", None), "matches"),
    ("matches_without_match_id", _set("tables.1.matches", 0), "matches"),
    # rule 10
    ("bad_schema_value", _set("tables.0.schema", "events"), "schema"),
    ("schema_name_mismatch", _set("tables.1.schema", "shots"), "schema"),
    ("canonical_without_schema", _canonical_without_schema, "schema"),
    ("freeze_frames_without_shots", _freeze_frames_without_shots, "freeze_frames"),
    # rule 11
    ("no_columns", _set("tables.1.columns", []), "columns"),
    ("duplicate_column", _duplicate_column, "columns"),
    ("empty_dtype", _set("tables.1.columns.0.dtype", ""), "dtype"),
    ("sort_by_unknown_column", _set("tables.0.sort_by", ["nope"]), "sort_by"),
    ("sort_by_duplicate", _set("tables.0.sort_by", ["event_id", "event_id"]), "sort_by"),
    # rule 12
    ("groups_without_keys", _groups_without_keys, "groups"),
    ("group_zero_rows", _set("tables.0.groups.2.rows", 0), "rows"),
    ("group_matches_gt_rows", _set("tables.0.groups.0.matches", 3), "matches"),
    ("group_negative_matches", _set("tables.0.groups.0.matches", -1), "matches"),
    ("group_matches_nullness", _set("tables.0.groups.0.matches", None), "matches"),
    ("group_bad_date", _set("tables.0.groups.0.first_date", "2024-02-30"), "first_date"),
    ("group_bad_date_format", _set("tables.0.groups.0.first_date", "06/01/2024"), "first_date"),
    ("group_date_newline", _set("tables.0.groups.0.last_date", "2024-01-13\n"), "last_date"),
    ("group_date_arabic_digits", _set("tables.0.groups.0.first_date", "٢٠٢٤-٠١-٠٦"), "first_date"),
    ("group_year_zero", _set("tables.0.groups.0.first_date", "0000-12-31"), "first_date"),
    ("group_signed_year", _set("tables.0.groups.0.last_date", "+10000-01-01"), "last_date"),
    ("group_one_date_null", _set("tables.0.groups.0.last_date", None), "date"),
    ("group_dates_reversed", _set("tables.0.groups.0.first_date", "2024-01-20"), "date"),
    ("groups_unsorted", _groups_unsorted, "groups"),
    ("groups_null_not_last", _groups_null_first, "groups"),
    ("groups_duplicate", _group_duplicate, "groups"),
    ("group_rows_sum", _group_rows_sum, "rows"),
    # rule 13
    ("files_unsorted", _files_unsorted, "files"),
    ("files_duplicate", _files_duplicate, "files"),
    ("path_slash", _with_files_digest(_set("files.1.path", "sub/lookup.parquet")), "path"),
    ("path_dotdot", _with_files_digest(_set("files.1.path", "..")), "path"),
    ("path_leading_dot", _with_files_digest(_set("files.1.path", ".lookup.parquet")), "path"),
    ("bad_sha256", _with_files_digest(_set("files.0.sha256", "D" * 64)), "sha256"),
    ("bad_sha256_short", _with_files_digest(_set("files.0.sha256", "d" * 63)), "sha256"),
    ("bad_sha256_newline", _with_files_digest(_set("files.0.sha256", "d" * 64 + "\n")), "sha256"),
    ("negative_bytes", _set("files.0.bytes", -1), "bytes"),
    ("bool_bytes", _set("files.0.bytes", False), "bytes"),
    # rule 14
    ("table_file_not_in_files", _table_file_not_in_files, "files"),
    ("file_without_table", _file_without_table, "files"),
    # rule 15
    ("parquet_bad_value", _set("parquet.compression_level", 3.5), "parquet"),
    ("parquet_null_value", _set("parquet.statistics", None), "parquet"),
    ("parquet_list_value", _set("parquet.compression", ["zstd"]), "parquet"),
    ("parquet_object_value", _set("parquet.compression", {"a": 1}), "parquet"),
    ("parquet_not_object", _set("parquet", ["zstd"]), "parquet"),
    ("environment_bad_value", _set("environment.python", 3.12), "environment"),
    ("environment_int_value", _set("environment.python", 3), "environment"),
    ("environment_list_value", _set("environment.polars", ["2"]), "environment"),
    ("environment_not_object", _set("environment", "py312"), "environment"),
]


def test_base_manifest_is_valid() -> None:
    """The fixture every reject case mutates must itself be accepted (fail for the right
    reason)."""
    snapshot.Manifest.from_json(sc.dumps(sc.base_manifest()))


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [pytest.param(mutate, needle, id=case) for case, mutate, needle in REJECT_CASES],
)
def test_from_json_rejects(mutate: Mutation, needle: str | None) -> None:
    result = mutate(sc.base_manifest())
    text = result if isinstance(result, str) else sc.dumps(result)
    with pytest.raises(snapshot.ManifestError) as info:
        snapshot.Manifest.from_json(text)
    assert isinstance(info.value, ValueError)
    if needle is not None:
        assert needle in str(info.value)


def test_from_json_rejects_every_spec_case_id() -> None:
    """Every case id of spec §2.2 rules 1-15 is in the parametrisation above."""
    spec_ids = {
        "invalid_json", "nan", "deep_nesting", "duplicate_key", "top_level_list", "missing_key",
        "missing_source_kind", "wrong_type", "bool_as_int", "manifest_version_2", "bad_name",
        "bad_builder", "bad_schema_version", "data_dir_absolute", "data_dir_escape",
        "data_dir_other_name", "bad_digest_hex", "digest_mismatch", "source_bad_kind",
        "source_short_revision", "source_files_on_git", "source_files_unsorted",
        "source_files_duplicate", "source_revision_mismatch", "source_path_newline",
        "content_sha256_bad_hex", "content_sha256_on_files", "no_tables", "tables_unsorted",
        "tables_duplicate", "bad_table_name", "table_file_mismatch", "negative_rows",
        "negative_matches", "matches_gt_rows", "matches_null_with_match_id",
        "matches_without_match_id", "bad_schema_value", "schema_name_mismatch",
        "canonical_without_schema", "freeze_frames_without_shots", "no_columns",
        "duplicate_column", "empty_dtype", "sort_by_unknown_column", "sort_by_duplicate",
        "groups_without_keys", "group_zero_rows", "group_matches_gt_rows",
        "group_matches_nullness", "group_bad_date", "group_year_zero", "group_signed_year",
        "group_one_date_null", "group_dates_reversed", "groups_unsorted", "group_rows_sum",
        "files_unsorted", "files_duplicate", "path_slash", "path_dotdot", "bad_sha256",
        "negative_bytes", "bool_bytes", "table_file_not_in_files", "file_without_table",
        "parquet_bad_value", "environment_bad_value",
    }  # fmt: skip
    assert spec_ids <= {case for case, _, _ in REJECT_CASES}


def test_from_json_accepts_canonical_tables_with_schema() -> None:
    m = sc.add_table(sc.base_manifest(), "shots", schema="shots")
    m = sc.add_table(m, "freeze_frames", schema="freeze_frames")
    parsed = snapshot.Manifest.from_json(sc.dumps(m))
    assert {t.name: t.schema for t in parsed.tables}["shots"] == "shots"


def test_from_json_accepts_sql_source_with_content_sha() -> None:
    m = sc.base_manifest()
    m["source"] = {
        "kind": "sql",
        "uri": "db:raw.table",
        "revision": "a" * 64,
        "content_sha256": "b" * 64,
        "files": [],
    }
    parsed = snapshot.Manifest.from_json(sc.dumps(m))
    assert parsed.source.content_sha256 == "b" * 64


def test_from_json_accepts_files_source_without_listing() -> None:
    """``list_files=False``: ``files`` empty, nothing to recompute (spec §1.2 Source)."""
    m = sc.base_manifest()
    m["source"]["files"] = []
    m["source"]["revision"] = "c" * 64
    assert snapshot.Manifest.from_json(sc.dumps(m)).source.files == ()


def test_manifest_error_names_key_path() -> None:
    with pytest.raises(snapshot.ManifestError, match=r"tables\[1\]\.rows"):
        snapshot.Manifest.from_json(sc.dumps(_set("tables.1.rows", "3")(sc.base_manifest())))


def test_unsupported_manifest_version_message() -> None:
    with pytest.raises(snapshot.ManifestError) as info:
        snapshot.Manifest.from_json(sc.dumps(_set("manifest_version", 2)(sc.base_manifest())))
    assert "unsupported manifest_version 2" in str(info.value)
    assert "reads 1" in str(info.value)


def test_read_manifest_missing_path(tmp_path: Path) -> None:
    with pytest.raises(snapshot.ManifestError, match="missing.json"):
        snapshot.read_manifest(tmp_path / "missing.json")


def test_read_manifest_directory(tmp_path: Path) -> None:
    (tmp_path / "dir.json").mkdir()
    with pytest.raises(snapshot.ManifestError, match="dir.json"):
        snapshot.read_manifest(tmp_path / "dir.json")


def test_manifest_dataclasses_are_frozen() -> None:
    m = snapshot.Manifest.from_json(sc.dumps(sc.base_manifest()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.name = "other"  # type: ignore[misc]


# --- Build decisions (build.md "Architect: ratify" items 7 and 15) ---------------------------


@pytest.mark.parametrize("path", ["a..b", "lookup..parquet", "x.._y"])
def test_from_json_rejects_double_dot_without_slash(path: str) -> None:
    """Item 7: rule 13 says "no ``..``"; the file-name regex alone accepts ``a..b``."""
    m = sc.base_manifest()
    m["files"][1]["path"] = path
    with pytest.raises(snapshot.ManifestError) as info:
        snapshot.Manifest.from_json(sc.dumps(sc.refresh_digest(m)))
    assert str(info.value).startswith("files[1].path: ")
    assert "not a bare file name" in str(info.value)


def test_read_manifest_rule_violation_names_path(tmp_path: Path) -> None:
    """Item 15: ``read_manifest`` prefixes the path; ``from_json`` does not."""
    m = sc.base_manifest()
    m["tables"][1]["rows"] = -1
    text = sc.dumps(m)
    with pytest.raises(snapshot.ManifestError) as bare:
        snapshot.Manifest.from_json(text)
    assert str(bare.value).startswith("tables[1].rows")
    path = tmp_path / "manifests" / "bad.json"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    with pytest.raises(snapshot.ManifestError) as info:
        snapshot.read_manifest(path)
    assert str(info.value) == f"manifest '{path}': {bare.value}"
