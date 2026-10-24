"""``write_snapshot``: argument checks, validation, the git-ignore guard, conflicts, recovery
and the writer self-check (C6 spec §2.1, §3.1, §5)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import decimal
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import schema_cases
import snapshot_cases as sc
from polars.testing import assert_frame_equal

from football_core import schema
from football_core.io import snapshot

# --- Basic result -----------------------------------------------------------------------------


def test_created_result_and_layout(tmp_path: Path) -> None:
    result = sc.write(tmp_path)
    assert result.status == "created"
    assert result.manifest_written is True
    assert os.path.samefile(result.manifest_path, sc.manifest_path(tmp_path))
    assert os.path.samefile(result.data_dir, sc.data_dir(tmp_path))
    assert sorted(p.name for p in sc.data_dir(tmp_path).iterdir()) == [
        "events.parquet",
        "freeze_frames.parquet",
        "shots.parquet",
    ]
    assert sorted(p.name for p in (tmp_path / "data" / "snapshots").iterdir()) == ["demo"]
    assert sorted(p.name for p in (tmp_path / "manifests").iterdir()) == ["demo.json"]


def test_root_accepts_str(tmp_path: Path) -> None:
    result = snapshot.write_snapshot(
        "demo", {"events": sc.generic_frame()}, sc.git_source(), root=str(tmp_path)
    )
    assert result.status == "created"
    assert sc.manifest_path(tmp_path).is_file()


def test_default_root_is_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    snapshot.write_snapshot("demo", {"events": sc.generic_frame()}, sc.git_source())
    assert sc.manifest_path(tmp_path).is_file()
    assert (sc.data_dir(tmp_path) / "events.parquet").is_file()


# --- Validation (spec §3.1 step 2) ------------------------------------------------------------


def test_invalid_shots_raise_and_write_nothing(tmp_path: Path) -> None:
    tables = sc.all_tables()
    tables["shots"] = schema_cases.with_value(tables["shots"], "is_goal", 0, False)
    with pytest.raises(schema.SchemaValidationError) as info:
        sc.write(tmp_path, tables)
    assert info.value.table == "shots"
    assert "is_goal_matches_outcome" in {f.check for f in info.value.failures}
    sc.assert_nothing_written(tmp_path, before=set())


def test_invalid_shots_dtype_raise_and_write_nothing(tmp_path: Path) -> None:
    tables = sc.all_tables()
    bad = pl.Series("outcome", ["goal", "saved", "blocked", "goall", "off_target"])
    tables["shots"] = schema_cases.replace_column(tables["shots"], bad)
    with pytest.raises(schema.SchemaValidationError):
        sc.write(tmp_path, tables)
    sc.assert_nothing_written(tmp_path, before=set())


def test_freeze_frames_checked_against_shots(tmp_path: Path) -> None:
    tables = sc.all_tables()
    tables["freeze_frames"] = schema_cases.with_value(
        tables["freeze_frames"], "player_id", 1, schema_cases.SHOOTER_1
    )
    with pytest.raises(schema.SchemaValidationError) as info:
        sc.write(tmp_path, tables)
    assert info.value.table == "freeze_frames"
    assert "shooter_not_in_frame" in {f.check for f in info.value.failures}
    sc.assert_nothing_written(tmp_path, before=set())


def test_freeze_frames_orphan_shot_rejected(tmp_path: Path) -> None:
    tables = sc.all_tables()
    tables["freeze_frames"] = schema_cases.with_value(
        tables["freeze_frames"], "shot_id", 0, "sb:00000000-0000-0000-0000-0000000000ff"
    )
    with pytest.raises(schema.SchemaValidationError) as info:
        sc.write(tmp_path, tables)
    assert "shot_exists" in {f.check for f in info.value.failures}


def test_freeze_frames_without_shots_rejected(tmp_path: Path) -> None:
    tables = {"freeze_frames": schema_cases.freeze_frames_frame()}
    with pytest.raises(ValueError, match="freeze_frames requires a shots table"):
        sc.write(tmp_path, tables)
    sc.assert_nothing_written(tmp_path, before=set())


def test_empty_shots_still_checks_freeze_frames(tmp_path: Path) -> None:
    tables = {
        "shots": schema_cases.shots_frame().head(0),
        "freeze_frames": schema_cases.freeze_frames_frame(),
    }
    with pytest.raises(schema.SchemaValidationError) as info:
        sc.write(tmp_path, tables)
    assert info.value.table == "freeze_frames"
    assert "shot_exists" in {f.check for f in info.value.failures}
    sc.assert_nothing_written(tmp_path, before=set())


def test_shots_alone_allowed(tmp_path: Path) -> None:
    result = sc.write(tmp_path, {"shots": schema_cases.shots_frame()})
    assert result.status == "created"


def test_generic_table_not_validated(tmp_path: Path) -> None:
    junk = pl.DataFrame(
        [
            pl.Series("x", [500.0, -3.0], dtype=pl.Float64),
            pl.Series("outcome", ["nonsense", "goal"], dtype=pl.String),
            pl.Series("shot_id", ["no-prefix", "no-prefix"], dtype=pl.String),
        ]
    )
    result = sc.write(tmp_path, {"events": junk, "shots_raw": junk})
    assert result.status == "created"
    m = sc.manifest_json(tmp_path)
    assert [t["schema"] for t in m["tables"]] == [None, None]


# --- Argument checks (spec §3.1 step 1) -------------------------------------------------------

BAD_NAMES = ["", "A", "Demo", "a/b", "..", ".", "-x", "_x", ".x", "a b", "a" * 65, "ü", "a\\b"]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_bad_name_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError) as info:
        sc.write(tmp_path, {"events": sc.generic_frame()}, name=name)
    assert snapshot.NAME_PATTERN in str(info.value)
    if name:
        assert repr(name) in str(info.value) or name in str(info.value)
    assert sc.tree(tmp_path) == set()


@pytest.mark.parametrize("name", ["demo\n", "demo\r", "\ndemo"])
def test_name_with_newline_rejected(tmp_path: Path, name: str) -> None:
    """Patterns are full matches: ``$`` must not accept a trailing newline (pinned choice)."""
    with pytest.raises(ValueError):
        sc.write(tmp_path, {"events": sc.generic_frame()}, name=name)
    assert sc.tree(tmp_path) == set()


@pytest.mark.parametrize("builder", ["StatsBomb loader", "", "Parquet", "a/b", "a" * 65, "x\n"])
def test_bad_builder_rejected(tmp_path: Path, builder: str) -> None:
    with pytest.raises(ValueError) as info:
        sc.write(tmp_path, {"events": sc.generic_frame()}, builder=builder)
    assert snapshot.NAME_PATTERN in str(info.value)
    if builder.strip():
        assert builder.strip() in str(info.value)
    assert sc.tree(tmp_path) == set()


@pytest.mark.parametrize("name", ["a", "0", "a" * 64, "statsbomb-open-v1", "x.y_z", "9-a"])
def test_good_names_accepted(tmp_path: Path, name: str) -> None:
    result = sc.write(tmp_path, {"events": sc.generic_frame()}, name=name, builder=name)
    assert result.status == "created"
    assert result.manifest.name == name
    assert result.manifest.builder == name
    assert sc.manifest_path(tmp_path, name).is_file()


@pytest.mark.parametrize(
    "table", ["Shots", "1a", "a-b", "a.b", "", "_a", "a" * 65, "ä", "shots\n", "a b"]
)
def test_bad_table_name_rejected(tmp_path: Path, table: str) -> None:
    with pytest.raises(ValueError) as info:
        sc.write(tmp_path, {table: sc.lookup_frame()})
    if table.strip():
        assert table.strip() in str(info.value)
    assert sc.tree(tmp_path) == set()


@pytest.mark.parametrize("table", ["a", "a" * 64, "a_1", "events_2024"])
def test_good_table_names_accepted(tmp_path: Path, table: str) -> None:
    result = sc.write(tmp_path, {table: sc.lookup_frame()})
    assert [t.name for t in result.manifest.tables] == [table]
    assert (sc.data_dir(tmp_path) / f"{table}.parquet").is_file()


def test_no_tables_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sc.write(tmp_path, {})
    assert sc.tree(tmp_path) == set()


def test_zero_column_table_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty_t"):
        sc.write(tmp_path, {"events": sc.generic_frame(), "empty_t": pl.DataFrame()})
    assert sc.tree(tmp_path) == set()


@pytest.mark.parametrize(
    "value",
    [
        pl.LazyFrame({"a": [1]}),
        {"a": [1]},
        pl.Series("a", [1]),
        None,
    ],
    ids=["lazyframe", "dict", "series", "none"],
)
def test_non_frame_rejected(tmp_path: Path, value: object) -> None:
    with pytest.raises(TypeError):
        sc.write(tmp_path, {"events": value})  # type: ignore[dict-item]
    assert sc.tree(tmp_path) == set()


@pytest.mark.parametrize(
    "source", ["https://x", None, {"kind": "git"}], ids=["str", "none", "dict"]
)
def test_non_source_rejected(tmp_path: Path, source: object) -> None:
    with pytest.raises(TypeError):
        snapshot.write_snapshot(
            "demo",
            {"events": sc.generic_frame()},
            source,  # type: ignore[arg-type]
            root=tmp_path,
        )
    assert sc.tree(tmp_path) == set()


def test_object_column_rejected_names_column(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        [
            pl.Series("k", ["a"], dtype=pl.String),
            pl.Series("payload", [object()], dtype=pl.Object),
        ]
    )
    with pytest.raises(ValueError) as info:
        sc.write(tmp_path, {"blobs": frame})
    assert "blobs" in str(info.value)
    assert "payload" in str(info.value)
    assert sc.tree(tmp_path) == set()


# --- sort_by ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sort_by", "error", "needles"),
    [
        ({"nope": ["event_id"]}, ValueError, ["nope"]),
        ({"events": ["no_such_column"]}, ValueError, ["no_such_column"]),
        ({"events": ["event_id", "event_id"]}, ValueError, ["event_id"]),
        ({"events": []}, ValueError, ["events"]),
        ({"events": ()}, ValueError, ["events"]),
        ({"events": "event_id"}, TypeError, ["events"]),
        ({"shots": ["outcome"]}, ValueError, ["outcome", "identity", "pre_shot"]),
        ({"shots": ["is_goal"]}, ValueError, ["is_goal", "outcome", "identity", "pre_shot"]),
        ({"shots": ["shot_id", "on_target"]}, ValueError, ["on_target"]),
        ({"shots": ["raw_type"]}, ValueError, ["raw_type", "lineage"]),
        ({"freeze_frames": ["shot_id", "no_col"]}, ValueError, ["no_col"]),
    ],
    ids=[
        "unknown_table",
        "unknown_column",
        "duplicate_key",
        "empty_list",
        "empty_tuple",
        "str_value",
        "label_outcome",
        "label_is_goal",
        "label_on_target",
        "lineage",
        "ff_unknown_column",
    ],
)
def test_sort_by_rejected(
    tmp_path: Path, sort_by: dict[str, Any], error: type[Exception], needles: list[str]
) -> None:
    with pytest.raises(error) as info:
        sc.write(tmp_path, sort_by=sort_by)
    for needle in needles:
        assert needle in str(info.value)
    assert sc.tree(tmp_path) == set()


@pytest.mark.parametrize(
    ("keys", "needles"),
    [
        (["goal_y"], ["goal_y", "post_shot"]),
        (["shot_id", "deflected"], ["deflected", "post_shot"]),
        (["big_chance"], ["big_chance", "restricted"]),
        (["end_y"], ["end_y", "outcome"]),
    ],
    ids=["post_shot_goal_y", "post_shot_deflected", "restricted_big_chance", "outcome_end_y"],
)
def test_sort_by_rejects_post_shot_restricted_and_end_location(
    tmp_path: Path, keys: list[str], needles: list[str]
) -> None:
    """Football code review r1: goal-mouth nullness and big chance (q214) leak the label."""
    with pytest.raises(ValueError) as info:
        sc.write(tmp_path, sort_by={"shots": keys})
    for needle in needles:
        assert needle in str(info.value)
    assert sc.tree(tmp_path) == set()


# The allowed roles are spelled out, not read from SORT_ROLES, so a widened SORT_ROLES fails.
_SORTABLE_ROLES = ("identity", "pre_shot")


def test_sort_roles_constant() -> None:
    assert snapshot.SORT_ROLES == _SORTABLE_ROLES


@pytest.mark.parametrize(
    ("column", "role"),
    [
        pytest.param(column, role, id=f"{role}-{column}")
        for role in schema.ROLES
        if role not in _SORTABLE_ROLES
        for column in schema.column_names("shots", roles=role)
    ],
)
def test_sort_by_rejects_every_non_feature_shots_column(
    tmp_path: Path, column: str, role: str
) -> None:
    with pytest.raises(ValueError) as info:
        sc.write(tmp_path, sort_by={"shots": ["shot_id", column]})
    message = str(info.value)
    assert repr(column) in message
    assert repr(role) in message
    assert sc.tree(tmp_path) == set()


def test_sort_by_non_sequence_rejected(tmp_path: Path) -> None:
    """build.md ratify item 11: ``None`` or an int is not a sequence of column names."""
    for value in (None, 3):
        with pytest.raises(TypeError, match="'events'"):
            sc.write(tmp_path, sort_by={"events": value})
    assert sc.tree(tmp_path) == set()


def test_sort_by_identity_and_pre_shot_accepted(tmp_path: Path) -> None:
    keys = ["match_date", "competition_id", "season_id", "match_id", "period", "shot_id"]
    result = sc.write(tmp_path, sort_by={"shots": keys, "freeze_frames": ("shot_id", "x")})
    entry = sc.table_entry(sc.manifest_json(tmp_path), "shots")
    assert entry["sort_by"] == keys
    assert sc.table_entry(sc.manifest_json(tmp_path), "freeze_frames")["sort_by"] == [
        "shot_id",
        "x",
    ]
    shots = next(t for t in result.manifest.tables if t.name == "shots")
    assert shots.sort_by == tuple(keys)
    back = pl.read_parquet(sc.data_dir(tmp_path) / "shots.parquet")
    expected = sc.sort_all(schema.validate_shots(schema_cases.shots_frame()), keys)
    assert_frame_equal(back, expected)


def test_sort_by_keys_then_all_columns(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        [
            pl.Series("k", [2, 1, 2, 1, None], dtype=pl.Int64),
            pl.Series("v", ["b", "b", "a", "a", "a"], dtype=pl.String),
            pl.Series("w", [1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64),
        ]
    )
    sc.write(tmp_path / "by_v", {"t": frame}, sort_by={"t": ["v"]})
    sc.write(tmp_path / "by_v_rev", {"t": frame.reverse()}, sort_by={"t": ["v"]})
    sc.write(tmp_path / "default", {"t": frame})
    by_v = pl.read_parquet(sc.data_dir(tmp_path / "by_v") / "t.parquet")
    assert by_v.rows() == [
        (1, "a", 4.0),
        (2, "a", 3.0),
        (None, "a", 5.0),
        (1, "b", 2.0),
        (2, "b", 1.0),
    ]
    assert by_v.columns == ["k", "v", "w"]  # sorting never reorders columns
    assert sc.files_bytes(tmp_path / "by_v") == sc.files_bytes(tmp_path / "by_v_rev")
    default = pl.read_parquet(sc.data_dir(tmp_path / "default") / "t.parquet")
    assert default.rows() == [
        (1, "a", 4.0),
        (1, "b", 2.0),
        (2, "a", 3.0),
        (2, "b", 1.0),
        (None, "a", 5.0),
    ]


def test_sort_places_nan_after_numbers_and_nulls_last(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        [pl.Series("f", [sc.NAN, None, 1.0, -sc.INF, sc.INF, -1.0], dtype=pl.Float64)]
    )
    sc.write(tmp_path, {"t": frame})
    back = pl.read_parquet(sc.data_dir(tmp_path) / "t.parquet")["f"].to_list()
    assert back[:4] == [-sc.INF, -1.0, 1.0, sc.INF]
    assert back[4] != back[4]  # NaN
    assert back[5] is None


def test_sort_by_normalised_to_tuple(tmp_path: Path) -> None:
    result = sc.write(tmp_path, {"events": sc.generic_frame()}, sort_by={"events": ["minute"]})
    (entry,) = result.manifest.tables
    assert entry.sort_by == ("minute",)
    assert isinstance(entry.sort_by, tuple)


# --- match_date range (spec §3.1 step 1; open issue 1: the alias fix) -------------------------


def test_grouped_date_table_written(tmp_path: Path) -> None:
    """A grouped table with a ``Date`` ``match_date`` is the normal case (every ``shots``
    table). The two-output range ``select`` must not fail on duplicate output names."""
    result = sc.write(tmp_path, {"events": sc.date_frame([0, 19_000, None], grouped=True)})
    assert result.status == "created"
    (group,) = sc.table_entry(sc.manifest_json(tmp_path), "events")["groups"]
    assert group["first_date"] == "1970-01-01"
    assert group["last_date"] == (dt.date(1970, 1, 1) + dt.timedelta(days=19_000)).isoformat()


def test_shots_table_groups_written(tmp_path: Path) -> None:
    sc.write(tmp_path, {"shots": schema_cases.shots_frame()})
    groups = sc.table_entry(sc.manifest_json(tmp_path), "shots")["groups"]
    assert groups == [
        {
            "competition_id": "sb:9001",
            "season_id": "sb:9002",
            "rows": 3,
            "matches": 2,
            "first_date": "2024-03-02",
            "last_date": "2024-03-16",
        },
        {
            "competition_id": "ws:9101",
            "season_id": "ws:9102",
            "rows": 2,
            "matches": 1,
            "first_date": "2024-03-09",
            "last_date": "2024-03-09",
        },
    ]


def _range_cases() -> list[Any]:
    cases = []
    for day in sc.OUT_OF_RANGE_DAYS:
        cases.append(pytest.param([day], id=f"alone_{day}"))
        cases.append(pytest.param([day, 19_000, None], id=f"with_in_range_{day}"))
    return cases


@pytest.mark.parametrize("days", _range_cases())
def test_out_of_range_match_date_rejected(tmp_path: Path, days: list[int | None]) -> None:
    present = [d for d in days if d is not None]
    with pytest.raises(ValueError) as info:
        sc.write(tmp_path, {"events": sc.date_frame(days, grouped=True)})
    message = str(info.value)
    assert "events" in message
    assert "'match_date'" in message
    assert str(min(present)) in message
    assert str(max(present)) in message
    assert "0001-01-01..9999-12-31" in message
    assert not isinstance(info.value, schema.SchemaValidationError)
    assert sc.tree(tmp_path) == set()


@pytest.mark.parametrize("day", [2_932_897, -800_000])
def test_out_of_range_match_date_on_shots_rejected(tmp_path: Path, day: int) -> None:
    """The canonical path: C3 accepts the date, so C6 step 1 is the only defence."""
    shots = schema_cases.shots_frame()
    shots = schema_cases.replace_column(
        shots, pl.Series("match_date", [day] * shots.height, dtype=pl.Int32).cast(pl.Date)
    )
    tables = {"shots": shots, "freeze_frames": schema_cases.freeze_frames_frame()}
    with pytest.raises(ValueError) as info:
        sc.write(tmp_path, tables)
    assert "shots" in str(info.value)
    assert "'match_date'" in str(info.value)
    assert str(day) in str(info.value)
    assert sc.tree(tmp_path) == set()


def test_match_date_boundaries_accepted(tmp_path: Path) -> None:
    frame = sc.date_frame([sc.DAY_MIN, sc.DAY_MAX], grouped=True)
    result = sc.write(tmp_path, {"events": frame})
    assert result.status == "created"
    (group,) = sc.table_entry(sc.manifest_json(tmp_path), "events")["groups"]
    assert group["first_date"] == "0001-01-01"
    assert group["last_date"] == "9999-12-31"
    assert snapshot.verify_snapshot(result.manifest_path).ok
    assert sc.write(tmp_path, {"events": frame}).status == "unchanged"


@pytest.mark.parametrize("day", sc.OUT_OF_RANGE_DAYS)
def test_out_of_range_date_without_groups_written(tmp_path: Path, day: int) -> None:
    frame = sc.date_frame([day, 0, None], grouped=False)
    result = sc.write(tmp_path, {"events": frame})
    assert result.status == "created"
    assert sc.table_entry(sc.manifest_json(tmp_path), "events")["groups"] is None
    back = pl.read_parquet(sc.data_dir(tmp_path) / "events.parquet")
    assert back.schema == frame.schema
    assert sorted(back["match_date"].to_physical().to_list(), key=lambda d: (d is None, d)) == [
        min(day, 0),
        max(day, 0),
        None,
    ]


@pytest.mark.parametrize("key_dtype", [pl.Float64(), pl.Boolean(), pl.Date()])
def test_out_of_range_date_with_unsupported_key_dtype_written(
    tmp_path: Path, key_dtype: pl.DataType
) -> None:
    """Keys present but not String / Enum / Categorical / integer: no groups, so no range
    check (spec §2.2 ``groups`` row, ``_groups_eligible``)."""
    frame = sc.date_frame([2_932_897, 0], grouped=True)
    values = {
        pl.Float64(): [1.0, 2.0],
        pl.Boolean(): [True, False],
        pl.Date(): [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
    }[key_dtype]
    frame = frame.with_columns(pl.Series("season_id", values, dtype=key_dtype))
    result = sc.write(tmp_path, {"events": frame})
    assert result.status == "created"
    assert sc.table_entry(sc.manifest_json(tmp_path), "events")["groups"] is None


def test_out_of_range_non_date_match_date_written(tmp_path: Path) -> None:
    """A grouped table whose ``match_date`` is an Int32 (not a Date) is not range-checked."""
    frame = sc.date_frame([0], grouped=True).with_columns(
        pl.Series("match_date", [100_000_000], dtype=pl.Int32)
    )
    sc.write(tmp_path, {"events": frame})
    (group,) = sc.table_entry(sc.manifest_json(tmp_path), "events")["groups"]
    assert group["first_date"] is None
    assert group["last_date"] is None


def test_all_null_match_date_grouped_written(tmp_path: Path) -> None:
    sc.write(tmp_path, {"events": sc.date_frame([None, None], grouped=True)})
    (group,) = sc.table_entry(sc.manifest_json(tmp_path), "events")["groups"]
    assert (group["first_date"], group["last_date"]) == (None, None)


# --- Self-check (spec §3.1 step 6; open issue 2) ----------------------------------------------


def test_writer_self_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = snapshot.Manifest.from_json

    def raising(cls: type[snapshot.Manifest], text: str) -> snapshot.Manifest:
        raise snapshot.ManifestError("tables[0].rows: injected")

    def changing(cls: type[snapshot.Manifest], text: str) -> snapshot.Manifest:
        return dataclasses.replace(original(text), name="other")

    # 1. the reader rejects the writer's text
    fresh = tmp_path / "fresh"
    monkeypatch.setattr(snapshot.Manifest, "from_json", classmethod(raising))
    with pytest.raises(RuntimeError) as info:
        sc.write(fresh)
    assert str(info.value).startswith("internal error")
    assert isinstance(info.value.__cause__, snapshot.ManifestError)
    assert not isinstance(info.value, snapshot.SnapshotError)
    sc.assert_nothing_written(fresh)

    # 2. the reader returns a different object
    monkeypatch.setattr(snapshot.Manifest, "from_json", classmethod(changing))
    with pytest.raises(RuntimeError, match="^internal error"):
        sc.write(fresh)
    sc.assert_nothing_written(fresh)

    # 3. an existing snapshot is left untouched
    monkeypatch.setattr(snapshot.Manifest, "from_json", original)
    existing = tmp_path / "existing"
    sc.write(existing)
    before_files = sc.files_bytes(existing)
    before_manifest = sc.manifest_path(existing).read_bytes()
    monkeypatch.setattr(snapshot.Manifest, "from_json", classmethod(raising))
    changed = {"events": sc.generic_frame().head(3)}
    with pytest.raises(RuntimeError, match="^internal error"):
        sc.write(existing, changed, overwrite=True)
    monkeypatch.setattr(snapshot.Manifest, "from_json", original)
    assert sc.files_bytes(existing) == before_files
    assert sc.manifest_path(existing).read_bytes() == before_manifest
    assert not sc.staging_dir(existing).exists()
    assert not sc.old_dir(existing).exists()


def test_self_check_catches_text_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``parsed.to_json() != text`` alone is enough to refuse (spec §3.1 step 6): the parsed
    object compares equal, only its re-serialisation drifts."""
    original_from_json = snapshot.Manifest.from_json
    original_to_json = snapshot.Manifest.to_json
    parsed_ids: set[int] = set()
    keep: list[snapshot.Manifest] = []

    def from_json(cls: type[snapshot.Manifest], text: str) -> snapshot.Manifest:
        parsed = original_from_json(text)
        parsed_ids.add(id(parsed))
        keep.append(parsed)
        return parsed

    def to_json(self: snapshot.Manifest) -> str:
        text = original_to_json(self)
        return text + " " if id(self) in parsed_ids else text

    monkeypatch.setattr(snapshot.Manifest, "from_json", classmethod(from_json))
    monkeypatch.setattr(snapshot.Manifest, "to_json", to_json)
    with pytest.raises(RuntimeError, match="^internal error"):
        sc.write(tmp_path)
    sc.assert_nothing_written(tmp_path)


# --- Conflicts, overwrite, unchanged (spec §3.1 steps 7-8) ------------------------------------


def test_conflict_raises_and_keeps_existing(tmp_path: Path) -> None:
    sc.write(tmp_path)
    before_files = sc.files_bytes(tmp_path)
    before_manifest = sc.manifest_path(tmp_path).read_bytes()
    tables = sc.all_tables()
    tables["events"] = tables["events"].head(4)
    with pytest.raises(snapshot.SnapshotExistsError) as info:
        sc.write(tmp_path, tables)
    message = str(info.value)
    assert "changed events.parquet" in message
    assert "shots.parquet" not in message
    assert message.rstrip().endswith("pass overwrite=True (fc snapshot --overwrite) to replace it")
    assert sc.files_bytes(tmp_path) == before_files
    assert sc.manifest_path(tmp_path).read_bytes() == before_manifest
    assert not sc.staging_dir(tmp_path).exists()
    assert not sc.old_dir(tmp_path).exists()


def test_conflict_message_lists_added_and_removed(tmp_path: Path) -> None:
    sc.write(tmp_path, {"events": sc.generic_frame(), "lookup": sc.lookup_frame()})
    with pytest.raises(snapshot.SnapshotExistsError) as info:
        sc.write(tmp_path, {"events": sc.generic_frame(), "extra_t": sc.lookup_frame()})
    message = str(info.value)
    assert "added extra_t.parquet" in message
    assert "removed lookup.parquet" in message
    assert "events.parquet" not in message
    assert not sc.staging_dir(tmp_path).exists()


def test_conflict_is_snapshot_error_not_value_error(tmp_path: Path) -> None:
    assert issubclass(snapshot.SnapshotExistsError, snapshot.SnapshotError)
    assert not issubclass(snapshot.SnapshotExistsError, ValueError)
    assert issubclass(snapshot.ManifestError, snapshot.SnapshotError)
    assert issubclass(snapshot.ManifestError, ValueError)
    assert issubclass(snapshot.SnapshotVerificationError, snapshot.SnapshotError)


def test_unchanged_rewrites_manifest_only_when_bytes_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_a: dict[str, str | bool | None] = {
        "python": "3.12.13",
        "polars": "2.0.0",
        "pandera": "0.34.1",
        "numpy": "2.5.4",
        "football_core": "0.1.0",
        "football_core_git": None,
        "football_core_dirty": None,
    }
    env_b = env_a | {"pandera": "0.99.0"}
    monkeypatch.setattr(snapshot, "environment", lambda: dict(env_a))
    sc.write(tmp_path)
    path = sc.manifest_path(tmp_path)
    before = path.read_bytes()
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    mtime = path.stat().st_mtime_ns

    same = sc.write(tmp_path)
    assert same.status == "unchanged"
    assert same.manifest_written is False
    assert path.stat().st_mtime_ns == mtime
    assert path.read_bytes() == before

    monkeypatch.setattr(snapshot, "environment", lambda: dict(env_b))
    newer = sc.write(tmp_path)
    assert newer.status == "unchanged"
    assert newer.manifest_written is True
    assert path.read_bytes() != before
    assert sc.manifest_json(tmp_path)["environment"]["pandera"] == "0.99.0"
    assert not path.with_suffix(".json.tmp").exists()


def test_unchanged_with_newer_source_rewrites_manifest(tmp_path: Path) -> None:
    sc.write(tmp_path)
    other = snapshot.source_from_git("https://example.invalid/synthetic", "f" * 40)
    result = sc.write(tmp_path, source=other)
    assert result.status == "unchanged"
    assert result.manifest_written is True
    assert sc.manifest_json(tmp_path)["source"]["revision"] == "f" * 40


def test_overwrite_replaces_and_removes_stale_table(tmp_path: Path) -> None:
    sc.write(tmp_path, {"events": sc.generic_frame(), "lookup": sc.lookup_frame()})
    result = sc.write(tmp_path, {"events": sc.generic_frame().head(2)}, overwrite=True)
    assert result.status == "replaced"
    assert result.manifest_written is True
    assert sorted(p.name for p in sc.data_dir(tmp_path).iterdir()) == ["events.parquet"]
    assert [t["name"] for t in sc.manifest_json(tmp_path)["tables"]] == ["events"]
    assert snapshot.verify_snapshot(sc.manifest_path(tmp_path)).ok
    assert not sc.old_dir(tmp_path).exists()
    assert not sc.staging_dir(tmp_path).exists()


def test_overwrite_on_unchanged_is_unchanged(tmp_path: Path) -> None:
    sc.write(tmp_path)
    assert sc.write(tmp_path, overwrite=True).status == "unchanged"


def test_orphan_data_dir_conflict(tmp_path: Path) -> None:
    orphan = sc.data_dir(tmp_path)
    orphan.mkdir(parents=True)
    (orphan / "old.parquet").write_bytes(b"keep me")
    with pytest.raises(snapshot.SnapshotExistsError, match="has no manifest"):
        sc.write(tmp_path)
    assert (orphan / "old.parquet").read_bytes() == b"keep me"
    assert sorted(p.name for p in orphan.iterdir()) == ["old.parquet"]
    assert not sc.manifest_path(tmp_path).exists()
    assert not sc.staging_dir(tmp_path).exists()
    result = sc.write(tmp_path, overwrite=True)
    assert result.status == "replaced"
    assert not (orphan / "old.parquet").exists()
    assert snapshot.verify_snapshot(result.manifest_path).ok


def test_orphan_with_only_ds_store_is_created(tmp_path: Path) -> None:
    orphan = sc.data_dir(tmp_path)
    orphan.mkdir(parents=True)
    (orphan / ".DS_Store").write_bytes(b"x")
    result = sc.write(tmp_path)
    assert result.status == "created"
    assert snapshot.verify_snapshot(result.manifest_path).ok


def test_empty_orphan_dir_is_created(tmp_path: Path) -> None:
    sc.data_dir(tmp_path).mkdir(parents=True)
    assert sc.write(tmp_path).status == "created"


def test_unreadable_manifest_conflict(tmp_path: Path) -> None:
    sc.write(tmp_path)
    before_files = sc.files_bytes(tmp_path)
    sc.manifest_path(tmp_path).write_text("{ broken", encoding="utf-8")
    with pytest.raises(snapshot.ManifestError):
        sc.write(tmp_path)
    assert sc.files_bytes(tmp_path) == before_files
    assert sc.manifest_path(tmp_path).read_text(encoding="utf-8") == "{ broken"
    assert not sc.staging_dir(tmp_path).exists()
    result = sc.write(tmp_path, overwrite=True)
    assert result.status == "replaced"
    assert snapshot.verify_snapshot(result.manifest_path).ok


def test_rebuild_after_data_deleted_is_unchanged(tmp_path: Path) -> None:
    first = sc.write(tmp_path)
    shutil.rmtree(sc.data_dir(tmp_path))
    again = sc.write(tmp_path)
    assert again.status == "unchanged"
    assert again.manifest_written is False
    assert again.manifest == first.manifest
    assert snapshot.verify_snapshot(again.manifest_path).ok


def test_rerun_heals_a_damaged_data_dir(tmp_path: Path) -> None:
    """Step 8 runs on ``unchanged`` too: staging replaces the damaged directory."""
    sc.write(tmp_path)
    sc.flip_byte(sc.data_dir(tmp_path) / "shots.parquet", 3)
    (sc.data_dir(tmp_path) / "junk.txt").write_bytes(b"x")
    result = sc.write(tmp_path)
    assert result.status == "unchanged"
    assert snapshot.verify_snapshot(result.manifest_path).ok


# --- Recovery (spec §3.1 step 5) --------------------------------------------------------------


def test_leftover_staging_removed(tmp_path: Path) -> None:
    staging = sc.staging_dir(tmp_path)
    staging.mkdir(parents=True)
    (staging / "junk.parquet").write_bytes(b"junk")
    result = sc.write(tmp_path)
    assert result.status == "created"
    assert not staging.exists()
    assert "junk.parquet" not in {p.name for p in sc.data_dir(tmp_path).iterdir()}


def test_leftover_old_removed(tmp_path: Path) -> None:
    sc.write(tmp_path)
    old = sc.old_dir(tmp_path)
    old.mkdir()
    (old / "stale.parquet").write_bytes(b"stale")
    result = sc.write(tmp_path)
    assert result.status == "unchanged"
    assert not old.exists()
    assert snapshot.verify_snapshot(result.manifest_path).ok


@pytest.mark.parametrize("rerun", ["same_data", "different_data"])
def test_interrupted_commit_rolled_back(tmp_path: Path, rerun: str) -> None:
    sc.write(tmp_path)
    before = sc.files_bytes(tmp_path)
    os.rename(sc.data_dir(tmp_path), sc.old_dir(tmp_path))  # crash between the two renames
    if rerun == "same_data":
        result = sc.write(tmp_path)
        assert result.status == "unchanged"
    else:
        tables = sc.all_tables()
        tables["events"] = tables["events"].head(2)
        with pytest.raises(snapshot.SnapshotExistsError, match="changed events.parquet"):
            sc.write(tmp_path, tables)
    assert sc.files_bytes(tmp_path) == before
    assert not sc.old_dir(tmp_path).exists()
    assert snapshot.verify_snapshot(sc.manifest_path(tmp_path)).ok


def test_two_snapshots_side_by_side(tmp_path: Path) -> None:
    sc.write(tmp_path, name="a")
    sc.write(tmp_path, {"lookup": sc.lookup_frame()}, name="b")
    assert snapshot.verify_snapshot(sc.manifest_path(tmp_path, "a")).ok
    assert snapshot.verify_snapshot(sc.manifest_path(tmp_path, "b")).ok
    assert sorted(p.name for p in (tmp_path / "data" / "snapshots").iterdir()) == ["a", "b"]


# --- Git-ignore guard (spec §3.1 step 4) ------------------------------------------------------


@sc.needs_git
def test_refuses_unignored_data_dir(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo", ignore="")
    before = sc.tree(repo)
    with pytest.raises(ValueError) as info:
        sc.write(repo)
    message = str(info.value)
    assert "would not be git-ignored" in message
    assert "events.parquet" in message  # the first file in table-name order
    sc.assert_nothing_written(repo, before=before)


@sc.needs_git
def test_refuses_subdir_root(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    (repo / "notebooks").mkdir()
    with pytest.raises(ValueError) as info:
        sc.write(repo / "notebooks")
    assert "would not be git-ignored" in str(info.value)
    assert "notebooks" in str(info.value)
    assert "events.parquet" in str(info.value)
    assert list((repo / "notebooks").iterdir()) == []


@sc.needs_git
def test_refuses_missing_subdir_root(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    before = sc.tree(repo)
    with pytest.raises(ValueError, match="would not be git-ignored"):
        sc.write(repo / "notebooks" / "new")
    assert not (repo / "notebooks").exists()
    assert sc.tree(repo) == before


@sc.needs_git
def test_missing_root_under_ignored_data_allowed(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    result = sc.write(repo / "data" / "new")
    assert result.status == "created"


@sc.needs_git
def test_missing_root_outside_repo_allowed(tmp_path: Path) -> None:
    result = sc.write(tmp_path / "nogit" / "deep")
    assert result.status == "created"


@sc.needs_git
def test_ignored_data_dir_in_repo_allowed(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    result = sc.write(repo)
    assert result.status == "created"
    status = sc.git(repo, "status", "--porcelain", "--untracked-files=all")
    assert "data/" not in status
    assert "manifests/demo.json" in status  # the manifest is meant to be committed


@sc.needs_git
def test_ignored_data_dir_in_non_ascii_repo_allowed(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "Masaüstü" / "repo")
    assert sc.write(repo).status == "created"


@sc.needs_git
def test_ignored_data_dir_relative_root_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    monkeypatch.chdir(repo)
    assert snapshot.write_snapshot("demo", sc.all_tables(), sc.git_source()).status == "created"
    assert sc.manifest_path(repo).is_file()


@sc.needs_git
def test_parquet_ignore_rule_allowed(tmp_path: Path) -> None:
    """Only the exit code of check-ignore counts, whatever rule matched."""
    repo = sc.git_repo(tmp_path / "repo", ignore="*.parquet\n")
    assert sc.write(repo).status == "created"


@sc.needs_git
@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_guard_git_error_fails_closed(tmp_path: Path) -> None:
    repo = sc.git_repo(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError) as info:
        sc.write(repo)
    assert "cannot check whether" in str(info.value)
    assert "symbolic link" in str(info.value)
    assert list(outside.iterdir()) == []
    assert not sc.manifest_path(repo).exists()


def _fake_git(
    rev_parse: subprocess.CompletedProcess[str] | None,
    check_ignore: dict[str, subprocess.CompletedProcess[str] | None] | None = None,
    default: subprocess.CompletedProcess[str] | None = None,
    calls: list[tuple[Path, tuple[str, ...]]] | None = None,
) -> Any:
    def fake(cwd: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
        if calls is not None:
            calls.append((Path(cwd), args))
        if args == ("rev-parse", "--is-inside-work-tree"):
            return rev_parse
        if args[:1] == ("check-ignore",):
            name = Path(args[-1]).name
            return (check_ignore or {}).get(name, default)
        return None  # git_revision / environment: no answer

    return fake


def _cp(rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["git"], rc, stdout, stderr)


def test_guard_git_no_answer_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open issue 2: git answers rev-parse, then gives no answer for check-ignore."""
    calls: list[tuple[Path, tuple[str, ...]]] = []
    monkeypatch.setattr(
        snapshot, "_run_git", _fake_git(_cp(0, "true\n"), default=None, calls=calls)
    )
    root = tmp_path / "root"
    with pytest.raises(ValueError) as info:
        sc.write(root)
    assert "cannot check whether" in str(info.value)
    assert "git did not answer" in str(info.value)
    assert "events.parquet" in str(info.value)
    sc.assert_nothing_written(root)
    assert not root.exists()
    ignore_calls = [args for _, args in calls if args[:1] == ("check-ignore",)]
    assert ignore_calls, "the guard never asked check-ignore"


@pytest.mark.parametrize(
    ("answers", "error"),
    [
        (
            {"events.parquet": _cp(0), "freeze_frames.parquet": _cp(0), "shots.parquet": _cp(0)},
            None,
        ),
        ({"events.parquet": _cp(1)}, "would not be git-ignored"),
        ({"events.parquet": _cp(0), "freeze_frames.parquet": _cp(1)}, "freeze_frames.parquet"),
        ({"events.parquet": _cp(128, "", "fatal: boom\nsecond line\n")}, "(git exit 128)"),
        ({"events.parquet": _cp(128, "", "fatal: boom\nsecond line\n")}, "fatal: boom"),
        ({"events.parquet": _cp(2, "", "")}, "cannot check whether"),
        ({"events.parquet": _cp(-9, "", "")}, "cannot check whether"),
    ],
    ids=["all_ignored", "rc1", "rc1_second", "rc128", "rc128_stderr", "rc2_no_stderr", "signal"],
)
def test_guard_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answers: dict[str, subprocess.CompletedProcess[str]],
    error: str | None,
) -> None:
    """Only the exit code of ``check-ignore -q`` is read: 0 ignored, 1 refuse, else fail
    closed (spec §3.1 step 4), with no git needed."""
    monkeypatch.setattr(snapshot, "_run_git", _fake_git(_cp(0, "true\n"), answers, default=_cp(0)))
    root = tmp_path / "root"
    if error is None:
        assert sc.write(root).status == "created"
        return
    with pytest.raises(ValueError) as info:
        sc.write(root)
    assert error in str(info.value)
    if error != "would not be git-ignored":
        assert "second line" not in str(info.value)
    sc.assert_nothing_written(root)


def test_guard_stdout_is_never_parsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """rc 0 with C-quoted or unrelated stdout is still 'ignored'."""
    quoted = _cp(0, '"/x/Masa\\303\\274st\\303\\274/data"\n')
    monkeypatch.setattr(snapshot, "_run_git", _fake_git(_cp(0, "true\n"), default=quoted))
    assert sc.write(tmp_path / "root").status == "created"


@pytest.mark.parametrize(
    "rev_parse",
    [None, _cp(128, "", "fatal: not a git repository"), _cp(0, "false\n")],
    ids=["none", "rc128", "false"],
)
def test_guard_skipped_outside_work_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rev_parse: subprocess.CompletedProcess[str] | None,
) -> None:
    calls: list[tuple[Path, tuple[str, ...]]] = []
    monkeypatch.setattr(snapshot, "_run_git", _fake_git(rev_parse, default=_cp(1), calls=calls))
    assert sc.write(tmp_path / "root").status == "created"
    assert not [args for _, args in calls if args[:1] == ("check-ignore",)]


def test_guard_runs_from_existing_anchor_with_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, tuple[str, ...]]] = []
    monkeypatch.setattr(
        snapshot, "_run_git", _fake_git(_cp(0, "true\n"), default=_cp(0), calls=calls)
    )
    (tmp_path / "a").mkdir()
    root = tmp_path / "a" / "b" / "c"
    sc.write(root)
    guard = [(cwd, args) for cwd, args in calls if args[:1] in (("rev-parse",), ("check-ignore",))]
    guard = [(cwd, args) for cwd, args in guard if args[:2] != ("rev-parse", "--show-toplevel")]
    assert guard[0] == ((tmp_path / "a").resolve(), ("rev-parse", "--is-inside-work-tree"))
    ignore = [(cwd, args) for cwd, args in guard if args[:1] == ("check-ignore",)]
    assert [cwd for cwd, _ in ignore] == [(tmp_path / "a").resolve()] * 3
    assert [args[:3] for _, args in ignore] == [("check-ignore", "-q", "--")] * 3
    assert [len(args) for _, args in ignore] == [4] * 3  # one path per call
    expected = [
        str(root.resolve() / "data" / "snapshots" / "demo" / f"{t}.parquet")
        for t in ("events", "freeze_frames", "shots")
    ]
    assert [args[3] for _, args in ignore] == expected


def test_existing_anchor(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "f").write_text("x")
    assert snapshot._existing_anchor(tmp_path / "a" / "b" / "c") == tmp_path / "a"
    assert snapshot._existing_anchor(tmp_path / "a") == tmp_path / "a"
    assert snapshot._existing_anchor(tmp_path / "f" / "x") == tmp_path


# --- Empty and hostile tables (spec §2.1 totality) --------------------------------------------


def test_empty_table_written(tmp_path: Path) -> None:
    frame = sc.generic_frame().head(0)
    result = sc.write(tmp_path, {"events": frame})
    entry = sc.table_entry(sc.manifest_json(tmp_path), "events")
    assert entry["rows"] == 0
    assert entry["matches"] == 0
    assert entry["groups"] == []
    assert snapshot.verify_snapshot(result.manifest_path).ok
    back = pl.read_parquet(sc.data_dir(tmp_path) / "events.parquet")
    assert back.schema == frame.schema
    assert back.height == 0


def test_empty_canonical_tables_written(tmp_path: Path) -> None:
    tables = {"shots": schema.empty_shots(), "freeze_frames": schema.empty_freeze_frames()}
    result = sc.write(tmp_path, tables)
    assert result.status == "created"
    assert [t.rows for t in result.manifest.tables] == [0, 0]


_DAYS = [-800_000, -719_163, 2_932_897, 100_000_000, None, 0, 1]


def _hostile_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            pl.Series("f", [sc.NAN, sc.INF, -sc.INF, 1e308, -1e308, None, 0.0], dtype=pl.Float64),
            pl.Series("i8", [-128, 127, None, 0, 1, -1, 5], dtype=pl.Int8),
            pl.Series("i16", [-32768, 32767, None, 0, 1, -1, 5], dtype=pl.Int16),
            pl.Series("i64", [-(2**63), 2**63 - 1, None, 0, 1, -1, 5], dtype=pl.Int64),
            pl.Series("s", [None, "", "\n", '"quoted"', "Masaüstü", "a,b", "\t"], dtype=pl.String),
            pl.Series("kickoff_date", _DAYS, dtype=pl.Int32).cast(pl.Date),
        ]
    )


def test_hostile_values_round_trip(tmp_path: Path) -> None:
    frame = _hostile_frame()
    result = sc.write(tmp_path, {"hostile": frame})
    assert result.status == "created"
    back = pl.read_parquet(sc.data_dir(tmp_path) / "hostile.parquet")
    assert back.schema == frame.schema
    as_ints = pl.col("kickoff_date").to_physical()
    assert_frame_equal(
        back.with_columns(as_ints),
        sc.sort_all(frame).with_columns(as_ints),
    )
    assert sc.table_entry(sc.manifest_json(tmp_path), "hostile")["groups"] is None
    assert sc.write(tmp_path, {"hostile": frame.reverse()}).status == "unchanged"


def test_all_writable_dtypes_round_trip(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        [
            pl.Series("n", [None, None], dtype=pl.Null),
            pl.Series("e", ["b", "a"], dtype=pl.Enum(["b", "a"])),
            pl.Series("c", ["y", "x"], dtype=pl.Categorical),
            pl.Series("poly", [[[1.0, 2.0]], None], dtype=pl.List(pl.Array(pl.Float64, 2))),
            pl.Series("st", [{"a": 1, "b": "x"}, None]),
            pl.Series("dec", [decimal.Decimal("1.25"), None], dtype=pl.Decimal(10, 2)),
            pl.Series(
                "ts",
                [dt.datetime(2024, 1, 1, 12, tzinfo=dt.UTC), None],
                dtype=pl.Datetime("us", "Europe/Istanbul"),
            ),
            pl.Series("dur", [dt.timedelta(seconds=90), None], dtype=pl.Duration("ms")),
            pl.Series("bin", [b"\x00\xff", None], dtype=pl.Binary),
            pl.Series("b", [True, None], dtype=pl.Boolean),
        ]
    )
    result = sc.write(tmp_path, {"dtypes": frame})
    back = pl.read_parquet(sc.data_dir(tmp_path) / "dtypes.parquet")
    assert back.schema == frame.schema
    assert_frame_equal(back, sc.sort_all(frame), categorical_as_str=True)
    (entry,) = result.manifest.tables
    assert entry.columns == tuple((name, str(dtype)) for name, dtype in frame.schema.items())


@pytest.mark.parametrize("uri", ["exp\udcff", "\ud800"])
def test_unencodable_uri_never_leaves_orphan_data(tmp_path: Path, uri: str) -> None:
    """Pinned choice (QA r3 m3): a str that cannot be encoded as UTF-8 fails before any data
    is renamed into place, either in ``Source`` or in ``write_snapshot``."""
    try:
        source = snapshot.source_from_git(uri, sc.GIT_SHA)
    except ValueError:
        return
    with pytest.raises(ValueError):
        sc.write(tmp_path, source=source)
    sc.assert_nothing_written(tmp_path)


# --- Build decisions (build.md "Architect: ratify" items 8-10) -------------------------------


def test_manifest_path_is_directory_with_overwrite(tmp_path: Path) -> None:
    """Item 8: the check runs before the commit renames, so no data is left without a
    manifest."""
    sc.manifest_path(tmp_path).mkdir(parents=True)
    with pytest.raises(snapshot.ManifestError):
        sc.write(tmp_path)
    with pytest.raises(snapshot.ManifestError, match="is a directory"):
        sc.write(tmp_path, overwrite=True)
    assert sc.manifest_path(tmp_path).is_dir()
    assert not os.path.lexists(sc.data_dir(tmp_path))
    assert not os.path.lexists(sc.staging_dir(tmp_path))
    assert not os.path.lexists(sc.old_dir(tmp_path))


def test_manifest_path_is_directory_keeps_existing_data(tmp_path: Path) -> None:
    sc.write(tmp_path)
    before = sc.files_bytes(tmp_path)
    sc.manifest_path(tmp_path).unlink()
    sc.manifest_path(tmp_path).mkdir()
    tables = sc.all_tables()
    tables["events"] = tables["events"].head(2)
    with pytest.raises(snapshot.ManifestError, match="is a directory"):
        sc.write(tmp_path, tables, overwrite=True)
    assert sc.files_bytes(tmp_path) == before
    assert not os.path.lexists(sc.staging_dir(tmp_path))


def test_stray_file_at_final_is_orphan(tmp_path: Path) -> None:
    """Item 9: a file where the data directory goes is orphan data, never silently
    unlinked."""
    stray = sc.data_dir(tmp_path)
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"keep me")
    with pytest.raises(snapshot.SnapshotExistsError, match="has no manifest"):
        sc.write(tmp_path)
    assert stray.read_bytes() == b"keep me"
    assert not sc.manifest_path(tmp_path).exists()
    assert not os.path.lexists(sc.staging_dir(tmp_path))
    result = sc.write(tmp_path, overwrite=True)
    assert result.status == "replaced"
    assert stray.is_dir()
    assert not os.path.lexists(sc.old_dir(tmp_path))
    assert snapshot.verify_snapshot(result.manifest_path).ok


def _outside(tmp_path: Path) -> Path:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "keep.parquet").write_bytes(b"keep me")
    return target


def test_symlinked_staging_unlinked_not_followed(tmp_path: Path) -> None:
    """Item 10: a symlink is unlinked; ``rmtree`` would refuse it, and following it would
    delete files outside the snapshot."""
    root = tmp_path / "root"
    target = _outside(tmp_path)
    sc.staging_dir(root).parent.mkdir(parents=True)
    sc.staging_dir(root).symlink_to(target, target_is_directory=True)
    result = sc.write(root)
    assert result.status == "created"
    assert not os.path.lexists(sc.staging_dir(root))
    assert (target / "keep.parquet").read_bytes() == b"keep me"
    assert snapshot.verify_snapshot(result.manifest_path).ok


def test_symlinked_old_unlinked_not_followed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = _outside(tmp_path)
    sc.write(root)
    sc.old_dir(root).symlink_to(target, target_is_directory=True)
    result = sc.write(root)
    assert result.status == "unchanged"
    assert not os.path.lexists(sc.old_dir(root))
    assert (target / "keep.parquet").read_bytes() == b"keep me"
    assert snapshot.verify_snapshot(result.manifest_path).ok


def test_remove_helper(tmp_path: Path) -> None:
    target = _outside(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    snapshot._remove(link)
    assert not os.path.lexists(link)
    assert (target / "keep.parquet").read_bytes() == b"keep me"
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "nowhere")
    snapshot._remove(dangling)
    assert not os.path.lexists(dangling)
    plain = tmp_path / "plain.txt"
    plain.write_bytes(b"x")
    snapshot._remove(plain)
    assert not os.path.lexists(plain)
    snapshot._remove(target)
    assert not os.path.lexists(target)
