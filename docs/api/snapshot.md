# io.snapshot

Immutable parquet snapshots with a committed JSON manifest. A model repo trains on a
snapshot, commits its manifest, and anyone can later prove that the data on disk is exactly
the data the model saw: `fc verify` names every file that changed, went missing or appeared.

```python
from football_core.io import snapshot

result = snapshot.write_snapshot("demo", {"shots": shots}, source, root=".")
report = snapshot.verify_snapshot("manifests/demo.json")
tables = snapshot.load_snapshot("manifests/demo.json")  # verifies first
```

## Layout

Relative to the root (the consuming repo, `--root`, default `.`):

| Path | Committed? | What |
|---|---|---|
| `data/snapshots/<name>/<table>.parquet` | **Never** (`data/*` is git-ignored) | One zstd parquet file per table, nothing else |
| `manifests/<name>.json` | Yes | The manifest |

The manifest stores `data_dir` relative to itself, always exactly
`"../data/snapshots/<name>"`, so it holds no absolute path or user name. Data kept elsewhere
is checked with `--data-dir`.

**Git-ignore guard.** Before writing anything, the writer asks git (`check-ignore`) whether
each data file would be ignored, from the nearest existing directory above it whose real
path is itself (so a symlinked `data/` cannot move the check outside the repo). A file
that git would track is refused with a message naming it; run from the repo root or add
`data/*` to `.gitignore`. Outside a work tree (or without git) the guard is skipped. When git
cannot answer, the write is refused (fails closed). A `data/` that is a **symlink** is
refused with git's reason ("beyond a symbolic link"): git only sees the link. Only data is
guarded; the manifest is meant to be committed. The transient `.<name>.staging/` and
`.<name>.old/` directories sit next to the data directory, so `data/*` covers them too.

## Determinism

A rerun on unchanged inputs in the same environment writes byte-identical data files and a
byte-identical manifest, so it produces no git diff:

- every table is sorted by its `sort_by` keys and then by **all** columns (nulls last), so
  input row order never reaches the file;
- the sorted frame is rechunked before writing, because the chunk layout of a sort depends
  on the thread count and can reach the bytes of nested columns;
- the writer settings are fixed: zstd level 3, statistics on, `row_group_size=100_000`;
- the manifest has **no timestamp**. Its creation time is its commit time in the consuming
  repo.

Limits: rows that are equal in every column except the sign of a zero or the payload of a NaN
keep their input order. The parquet file embeds the polars version, so a different polars
may write different bytes; the manifest records the versions, which explains such a change.

"Same environment" includes football-core's own commit. When football-core runs from its git
checkout, the manifest records that commit (`football_core_git`) and whether the checkout has
uncommitted changes (`football_core_dirty`). Write snapshots in the consuming repo, not inside
football-core: there, committing the manifest moves football-core's HEAD, so every rerun
rewrites the manifest (`unchanged (manifest updated)`) although the data files and their
sha256 values stay the same.

Canonical tables (`shots`, `freeze_frames`) accept only `identity` and `pre_shot` columns as
`sort_by` keys, so no file is ordered by a label such as `outcome`.

## What is validated

A table named `shots` passes `schema.validate_shots`. A table named `freeze_frames` passes
`schema.validate_freeze_frames` against those shots, so it needs a `shots` table in the same
snapshot. Every other table is generic and written as given (any dtype polars can write,
except `Object`). All checks run before anything touches the disk.

One value-dependent refusal: in a table with `competition_id` / `season_id` keys and a `Date`
`match_date`, a date outside 0001-01-01..9999-12-31 is refused. Such a date is almost always
a loader reading the epoch with the wrong unit.

## Manifest

```json
{
  "manifest_version": 1,
  "name": "demo",
  "builder": "parquet",
  "schema_version": "1.1",
  "data_dir": "../data/snapshots/demo",
  "data_digest": "<64 hex>",
  "source": {
    "kind": "files",
    "uri": "exports",
    "revision": "<64 hex>",
    "content_sha256": null,
    "files": [{"path": "events.parquet", "sha256": "<64 hex>", "bytes": 1234}]
  },
  "tables": [
    {
      "name": "events",
      "file": "events.parquet",
      "rows": 4,
      "matches": null,
      "schema": null,
      "sort_by": [],
      "columns": [{"name": "event_id", "dtype": "String"}],
      "groups": null
    }
  ],
  "files": [{"path": "events.parquet", "sha256": "<64 hex>", "bytes": 1234}],
  "parquet": {"compression": "zstd", "compression_level": 3, "statistics": true, "row_group_size": 100000},
  "environment": {"python": "3.12.13", "polars": "2.0.0", "pandera": "0.34.1", "numpy": "2.5.4",
                  "football_core": "0.1.0", "football_core_git": "<40 hex>|null", "football_core_dirty": false}
}
```

| Field | Meaning |
|---|---|
| `manifest_version` | The format version (1). Other values are refused |
| `name`, `builder` | Snapshot name; the builder that produced it (null when written directly) |
| `schema_version` | `football_core.schema.SCHEMA_VERSION` at write time |
| `data_dir` | Always `"../data/snapshots/<name>"`, relative to the manifest |
| `data_digest` | sha256 over the file hashes: the single id of the snapshot's data, for model cards |
| `source` | `git` (a commit SHA), `files` (a digest of the input files, optionally listed) or `sql` (a hash of the query text plus `content_sha256` of the fetched rows) |
| `tables[]` | Sorted by name: rows, matches, the C3 schema the table passed, sort keys, columns with dtypes, and `groups` |
| `tables[].matches` | Distinct non-null `match_id` values **present in this table** (on `shots`: matches with at least one shot), not the matches loaded |
| `tables[].groups` | Per (competition, season), when both keys are String, Enum, Categorical or integer: rows, matches and the first / last `match_date` (null unless `match_date` is a `Date`). Sorted by the keys as strings, nulls last. Aggregates only |
| `files[]` | Sorted by path: sha256 and size of each data file |
| `parquet` | The writer settings |
| `environment` | python, polars, pandera, numpy and football-core versions, and football-core's own commit when it runs from its checkout (cached per process: a long notebook session records the commit it started on). Informational: verification never reads it |

`Manifest.from_json` rejects every violation of the format with a `ManifestError` naming the
key path (for example `tables[1].rows`): invalid JSON, `NaN`, duplicate keys, a missing key
or wrong type, another `manifest_version`, bad names, another `data_dir`, a digest that does
not match, an invalid source, unsorted or duplicate tables, files or groups, counts out of
range, impossible dates, and files that do not match the tables. Unknown keys are ignored, so
a later addition stays readable. The digest is the `sha256sum` line format, so a directory
digest can be reproduced by hand with
`(cd dir && shasum -a 256 * | LC_ALL=C sort -k2) | shasum -a 256`.

## Rerun, conflicts and recovery

| Situation | Result |
|---|---|
| No manifest, no data | `created` |
| The new files hash like the existing manifest's | `unchanged`; the data directory is restored from the rebuild, and the manifest is rewritten only if its bytes differ (for example a newer pandera) |
| Different data, or data without a manifest | `SnapshotExistsError` listing `changed` / `added` / `removed` files, unless `overwrite=True` (`replaced`) |
| An unreadable existing manifest | `ManifestError`, unless `overwrite=True` |

Files are staged first and renamed into place, data first and the manifest last. Leftovers of
an interrupted run are recovered on the next run. A crash between the data rename and the
manifest write leaves new data under the old manifest: `verify` reports `changed` and the
next run needs `overwrite=True`. Concurrent writers to the same name are not supported.

## Verify

`verify_snapshot` never raises on the state of the data; each problem is one line:

| Kind | When |
|---|---|
| `changed` | A different size, or the same size with a different sha256 |
| `missing` | A listed file is absent, or is not a regular file |
| `extra` | Anything else in the data directory (a directory shows as `name/`); only `.DS_Store` is ignored |
| `unreadable` | A file or the data directory cannot be read |

## CLI

```bash
uv run fc snapshot NAME [--builder B] [--input PATH]... [--param KEY=VALUE]... [--root DIR] [--overwrite]
uv run fc verify manifests/*.json [--data-dir DIR]
uv run fc builders
```

| Exit code | When |
|---|---|
| 0 | Success; every manifest verified clean |
| 1 | `verify`: a problem in any manifest. `snapshot`: a schema error, a conflict, an unreadable manifest, or a refusal by the builder or the writer |
| 2 | Usage errors (including an unknown builder or a malformed `--param`); `verify` of a manifest that cannot be read. With several manifests, the maximum |

`fc` is also a builtin of zsh and bash (`fc` edits shell history). Run it as `uv run fc`, or
use the alias `football-core` in an activated shell. Parameter values are passed to the
builder but never written to the manifest; secrets belong in environment variables.

## Builders: how providers plug in

A builder is `Callable[[BuildRequest], BuildResult]`. The built-in `parquet` builder makes one
table per input parquet file; the built-in `statsbomb` builder
(`football_core.io.statsbomb:build_snapshot`, C4) syncs and loads the open-data cache. A
provider module adds one line to `BUILTIN_BUILDERS`; `register_builder` adds one at runtime.
A sketch of the StatsBomb builder (the real one also checks its params and logs skipped 360
files):

```python
def build_snapshot(request: BuildRequest) -> BuildResult:
    (cache,) = request.inputs  # e.g. data/raw/statsbomb/open-data
    commit = request.params.get("commit", OPEN_DATA_COMMIT)
    data = load(sync(cache, commit=commit))  # every file was blob-checked against the commit
    if not reconcile_goals(data.shots, data.matches).is_empty():
        raise ValueError("goals do not reconcile with the score")  # no snapshot is written
    return BuildResult(
        {"shots": data.shots, "freeze_frames": data.freeze_frames, "matches": data.matches},
        source_from_git(OPEN_DATA_URL, commit),
        sort_by={"shots": ("match_date", "competition_id", "season_id", "match_id")},
    )
```

`sort_by` is optional: it sets the leading sort keys (identity or pre-shot columns only on
`shots` / `freeze_frames`); every remaining column follows as a tie-breaker.

For a builder that reads a git clone, `git_revision` gives a commit only when the input is
the **clone root**; a subfolder silently takes the file-digest fallback. A git call that
times out (`GIT_TIMEOUT`, 60 s) also falls back, so the manifest's source changes while the
data does not. A private source (WhoScored) uses `list_files=False`, and a `sql` source must
pass `content_sha256=content_digest(...)` of the fetched rows.

!!! warning "The `parquet` builder lists file names"
    The built-in `parquet` builder always lists every input file name and size in
    `source.files`, and uses the name of the inputs' common folder as `source.uri`. Do not
    point it at private (WhoScored) exports: those names would land in the committed
    manifest. Private data goes through its own builder with `list_files=False`.

::: football_core.io.snapshot
