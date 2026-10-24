# Data policy

| Source | Use | Store | Publish |
|---|---|---|---|
| StatsBomb open data (GitHub JSON, pinned commit) | Public training, benchmarks, every published chart | `data/raw/statsbomb/` (git-ignored) | Results and charts only, always with the StatsBomb logo. The licence forbids redistributing the data itself |
| WhoScored / Opta event CSV archive (scraped; 4,025 matches, 11 competitions, 2025-26 and 2026-27) | Private training and research | Parquet snapshots in `data/snapshots/` (git-ignored) | **Never**: no raw rows, no derived tables, no files in any repo |
| WhoScored raw JSON in `opposition-engine` (`raw.raw_match_json`) | Lossless source for new matches | Same as above | **Never** |
| Public tracking samples | Tracking models | `data/raw/<provider>/` (git-ignored) | Per that dataset's licence |

## Snapshots
A snapshot is an immutable, zstd-compressed parquet export written with **polars**
(`football_core.io.snapshot`, the `fc snapshot` command). The data goes to
`data/snapshots/<name>/<table>.parquet` (git-ignored; the writer refuses a location that git
would track) and its manifest to `manifests/<name>.json`, which records:

- the schema version (`football_core.schema.SCHEMA_VERSION`, also written on every row) and
  the C3 schema each table passed
- the source: a git commit SHA (StatsBomb open data), a digest of every input file (the
  archive), or a hash of the export SQL plus a digest of the fetched rows (`raw.raw_match_json`)
- per data file: name, sha256 and bytes, plus one `data_digest` over them
- per table: rows, match count, columns with dtypes, and rows, matches and date range per
  competition-season
- the library versions and football-core's own commit

The manifest has **no timestamp**: a rerun on unchanged inputs writes identical bytes, and
the git commit time of the manifest stands in for a creation time.

Manifests are committed; data never is. `fc verify` re-hashes a snapshot against its manifest.

**Binding rule:** a manifest for a private source (WhoScored) holds aggregates and digests
only: `list_files=False`, no per-match ids or file names.

## Known archive issues the export must handle
- One match is stored twice under different IDs (`Belgium_Pro_League/…_1904939` is a copy of
  `Championship/…_1908515`). Matches whose event IDs already exist are dropped.
- Column order and presence vary between files; `isGoal` and `isOwnGoal` are sometimes missing.
  Columns are read by name; goals come from `type`, own goals from qualifier 28.
- `eventId` is numbered per team. The global `id` identifies shots; WhoScored reuses it on some
  synthetic `OffsideGiven` events, so it is not assumed unique across all events.

## Rule of thumb
If a file came from WhoScored, it stays in a git-ignored folder on this machine.
Aggregate findings (e.g. "model log loss on private data: 0.29") may be written in docs.
