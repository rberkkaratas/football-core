# Data policy

| Source | Use | Store | Publish |
|---|---|---|---|
| StatsBomb open data (GitHub JSON, pinned commit) | Public training, benchmarks, every published chart | `data/raw/statsbomb/` (git-ignored) | Results and charts only, always with the StatsBomb logo. The licence forbids redistributing the data itself |
| WhoScored / Opta event CSV archive (scraped; 4,025 matches, 11 competitions, 2025-26 and 2026-27) | Private training and research | Parquet snapshots in `data/snapshots/` (git-ignored) | **Never**: no raw rows, no derived tables, no files in any repo |
| WhoScored raw JSON in `opposition-engine` (`raw.raw_match_json`) | Lossless source for new matches | Same as above | **Never** |
| Public tracking samples | Tracking models | `data/raw/<provider>/` (git-ignored) | Per that dataset's licence |

## StatsBomb open data: cache and logo
`statsbomb.sync` (run by `fc snapshot … --builder statsbomb`) downloads the open-data files of
one pinned commit (`OPEN_DATA_COMMIT`) into `data/raw/statsbomb/open-data/<commit>/`: a
`tree.json` listing each file's git blob id and size, and every file gzip-compressed
(`<path>.gz`). Each download is checked against its blob id before it is written, and the
cache is refused before any write unless git ignores it. A complete cache needs no network.

The StatsBomb logo is fetched once with `statsbomb.fetch_logo()` into the same ignored cache
(checked against its blob id) and is never committed. `football_core.viz` adds it, with
"Data: StatsBomb", to every chart drawn from StatsBomb rows, and refuses to draw such a chart
when the logo is missing. Charts from WhoScored rows carry the text "Data: Opta via WhoScored"
and stay private (table above).

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
