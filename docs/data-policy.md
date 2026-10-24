# Data policy

| Source | Use | Store | Publish |
|---|---|---|---|
| StatsBomb open data (GitHub JSON, pinned commit) | Public training, benchmarks, every published chart | `data/raw/statsbomb/` (git-ignored) | Results and charts only, always with the StatsBomb logo. The licence forbids redistributing the data itself |
| WhoScored / Opta event CSV archive (scraped; 4,025 matches, 11 competitions, 2025-26 and 2026-27) | Private training and research | Parquet snapshots in `data/snapshots/` (git-ignored) | **Never**: no raw rows, no derived tables, no files in any repo |
| WhoScored raw JSON in `opposition-engine` (`raw.raw_match_json`) | Lossless source for new matches | Same as above | **Never** |
| Public tracking samples | Tracking models | `data/raw/<provider>/` (git-ignored) | Per that dataset's licence |

## Snapshots
A snapshot is an immutable, zstd-compressed parquet export written with **polars**, together
with a `manifest.json` that records:

- schema version, creation time and git SHA of `football-core`
- the source: StatsBomb commit SHA, or the archive path plus a hash of every input file
- per output file: path, sha256, rows, bytes
- row counts, match counts and date range per competition-season
- library versions

Manifests are committed; data never is. `fc verify` re-hashes a snapshot against its manifest.

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
