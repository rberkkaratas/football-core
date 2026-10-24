# Data policy

| Source | Use | Store | Publish |
|---|---|---|---|
| StatsBomb open data | Training, benchmarks, public demos | `data/raw/statsbomb/` (git-ignored) | Results and charts yes, with StatsBomb attribution as its licence requires |
| WhoScored / Opta events (scraped, exported from `opposition-engine`) | Private training & research | Parquet snapshots in `data/raw/whoscored/` (git-ignored) | **Never**: no raw rows, no derived tables, no files in any repo |
| Public tracking samples | Tracking models | `data/raw/<provider>/` (git-ignored) | Per that dataset's licence |

## WhoScored snapshots
A snapshot is a parquet export of events from the local `opposition-engine` Postgres
database. Each snapshot is immutable and named with its export date, e.g.
`whoscored_shots_2026-10-24.parquet`, together with a small JSON manifest (competitions,
seasons, row counts, engine version) so models trained on it are reproducible.

The export script will live in `football_core.io.whoscored` (see [Modules](modules.md)).

## Rule of thumb
If a file came from WhoScored, it stays in a git-ignored folder on this machine.
Aggregate findings (e.g. "model log loss on private data: 0.29") may be written in docs.
