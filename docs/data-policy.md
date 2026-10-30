# Data policy

| Source | Use | Store | Publish |
|---|---|---|---|
| StatsBomb open data (GitHub JSON, pinned commit) | Public training, benchmarks, every published chart | `data/raw/statsbomb/` (git-ignored) | Results and charts only, always with the StatsBomb logo. The licence forbids redistributing the data itself |
| WhoScored / Opta event CSV archive (scraped; 4,075 `match_ids` rows, 4,025 event files, 11 competitions, 2025-26 and 2026-27) | Private training and research | Parquet snapshots in `data/snapshots/` (git-ignored) | **Never**: no raw rows, no derived tables, no files in any repo |
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
  the schema each table passed
- the source: a git commit SHA (StatsBomb open data), a digest of every input file (the
  archive), or a hash of the export SQL plus a digest of the fetched rows (`raw.raw_match_json`)
- per data file: name, sha256 and bytes, plus one `data_digest` over them
- per table: rows, match count, columns with dtypes, and rows, matches and date range per
  competition-season
- the library versions and football-core's own commit

The manifest has **no timestamp**: a rerun on unchanged inputs writes identical bytes, and
the git commit time of the manifest stands in for a creation time.

Manifests of public sources are committed; data never is. `fc verify` re-hashes a snapshot
against its manifest.

**Binding rule:** a manifest for a private source (WhoScored) holds aggregates and digests
only: `list_files=False`, no per-match ids or file names. It still holds per
competition-season counts and dates, so `manifests/whoscored-*.json` is git-ignored in every
repo and never committed; the consumer's copy (football-shot-models) is the authoritative one.
The name glob is not the only guard: the `whoscored` builder returns `BuildResult(private=True)`,
and `write_snapshot` then refuses, before writing anything, a manifest path that git would not
ignore (for example `uv run fc snapshot ws-2027 --builder whoscored`, outside the glob).

## Known archive issues the export must handle
- One match is stored twice under different IDs: `Championship/…_1908515` is the real one and
  `Belgium_Pro_League/…_1904939` the copy, which `match_ids` marks `wrong_competition`. The
  loader keeps 1908515 and drops the `wrong_competition` copy whatever the read order.
- Column order and presence vary between files; `isGoal` and `isOwnGoal` are sometimes missing.
  Columns are read by name; goals come from `type`, own goals from qualifier 28.
- `eventId` is numbered per team. The global `id` identifies shots; WhoScored reuses it on some
  synthetic `OffsideGiven` events, so it is not assumed unique across all events.

## WhoScored archive: what `io.whoscored` keeps
- **Keep rule, status first.** Every `match_ids` row is counted once: a status other than `ok`
  drops the match under that status (`wrong_competition`, `pending`, or `unvalidated` for an
  empty status) and its files are never opened; an `ok` row without an event file is
  `no_event_file`; an event file without a `match_ids` row is `unvalidated`. On the archive:
  3,802 kept; dropped wrong_competition 1, pending 6, unvalidated 252, no_event_file 14. The
  builder logs each count.
- **Home and away are never inferred.** They come from the metadata JSON (its two team ids must
  be the file's) or else from the validated team names in `match_ids`, each matched to exactly
  one team of the file.
- **Scores.** 1,612 kept matches have no metadata and no score. In the 4 extra-time matches the
  metadata holds the 90-minute score: it must equal the goals of periods 1–2, and is then
  nulled, because the canonical score is the final score. Goals reconcile with the score in
  every one of the 2,186 scored matches (shoot-outs excluded).
- **Short matches.** 4 kept matches have their last event before minute 85. They are kept as
  recorded and counted; nothing in the data tells an abandoned match from a cut-short scrape.
- **Opta flags are not StatsBomb's.** `first_time` is Opta q328 and `one_on_one` q89. On men's
  big-five league shots their shares are 0.511 vs 0.315 (StatsBomb `first_time`) and 0.019 vs
  0.055 (StatsBomb `one_on_one`), so a model fitted on one provider is not reused on the other.

## Rule of thumb
If a file came from WhoScored, it stays in a git-ignored folder on this machine.
Aggregate findings (e.g. "model log loss on private data: 0.29") may be written in docs.
