# Roadmap

Built just ahead of what `football-shot-models` needs. The plan has six milestones; the
component IDs (C*, M*, P*) inside them stay the same across both repos. The full plan, with
model milestones, is in the `football-shot-models` roadmap.

## Done before the milestones
| ID | Work | Result |
|---|---|---|
| S0.2 | Coordinate conventions rewritten (piecewise mapping) | ✅ [Conventions](conventions.md) match the C2 tests |
| S0.3 | Zensical docs, mypy, hypothesis, `make check` | ✅ All five repos pass `make check` |
| C2 | `pitch` v1 | ✅ Landmark, property, doctest and mplsoccer-oracle tests pass (≤ 1e-6 m on 5,000 random points per provider) |

## Core work by milestone
| Milestone | ID | Component | Done when |
|---|---|---|---|
| 1 · Data foundation ✅ | C3 | `schema` v1 | ✅ 47-column shot table and 6-column freeze-frame table, each column tagged with a leakage role. Synthetic frames validate; bad enums, out-of-bounds coordinates, wrong dtypes, missing columns and duplicate `shot_id` fail with messages naming the column and value (516 schema tests and doctests pass) |
| | C6 | Snapshots + manifest CLI | ✅ `io.snapshot` and the `fc` CLI. A rerun on unchanged inputs writes byte-identical parquet files and manifest (also after shuffling or re-chunking the input rows and in a fresh process); `fc verify` exits 1 and names the file on a one-byte flip, append or truncation, a missing file and an extra file (635 snapshot tests pass) |
| 2 · StatsBomb xG baseline ✅ | C4 | `io.statsbomb` | ✅ All 101,614 shots (101,227 Shot + 387 own goals) of 3,961 matches at the pinned open-data commit load with zero schema violations, plus 1,305,559 freeze-frame rows; goals reconcile with the score in every match (shoot-outs excluded). 418 of 426 matches with 360 files give visible areas; 8 upstream-defective files are named and skipped. Coordinates agree with kloppy to 1e-9 m in both boxes (x) and everywhere (y); elsewhere x differs by ≤ 0.57 m (kloppy's two extra x landmarks) |
| | C7 | `viz` | ✅ Image tests (Agg, tolerance 0.5) pass for the shot map (marker area ∝ xG) and the front-on 7.32 × 2.44 m goal-mouth plot; the StatsBomb logo and attribution are added automatically and a missing logo fails before drawing |
| 3 · xG model ✅ | C3.1 | `schema` 1.1: canonical matches table (A10) | ✅ 14-column `matches` table (one row per match, `gender` the only `pre_shot` column) with 4 rules, and `validate_shots(..., matches=)` checks each shot's match exists and agrees with it. The rebuilt `statsbomb-open-v1` snapshot writes 3,961 matches (2,651 men's, 1,310 women's) with zero violations; 101,614 shots and 1,305,559 freeze-frame rows unchanged, and the rebuild leaves every MS2 card metric unchanged |
| 4 · WhoScored xG v1 ✅ | C5 | `io.whoscored` (CSV archive) | ✅ The private archive loads into 3,802 matches and 98,261 shots (empty freeze-frame table) with zero schema violations. One keep rule, status first, partitions the 4,075 `match_ids` rows, so the duplicated match is kept once (its copy is `wrong_competition`) whatever the read order; each drop is logged with its count: wrong_competition 1, pending 6, unvalidated 252, no_event_file 14. Kept without a score: 1,612 matches with no metadata and 4 extra-time matches (the provider's 90-minute score is checked against the periods 1–2 goals, then nulled); 4 kept matches end before minute 85 (counted). Goals reconcile with the metadata score in all 2,186 scored matches. Raw Opta 0–100 coordinates map through the `pitch` landmarks; synthetic fixtures cover q82, q28, q9, q214, q55 and three header layouts. The manifest lists no file (`list_files=False`), is git-ignored, and `fc verify` passes. No schema bump (1.1) |

## Decisions
- **Docs:** Zensical (pinned), reading `mkdocs.yml`.
- **DataFrames:** polars; parquet snapshots are the working format.
- **Snapshots:** data in `data/snapshots/<name>/` (git-ignored), manifest in `manifests/<name>.json` (committed for public sources; a WhoScored manifest is git-ignored), no DVC. The manifest holds no timestamp, so reruns produce no diff.
- **Private data:** the WhoScored CSV archive, with raw JSON from `opposition-engine` for new matches.
