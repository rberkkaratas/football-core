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
| 2 · StatsBomb xG baseline | C4 | `io.statsbomb` | ~100k shots, zero schema violations, goals reconcile with scores |
| | C7 | `viz` | Image tests pass for shot map and goal-mouth plot |
| 4 · WhoScored xG v1 | C5 | `io.whoscored` (CSV archive) | Duplicate match dropped; goals reconcile with metadata scores; fixtures cover q82/q28/q9/q214/q55 |

## Decisions
- **Docs:** Zensical (pinned), reading `mkdocs.yml`.
- **DataFrames:** polars; parquet snapshots are the working format.
- **Snapshots:** data in `data/snapshots/<name>/` (git-ignored), manifest in `manifests/<name>.json` (committed), no DVC. The manifest holds no timestamp, so reruns produce no diff.
- **Private data:** the WhoScored CSV archive, with raw JSON from `opposition-engine` for new matches.
