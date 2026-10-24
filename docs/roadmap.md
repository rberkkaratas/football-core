# Roadmap

Built just ahead of what `football-shot-models` needs. Milestone IDs match the shot-models
roadmap.

| ID | Milestone | Done when |
|---|---|---|
| S0.2 | Coordinate conventions rewritten (piecewise mapping) | ✅ [Conventions](conventions.md) match the C2 tests |
| S0.3 | Zensical docs, mypy, hypothesis, `make check` | ✅ All five repos pass `make check` |
| C2 | `pitch` v1 | Landmark, property and mplsoccer-oracle tests pass |
| C3 | `schema` v1 | Synthetic frames validate; bad enums / coordinates fail clearly |
| C6 | Snapshots + manifest CLI | Identical hashes on rerun; `verify` catches a one-byte change |
| C4 | `io.statsbomb` | ~100k shots, zero schema violations, goals reconcile with scores |
| C5 | `io.whoscored` (CSV archive) | Duplicate match dropped; goals reconcile with metadata scores; fixtures cover q82/q28/q9/q214/q55 |
| C7 | `viz` | Image tests pass for shot map and goal-mouth plot |

## Decisions
- **Docs:** Zensical (pinned), reading `mkdocs.yml`.
- **DataFrames:** polars; parquet snapshots are the working format.
- **Private data:** the WhoScored CSV archive, with raw JSON from `opposition-engine` for new matches.
