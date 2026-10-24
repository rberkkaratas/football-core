# data/

Everything in this folder except this README is git-ignored.

| Folder | Contents |
|---|---|
| `raw/` | Untouched source files (StatsBomb open data JSON, WhoScored parquet snapshots) |
| `interim/` | Normalised tables in the shared schema (105 x 68 m coordinates) |
| `processed/` | Model-ready feature tables |

**Data policy:** WhoScored/Opta event data is scraped and must never be committed,
pushed or published. Public examples and benchmarks use StatsBomb open data only.
See `football-core/docs/data-policy.md`.
