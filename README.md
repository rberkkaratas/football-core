# football-core

Shared foundation for the football model repositories:

| Repo | Scope |
|---|---|
| **football-core** | Data loaders, canonical schema, pitch coordinates, plotting |
| [football-shot-models](../football-shot-models) | xG, xGOT / PSxG, goalkeeper shot-stopping |
| [football-passing-models](../football-passing-models) | xPass, xA, xGChain / xGBuildup, packing |
| [football-possession-value](../football-possession-value) | xT, VAEP, OBV / PV-style action valuation |
| [football-tracking-models](../football-tracking-models) | Pitch control, EPV, OBSO, pressing intensity |

> **Status:** `pitch` (C2), `schema` (C3; 1.1 adds the canonical matches table, C3.1; 1.2 adds the facing keeper `gk_id`, P2.1), `io.snapshot` with the `fc` CLI (C6), the StatsBomb open-data loader `io.statsbomb` (C4), the private WhoScored / Opta archive loader `io.whoscored` (C5, git-ignored snapshot and manifest), `viz` (C7, shot map and goal-mouth plot) and the goal-mouth harmonisation with `goalmouth` and `fc goalmouth` (P2.1, MS5) are implemented and tested. Next: the goalkeeper models of football-shot-models (MS6). See [`docs/roadmap.md`](docs/roadmap.md) and [`docs/modules.md`](docs/modules.md).

## Why a shared package
Every model repo needs the same three things: read events from different providers, put them
on one pitch, and draw them. Doing that once keeps the model repos focused on modelling and
guarantees that a coordinate in one repo means the same thing in every other.

## Conventions (summary)
- **Coordinates:** metres on a 105 x 68 pitch; the team in possession attacks towards `x = 105`.
  Provider grids are mapped piecewise between pitch markings, never scaled linearly.
- **Data:** scraped WhoScored data never leaves this machine; public work uses StatsBomb open data.

Full detail: [`docs/conventions.md`](docs/conventions.md), [`docs/data-policy.md`](docs/data-policy.md).

## Snapshots
Freeze a set of parquet tables and check later that they have not changed. Data goes to
`data/snapshots/<name>/` (git-ignored); commit only `manifests/<name>.json`.
```bash
uv run fc snapshot demo --input exports/      # one table per .parquet file
uv run fc verify manifests/demo.json          # exit 1 and the file name if anything changed
```
Run it as `uv run fc` (zsh and bash have a builtin `fc`) or as `football-core`. Details:
[`docs/api/snapshot.md`](docs/api/snapshot.md).

StatsBomb open data is downloaded once into the git-ignored cache and frozen as a snapshot:
```bash
uv run fc snapshot statsbomb-open-v1 --builder statsbomb --input data/raw/statsbomb/open-data
```
Never commit or redistribute the data; every StatsBomb chart carries the StatsBomb logo
([`docs/api/statsbomb.md`](docs/api/statsbomb.md), [`docs/api/viz.md`](docs/api/viz.md)).

## Development
```bash
uv sync --all-groups
make check                  # ruff, mypy, pytest, strict docs build
make docs-serve             # documentation site at http://127.0.0.1:8000
```

## License
MIT. See [LICENSE](LICENSE).
