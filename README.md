# football-core

Shared foundation for the football model repositories:

| Repo | Scope |
|---|---|
| **football-core** | Data loaders, canonical schema, pitch coordinates, plotting |
| [football-shot-models](../football-shot-models) | xG, xGOT / PSxG, goalkeeper shot-stopping |
| [football-passing-models](../football-passing-models) | xPass, xA, xGChain / xGBuildup, packing |
| [football-possession-value](../football-possession-value) | xT, VAEP, OBV / PV-style action valuation |
| [football-tracking-models](../football-tracking-models) | Pitch control, EPV, OBSO, pressing intensity |

> **Status:** skeleton. No modules are implemented yet; see [`docs/roadmap.md`](docs/roadmap.md).

## Why a shared package
Every model repo needs the same three things: read events from different providers, put them
on one pitch, and draw them. Doing that once keeps the model repos focused on modelling and
guarantees that a coordinate in one repo means the same thing in every other.

## Conventions (summary)
- **Coordinates:** metres on a 105 x 68 pitch; the team in possession attacks towards `x = 105`.
- **Data:** scraped WhoScored data never leaves this machine; public work uses StatsBomb open data.

Full detail: [`docs/conventions.md`](docs/conventions.md), [`docs/data-policy.md`](docs/data-policy.md).

## Development
```bash
uv sync --all-groups
make check                  # ruff, mypy, pytest, strict docs build
make docs-serve             # documentation site at http://127.0.0.1:8000
```

## License
MIT. See [LICENSE](LICENSE).
