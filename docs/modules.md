# Modules

```
src/football_core/
├── pitch.py             # provider → 105 x 68 m piecewise mapping, goal mouth, distance & angle
├── schema.py            # shot + freeze-frame tables: dtypes, pandera models, column roles
├── cli.py               # the `fc` console script (alias `football-core`): snapshot, verify, builders
├── io/
│   ├── __init__.py      # package docstring only; loaders are imported by name
│   ├── statsbomb.py     # open data at a pinned commit → shots, freeze frames, matches; logo
│   └── snapshot.py      # parquet + manifest writer and verifier, builder registry
└── viz.py               # shot map, front-on goal-mouth plot, attribution
```

Planned (C5, not written yet): `io/whoscored.py`, the WhoScored CSV archive → canonical
shots, dedup and snapshot export.

| Module | Status | Milestone |
|---|---|---|
| `pitch` | ✅ Done | C2 |
| `schema` | ✅ Done | C3 |
| `io.snapshot` | ✅ Done | C6 |
| `cli` | ✅ Done | C6 |
| `io.statsbomb` | ✅ Done | C4 |
| `io.whoscored` | Planned | C5 |
| `viz` | ✅ Done (`viz` extra: matplotlib) | C7 |

Every module ships with unit tests before a model repo depends on it. Reference libraries
(mplsoccer, kloppy, socceraction) are used as **test oracles only**, never as runtime
dependencies for the coordinate or schema logic.
