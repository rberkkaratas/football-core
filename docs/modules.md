# Modules

```
src/football_core/
├── pitch.py             # provider → 105 x 68 m piecewise mapping, goal mouth, distance & angle
├── schema.py            # canonical shot table (polars dtypes, pandera model), SCHEMA_VERSION
├── io/
│   ├── statsbomb.py     # open-data events + shot freeze frames (GitHub JSON, pinned SHA)
│   ├── whoscored.py     # CSV archive / raw JSON → canonical shots, dedup, snapshot export
│   └── snapshot.py      # parquet + manifest writer, `fc snapshot` / `fc verify`
└── viz/                 # shot map, front-on goal-mouth plot
```

| Module | Status | Milestone |
|---|---|---|
| `pitch` | In progress | C2 |
| `schema` | Planned | C3 |
| `io.snapshot` | Planned | C6 |
| `io.statsbomb` | Planned | C4 |
| `io.whoscored` | Planned | C5 |
| `viz` | Planned | C7 |

Every module ships with unit tests before a model repo depends on it. Reference libraries
(mplsoccer, kloppy, socceraction) are used as **test oracles only**, never as runtime
dependencies for the coordinate or schema logic.
