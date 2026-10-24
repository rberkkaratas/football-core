# Modules (planned)

```
src/football_core/
├── io/
│   ├── statsbomb.py     # read StatsBomb open data (events, 360 freeze frames)
│   ├── whoscored.py     # export Postgres → parquet snapshot; read snapshots
│   └── tracking.py      # public tracking sample loaders
├── schema.py            # canonical event / shot tables and column contracts
├── pitch.py             # provider → 105 x 68 m conversions, distance & angle helpers
└── viz/                 # pitch, shot map, goal-mouth plots
```

Each module gets unit tests before a model repo depends on it.
