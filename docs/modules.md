# Modules

```
src/football_core/
├── pitch.py             # provider → 105 x 68 m piecewise mapping, goal mouth and goal frame,
│                        #   goal-line projection, distance & angle
├── goalmouth.py         # goal-mouth census (goals in the frame, woodwork), the Opta zone check;
│                        #   StatsBomb recording grid per match, grid census, lattice check
├── schema.py            # matches, shot + freeze-frame tables: dtypes, pandera models, roles
├── cli.py               # the `fc` console script (alias `football-core`): snapshot, verify,
│                        #   builders, goalmouth
├── io/
│   ├── __init__.py      # package docstring only; loaders are imported by name
│   ├── statsbomb.py     # open data at a pinned commit → shots, freeze frames, matches; logo;
│   │                    #   stop points projected to the goal line and snapped to the 0.1 yd
│   │                    #   lattice, facing keeper, keeper names from the lineups
│   ├── whoscored.py     # private WhoScored / Opta CSV archive → shots and matches (no frames);
│   │                    #   facing keeper from the lineup events
│   └── snapshot.py      # parquet + manifest writer and verifier, builder registry
└── viz.py               # shot map, front-on goal-mouth plot, attribution
```

| Module | Status | Milestone |
|---|---|---|
| `pitch` | ✅ Done (goal frame, goal-mouth landmarks, projection, inverse goal-mouth map) | C2, P2.1, MS6 |
| `goalmouth` | ✅ Done (MS6: `match_grids`, `grid_census`, lattice check) | P2.1, MS6 |
| `schema` | ✅ Done (1.1: matches table; 1.2: `gk_id`) | C3, C3.1, P2.1 |
| `io.snapshot` | ✅ Done | C6 |
| `cli` | ✅ Done (`fc goalmouth`; MS6: grid census and lattice line) | C6, P2.1, MS6 |
| `io.statsbomb` | ✅ Done (goal-line projection, `gk_id`; MS6: lattice snap, `player_names`) | C4, P2.1, MS6 |
| `io.whoscored` | ✅ Done (builder `whoscored`; manifest git-ignored; `gk_id`) | C5, P2.1 |
| `viz` | ✅ Done (`viz` extra: matplotlib) | C7 |

Every module ships with unit tests before a model repo depends on it. Reference libraries
(mplsoccer, kloppy, socceraction) are used as **test oracles only**, never as runtime
dependencies for the coordinate or schema logic.
