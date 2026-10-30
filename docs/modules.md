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
│   │                    #   facing keeper from the lineup events; previous play event and
│   │                    #   rebound flag of each shot
│   └── snapshot.py      # parquet + manifest writer and verifier, builder registry
├── viz.py               # shot map, front-on goal-mouth plot, attribution; light and dark
│                        #   themes, the chart canvas (title, subtitle, stat strip, body, footer),
│                        #   pitch crop, shot marks, machine checks (overlap, clipping, contrast)
└── fonts/               # Inter 4.0 static TTFs and the SIL OFL licence (package data)
src/football_docs/       # docs-site tooling for every football-* site (outside football_core)
├── roles.py             # griffe extension: Sphinx roles → links or plain code at build time
├── pages.py             # internal-code and number checks, results block, built-site role check
├── theme.py             # site CSS and logo generated from viz.THEMES (`make docs-theme`)
└── meta.py              # repository metadata check: author, CI and Pages workflows
```

| Module | Status | Milestones |
|---|---|---|
| `pitch` | ✅ Done (goal frame, goal-mouth landmarks, projection, inverse goal-mouth map) | before the milestones; xGOT; goalkeepers |
| `goalmouth` | ✅ Done (`match_grids`, `grid_census`, lattice check) | xGOT; goalkeepers |
| `schema` | ✅ Done (1.1: matches table; 1.2: `gk_id`; 1.3: `prev_action`, `rebound`) | data foundation; xG model; xGOT; polish round 1 |
| `io.snapshot` | ✅ Done | data foundation |
| `cli` | ✅ Done (`fc goalmouth`, grid census and lattice line) | data foundation; xGOT; goalkeepers |
| `io.statsbomb` | ✅ Done (goal-line projection, `gk_id`, lattice snap, `player_names`) | StatsBomb xG baseline; xGOT; goalkeepers |
| `io.whoscored` | ✅ Done (builder `whoscored`; manifest git-ignored; `gk_id`; `previous_play`, `shot_context`) | WhoScored xG v1; xGOT; polish round 1 |
| `viz` | ✅ Done (`viz` extra: matplotlib; theme layer, Inter, canvas, checks) | StatsBomb xG baseline; chart round 1 |
| `football_docs.roles`, `.pages`, `.theme`, `.meta` | ✅ Done (docs-site tooling: roles rendered at build time, page checks and results block, site theme from `viz.THEMES`, repository metadata check: author, CI and Pages wiring) | docs round 1 |

Every module ships with unit tests before a model repo depends on it. Reference libraries
(mplsoccer, kloppy, socceraction) are used as **test oracles only**, never as runtime
dependencies for the coordinate or schema logic.
