# football-core

football-core is the shared foundation of the `football-*` model repositories: provider loaders,
one canonical shot schema, pitch coordinates on a 105 × 68 m frame, snapshots with manifests,
and the chart theme every published chart is drawn on. It reads StatsBomb open data, which is
public, and a private WhoScored / Opta event archive, which never leaves the author's machine:
only counts and metrics from it are published.

## What it guarantees

- **One pitch.** Provider grids map to metres piecewise between IFAB pitch markings, never by
  linear scaling; the mapping agrees with the mplsoccer oracle to within 1e-6 m on 5,000 random
  points per provider ([Conventions](conventions.md)).
- **Every StatsBomb shot loads.** All 101,614 shots of 3,961 matches at the pinned open-data
  commit load with zero schema violations, plus 1,305,559 freeze-frame rows, and goals reconcile
  with the recorded score in every match (shoot-outs excluded; see the [Roadmap](roadmap.md)).
- **The private archive loads the same way.** 3,802 matches and 98,261 shots with zero schema
  violations; goals reconcile with the metadata score in all 2,186 scored matches.
- **One goal frame.** Both providers' goal mouths map through one landmark rule: 11,309 of
  11,328 StatsBomb goals and 10,391 of 10,391 WhoScored goals land inside the frame within
  0.15 m.
- **Reproducible data.** A snapshot rerun on unchanged inputs writes byte-identical parquet
  files and manifest, and `fc verify` names any file that changed.

## Install and reproduce

The model repositories expect football-core checked out next to them:

```text
workspace/
├── football-core/
└── football-shot-models/
```

```bash
cd football-core
uv sync --all-groups   # runtime, viz, dev and docs dependencies
make check             # ruff, mypy, pytest with doctests, strict docs build
```

The public data chain downloads StatsBomb open data at its pinned commit into the git-ignored
`data/` folder and freezes it as a verified snapshot:

```bash
uv run fc snapshot statsbomb-open-v1 --builder statsbomb --input data/raw/statsbomb/open-data
uv run fc verify manifests/statsbomb-open-v1.json
```

The WhoScored loader needs the private archive, so its results cannot be reproduced from this
repository alone. `make docs-serve` serves this site locally.

## Read next

<div class="grid cards" markdown>

- [**Conventions**](conventions.md): the coordinate frame, landmarks and goal mouth.
- [**Data policy**](data-policy.md): what may be used, stored and published.
- [**Modules**](modules.md): the package layout and what each module does.
- [**Roadmap**](roadmap.md): what was built, in which order, with its results.
- [**API reference**](api/pitch.md): every public function, with examples.

</div>
