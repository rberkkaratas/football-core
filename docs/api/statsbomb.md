# io.statsbomb

StatsBomb open data at a pinned commit, cached under the git-ignored `data/` folder and
loaded into the C3 `shots` and `freeze_frames` tables plus a `matches` table.

!!! warning "Licence"
    Never commit or redistribute StatsBomb data. Every chart and every published result
    (model cards, posts) credits StatsBomb as the data source, with the StatsBomb logo on
    charts.

```bash
uv run fc snapshot statsbomb-open-v1 --builder statsbomb --input data/raw/statsbomb/open-data
```

```python
from football_core.io import statsbomb

checkout = statsbomb.sync("data/raw/statsbomb/open-data")  # no network once complete
data = statsbomb.load(checkout, match_ids=[3857276])
assert statsbomb.reconcile_goals(data.shots, data.matches).is_empty()
```

::: football_core.io.statsbomb
