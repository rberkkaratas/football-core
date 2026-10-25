# viz

Shot map and front-on goal-mouth plot on the canonical frame, with the data source credited
automatically. Install the `viz` extra (`uv add 'football-core[viz]'`; matplotlib only).

```python
from football_core import viz
from football_core.io import statsbomb

statsbomb.fetch_logo()  # once: into the git-ignored cache, never committed
ax = viz.shot_map(shots, xg="provider_xg")
ax.figure.savefig("outputs/shot-map.png", dpi=200, bbox_inches="tight")
```

!!! note "StatsBomb saves"
    The goal-mouth point is the provider's goal-plane point. For a StatsBomb save it is where
    the keeper stopped the ball, not a projection onto the goal line, so saves can sit outside
    the frame. Projecting them onto the line is an xGOT follow-up.

::: football_core.viz
