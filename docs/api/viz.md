# viz

Shot map and front-on goal-mouth plot on the canonical frame, with the data source credited
automatically, and the theme layer every publishable chart is drawn on. Install the `viz` extra (`uv add 'football-core[viz]'`; matplotlib only).

```python
from football_core import viz
from football_core.io import statsbomb

statsbomb.fetch_logo()  # once: into the git-ignored cache, never committed
ax = viz.shot_map(shots, xg="provider_xg")
ax.figure.savefig("outputs/shot-map.png", dpi=200, bbox_inches="tight")
```

!!! note "StatsBomb saves"
    The goal-mouth point is the provider's goal-plane point. For a StatsBomb stop point (a save,
    or any shot the keeper stopped) the loader has already projected `goal_y` to the goal line
    from the shot location and snapped it to the 0.1 yd lattice (see
    [Conventions](../conventions.md#goal-mouth)); `goal_z` is the height where the ball was
    stopped, so a save can sit above or beside the frame. Do not project the points again.

## Theme layer

Publishable charts are drawn on a themed canvas rather than on matplotlib's defaults.

- **Themes.** `THEMES["light"]` (Berk's deck: `#f8fafc` background, slate titles, `#e4002b`
  accent for goals only) and `THEMES["dark"]` (for social feeds) share one set of named tokens;
  every text and mark token meets WCAG contrast on the surfaces it sits on (4.5:1 body text,
  3:1 large text and marks).
- **Fonts.** Inter 4.0 (Regular, Medium, SemiBold, Bold; SIL Open Font License, `fonts/LICENSE.txt`)
  ships as package data and is registered at import without touching `rcParams`, so the earlier
  `shot_map` / `goal_mouth` keep matplotlib's default font. If a file is missing or does not
  resolve, `register_fonts` warns and the charts use DejaVu Sans (`FALLBACK_FAMILY`).
- **Layout.** `canvas(theme, size, title=..., subtitle=..., footer=Footer(...), stats=...)` lays out
  a statement title (stepping down from 21 to 19 and 17 pt before it refuses a third line; the
  wider fallback font also tries 15 pt), a grey context subtitle, an optional stat strip on a
  rounded panel, the body, and a footer with the data source, the model, the not-shown note,
  "Prepared by R. Berk Karataş" and the provider logo, at 1080 × 1350 (`portrait`) or
  1080 × 1080 px (`square`). Every width is measured, never guessed: a part that does not fit
  raises `ValueError` naming it.
- **Shots.** `crop_pitch` crops the half pitch to the shots (depth capped at 35 m; the caller
  prints the undrawn shots), `ink_scale` sizes markers by an ink budget, `draw_shots` draws
  misses as rings under goals, and `shot_keys` builds the legend.
- **Checks.** `check(canvas)` returns the problems as sentences: text overlap, text on a mark,
  clipping, panel containment, text and mark contrast measured on the rendered pixels, and key
  size. Every chart passes it before it is written.

```python
from football_core import viz

c = viz.canvas(
    "dark",
    "square",
    title="Most goals came from inside the box",
    subtitle="Team · League 2015/2016",
    stats=[viz.Stat("512", "Shots")],
    footer=viz.Footer(("statsbomb",), model="xG: own model"),
)
assert viz.check(c) == []
viz.save(c, "outputs/chart.png")  # git-ignored; a consumer's make docs-charts publishes
```

::: football_core.viz
