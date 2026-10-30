# Conventions

These rules apply to **all** `football-*-models` repositories.

## Canonical pitch frame

All internal tables use metres on a standard **105 x 68 m** pitch.

| Item | Value |
|---|---|
| Origin `(0, 0)` | Bottom-left corner when the pitch is drawn with play running left to right |
| `x` | `0 → 105` along the length; the team in possession **always attacks towards `x = 105`** |
| `y` | `0 → 68` across the width, pointing **up**; `y = 0` is the attacking team's **right** touchline |
| Opponent goal centre | `(105, 34)` |
| Goal posts | `y = 30.34` (shooter's right) and `y = 37.66` (shooter's left), goal width 7.32 m |
| Goal-mouth height `z` | `0 → 2.44` m (crossbar) |

This is the same orientation as Opta, SPADL and mplsoccer's `custom` pitch, so Opta data needs
no flip and StatsBomb data needs a y-flip.

!!! note "Assumed pitch size"
    Real pitch dimensions vary by stadium and providers do not report them, so every event is
    mapped to the standard 105 x 68 m pitch. Distances are approximations that are consistent
    across providers.

## Provider → canonical: piecewise mapping between pitch markings

Provider grids are **not to scale**. Opta's 0–100 grid stretches the penalty area and squeezes
the goal, so a single linear scale factor is wrong where it matters most for shot models:

| Landmark | True (m) | Opta, linear `x·1.05`, `y·0.68` | StatsBomb, linear |
|---|---:|---:|---:|
| Penalty spot to goal line | 11.00 | 12.08 | 10.50 |
| Penalty-area depth | 16.50 | 17.85 | 15.75 |
| Goal width | 7.32 | 6.53 | 6.80 |

Instead, coordinates are mapped **piecewise-linearly between IFAB pitch markings**: each
provider's landmark lands exactly on the same landmark in metres, and points in between are
interpolated. This is the approach used by kloppy and mplsoccer's `Standardizer`.

### Length axis (`x`)

| Landmark | Opta | StatsBomb | Metres |
|---|---:|---:|---:|
| Own goal line | 0 | 0 | 0 |
| Own six-yard line | 5.8 | 6 | 5.5 |
| Own penalty spot | 11.5 | 12 | 11.0 |
| Own penalty-area line | 17.0 | 18 | 16.5 |
| Halfway line | 50 | 60 | 52.5 |
| Opponent penalty-area line | 83.0 | 102 | 88.5 |
| Opponent penalty spot | 88.5 | 108 | 94.0 |
| Opponent six-yard line | 94.2 | 114 | 99.5 |
| Opponent goal line | 100 | 120 | 105 |

### Width axis (`y`)

| Landmark | Opta (`y` up) | StatsBomb (`y` down) | Metres (`y` up) |
|---|---:|---:|---:|
| Right touchline | 0 | 80 | 0 |
| Penalty-area side | 21.1 | 62 | 13.84 |
| Six-yard side | 36.8 | 50 | 24.84 |
| Right post | 45.2 | 44 | 30.34 |
| Left post | 54.8 | 36 | 37.66 |
| Six-yard side | 63.2 | 30 | 43.16 |
| Penalty-area side | 78.9 | 18 | 54.16 |
| Left touchline | 100 | 0 | 68 |

"Right" and "left" are from the attacking team's point of view. Landmark values match
mplsoccer 1.8.1's `opta`, `statsbomb` and `custom` (105 x 68) pitch definitions, which the
`football_core.pitch` tests use as an oracle.

### Goal mouth

Shot end locations in the goal mouth are stored in the **same `y` axis** as the pitch
(`goal_y`, posts at 30.34 / 37.66 m) with the height `goal_z` in metres above the ground.
Both providers map through one landmark rule, `pitch.goal_mouth_to_canonical`: the width
landmarks above plus the **outer face of each post**, and in height the ground, the underside
of the crossbar and its top. Posts and bar are 12 cm wide (IFAB Law 1, `pitch.POST_WIDTH`).

| Landmark | Metres (`y` / `z`) | Opta `goalMouthY` / `goalMouthZ` | StatsBomb `end_location`, yd |
|---|---:|---:|---:|
| Right post, outer face | 30.22 | 44.2 | 44.0 + 0.12 · 6 / 5.5 |
| Right post, inner face | 30.34 | 45.2 | 44.0 |
| Left post, inner face | 37.66 | 54.8 | 36.0 |
| Left post, outer face | 37.78 | 55.8 | 36.0 − 0.12 · 6 / 5.5 |
| Ground | 0 | 0 | 0 |
| Crossbar, underside | 2.44 | 38 | 2.44 / 0.9144 |
| Crossbar, top | 2.56 | 42 | 2.56 / 0.9144 |

Opta draws the woodwork as a **band**: post hits are coded in [44.2, 45.2] and [54.8, 55.8],
bar hits at heights in (38, 42), and no goal, save or miss lies in either band. The band maps
onto the 12 cm of the post or bar, so a woodwork hit lands on the woodwork. Inside the frame
([45.2, 54.8] × [0, 38]) the Opta map is unchanged. StatsBomb's extra knots are collinear, so
its map is unchanged. Above the bar top the height keeps the below-bar scale, and a negative
height clips to 0.

**Goal frame.** `pitch.to_goal_frame(goal_y, goal_z)` gives metres from the goal centre,
**y positive = the shooter's right**, z up from the ground: the right post is at y = +3.66,
the left post at −3.66, the crossbar at z = 2.44. The stored columns stay in pitch-width metres
(schema 1.2); every goal-mouth check (`football_core.goalmouth`) and every xGOT feature reads
the goal frame through this one function. `pitch.frame_distance` is the distance from a
goal-frame point to the opening (0 inside it), `pitch.woodwork_distance` the distance to the
posts and bar as 12 cm solids.

**StatsBomb stop points.** StatsBomb records a save, or any shot the keeper stopped, where the
ball was stopped, not where it crossed the line. The loader projects such a point to the goal
line along the line from the shot location (`pitch.project_to_goal_line`, in metres after the
piecewise map). The rule is outcome-blind and grid-aware: a stop point within one grid cell of
the line (raw end x ≥ 120 − 2c, with c the match's cell offset, so 1 yd on a coarse match and
0.1 yd on a fine one) counts as on the line and is kept as recorded; every other point is
projected, whatever the outcome (`io.statsbomb.is_projected`). The goal-mouth y is recorded in
tenths of a yard in every match, coarse ones included (even tenths are over-represented there,
but no match keeps a coarser grid), so a projected point is **snapped to the centre of its
0.1 yd goal-mouth cell** (`io.statsbomb.snap_goal_y`): projected StatsBomb points sit on the
same 0.1 yd lattice as recorded ones, and the lattice no longer separates goals from saves.
`fc goalmouth` prints the grid census (`goalmouth.grid_census`, counts by grid, outcome and
stop point) and `--check` fails on a `goal_y` off the lattice (`goalmouth.lattice_problems`).
`goal_z` keeps the recorded height at the stop point (no trajectory model). Opta's `goalMouthY` / `goalMouthZ` is already the goal-plane point, so
WhoScored shots are not projected.

**Opta direction and crossbar, confirmed.** The crossbar at 38: saves reach at most 38.0
and bar hits start above it. The direction of `goalMouthY`: every shot with a left zone
qualifier lies on the shooter's left (raw > 50, goal-frame y < 0) and every shot with a right
zone qualifier on the shooter's right, with 0 exceptions (`goalmouth.zone_sides`,
`fc goalmouth`; the "Close" zones 83–87 hold no shot in the archive):

| Opta zone qualifier | Named side | Goal-frame side |
|---|---|---|
| 76 LowLeft, 77 HighLeft, 83 CloseLeft, 86 CloseLeftAndHigh | left | y < 0 on every shot (shooter's left) |
| 80 LowRight, 81 HighRight, 84 CloseRight, 87 CloseRightAndHigh | right | y > 0 on every shot (shooter's right) |
| 73 MissLeft / 75 MissRight (reported apart) | left / right | all on their named side |
| 74, 78, 79, 85 | no side | — |

A physical anchor agrees: of 510 saves that are more than 2 m off centre both on
`goalMouthY` and on the line from the shot to the save point, 437 are on the same side and 73 on
the opposite one (correlation 0.624; parries explain the opposite cases).

**Facing keeper.** `gk_id` is the goalkeeper of the defending team on the pitch at the strike,
from the lineup state, never from save or goal-keeper events (those exist per outcome). StatsBomb
walks `Starting XI`, `Tactical Shift` (position id 1) and keeper `Substitution`s in event index
order and cross-checks the freeze-frame keeper of every non-penalty shot. WhoScored walks
`FormationSet` / `FormationChange` (slot 1 of q131 paired with q30) and `SubstitutionOn` into
slot 1 (q145) in clock order. Own goals get null; a shot before any lineup, or a lineup without
exactly one keeper, raises.

### Special cases
- **Own goals** (Opta qualifier 28) are recorded from the scoring side's perspective. The
  canonical shot table stores an own goal in the frame in which the goal the ball entered is
  at `x = 105`, so `distance_m` and `angle_rad` are measured to that goal; the loader flips
  with `(105 − x, 68 − y)` when needed. `team_id`, `player_id` and `is_home` still describe
  the player who put the ball in.
- **StatsBomb cell offset:** a StatsBomb value names the cell that *ends* at it, so the loader
  (`io.statsbomb.cell_offset`) subtracts half a cell before mapping, as kloppy does: 0.5 yd when
  every shot location of the match is an integer (coarse grid), else 0.05 yd (fine grid). The
  evidence: shifted penalties land on the spot (coarse mean 108.08, fine 107.99 for a true 108);
  `metadata.shot_fidelity_version` agrees with the inferred grid on all 3,764 matches that carry
  it, and a disagreement raises; goals sit at the right post (y = 44.0) 57 times but at the left
  post (36.0) only 8 times, the asymmetry of a ceiling-to-0.1 grid, and coarse matches record
  goal-mouth y in tenths. The same offset applies to freeze-frame players, `gk_*` and
  `end_location`, with two deliberate deviations from kloppy, both on post-shot columns only:
  an end x of 120 is the goal line and is not shifted (105 m), and goal-mouth height `z` is not
  shifted (recorded heights start at 0.0). A 3-value end location uses y − 0.05 on both grids.
  27 coarse-grid shots have tenths in their freeze frames and still get 0.5 (error ≤ 0.45 yd).
  `visible_area` polygons are not shifted. The pitch functions take the offset as an explicit
  argument (`cell_offset=0.0` after the loader has shifted).
- **Out-of-bounds values** are clipped to the pitch by the interpolation.

## Geometry

With `X = 105 − x` (distance to the goal line) and `C = |y − 34|` (offset from the centre line):

$$
d = \sqrt{X^2 + C^2}, \qquad
\theta = \operatorname{atan2}\!\left(7.32\,X,\; X^2 + C^2 - 3.66^2\right) \in [0, \pi]
$$

`θ` is the angle the two posts subtend at the shot location: 0.6424 rad (36.8°) at the penalty
spot and π on the goal line between the posts.

## Naming
- Code, docs, commits and notebooks: **English**.
- Columns: `snake_case`; coordinates `x`, `y`, end location `end_x`, `end_y`, `end_z`, goal
  mouth `goal_y`, `goal_z`, goalkeeper `gk_x`, `gk_y`. The full column lists are in
  [`schema`](api/schema.md).
- Identifiers are provider-prefixed strings (`sb:3788741`, `ws:2895397135`). For WhoScored
  events use the global Opta `id`, not `eventId`, which is numbered per team.
- DataFrames: **polars**. Convert with `.to_numpy()` / `.to_pandas()` only at library boundaries.

## Canonical tables and leakage

Every loader writes the three tables in [`football_core.schema`](api/schema.md): `matches`
(one row per match), `shots` (one row per shot) and `freeze_frames` (one row per visible player
other than the shooter). A table must pass `validate_matches` / `validate_shots` /
`validate_freeze_frames` before it is used or saved; with `matches=`, `validate_shots` also
checks that every shot's match exists and agrees with it (date, competition, season, teams,
home side).

- **Column roles.** Each column has a role: `identity`, `pre_shot`, `post_shot`, `outcome`,
  `restricted` or `lineage`. Model feature lists are built from the schema, for example
  `column_names("shots", roles="pre_shot")` for xG, and are never typed by hand. Only
  `pre_shot` columns are xG features; `big_chance` (`restricted`) is allowed only in a
  labelled ablation.
- **Missing values.** Null is the only missing marker: no NaN, no ±inf, and no filled-in
  default such as 0. For a yes / no flag, `False` means the provider records the attribute
  for that competition-season and did not tag it; null means it is not recorded there.
- **Version.** Every row carries `schema_version` (currently `1.1`: 1.1 added the `matches`
  table), and validation rejects any other value.

## Validation splits
Never split events randomly by row. Split by **match**, **season** or **league** to avoid
leakage between events from the same game.
