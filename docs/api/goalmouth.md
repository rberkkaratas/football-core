# goalmouth

The goal-mouth checks of the xGOT milestone, counts only. Every distance is computed in the goal frame of
`pitch.to_goal_frame` (metres from the goal centre, y positive = the shooter's right); see
[Conventions](../conventions.md#goal-mouth).

- `census(shots)` gives, per provider and outcome group (goal, saved, post, off target,
  blocked, own goal), the shots, those with a goal-mouth point, those inside the frame within
  `GOAL_TOLERANCE` (0.15 m) and those within `WOODWORK_TOLERANCE` (0.3 m) of the posts and bar
  as 12 cm solids. Callers divide by `shots` for a share.
- `zone_sides(shots)` counts, per Opta goal-mouth zone qualifier of `OPTA_ZONES`, the WhoScored
  shots on the shooter's left, right and centre; `confirms_direction` is True when every left
  zone of 76–87 lies on the left and every right zone on the right. Zones 73 / 75 (MissLeft /
  MissRight) are reported apart.
- `fc goalmouth MANIFEST...` prints both tables and the goals-inside-the-frame share per provider
  for verified snapshots and writes nothing. With `--check` it exits 1 when the xGOT milestone's criterion
  fails (`criterion_problems`: under 99 % of a provider's goals inside the frame, or the Opta
  direction not confirmed), so a loader regression fails a command after a snapshot rebuild.

The goalkeeper milestone adds the StatsBomb recording grid, counts only:

- `match_grids(shots)` gives each StatsBomb match's recording grid, `"coarse"` (1 yd) or
  `"fine"` (0.1 yd), by the loader's `cell_offset` rule over the raw shot locations (null when no
  row has a usable location). It is the one rule behind `grid_census` and the shot-models
  population filter that leaves coarse-grid matches out of the StatsBomb xGOT fit and GSAx.
- `grid_census(shots)` counts end locations on whole yards and on odd tenths by grid, outcome and
  stop point; `lattice_problems` / `lattice_counts` check that every stored StatsBomb `goal_y` is on
  the 0.1 yd goal-mouth lattice. `fc goalmouth` prints the fidelity mapping, the census and the
  lattice line, and `--check` also fails when a value is off the lattice.

::: football_core.goalmouth
