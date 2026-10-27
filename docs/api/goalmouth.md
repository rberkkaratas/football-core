# goalmouth

The P2.1 goal-mouth checks, counts only. Every distance is computed in the goal frame of
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
- `fc goalmouth MANIFEST...` prints both tables for verified snapshots and writes nothing.

::: football_core.goalmouth
