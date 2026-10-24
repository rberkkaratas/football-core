# Conventions

These rules apply to **all** `football-*-models` repositories.

## Coordinate system

All internal tables use metres on a standard **105 x 68 m** pitch.

| Item | Value |
|---|---|
| Origin `(0, 0)` | Corner where the attacking team's own goal line meets the touchline on its right-hand side, viewed with play running left to right |
| `x` | `0 → 105` along the pitch length; the team in possession **always attacks towards `x = 105`** |
| `y` | `0 → 68` across the pitch width |
| Opponent goal centre | `(105, 34)` |
| Goal posts | `y = 34 ± 3.66` (goal width 7.32 m) |
| Goal height | 2.44 m (used for goal-mouth `z`) |

!!! note "Assumed pitch size"
    Real pitch dimensions vary by stadium and providers do not report them, so every
    event is mapped to the standard 105 x 68 m pitch. Distances are therefore
    approximations, consistent across providers.

### Provider conversions

| Provider | Native frame | To metres |
|---|---|---|
| Opta / WhoScored | `x, y ∈ [0, 100]`, attacking towards `x = 100`, `y = 0` on the right | `x · 1.05`, `y · 0.68` |
| StatsBomb | `x ∈ [0, 120]`, `y ∈ [0, 80]`, `y = 0` at the top | `x · 105/120`, `(80 − y) · 68/80` |
| Tracking (e.g. Metrica, SkillCorner) | provider-specific, often centred metres | per-loader; documented in the loader |

Exact orientation rules for each provider will be confirmed against real samples and
covered by unit tests in `football_core.pitch` before any model uses them.

## Naming
- Code, docs, commits and notebooks: **English**.
- Columns: `snake_case`; coordinates `x`, `y` (start), `end_x`, `end_y`, goal mouth `goal_y`, `goal_z`.
- Identifiers: `match_id`, `team_id`, `player_id`, `event_id` are always provider-prefixed
  strings (e.g. `sb:3788741`, `ws:1903213`) so sources can be mixed without collisions.

## Validation splits
Never split events randomly by row. Split by **match** or **season** to avoid leakage
between events from the same game.
