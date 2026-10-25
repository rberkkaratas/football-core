# schema

The canonical **shot** table, the long-format **freeze-frame** table and the **matches**
table that every loader (StatsBomb in C4, WhoScored / Opta in C5) writes, with polars dtypes, pandera models and a
leakage role per column. Coordinates are canonical metres from [`pitch`](pitch.md): the
shooting team attacks towards `x = 105`, `y = 0` is the shooter's right touchline.

```python
from football_core import schema

shots = schema.validate_shots(shots)  # raises SchemaValidationError listing every problem
features = schema.column_names("shots", roles="pre_shot")  # the xG allow-list
```

## Column roles

| Role | Meaning | May be a model feature? |
|---|---|---|
| `identity` | Keys and grouping (who, which match, when) | Never; used for joins and CV groups |
| `pre_shot` | Fully determined at the instant the ball is struck: **both the value and whether it is null** | xG (M1) and everything after |
| `post_shot` | Ball flight after the strike: `goal_y`, `goal_z`, `deflected` | xGOT / PSxG (P2) only, never xG |
| `outcome` | Labels, population definitions, and where the ball was stopped (`end_x`, `end_y`, `end_z`) | Never as a feature of the same shot |
| `restricted` | Pre-shot in time, but an annotator judgement that leaks the outcome (`big_chance`) | Labelled ablation only |
| `lineage` | Provenance, provider model outputs, raw payloads, version | Never |

The xG allow-list is `column_names("shots", roles="pre_shot")`. It is an *allow*-list: a new
column stays out of the features until someone deliberately gives it `pre_shot`.

The stop point (`end_x`, `end_y`, `end_z`) has role `outcome`, not `post_shot`. Where the
keeper or a blocker stopped the ball is the keeper's action, and on the on-target population
its value (and, for WhoScored, its nullness alone) separates goals from saves. A PSxG model
that saw it would make the goals-prevented measure circular.

## Missing values

Null is the only missing marker (no NaN, no ±inf), and loaders never fill a default.

| Value | Meaning (Boolean attribute flags) |
|---|---|
| `True` | The provider tagged this attribute on this shot. |
| `False` | The provider records this attribute for this competition-season and did not tag it. |
| `null` | The provider does not record this attribute for this competition-season, or the loader cannot tell. |

- **StatsBomb** defines `first_time`, `follows_dribble`, `open_goal`, `one_on_one` and
  `deflected` for every shot, so absent means `False`. `big_chance` is always null.
- **Opta / WhoScored** qualifier flags are `False` only if the qualifier occurs among the
  shots of that competition-season; otherwise the column is null there. `follows_dribble`
  is null for WhoScored; `technique` is `volley` (q108) or null, never `normal`.
- `gk_x` / `gk_y`, `provider_xg`, `provider_psxg` and `end_*` are null when the provider
  gives no value. Never 0 and never a value the loader computes itself.
- **Nullness of a `pre_shot` column is part of its value**: it may depend on the provider,
  the competition-season and pre-shot context, never on what happened after the strike.

**Goal mouth.** `goal_y` / `goal_z` are the provider's recorded point on the goal plane. For
goals, posts and off-target shots that is the crossing point (it may lie outside the posts).
For **saved** shots it is the provider's *estimate*, not an observed crossing: Opta q102 /
q103 project the shot to the line, StatsBomb gives the y / z of the save point. They are null
after a block by an outfield player (rule `blocked_has_no_goal_mouth`), for StatsBomb
`Wayward`, and when the provider gives none. `goal_z` may be null while `goal_y` is set.

**Own goals** are expressed in the frame in which the goal the ball entered is at `x = 105`,
so `distance_m` and `angle_rad` are measured to that goal. `team_id`, `player_id` and
`is_home` still describe the player who put the ball in.

## Shot table

`shots` has 47 columns, in this canonical order. Ranges are inclusive; "id" means a
provider-prefixed string (`sb:…` / `ws:…`) whose prefix matches `provider`.

| # | Column | Dtype | Null | Allowed / range | Role | Description |
|---|---|---|---|---|---|---|
| 1 | `shot_id` | String | no | id, unique | identity | Provider event id, the primary key (Opta: the global id, never eventId). |
| 2 | `provider` | Enum(PROVIDERS) | no | statsbomb, whoscored | identity | Data source (whoscored uses the opta pitch grid). |
| 3 | `competition_id` | String | no | id | identity | Competition id. |
| 4 | `season_id` | String | no | id | identity | Season id. |
| 5 | `match_id` | String | no | id | identity | Match id, the group key for match-level cross-validation. |
| 6 | `match_date` | Date | no |  | identity | Match date. |
| 7 | `team_id` | String | no | id | identity | Team of the player who struck the ball (also for own goals). |
| 8 | `opponent_team_id` | String | no | id | identity | The other team. |
| 9 | `player_id` | String | no | id | identity | Player who struck the ball. |
| 10 | `is_home` | Boolean | no |  | pre_shot | Whether team_id is the home side (nominal at neutral venues). |
| 11 | `period` | Int8 | no | {1, 2, 3, 4, 5} | pre_shot | Period: 1-2 regulation, 3-4 extra time, 5 penalty shoot-out. |
| 12 | `minute` | Int16 | no | [0, 150] | pre_shot | Provider match-clock minute (period 2 starts at 45). |
| 13 | `second` | Int8 | no | [0, 59] | pre_shot | Second within the minute. |
| 14 | `seconds_in_period` | Float64 | no | [0, 4500] | pre_shot | Seconds since the period kicked off. |
| 15 | `x` | Float64 | no | [0, 105] | pre_shot | Shot location, metres. |
| 16 | `y` | Float64 | no | [0, 68] | pre_shot | Shot location, metres. |
| 17 | `end_x` | Float64 | yes | [0, 105] | outcome | Where the ball was stopped (save, block or line); null when not given. |
| 18 | `end_y` | Float64 | yes | [0, 68] | outcome | As end_x. |
| 19 | `end_z` | Float64 | yes | ≥ 0 | outcome | Height at the end location, metres. |
| 20 | `goal_y` | Float64 | yes | [0, 68] | post_shot | Provider's point on the goal plane (an estimate for saves); null if blocked. |
| 21 | `goal_z` | Float64 | yes | ≥ 0 | post_shot | Height on the goal plane, metres (crossbar at 2.44); null if blocked. |
| 22 | `gk_x` | Float64 | yes | [0, 105] | pre_shot | Goalkeeper defending the target goal at the strike. |
| 23 | `gk_y` | Float64 | yes | [0, 68] | pre_shot | As gk_x. |
| 24 | `distance_m` | Float64 | no | ≥ 0 | pre_shot | pitch.shot_distance(x, y), stored so nobody recomputes it differently. |
| 25 | `angle_rad` | Float64 | no | [0, π] | pre_shot | pitch.opening_angle(x, y). |
| 26 | `visible_area` | List(Array(Float64, 2)) | yes | check `polygon` | pre_shot | StatsBomb 360 camera polygon in metres; null without a 360 frame. |
| 27 | `body_part` | Enum(BODY_PARTS) | yes | right_foot, left_foot, head, other | pre_shot | Body part used. |
| 28 | `situation` | Enum(SITUATIONS) | no | open_play, fast_break, from_corner, from_free_kick, from_throw_in, direct_free_kick, direct_corner, penalty, kick_off | pre_shot | Open play, fast break, set-piece phase or direct set piece. |
| 29 | `assist_type` | Enum(ASSIST_TYPES) | yes | none, pass, cross, through_ball, cutback | pre_shot | none = recorded and unassisted; null = the provider cannot tell. |
| 30 | `technique` | Enum(TECHNIQUES) | yes | normal, volley, half_volley, lob, backheel, diving_header, overhead_kick | pre_shot | Strike technique; normal only where the full vocabulary is recorded. |
| 31 | `outcome` | Enum(OUTCOMES) | no | goal, saved, blocked, post, off_target | outcome | What happened to the shot. |
| 32 | `is_goal` | Boolean | no |  | outcome | The ball went in (own goals included). |
| 33 | `is_own_goal` | Boolean | no |  | outcome | Own goal (xG trains on the other shots). |
| 34 | `is_penalty` | Boolean | no |  | pre_shot | Penalty kick (situation == penalty). |
| 35 | `on_target` | Boolean | no |  | outcome | Outcome goal or saved, and not an own goal. |
| 36 | `first_time` | Boolean | yes |  | pre_shot | Tri-state: first-time shot. |
| 37 | `follows_dribble` | Boolean | yes |  | pre_shot | Tri-state: shot follows a dribble (StatsBomb only). |
| 38 | `open_goal` | Boolean | yes |  | pre_shot | Tri-state: open goal. |
| 39 | `one_on_one` | Boolean | yes |  | pre_shot | Tri-state: one on one. |
| 40 | `big_chance` | Boolean | yes |  | restricted | Tri-state: Opta big chance; null for StatsBomb. |
| 41 | `deflected` | Boolean | yes |  | post_shot | Tri-state: shot deflected. |
| 42 | `provider_xg` | Float64 | yes | [0, 1] | lineage | Provider xG, a benchmark only. |
| 43 | `provider_psxg` | Float64 | yes | [0, 1] | lineage | Provider post-shot xG, a benchmark only. |
| 44 | `assist_event_id` | String | yes | id | lineage | Event id of the assisting pass (Opta: resolved to the global id). |
| 45 | `raw_type` | String | no | check `non_empty` | lineage | Provider event type (encodes the outcome, so never a feature). |
| 46 | `raw_qualifiers` | String | no | check `json` | lineage | Raw provider shot object / qualifier list as JSON text. |
| 47 | `schema_version` | String | no | check `schema_version` | lineage | Schema version the row was written with. |

Cross-column rules (`SHOT_RULES`): ids match the provider's prefix; the two teams differ;
`distance_m` and `angle_rad` agree with `pitch.shot_distance` / `pitch.opening_angle` (to
1e-9); `is_goal` and `on_target` agree with `outcome` and `is_own_goal`; an own goal is a
goal; a blocked shot has no
goal-mouth point; `is_penalty` agrees with `situation`; a shoot-out kick (period 5) is a
penalty; outside shoot-outs the minute is not before the period's start and
`seconds_in_period` matches minute and second within 1 s; an unassisted shot has no
assist event.

## Freeze-frame table

One row per player visible at the shot **other than the shooter**, in the same frame as the
shot. No row order is implied.

| # | Column | Dtype | Null | Allowed / range | Role | Description |
|---|---|---|---|---|---|---|
| 1 | `shot_id` | String | no | id | identity | The shot this frame belongs to (not unique here). |
| 2 | `player_id` | String | yes | id | identity | Visible player; null for id-less sources. |
| 3 | `is_teammate` | Boolean | no |  | pre_shot | Teammate of the shooter. |
| 4 | `is_keeper` | Boolean | no |  | pre_shot | Goalkeeper. |
| 5 | `x` | Float64 | no | [0, 105] | pre_shot | Player location, metres. |
| 6 | `y` | Float64 | no | [0, 68] | pre_shot | Player location, metres. |

Rules (`FREEZE_FRAME_RULES`): `player_id` has the same prefix as `shot_id`; a player appears
once per shot; at most one keeper per side per shot. With `validate_freeze_frames(ff,
shots=shots)`, every `shot_id` must exist in `shots` and the shooter must not be in their own
frame.

## Matches table

One row per match, provider-neutral, so StatsBomb (C4) and WhoScored (C5) fill the same
table. Nothing in it is ever filled with a default. `gender` is the one `pre_shot` column:
the shot models join it onto the shots by `match_id`
(`column_names("matches", roles="pre_shot")`).

| # | Column | Dtype | Null | Allowed / range | Role | Description |
|---|---|---|---|---|---|---|
| 1 | `match_id` | String | no | id, unique | identity | Provider match id, the primary key. |
| 2 | `provider` | Enum(PROVIDERS) | no | statsbomb, whoscored | identity | Data source. |
| 3 | `competition_id` | String | no | id | identity | Competition id. |
| 4 | `season_id` | String | no | id | identity | Season id. |
| 5 | `match_date` | Date | no |  | identity | Match date, as in the shots table. |
| 6 | `home_team_id` | String | no | id | identity | Home side as the provider lists it (nominal at neutral venues); shots.is_home refers to it. |
| 7 | `away_team_id` | String | no | id | identity | The other team. |
| 8 | `competition_name` | String | no | check `non_empty` | identity | Competition name, a label for cards. |
| 9 | `season_name` | String | no | check `non_empty` | identity | Season as football writes it ('2015/2016', '2026'), a label for cards. |
| 10 | `gender` | Enum(GENDERS) | no | male, female | pre_shot | Men's or women's football, known before kick-off. |
| 11 | `home_score` | Int16 | yes | [0, 99] | outcome | Recorded final score without the shoot-out; null when the provider gives none. |
| 12 | `away_score` | Int16 | yes | [0, 99] | outcome | As home_score. |
| 13 | `has_360` | Boolean | yes |  | lineage | A usable StatsBomb 360 file; null for every other provider. |
| 14 | `schema_version` | String | no | check `schema_version` | lineage | Schema version the row was written with. |

Columns 1-5 and 14 have the same name, dtype and role as in `shots`; no `pre_shot` matches
column shares a name with a shots column.

Rules (`MATCH_RULES`): the match, competition, season and team ids start with the provider's
prefix; the two teams differ; `home_score` and `away_score` are both set or both null;
`has_360` is set for StatsBomb rows and null for every other provider. With
`validate_shots(shots, matches=matches)`, every shot's `match_id` must exist in `matches`
(`match_exists`), and its competition, season, date and team pair (home / away read from
`is_home`) must equal that match's row (`shot_agrees_with_match`).

## Validation and errors

`validate_shots` / `validate_freeze_frames` / `validate_matches` never cast. Pass 1 checks structure (missing and
extra columns, dtypes, and the invalid values of a mis-typed enum column); pass 2 checks
nulls, ranges, finiteness, uniqueness, id / JSON / polygon / version formats, and then the
rules. Pass 2 runs only when pass 1 finds nothing. Both passes pin pandera's config, so
`PANDERA_VALIDATION_ENABLED` cannot switch them off. One `SchemaValidationError` lists every
problem, naming the column and the bad values:

```text
shots: 2 schema problem(s) (schema 1.1)
  - column 'shot_id': duplicate value(s) 'sb:a1' (rows 0, 1)
  - rule 'is_goal_matches_outcome' (shot_id, is_goal, outcome): is_goal must equal outcome == 'goal'; failed for shot_id='sb:a2', is_goal=False, outcome='goal' (rows 2)
```

!!! warning "Build frames column-wise or with `cast_to_schema`"
    `pl.DataFrame(list_of_row_dicts, schema=SHOT_SCHEMA)` silently turns an invalid enum
    value into null and truncates `1.5` to `1` in an integer column. Build frames column by
    column with strict dtypes, or from strings with `cast_to_schema`, which casts along safe
    paths only and names every value it cannot cast.

## Versioning

`SCHEMA_VERSION = "1.1"` (`MAJOR.MINOR`), written on every row and checked exactly. A MINOR
bump adds a nullable column or a rule that every existing snapshot satisfies. A MAJOR bump
removes or renames a column, changes a dtype, changes **any** enum's categories, changes a
column's role or tightens a range.

Adding a canonical table is a MINOR bump. Snapshots that hold a table of that name without a
schema (`schema: null`) are refused by the manifest reader and must be rebuilt, never
migrated: a 1.0 `matches` table (String `gender`, `schema: null`) is rebuilt from the raw
data. 1.1 added the matches table; no existing enum changed.

::: football_core.schema
