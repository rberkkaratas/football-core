"""The facing goalkeeper on WhoScored shots (spec ``runs/MS5/goalmouth`` §3.2, §10.6).

Lineup state by match clock: rows filtered to ``FormationSet``, ``FormationChange``,
``SubstitutionOn`` and the shot types, sorted by ``(period rank, minute, second, id)``.
``FormationSet`` / ``FormationChange`` pair q30 with q131 (keeper = slot "1");
``SubstitutionOn`` with q145 "1" makes its player the keeper. A shot gets the other team's
keeper, an own goal (q28) null. Every id and name is invented (``whoscored_cases``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import whoscored_cases as wsc
from whoscored_cases import Ev, q

from football_core import schema
from football_core.io import whoscored

STEM = f"16082025_{wsc.MAIN}"  # the MAIN event file
FS_A, FS_B = wsc._M + 1, wsc._M + 2  # MAIN's FormationSet rows (team A, team B), file order

# The KEEPERS match (Championship 2025-26, names route): R acts first and is home.
R, S = 9701, 9702
KEEPERS = 7000601
_K = 2_907_000_000
K01, K02, K03, K04, K05, OG_ET, K06, K07, K08 = (_K + i for i in range(101, 110))
SUB_S1, SUB_S10, SUB_S_ET, OFF_S, CHANGE_R = (_K + i for i in range(201, 206))


def _shot(gid: int, eid: int, team: int, period: str, minute: int, second: int = 0) -> Ev:
    quals = [q(9), q(214), q(20)] if period == "PenaltyShootout" else [q(22), q(20)]
    return Ev(gid, eid, team, "MissedShots", period, minute, second, "85.0", "50.0",
              team * 100 + 9, quals=quals, mouth=("60.0", "15.0"))  # fmt: skip


def _sub_on(gid: int, eid: int, team: int, period: str, minute: int, player: int, slot: str) -> Ev:
    return Ev(gid, eid, team, "SubstitutionOn", period, minute, 0, None, None, player,
              quals=[q(145, slot)])  # fmt: skip


def _change_quals() -> list[dict[str, Any]]:
    """R's FormationChange after its keeper is sent off: outfielder 970105 to slot 1."""
    players = [R * 100 + k for k in range(2, 12)]  # 10 players; 970105 is the 4th
    slots = ["2", "3", "4", "1", "6", "7", "8", "9", "10", "11"]
    return [q(30, ", ".join(map(str, players))), q(131, ", ".join(slots))]


def keepers_match() -> wsc.Match:
    first = [
        _shot(K01, 20, R, "FirstHalf", 10),
        _shot(K02, 20, S, "FirstHalf", 20),
        Ev(OFF_S, 21, S, "SubstitutionOff", "FirstHalf", 30, None, None, None, S * 10 + 1),
        _sub_on(SUB_S1, 22, S, "FirstHalf", 30, S * 100 + 50, "1"),  # keeper change
        _sub_on(SUB_S10, 23, S, "FirstHalf", 31, S * 100 + 51, "10"),  # near miss: slot 10
        _shot(K03, 21, R, "FirstHalf", 35),
    ]
    second = [_shot(K04, 24, S, "SecondHalf", 55), _shot(K05, 25, S, "SecondHalf", 70)]
    extra = [
        Ev(OG_ET, 22, R, "Goal", "FirstPeriodOfExtraTime", 100, 0, "4.0", "52.0", R * 100 + 7,
           quals=[q(22), q(72), q(28)], mouth=("50.0", "5.0")),
        _shot(K06, 26, S, "FirstPeriodOfExtraTime", 104),
        _sub_on(SUB_S_ET, 27, S, "SecondPeriodOfExtraTime", 118, S * 100 + 60, " 1"),
    ]  # fmt: skip
    shootout = [
        _shot(K07, 23, R, "PenaltyShootout", 121, 0),
        _shot(K08, 28, S, "PenaltyShootout", 121, 30),
    ]
    events = wsc._frame(_K, R, S, first, second, extra_time=extra, shootout=shootout,
                        postgame_minute=122)  # fmt: skip
    # Listed after PostGame (the archive puts formation rows at the end) but applied by clock.
    events.append(Ev(CHANGE_R, 24, R, "FormationChange", "SecondHalf", 60, 0, None, None,
                     quals=_change_quals()))  # fmt: skip
    return wsc.Match("keepers", KEEPERS, "Championship", "2025-2026", "11102025", R, S,
                     {R: "Romeo Town", S: "Sierra City"}, events,
                     second_suffix=".0")  # fmt: skip


def _gk(archive: Any) -> dict[str, str | None]:
    shots = archive.shots
    return dict(zip(shots["shot_id"].to_list(), shots["gk_id"].to_list(), strict=True))


def test_keeper_states(tmp_path: Path) -> None:
    m = keepers_match()
    # Preconditions: the FormationChange is the file's last row, after PostGame; FormationSet
    # seconds are written "0.0"; the SubstitutionOff (not a lineup row) has an empty second.
    assert m.events[-1].gid == CHANGE_R and m.events[-2].period == "PostGame"
    assert m.events[0].type == "FormationSet" and m.events[0].raw["second"] == "0.0"
    assert m.event(OFF_S).second is None
    loaded = wsc.load_fixture(tmp_path, wsc.Fixture([m]))
    assert _gk(loaded.archive) == {
        f"ws:{K01}": "ws:97021",
        f"ws:{K02}": "ws:97011",
        f"ws:{K03}": "ws:970250",  # after the slot-1 SubstitutionOn (slot "10" changes nothing)
        f"ws:{K04}": "ws:97011",  # before the FormationChange at minute 60
        f"ws:{K05}": "ws:970105",  # after it, although it is listed last in the file
        f"ws:{OG_ET}": None,  # own goal in extra time
        f"ws:{K06}": "ws:970105",
        f"ws:{K07}": "ws:970260",  # shoot-out kick after the keeper sub in extra time
        f"ws:{K08}": "ws:970105",
    }


def test_standard_keepers(ws_standard: Any) -> None:
    """Every shot of the standard archive faces the other team's FormationSet keeper; the own
    goal (OG match) is null; shoot-out kicks are covered."""
    shots = ws_standard.archive.shots
    assert shots.schema["gk_id"] == pl.String()
    assert shots.filter(pl.col("period") == 5).height == 2  # precondition: SO_GOAL, SO_MISS
    for r in shots.iter_rows(named=True):
        keeper = f"ws:{wsc.keeper_of(int(r['opponent_team_id'][3:]))}"
        assert r["gk_id"] == (None if r["is_own_goal"] else keeper), r["shot_id"]
    assert shots.filter(pl.col("is_own_goal"))["shot_id"].to_list() == [f"ws:{wsc.OWN_GOAL}"]


# --- Refusals (MAIN match of the standard archive) -------------------------------------------


def _value_error(
    fx: wsc.Fixture, tmp_path: Path, *needles: str, any_of: tuple[str, ...] = ()
) -> None:
    wsc.write_archive(tmp_path, fx)
    with pytest.raises(ValueError) as info:
        whoscored.load(tmp_path)
    msg = str(info.value)
    assert not isinstance(info.value, schema.SchemaValidationError), msg
    for needle in needles:
        assert needle in msg, (needle, msg)
    if any_of:
        assert any(n in msg for n in any_of), (any_of, msg)


def _formation(fx: wsc.Fixture, gid: int) -> Ev:
    return fx.match("main").event(gid)


def _slots(*slots: str) -> Callable[[wsc.Fixture], None]:
    def mutate(fx: wsc.Fixture) -> None:
        ev = _formation(fx, FS_B)
        ev.quals = [ev.quals[0], q(131, ",".join(slots))]

    return mutate


@pytest.mark.parametrize(
    "slots",
    [
        ("2", "3", "12", *(str(k) for k in range(4, 12)), "0", "0"),
        ("2", "3", "1", "1", *(str(k) for k in range(5, 12)), "0", "0"),
    ],
    ids=["no_slot_1", "two_slot_1"],
)
def test_slot_one_count_raises(tmp_path: Path, slots: tuple[str, ...]) -> None:
    fx = wsc.standard()
    assert len(slots) == 13  # precondition: same length as q30, so only the count is wrong
    _slots(*slots)(fx)
    _value_error(fx, tmp_path, STEM, str(FS_B))


def test_q30_q131_length_raises(tmp_path: Path) -> None:
    fx = wsc.standard()
    _slots("2", "3", "1", *(str(k) for k in range(4, 12)), "0")(fx)  # 12 slots for 13 players
    _value_error(fx, tmp_path, STEM, str(FS_B))


def test_no_formation_raises(tmp_path: Path) -> None:
    """No FormationSet for B: A's first shot (BLOCKED, eventId 30) has no keeper to face."""
    fx = wsc.standard()
    main = fx.match("main")
    main.events = [e for e in main.events if e.gid != FS_B]
    _value_error(fx, tmp_path, STEM, any_of=(str(wsc.BLOCKED), "30"))


@pytest.mark.parametrize(
    ("gid", "column", "text"),
    [
        (FS_A, "id", "2901000001x"),
        (FS_A, "minute", "0.5"),
        (FS_B, "second", "zero"),
        (FS_B, "second", ""),
    ],
    ids=["id", "minute", "second", "second_empty"],
)
def test_lineup_bad_int_text_raises(tmp_path: Path, gid: int, column: str, text: str) -> None:
    fx = wsc.standard()
    _formation(fx, gid).raw[column] = text
    _value_error(fx, tmp_path, STEM, *([text] if text else []))


def test_own_goal_before_any_lineup_is_null(tmp_path: Path) -> None:
    """Build ratify 4: an own goal is decided before the keeper is looked up, so an own goal
    earlier on the clock than every lineup row loads with a null ``gk_id`` (no raise)."""
    fx = wsc.standard()
    og = fx.match("og")
    for ev in og.events:
        if ev.type == "FormationSet":
            ev.period, ev.minute = "FirstHalf", 31  # after the own goal at 30:12
    assert og.event(wsc.OWN_GOAL).minute == 30  # precondition
    gk = _gk(wsc.load_fixture(tmp_path, fx).archive)
    assert gk[f"ws:{wsc.OWN_GOAL}"] is None
    assert gk[f"ws:{wsc.H_GOAL}"] == f"ws:{wsc.keeper_of(og.away)}"


@pytest.mark.parametrize("period", ["", "HalfTime"], ids=["null", "unknown"])
def test_lineup_bad_period_raises(tmp_path: Path, period: str) -> None:
    """Build ratify 5: a lineup row with a null or unknown period raises (file, event,
    period) instead of sorting first on the clock."""
    fx = wsc.standard()
    _formation(fx, FS_B).raw["period"] = period
    _value_error(fx, tmp_path, STEM, str(FS_B))
