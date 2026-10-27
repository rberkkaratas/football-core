"""The facing goalkeeper on StatsBomb shots (spec ``runs/MS5/goalmouth`` §3.1, §10.6).

``gk_id`` comes from lineup state walked in event-file (``index``) order: ``Starting XI`` and
``Tactical Shift`` set a team's keeper (the one lineup entry with ``position.id == 1``); a
``Substitution`` of the current keeper hands the role to the replacement; a shot gets the other
team's keeper; an ``Own Goal Against`` gets null. A freeze-frame keeper (non-teammate) must agree.
All ids are the fixture's invented ones (``statsbomb_cases``: keepers 1001 / 2001).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import statsbomb_cases as sbc

from football_core import schema
from football_core.io import statsbomb

A = sbc.A  # one_match's default match id
EVENTS = f"data/events/{A}.json"


def _keeper(pid: int) -> dict[str, Any]:
    return sbc.ff_player(pid, (119.05, 40.05), teammate=False, keeper=True)


def _substitution(eid: str, team_id: int, off: int, on: int, period: int, minute: int) -> Any:
    return sbc.event(
        eid, "Substitution", team_id=team_id, period=period, minute=minute, player_id=off,
        substitution={"outcome": {"id": 103, "name": "Injury"}, "replacement": sbc.player(on)},
    )  # fmt: skip


def _tactical_shift(
    eid: str, team_id: int, lineup: Sequence[tuple[int, int]], period: int, minute: int
) -> Any:
    entries = [
        {
            "player": sbc.player(pid),
            "position": {"id": pos, "name": "Goalkeeper" if pos == 1 else "Center Forward"},
            "jersey_number": pid % 100,
        }
        for pid, pos in lineup
    ]
    return sbc.event(eid, "Tactical Shift", team_id=team_id, period=period, minute=minute,
                     tactics={"formation": 4411, "lineup": entries})  # fmt: skip


def _load(tmp_path: Path, match: sbc.Match, edit: Callable[[Any], Any] | None = None) -> Any:
    files = sbc.repo_files([match])
    if edit is not None:
        sbc.edit_json(files, EVENTS, edit)
    return statsbomb.load(sbc.write_cache(tmp_path / "cache", files))


def _value_error(call: Callable[[], object], *needles: str) -> None:
    with pytest.raises(ValueError) as info:
        call()
    assert not isinstance(info.value, schema.SchemaValidationError), info.value
    for needle in needles:
        assert needle in str(info.value), (needle, str(info.value))


def _gk(shots: pl.DataFrame) -> dict[str, str | None]:
    return dict(zip(shots["shot_id"].to_list(), shots["gk_id"].to_list(), strict=True))


# --- States -----------------------------------------------------------------------------------

# Away lineup after the keeper (2012, on for 2001) is sent off: outfielder 2005 goes in goal.
SHIFT_LINEUP = [(2002, 23), (2003, 23), (2004, 23), (2005, 1), (2006, 23), (2007, 23),
                (2008, 23), (2009, 23), (2010, 23), (2011, 23)]  # fmt: skip


def keeper_match() -> sbc.Match:
    H, W = sbc.HOME, sbc.AWAY
    pen = {"shot_type": "Penalty", "location": (108.05, 40.05), "period": 5}
    # fmt: off
    events = [
        sbc.shot("fake-shot-k01", minute=10),                                     # vs 2001
        sbc.shot("fake-shot-k02", team_id=W, freeze_frame=sbc.default_frame(W),
                 location=(101.35, 42.65), minute=20),                            # vs 1001
        _substitution("fake-sub-k01", H, off=1006, on=1012, period=1, minute=25),  # not a keeper
        _substitution("fake-sub-k02", W, off=2001, on=2012, period=1, minute=30),  # keeper change
        sbc.shot("fake-shot-k03", freeze_frame=[_keeper(2012)], minute=35),       # vs 2012
        sbc.own_goal_against("fake-oga-k01", team_id=W, player_id=2004, location=(5.35, 30.25),
                             period=1, minute=40),                                # null
        sbc.event("fake-card-k01", "Bad Behaviour", team_id=W, period=2, minute=60,
                  player_id=2012, bad_behaviour={"card": {"id": 5, "name": "Red Card"}}),
        _tactical_shift("fake-ts-k01", W, SHIFT_LINEUP, period=2, minute=61),
        sbc.shot("fake-shot-k04", freeze_frame=[_keeper(2005)], period=2, minute=62),  # vs 2005
        _substitution("fake-sub-k03", H, off=1001, on=1013, period=3, minute=100),
        sbc.shot("fake-shot-k05", team_id=W, freeze_frame=None, location=(101.35, 42.65),
                 period=3, minute=101),                                           # vs 1013
        sbc.shot("fake-shot-k06", team_id=W, outcome="Goal", freeze_frame=[_keeper(1013)],
                 minute=120, second=30, **pen),                                   # vs 1013
        sbc.shot("fake-shot-k07", freeze_frame=[_keeper(2005)], minute=121, **pen),  # vs 2005
    ]
    # fmt: on
    return sbc.one_match(*events)


def test_keeper_states(tmp_path: Path) -> None:
    shots = _load(tmp_path, keeper_match()).shots
    assert _gk(shots) == {
        "sb:fake-shot-k01": "sb:2001",
        "sb:fake-shot-k02": "sb:1001",
        "sb:fake-shot-k03": "sb:2012",  # after the keeper substitution
        "sb:fake-oga-k01": None,
        "sb:fake-shot-k04": "sb:2005",  # an outfielder in goal after the red card
        "sb:fake-shot-k05": "sb:1013",  # extra-time keeper substitution
        "sb:fake-shot-k06": "sb:1013",  # shoot-out kick after it
        "sb:fake-shot-k07": "sb:2005",
    }
    assert shots.schema["gk_id"] == pl.String()


def test_own_goal_null(sb_standard: Any) -> None:
    """§3.1 / §4: every shot of the standard fixture faces the other team's lineup keeper; own
    goals (one per match) are null."""
    shots = sb_standard.shots
    assert shots["is_own_goal"].sum() == 2  # precondition: OWN_GOAL_A and OWN_GOAL_B
    keeper = {"sb:101": "sb:2001", "sb:202": "sb:1001"}
    for r in shots.iter_rows(named=True):
        expected = None if r["is_own_goal"] else keeper[r["team_id"]]
        assert r["gk_id"] == expected, r["shot_id"]
    assert sbc.row(shots, sbc.SHOOTOUT_SAVE)["gk_id"] == "sb:1001"  # period 5, no frame


# --- Refusals ---------------------------------------------------------------------------------


def _set_positions(eid: str, keepers: Sequence[int]) -> Callable[[Any], None]:
    """Edit the lineup of event ``eid`` so exactly the players ``keepers`` have position 1."""

    def edit(events: list[Any]) -> None:
        (ev,) = [e for e in events if e["id"] == eid]
        for entry in ev["tactics"]["lineup"]:
            pid = entry["player"]["id"]
            entry["position"] = {"id": 1 if pid in keepers else 23, "name": "Fixture"}

    return edit


@pytest.mark.parametrize(
    ("event_id", "keepers"),
    [
        (f"fake-xi-{A}-a", ()),
        (f"fake-xi-{A}-a", (2001, 2002)),
        ("fake-ts-k01", ()),
        ("fake-ts-k01", (2005, 2006)),
    ],
    ids=["xi_none", "xi_two", "shift_none", "shift_two"],
)
def test_lineup_keeper_count_raises(
    tmp_path: Path, event_id: str, keepers: tuple[int, ...]
) -> None:
    edit = _set_positions(event_id, keepers)
    _value_error(lambda: _load(tmp_path, keeper_match(), edit), str(A), event_id)


def test_frame_keeper_disagrees_raises(tmp_path: Path) -> None:
    match = sbc.one_match(sbc.shot("fake-shot-f01", freeze_frame=[_keeper(2009)]))
    _value_error(lambda: _load(tmp_path, match), str(A), "fake-shot-f01", "2001", "2009")


def test_shot_before_lineup_raises(tmp_path: Path) -> None:
    """No keeper yet for the defending team -> ValueError (match, shot); the shooter's own
    lineup is not needed (near miss: only the home Starting XI removed)."""
    match = sbc.one_match(sbc.shot("fake-shot-b01", minute=10))

    def drop(xi: str) -> Callable[[Any], Any]:
        return lambda events: [e for e in events if e["id"] != xi]

    _value_error(lambda: _load(tmp_path / "a", match, drop(f"fake-xi-{A}-a")),
                 str(A), "fake-shot-b01")  # fmt: skip
    shots = _load(tmp_path / "h", match, drop(f"fake-xi-{A}-h")).shots
    assert _gk(shots) == {"sb:fake-shot-b01": "sb:2001"}


@pytest.mark.parametrize("step", [0, -1], ids=["repeated_index", "decreasing_index"])
def test_events_not_in_index_order_raises(tmp_path: Path, step: int) -> None:
    match = sbc.one_match(
        sbc.shot("fake-shot-i01", minute=10), sbc.shot("fake-shot-i02", minute=20)
    )

    def edit(events: list[Any]) -> None:
        j = next(i for i, e in enumerate(events) if e["id"] == "fake-shot-i02")
        events[j]["index"] = events[j - 1]["index"] + step

    _value_error(lambda: _load(tmp_path, match, edit), str(A), "fake-shot-i02")


def test_penalty_frame_keeper_not_checked(tmp_path: Path) -> None:
    """Build ratify 7: penalty freeze frames are not read, so a penalty whose frame names
    another keeper loads with the lineup keeper (near miss: the same frame on an open-play
    shot raises, ``test_frame_keeper_disagrees_raises``)."""
    pen = sbc.shot("fake-shot-p01", shot_type="Penalty", location=(108.05, 40.05),
                   freeze_frame=[_keeper(2009)], minute=10)  # fmt: skip
    shots = _load(tmp_path, sbc.one_match(pen)).shots
    assert sbc.row(shots, "fake-shot-p01")["is_penalty"]  # precondition
    assert _gk(shots) == {"sb:fake-shot-p01": "sb:2001"}


def test_every_frame_keeper_checked(tmp_path: Path) -> None:
    """Build ratify 7: every non-teammate keeper in a frame must be the lineup keeper, not
    only the first one listed."""
    match = sbc.one_match(
        sbc.shot("fake-shot-f02", freeze_frame=[_keeper(2001), _keeper(2009)], minute=10)
    )
    _value_error(lambda: _load(tmp_path, match), str(A), "fake-shot-f02", "2009")
