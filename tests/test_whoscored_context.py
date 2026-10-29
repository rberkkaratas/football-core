"""``previous_play`` and ``shot_context``: the pre-shot context columns ``prev_action`` and
``rebound`` (spec ``runs/PR1/xg-context`` §2.2, §4).

Events are hand-written one-match frames in file order, every cell a string as the archive
reads it (ids normalised, ``second`` as ``"12"`` or ``"12.0"``, qualifiers as a Python literal).
Ids, teams and clocks are invented. Expected levels come from the spec's tables, never from the
module under test. The loader tests read the standard archive of ``whoscored_cases``.
"""

from __future__ import annotations

import re
from typing import Any

import hypothesis.strategies as st
import polars as pl
import pytest
import whoscored_cases as wsc
from hypothesis import given, settings

from football_core.io import whoscored

A, B = "9101", "9102"  # shooting team, opponent
NAMES = {94: "Def block", 170: "LeadingToGoal", 169: "LeadingToAttempt", 233: "OppositeRelated",
         22: "RegularPlay", 20: "RightFoot", 212: "Length"}  # fmt: skip
COLUMNS = ("id", "eventId", "teamId", "period", "minute", "second", "type", "qualifiers")

# spec §2.2, typed by hand
SPEC_SKIP = frozenset({
    "Start", "End", "FormationSet", "FormationChange", "SubstitutionOn", "SubstitutionOff",
    "Card", "CornerAwarded", "OffsideGiven", "OffsideProvoked", "Error", "ChanceMissed",
    "GoodSkill", "PenaltyFaced", "CrossNotClaimed", "ShieldBallOpp", "KeeperSweeper",
    "Challenge", "Aerial", "Foul", "Dispossessed",
})  # fmt: skip
SHOT = "MissedShots"


def q(qid: int, value: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": {"value": qid, "displayName": NAMES.get(qid, f"Q{qid}")}}
    if value is not None:
        out["value"] = value
    return out


def ev(
    gid: int, eid: int, team: str, type_: str, minute: int | None, second: int | str | None,
    *quals: dict[str, Any], period: str = "FirstHalf",
) -> dict[str, str | None]:  # fmt: skip
    """One event row; a shot gets a pattern and a body part, as the archive writes it."""
    if type_ in ("Goal", "SavedShot", "MissedShots", "ShotOnPost") and not quals:
        quals = (q(22), q(20))
    return {
        "id": str(gid),
        "eventId": str(eid),
        "teamId": team,
        "period": period,
        "minute": None if minute is None else str(minute),
        "second": None if second is None else str(second),
        "type": type_,
        "qualifiers": repr(list(quals)),
    }


def frame(*rows: dict[str, str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {c: [r[c] for r in rows] for c in COLUMNS}, schema=dict.fromkeys(COLUMNS, pl.String())
    )


def start(period: str = "FirstHalf", minute: int = 0, base: int = 1) -> list[dict[str, Any]]:
    """The period's Start pair (skip type, null second as in the archive)."""
    return [ev(base, 1, A, "Start", minute, None, period=period),
            ev(base + 1, 1, B, "Start", minute, None, period=period)]  # fmt: skip


def context(*rows: dict[str, Any]) -> dict[str, tuple[str, bool]]:
    return whoscored.shot_context(frame(*rows), "'fixture.csv'")


# --- Constants ---------------------------------------------------------------------------------


def test_constants() -> None:
    """§2.2: the skip set, the q-ids, the save types of the q233 guard and the 3 s window."""
    assert frozenset(whoscored.PLAY_SKIP) == SPEC_SKIP
    assert (whoscored.OUTFIELD_BLOCK, whoscored.OPPOSITE_RELATED) == (94, 233)
    assert frozenset({"Save", "Punch", "Claim", "Smother"}) == whoscored.SAVE_TYPES
    assert whoscored.REBOUND_S == 3
    assert whoscored.EVENT_COLUMNS == wsc.REQUIRED  # unchanged (§2.2)


# --- prev_action levels ------------------------------------------------------------------------

LEVELS: dict[str, tuple[str, str, tuple[dict[str, Any], ...], str]] = {
    # case: (team of the previous event, its type, its qualifiers, expected level)
    "pass": (A, "Pass", (q(212, "10.0"),), "pass"),
    "take_on": (A, "TakeOn", (), "take_on"),
    "ball_recovery": (A, "BallRecovery", (), "regain"),
    "interception": (A, "Interception", (), "regain"),
    "tackle": (A, "Tackle", (), "regain"),
    "blocked_pass": (A, "BlockedPass", (), "regain"),
    "own_saved_shot": (A, "SavedShot", (q(22), q(20)), "shot"),
    "own_missed_shot": (A, "MissedShots", (q(22), q(20)), "shot"),
    "own_post": (A, "ShotOnPost", (q(22), q(20)), "shot"),
    "own_goal_type": (A, "Goal", (q(22), q(20)), "shot"),
    "own_touch": (A, "BallTouch", (), "other"),
    "own_keeper_save": (A, "Save", (), "other"),
    "opp_clearance": (B, "Clearance", (), "opp_clearance"),
    "opp_save_q94": (B, "Save", (q(94),), "opp_block"),
    "opp_save": (B, "Save", (), "opp_keeper"),
    "opp_save_other_q": (B, "Save", (q(212, "1.0"),), "opp_keeper"),  # near miss on q94
    "opp_punch": (B, "Punch", (), "opp_keeper"),
    "opp_claim": (B, "Claim", (), "opp_keeper"),
    "opp_smother": (B, "Smother", (), "opp_keeper"),
    "opp_pickup": (B, "KeeperPickup", (), "opp_keeper"),
    "opp_pass": (B, "Pass", (), "opp_other"),
    "opp_tackle": (B, "Tackle", (), "opp_other"),
    "opp_shot": (B, "SavedShot", (q(22), q(20)), "opp_other"),
    "opp_unknown_type": (B, "SomethingNew", (), "opp_other"),
}


@pytest.mark.parametrize("case", sorted(LEVELS))
def test_prev_action_levels(case: str) -> None:
    """§2.2 level table: same team via SAME_ACTION (else "other"); an opponent Save is
    ``opp_block`` with q94 and ``opp_keeper`` without; other opponent types via OPP_ACTION
    (else ``opp_other``). The previous event is 10 s earlier, so ``rebound`` is False."""
    team, kind, quals, level = LEVELS[case]
    rows = [*start(), ev(11, 5, A, "Pass", 9, 30), ev(12, 6, team, kind, 10, 0, *quals),
            ev(13, 7, A, SHOT, 10, 10)]  # fmt: skip
    out = context(*rows)
    assert out["13"] == (level, False)
    shots = {r["id"] for r in rows if r["type"] in ("Goal", "SavedShot", "MissedShots",
                                                     "ShotOnPost")}  # fmt: skip
    assert set(out) == shots  # every SHOT_TYPES row, nothing else


def test_no_previous_play_is_other() -> None:
    """§2.2: the first play row of its period has no previous play: "other" (the Start pair is
    skipped; a play row of the period before does not count)."""
    rows = [*start(), ev(11, 5, A, "Pass", 44, 0), *start("SecondHalf", 45, base=20),
            ev(30, 8, A, SHOT, 45, 3, period="SecondHalf")]  # fmt: skip
    assert context(*rows)["30"] == ("other", False)


# --- rebound -----------------------------------------------------------------------------------


@pytest.mark.parametrize(("gap", "rebound"), [(0, True), (3, True), (4, False)])
def test_rebound_window(gap: int, rebound: bool) -> None:
    """§2.2: the team's previous shot in the period at most ``REBOUND_S`` = 3 s earlier on the
    clock (inclusive; 4 s is out). The keeper's save in between makes ``prev_action``
    ``opp_keeper`` either way; the clock crosses a minute boundary."""
    first = (20, 58)
    t = first[0] * 60 + first[1] + gap
    rows = [*start(), ev(11, 5, A, SHOT, *first), ev(12, 9, B, "Save", *first),
            ev(13, 6, A, SHOT, t // 60, t % 60)]  # fmt: skip
    out = context(*rows)
    assert out["13"] == ("opp_keeper", rebound)
    assert out["11"] == ("other", False)


def test_rebound_reads_the_latest_own_shot_only() -> None:
    """Near misses: the opponent's shot 1 s before is not the team's (A's own previous shot is
    4 s back: False); with the team's shots 5 s and 2 s back, the latest decides (True)."""
    rows = [*start(), ev(11, 5, A, SHOT, 30, 0), ev(12, 8, B, SHOT, 30, 3),
            ev(13, 6, A, SHOT, 30, 4)]  # fmt: skip
    out = context(*rows)
    assert out["12"] == ("opp_other", False)  # B's first shot; A's shot is not B's
    assert out["13"] == ("opp_other", False)  # A's previous shot 4 s before
    rows = [*start(), ev(11, 5, A, SHOT, 30, 0), ev(12, 6, A, SHOT, 30, 3),
            ev(13, 7, A, SHOT, 30, 5)]  # fmt: skip
    out = context(*rows)
    assert out["13"] == ("shot", True)  # 2 s after the latest, 5 s after the older one


def test_rebound_other_period() -> None:
    """§2.2 "in the period": a shot late in the first half does not make the first shot of the
    second half a rebound, although the clocks are 2 s apart."""
    rows = [
        *start(),
        ev(11, 5, A, SHOT, 45, 59),
        ev(12, 6, A, "End", 46, None),
        *start("SecondHalf", 45, base=20),
        ev(30, 7, A, SHOT, 46, 1, period="SecondHalf"),
    ]
    assert context(*rows)["30"] == ("other", False)


# --- leakage guards (§4) -----------------------------------------------------------------------


def test_error_with_q170_is_skipped() -> None:
    """§2.2 / §4: an opponent ``Error`` carrying q170 LeadingToGoal (the goal's eventId) just
    before the goal is skipped: the previous play is the team's pass (mutant: drop "Error"
    from PLAY_SKIP → ``opp_other``)."""
    rows = [*start(), ev(11, 5, A, "Pass", 50, 0), ev(12, 9, B, "Error", 50, 1, q(170, "6")),
            ev(13, 6, A, "Goal", 50, 2)]  # fmt: skip
    assert context(*rows)["13"] == ("pass", False)
    prev = whoscored.previous_play(frame(*rows), "'f'")
    assert prev["13"] == "11"
    assert prev["12"] == "11"  # a skipped row still has a previous play row


def test_chance_missed_is_skipped() -> None:
    """§2.2: ``ChanceMissed`` (an annotator's grading label) is skipped."""
    rows = [*start(), ev(11, 5, A, "TakeOn", 60, 0), ev(12, 6, A, "ChanceMissed", 60, 1),
            ev(13, 7, A, SHOT, 60, 2)]  # fmt: skip
    assert context(*rows)["13"] == ("take_on", False)


@pytest.mark.parametrize("kind", sorted(SPEC_SKIP))
def test_skip_types(kind: str) -> None:
    """§2.2: every ``PLAY_SKIP`` type between a pass and the shot is skipped (on a set piece the
    previous play is the play before the stoppage)."""
    rows = [*start(), ev(11, 5, A, "Pass", 70, 0), ev(12, 9, B, kind, 70, 2),
            ev(13, 6, A, SHOT, 70, 30)]  # fmt: skip
    assert context(*rows)["13"] == ("pass", False)


def test_save_of_this_shot_after_it_is_ignored() -> None:
    """§4: the keeper's Save of this shot, on the same second and *after* it in file order, does
    not change ``prev_action`` (it is not earlier) and does not trip the q233 guard."""
    rows = [*start(), ev(11, 5, A, "Pass", 75, 0), ev(12, 6, A, "SavedShot", 75, 4),
            ev(13, 9, B, "Save", 75, 4, q(233, "6"))]  # fmt: skip
    assert context(*rows)["12"] == ("pass", False)


@pytest.mark.parametrize("kind", ["Save", "Punch", "Claim", "Smother"])
def test_q233_guard_raises(kind: str) -> None:
    """§2.2: a previous opponent save-type row whose q233 names this shot's eventId means a save
    of this very shot sits before it → ValueError naming ``where``. Near miss: q233 naming
    another eventId passes."""
    rows = [*start(), ev(11, 5, A, "Pass", 80, 0), ev(12, 9, B, kind, 80, 3, q(233, "6")),
            ev(13, 6, A, SHOT, 80, 3)]  # fmt: skip
    with pytest.raises(ValueError, match="fixture.csv"):
        context(*rows)
    rows[-1] = ev(13, 7, A, SHOT, 80, 3)  # q233 names eventId 6, the shot is 7
    assert context(*rows)["13"][0] == "opp_keeper"


# --- fail-closed clock checks ------------------------------------------------------------------


def test_clock_inversion_raises() -> None:
    """§2.2: a play row earlier in file order and later on the clock than a later play row of
    its period → ValueError naming ``where``, the event id and both clocks. Near misses: an
    inverted skip row, equal clocks, and an inversion across periods pass."""
    bad = [*start(), ev(11, 5, A, "Pass", 10, 15), ev(12, 6, A, SHOT, 10, 14)]
    with pytest.raises(ValueError, match="fixture.csv") as err:
        context(*bad)
    msg = str(err.value)
    assert re.search(r"\b1[12]\b", msg), msg  # the event id
    assert "15" in msg and "14" in msg, msg  # both clocks (as 615 / 614 s or 10:15 / 10:14)
    ok = [*start(), ev(11, 5, A, "Pass", 10, 15), ev(12, 9, B, "Foul", 10, 20),
          ev(13, 6, A, SHOT, 10, 15), ev(14, 7, A, "End", 47, None),
          ev(15, 8, B, "Pass", 46, 30), *start("SecondHalf", 45, base=20),
          ev(30, 9, A, SHOT, 45, 10, period="SecondHalf")]  # fmt: skip
    assert context(*ok)["13"] == ("pass", False)


@pytest.mark.parametrize("second", [None, "", "12.5", "x1", "-3"])
def test_bad_clock_on_a_play_row_raises(second: str | None) -> None:
    """§2.2: a play row with a null or non-``INT_TEXT`` ``second`` → ValueError naming
    ``where``. The archive's null-second rows are ``OffsideGiven`` (skipped)."""
    row = ev(11, 5, A, "Pass", 20, 0)
    row["second"] = second if second != "" else None
    with pytest.raises(ValueError, match="fixture.csv"):
        context(*start(), row, ev(12, 6, A, SHOT, 20, 30))


def test_null_clock_on_a_skip_row_passes() -> None:
    """§2.2 near misses: a skipped type (``OffsideGiven``) and a non-play-period row with a null
    ``second`` pass; ``"12.0"`` is ``INT_TEXT``."""
    rows = [
        ev(1, 1, A, "FormationSet", 0, None, period="PreMatch"),
        *start(base=2),
        ev(11, 5, A, "Pass", 20, "0.0"),
        ev(12, 9, B, "OffsideGiven", 20, None),
        ev(13, 6, A, SHOT, 20, "12.0"),
        ev(14, 7, A, "Pass", 95, None, period="PostGame"),
    ]
    assert context(*rows)["13"] == ("pass", False)


# --- previous_play -----------------------------------------------------------------------------


def test_previous_play_custom_skip_keeps_foul() -> None:
    """§2.2, §8: ``previous_play`` maps every row id to the id of the last earlier non-skip row
    of its period (None for the first); with a custom ``skip`` (``PLAY_SKIP`` without ``Foul``,
    the passing-models use) the foul drawn is the shot's previous play."""
    rows = [*start(), ev(11, 5, A, "Pass", 30, 0), ev(12, 9, B, "Foul", 30, 4),
            ev(13, 6, A, SHOT, 30, 20)]  # fmt: skip
    df = frame(*rows)
    default = whoscored.previous_play(df, "'f'")
    assert default == {"1": None, "2": None, "11": None, "12": "11", "13": "11"}
    custom = whoscored.previous_play(df, "'f'", skip=whoscored.PLAY_SKIP - {"Foul"})
    assert custom["13"] == "12" and custom["12"] == "11"


def test_previous_play_bare_str_skip() -> None:
    """§2.2: ``skip`` as a bare str → TypeError."""
    df = frame(*start(), ev(11, 5, A, "Pass", 30, 0))
    with pytest.raises(TypeError):
        whoscored.previous_play(df, "'f'", skip="Foul")  # type: ignore[arg-type]


_TYPES = ("Pass", "TakeOn", "Foul", "Aerial", "Error", "OffsideGiven", "Clearance", "Card")


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.sampled_from(_TYPES), st.booleans(), st.integers(0, 3)),
                min_size=1, max_size=25))  # fmt: skip
def test_previous_play_property(events: list[tuple[str, bool, int]]) -> None:
    """Property: on any non-decreasing clock, each row's previous play is the last earlier row
    of the period whose type is not skipped (the brute-force definition), for the default
    skip and for the empty skip."""
    rows = [ev(1, 1, A, "Start", 0, 0), ev(2, 1, B, "Start", 0, 0)]  # a clock: no skip needed
    t = 0
    for i, (kind, home, step) in enumerate(events):
        t += step
        rows.append(ev(100 + i, 10 + i, A if home else B, kind, t // 60, t % 60))
    df = frame(*rows)
    for skip in (whoscored.PLAY_SKIP, frozenset()):
        got = whoscored.previous_play(df, "'f'", skip=skip)
        last: str | None = None
        for r in rows:
            assert got[str(r["id"])] == last, (r, skip)
            if r["type"] not in skip:
                last = str(r["id"])


# --- the loader --------------------------------------------------------------------------------

LOADED = {
    "first_shot_of_period": (wsc.SAVED, "other", False),
    "after_own_pass": (wsc.GOAL, "pass", False),
    "after_opponent_keeper_save": (wsc.CLEARED, "opp_keeper", False),
    "after_opponent_shot": (wsc.BLOCKED, "opp_other", False),
    "after_own_touch": (wsc.NON_PASS, "other", False),
    "penalty": (wsc.PENALTY, "opp_other", False),
    "shootout_first_kick": (wsc.SO_GOAL, "other", False),
    "shootout_second_kick": (wsc.SO_MISS, "opp_other", False),
    "extra_time": (wsc.ET_K_ET_GOAL, "other", False),
    "own_goal": (wsc.OWN_GOAL, None, None),
}


@pytest.mark.parametrize("case", sorted(LOADED))
def test_loader_writes_context(ws_standard: Any, case: str) -> None:
    """§2.2: ``_add_shots`` writes ``prev_action`` / ``rebound`` on every shot (penalties and
    period 5 included); an own goal gets (null, null)."""
    gid, level, rebound = LOADED[case]
    shots = ws_standard.archive.shots
    row = shots.filter(pl.col("shot_id") == f"ws:{gid}").select("prev_action", "rebound")
    assert row.rows() == [(level, rebound)]
    own = shots["is_own_goal"]
    assert shots.filter(~own)["prev_action"].null_count() == 0
    assert shots.filter(~own)["rebound"].null_count() == 0
    assert shots.filter(own)["prev_action"].null_count() == int(own.sum())


REB_SAVE, REB_SHOT = 2_901_000_301, 2_901_000_302  # MAIN-range ids unused by the archive


def test_loader_rebound_and_outfield_block(ws_standard: Any, tmp_path: Any) -> None:
    """Integration census (PR1.xg-context): the standard archive loads no ``rebound`` True and
    no ``opp_block`` shot, so the loader is also run on MAIN with an outfield block of BLOCKED
    (B's ``Save`` with q94, 8:31) and A's shot one second later (8:32): it loads as
    (opp_block, True), and BLOCKED (8:30) keeps its own value."""
    std = ws_standard.archive.shots
    assert not std["rebound"].any() and "opp_block" not in set(std["prev_action"].drop_nulls())
    fx = wsc.mini()
    m = fx.matches[0]
    i = next(k for k, e in enumerate(m.events) if e.gid == wsc.BLOCKED)
    m.events[i + 1 : i + 1] = [
        wsc.Ev(REB_SAVE, 61, wsc.B, "Save", "FirstHalf", 8, 31, "10.0", "45.0", wsc.B * 10 + 4,
               quals=[q(94)]),
        wsc.Ev(REB_SHOT, 41, wsc.A, "MissedShots", "FirstHalf", 8, 32, "92.0", "45.0",
               quals=[q(22), q(20)], mouth=("55.0", "20.0")),
    ]  # fmt: skip
    shots = wsc.load_fixture(tmp_path, fx).archive.shots
    got = dict(shots.select(pl.col("shot_id"), pl.struct("prev_action", "rebound")).rows())
    before = std.filter(pl.col("shot_id") == f"ws:{wsc.BLOCKED}").select("prev_action", "rebound")
    assert got[f"ws:{REB_SHOT}"] == {"prev_action": "opp_block", "rebound": True}
    assert got[f"ws:{wsc.BLOCKED}"] == before.row(0, named=True)


# --- build ratify (runs/PR1/xg-context/build.md §2.2 items 1-2) --------------------------------


def test_repeated_id_maps_from_last_row_and_context_reads_rows() -> None:
    """Ratify 1: a repeated event id (a synthetic ``OffsideGiven`` can reuse one) maps, in
    ``previous_play``, from its *last* row; ``shot_context`` reads the previous play row by
    index, so the shot after B's clearance id 12 is ``opp_clearance`` although a later row
    (A's pass) reuses id 12. A lookup by id would give that pass ("pass")."""
    rows = [*start(), ev(11, 5, A, "Pass", 10, 0), ev(12, 9, B, "Clearance", 10, 2),
            ev(13, 6, A, SHOT, 10, 4), ev(12, 7, A, "Pass", 10, 20)]  # fmt: skip
    df = frame(*rows)
    assert df["id"].to_list().count("12") == 2  # precondition: the id repeats
    prev = whoscored.previous_play(df, "'f'")
    assert prev["12"] == "13"  # the last row's previous play (the first row's would be "11")
    assert prev["13"] == "12"
    assert context(*rows)["13"] == ("opp_clearance", False)


@pytest.mark.parametrize("period", ["PreMatch", "PostGame", None])
def test_shot_context_refuses_a_shot_outside_periods_1_to_5(period: str | None) -> None:
    """Ratify 2: ``shot_context`` (public, total) raises ValueError naming ``where`` and the shot
    for a shot row outside periods 1-5 (a non-play or null period), before reading anything else;
    the same row type in FirstHalf passes (near miss)."""
    ok = [*start(), ev(11, 5, A, "Pass", 10, 0), ev(13, 6, A, SHOT, 10, 4)]
    assert context(*ok)["13"] == ("pass", False)
    bad = [*ok, ev(14, 7, A, SHOT, 95, 0)]
    bad[-1]["period"] = period
    with pytest.raises(ValueError, match=r"fixture\.csv.*'14'.*shot period"):
        context(*bad)
