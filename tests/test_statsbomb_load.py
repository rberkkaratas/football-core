"""``statsbomb.load``: mapping to the C3 tables, cell offset, penalties, own goals, 360 and bad
input (spec ``runs/MS2/c4/spec.md`` §1, §2, §3 "Reading", §6, §7)."""

from __future__ import annotations

import datetime as dt
import json
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hypothesis.strategies as st
import polars as pl
import pytest
import statsbomb_cases as sbc
from hypothesis import given, settings
from statsbomb_cases import COARSE, FINE, YARD

from football_core import pitch, schema
from football_core.io import statsbomb

TOL = 1e-9
EVENTS_A = f"data/events/{sbc.A}.json"
MATCHES_FILE = f"data/matches/{sbc.COMPETITION}/{sbc.SEASON}.json"
SIXTY_A = f"data/three-sixty/{sbc.A}.json"
FLAGS = ("first_time", "follows_dribble", "open_goal", "one_on_one", "deflected")


def _same(actual: Any, expected: Any) -> bool:
    if expected is None or isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, float):
        return isinstance(actual, float) and math.isclose(actual, expected, rel_tol=0, abs_tol=TOL)
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return bool(actual == expected) and type(actual) is type(expected)


def _value_error(call: Callable[[], object], *needles: str) -> ValueError:
    """A plain ValueError (a SchemaValidationError would be a loader bug) naming ``needles``."""
    with pytest.raises(ValueError) as info:
        call()
    assert not isinstance(info.value, schema.SchemaValidationError), info.value
    for needle in needles:
        assert needle in str(info.value), (needle, str(info.value))
    return info.value


def _load(tmp_path: Path, *matches: sbc.Match) -> Any:
    return sbc.load_matches(tmp_path / "cache", list(matches))


def _shot_only(tmp_path: Path, *events: sbc.Event, **match: Any) -> Any:
    return _load(tmp_path, sbc.one_match(*events, **match))


def _goal_line_y(r: dict[str, Any]) -> float:
    """MS5.goalmouth §3.1: where the line from (x, y) through the stop point meets x = 105."""
    k = (105.0 - r["x"]) / (r["end_x"] - r["x"])
    return min(max(r["y"] + (r["end_y"] - r["y"]) * k, 0.0), 68.0)


# --- zero violations --------------------------------------------------------------------------


def test_load_validates(sb_standard: Any) -> None:
    shots, ff = sb_standard.shots, sb_standard.freeze_frames
    assert shots.height == sbc.SHOT_ROWS
    assert ff.height == sbc.FREEZE_FRAME_ROWS
    assert shots.schema == schema.SHOT_SCHEMA
    assert ff.schema == schema.FREEZE_FRAME_SCHEMA
    schema.validate_shots(shots)
    schema.validate_freeze_frames(ff, shots=shots)
    assert set(shots["outcome"].cast(pl.String)) == set(schema.OUTCOMES)
    assert set(shots["situation"].cast(pl.String)) == set(schema.SITUATIONS)
    assert shots["is_penalty"].any() and shots["is_own_goal"].any()
    assert (shots["period"] == 5).any()
    assert shots["visible_area"].is_not_null().any()
    assert set(shots["provider"].cast(pl.String)) == {"statsbomb"}


def _tenths(lo: int, hi: int) -> st.SearchStrategy[float]:
    return st.integers(lo * 10, hi * 10).map(lambda k: k / 10)


@st.composite
def _random_shot(draw: st.DrawFn) -> sbc.Event:
    period = draw(st.integers(1, 5))
    shot_type = "Penalty" if period == 5 else draw(st.sampled_from(sorted(sbc.SHOT_TYPE_IDS)))
    outcome = draw(st.sampled_from(sorted(sbc.OUTCOME_IDS)))
    start = sbc.PERIOD_START[period]
    if outcome in sbc.TWO_VALUE_END:
        end: tuple[float, ...] = (draw(_tenths(0, 120)), draw(_tenths(0, 80)))
    else:
        end = (draw(_tenths(0, 120)), draw(_tenths(0, 80)), draw(_tenths(0, 7)))
    opponents = draw(st.lists(st.integers(2002, 2011), unique=True, max_size=4))
    mates = draw(
        st.lists(st.integers(1002, 1011).filter(lambda p: p != 1009), max_size=3, unique=True)
    )
    # MS5.goalmouth §3.1: a frame keeper is a lineup keeper (2001 facing, or the shooter's 1001)
    keeper = draw(st.sampled_from([None, 2001, 1001]))
    frame = [
        sbc.ff_player(
            pid,
            (draw(_tenths(0, 120)), draw(_tenths(0, 80))),
            teammate=pid < 2000,
            keeper=keeper is not None and i == 0,
        )
        for i, pid in enumerate(([keeper] if keeper else []) + opponents + mates)
    ]
    flags = draw(st.sets(st.sampled_from(FLAGS)))
    return sbc.shot(
        "fake-shot-h01",
        location=(draw(_tenths(0, 120)), draw(_tenths(0, 80))),
        outcome=outcome,
        shot_type=shot_type,
        play_pattern=draw(st.sampled_from(sorted(sbc.PLAY_PATTERN_IDS))),
        body_part=draw(st.sampled_from(sorted(sbc.BODY_PART_IDS))),
        technique=draw(st.sampled_from(sorted(sbc.TECHNIQUE_IDS))),
        end_location=end,
        freeze_frame=frame if frame or draw(st.booleans()) else None,
        flags=flags,
        xg=draw(st.floats(0.0, 1.0)),
        period=period,
        minute=draw(st.integers(start, min(start + 45, 150))),
        second=draw(st.integers(0, 59)),
        millis=draw(st.integers(0, 999)),
    )


@settings(max_examples=40, deadline=None)
@given(ev=_random_shot())
def test_property_random_fine_shot_validates(ev: sbc.Event) -> None:
    """Any vocabulary combination on the 0.1 yd grid loads into schema-valid tables."""
    with tempfile.TemporaryDirectory() as tmp:
        data = _shot_only(Path(tmp), ev, fidelity=None)
    assert data.shots.height == 1
    schema.validate_shots(data.shots)
    schema.validate_freeze_frames(data.freeze_frames, shots=data.shots)


# --- mapping ----------------------------------------------------------------------------------


def _open_shot_expected() -> dict[str, Any]:
    """Spec §2.1 for ``OPEN_SHOT`` (fine grid), computed by hand from the fixture."""
    ev = next(e for e in sbc.full_events(sbc.match_a()) if e["id"] == sbc.OPEN_SHOT)
    x, y = sbc.metres(102.35, 35.65, FINE)
    gk_x, gk_y = sbc.metres(118.25, 40.15, FINE)
    goal_y = sbc.metres_y(38.35 - FINE)
    return {
        "shot_id": "sb:fake-shot-a01",
        "provider": "statsbomb",
        "competition_id": "sb:9001",
        "season_id": "sb:9002",
        "match_id": "sb:900001",
        "match_date": dt.date(2024, 1, 6),
        "team_id": "sb:101",
        "opponent_team_id": "sb:202",
        "player_id": "sb:1009",
        "gk_id": "sb:2001",  # MS5.goalmouth §3.1: the away lineup keeper
        "is_home": True,
        "period": 1,
        "minute": 23,
        "second": 41,
        "seconds_in_period": 23 * 60 + 41.25,
        "x": x,
        "y": y,
        "end_x": 105.0,  # end x = 120 is the goal line: not shifted
        "end_y": goal_y,  # 3-value end: y - FINE on both grids
        "end_z": 0.85 * YARD,
        "goal_y": goal_y,
        "goal_z": 0.85 * YARD,
        "gk_x": gk_x,  # the opponent keeper, never the shooter's own (also in the frame)
        "gk_y": gk_y,
        "distance_m": float(pitch.shot_distance(x, y)),
        "angle_rad": float(pitch.opening_angle(x, y)),
        "visible_area": sbc.polygon(sbc.AREA_A01),
        "body_part": "left_foot",
        "situation": "open_play",
        "assist_type": "pass",
        "technique": "half_volley",
        "outcome": "goal",
        "is_goal": True,
        "is_own_goal": False,
        "is_penalty": False,
        "on_target": True,
        "first_time": True,
        "follows_dribble": False,
        "open_goal": False,
        "one_on_one": False,
        "prev_action": None,  # PR1.xg-context: not derived for StatsBomb
        "rebound": None,
        "big_chance": None,
        "deflected": False,
        "provider_xg": 0.31,
        "provider_psxg": None,
        "assist_event_id": "sb:fake-pass-a01",
        "raw_type": "Shot",
        "raw_qualifiers": sbc.raw_qualifiers(ev),  # keeps under_pressure, drops freeze_frame
        "schema_version": schema.SCHEMA_VERSION,
    }


@pytest.mark.parametrize("column", schema.column_names("shots"))
def test_shot_columns(sb_standard: Any, column: str) -> None:
    actual = sbc.row(sb_standard.shots, sbc.OPEN_SHOT)[column]
    expected = _open_shot_expected()[column]
    assert _same(actual, expected), (column, actual, expected)


def test_gk_null_without_opponent_keeper(tmp_path: Path) -> None:
    own_keeper_only = [
        sbc.ff_player(sbc.HOME_KEEPER, (30.05, 40.05), teammate=True, keeper=True),
        sbc.ff_player(2004, (110.15, 38.05), teammate=False),
    ]
    data = _shot_only(
        tmp_path,
        sbc.shot("fake-shot-g01", freeze_frame=own_keeper_only),
        sbc.shot("fake-shot-g02", freeze_frame=None, minute=20),
    )
    for sid in ("fake-shot-g01", "fake-shot-g02"):
        r = sbc.row(data.shots, sid)
        assert r["gk_x"] is None and r["gk_y"] is None
    ff = data.freeze_frames
    assert ff.filter(pl.col("shot_id") == "sb:fake-shot-g01").height == 2
    assert ff.filter(pl.col("shot_id") == "sb:fake-shot-g02").height == 0


@pytest.mark.parametrize(
    ("raw", "outcome", "is_goal", "on_target"),
    [
        ("Goal", "goal", True, True),
        ("Saved", "saved", False, True),
        ("Saved to Post", "saved", False, True),
        ("Blocked", "blocked", False, False),
        ("Post", "post", False, False),
        ("Off T", "off_target", False, False),
        ("Wayward", "off_target", False, False),
        ("Saved Off Target", "off_target", False, False),
    ],
)
def test_outcome_mapping(
    tmp_path: Path, raw: str, outcome: str, is_goal: bool, on_target: bool
) -> None:
    r = sbc.row(_shot_only(tmp_path, sbc.shot("fake-shot-o01", outcome=raw)).shots, "fake-shot-o01")
    assert (r["outcome"], r["is_goal"], r["on_target"]) == (outcome, is_goal, on_target)


@pytest.mark.parametrize(
    ("shot_type", "play_pattern", "situation"),
    [
        ("Penalty", "Regular Play", "penalty"),
        ("Penalty", "From Counter", "penalty"),
        ("Free Kick", "From Free Kick", "direct_free_kick"),
        ("Free Kick", "Regular Play", "direct_free_kick"),
        ("Corner", "From Corner", "direct_corner"),
        ("Kick Off", "From Kick Off", "kick_off"),
        ("Open Play", "From Corner", "from_corner"),
        ("Open Play", "From Free Kick", "from_free_kick"),
        ("Open Play", "From Throw In", "from_throw_in"),
        ("Open Play", "From Counter", "fast_break"),
        ("Open Play", "Regular Play", "open_play"),
        ("Open Play", "From Goal Kick", "open_play"),
        ("Open Play", "From Keeper", "open_play"),
        ("Open Play", "From Kick Off", "open_play"),
        ("Open Play", "Other", "open_play"),
    ],
)
def test_situation_mapping(
    tmp_path: Path, shot_type: str, play_pattern: str, situation: str
) -> None:
    ev = sbc.shot("fake-shot-p01", shot_type=shot_type, play_pattern=play_pattern)
    r = sbc.row(_shot_only(tmp_path, ev).shots, "fake-shot-p01")
    assert r["situation"] == situation
    assert r["is_penalty"] is (shot_type == "Penalty")


@pytest.mark.parametrize(
    ("pass_flags", "assist_type"),
    [
        ({"cut_back": True, "cross": True}, "cutback"),
        ({"cut_back": True, "through": True}, "cutback"),
        ({"through": True, "cross": True}, "through_ball"),
        ({"cross": True}, "cross"),
        ({}, "pass"),
        (None, "none"),
    ],
    ids=["cut_back+cross", "cut_back+through", "through+cross", "cross", "plain", "none"],
)
def test_assist_precedence(
    tmp_path: Path, pass_flags: dict[str, bool] | None, assist_type: str
) -> None:
    events = [] if pass_flags is None else [sbc.key_pass("fake-pass-k01", **pass_flags)]
    key = None if pass_flags is None else "fake-pass-k01"
    events.append(sbc.shot("fake-shot-k01", key_pass_id=key, minute=11))
    r = sbc.row(_shot_only(tmp_path, *events).shots, "fake-shot-k01")
    assert r["assist_type"] == assist_type
    assert r["assist_event_id"] == (None if key is None else "sb:fake-pass-k01")


def test_goal_mouth_null_for_blocked_and_wayward(tmp_path: Path) -> None:
    data = _shot_only(
        tmp_path,
        sbc.shot("fake-shot-m01", outcome="Blocked", end_location=(110.15, 38.05)),
        sbc.shot("fake-shot-m02", outcome="Wayward", end_location=(115.05, 60.05), minute=11),
        # adversarial: a Blocked shot with 3 values still has no goal-mouth point (C3 rule)
        sbc.shot("fake-shot-m03", outcome="Blocked", end_location=(110.15, 38.05, 0.4), minute=12),
        sbc.shot("fake-shot-m04", outcome="Saved", end_location=(118.15, 39.05, 0.4), minute=13),
    )
    end_x, end_y = sbc.metres(110.15, 38.05, FINE)
    for sid in ("fake-shot-m01", "fake-shot-m02", "fake-shot-m03"):
        r = sbc.row(data.shots, sid)
        assert r["goal_y"] is None and r["goal_z"] is None, sid
    m01, m03 = sbc.row(data.shots, "fake-shot-m01"), sbc.row(data.shots, "fake-shot-m03")
    assert _same([m01["end_x"], m01["end_y"]], [end_x, end_y])
    assert m01["end_z"] is None
    assert _same(m03["end_z"], 0.4 * YARD)
    m04 = sbc.row(data.shots, "fake-shot-m04")
    # MS5.goalmouth §3.1: a save stopped at 118.15 yd (fine grid) is projected to the goal line,
    # MS6 xgot-ready §2.2: then moved to the centre of its 0.1 yd goal-mouth cell
    assert _same([m04["goal_y"], m04["goal_z"]], [sbc.snapped(_goal_line_y(m04)), 0.4 * YARD])


def test_end_z_in_metres(tmp_path: Path) -> None:
    zs = {"fake-shot-z01": 1.5, "fake-shot-z02": 0.0, "fake-shot-z03": 7.8}
    events = [
        sbc.shot(sid, outcome="Saved", end_location=(118.15, 39.05, z), minute=10 + i)
        for i, (sid, z) in enumerate(zs.items())
    ]
    shots = _shot_only(tmp_path, *events).shots
    for sid, z in zs.items():
        r = sbc.row(shots, sid)
        assert _same(r["end_z"], z * YARD), sid  # z is never shifted: 0.0 stays 0.0
        assert _same(r["goal_z"], z * YARD), sid


def test_flags_absent_are_false_big_chance_null(tmp_path: Path) -> None:
    flags = FLAGS
    data = _shot_only(
        tmp_path,
        sbc.shot("fake-shot-f01", xg=0.05),
        sbc.shot("fake-shot-f02", flags=flags, minute=11),
    )
    plain, flagged = sbc.row(data.shots, "fake-shot-f01"), sbc.row(data.shots, "fake-shot-f02")
    for flag in flags:
        assert plain[flag] is False and flagged[flag] is True, flag
    for r in (plain, flagged):
        assert r["big_chance"] is None and r["provider_psxg"] is None
    assert _same(plain["provider_xg"], 0.05)


# --- bad input --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "needles"),
    [
        (
            {"outcome": "Saved To Post", "end_location": (119.95, 43.95, 1.25)},
            ("Saved To Post", "outcome"),
        ),
        ({"shot_type": "Indirect Free Kick"}, ("Indirect Free Kick", "type")),
        ({"play_pattern": "From Penalty"}, ("From Penalty", "play_pattern")),
        ({"body_part": "Shoulder"}, ("Shoulder", "body_part")),
        ({"technique": "Scissor Kick"}, ("Scissor Kick", "technique")),
        ({"fidelity": "3"}, ("'3'", "shot_fidelity_version")),
    ],
    ids=["outcome", "shot_type", "play_pattern", "body_part", "technique", "shot_fidelity"],
)
def test_unknown_value_raises(
    tmp_path: Path, kwargs: dict[str, Any], needles: tuple[str, ...]
) -> None:
    kwargs = dict(kwargs)
    match_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k == "fidelity"}
    ev = sbc.shot("fake-shot-u01", **kwargs)
    _value_error(lambda: _shot_only(tmp_path, ev, **match_kwargs), str(sbc.A), *needles)


def _ff(location: Any, *, teammate: Any = False) -> list[dict[str, Any]]:
    keeper = sbc.ff_player(sbc.AWAY_KEEPER, location, teammate=False, keeper=True)
    return [keeper, {**sbc.ff_player(2003, (110.15, 38.05), teammate=False), "teammate": teammate}]


# Spec §2.1: a bool or str coordinate, a non-numeric number, a non-bool flag, an odd
# visible_area or a foreign team is a ValueError naming the match, never a silent float().
# fmt: off
BAD_VALUES: dict[str, tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]] = {
    "bool_x_coarse": ({"location": (True, 40)}, {"fidelity": "1"}, ("location",)),
    "bool_y_coarse": ({"location": (109, True)}, {"fidelity": "1"}, ("location",)),
    "str_x": ({"location": ("102.35", 35.65)}, {}, ("location",)),
    "str_y": ({"location": (102.35, "35.65")}, {}, ("location",)),
    "bool_y_freeze_frame": ({"freeze_frame": _ff((118.25, True))}, {}, ("location",)),
    "bool_y_end_location": ({"end_location": (118.15, True, 0.55)}, {}, ("location",)),
    "str_end_z": ({"end_location": (118.15, 39.05, "1.5")}, {}, ("end_location z",)),
    "bool_end_z": ({"end_location": (118.15, 39.05, True)}, {}, ("end_location z",)),
    "four_value_end": ({"end_location": (118.15, 39.05, 0.55, 1.0)}, {}, ("end_location",)),
    "str_xg": ({"xg": "0.3"}, {}, ("statsbomb_xg",)),
    "bool_xg": ({"xg": True}, {}, ("statsbomb_xg",)),
    "str_teammate": ({"freeze_frame": _ff((118.25, 40.15), teammate="yes")}, {}, ("teammate",)),
    "foreign_team": ({"team": sbc.team(303)}, {}, ("303", "not in the match")),
}
# fmt: on


@pytest.mark.parametrize("case", list(BAD_VALUES))
def test_bad_value_raises(tmp_path: Path, case: str) -> None:
    kwargs, match_kwargs, needles = BAD_VALUES[case]
    ev = sbc.shot("fake-shot-v01", **kwargs)
    _value_error(lambda: _shot_only(tmp_path, ev, **match_kwargs), str(sbc.A), *needles)


def test_mixed_flag_types_raise(tmp_path: Path) -> None:
    """One shot with ``first_time: true`` and one with ``"yes"``: a ValueError naming the match
    and the field, not a polars TypeError from building the Boolean column."""
    good = sbc.shot("fake-shot-v02", flags=("first_time",))
    bad = sbc.shot("fake-shot-v03", minute=12)
    bad["shot"]["first_time"] = "yes"
    _value_error(lambda: _shot_only(tmp_path, good, bad), str(sbc.A), "shot.first_time", "'yes'")


def test_odd_visible_area_raises(tmp_path: Path) -> None:
    ev = sbc.shot("fake-shot-v04")
    sixty = [sbc.visible_area("fake-shot-v04", [100.0, 30.0, 110.0, 30.0, 105.0])]
    _value_error(
        lambda: _shot_only(tmp_path, ev, three_sixty=sixty), str(sbc.A), "odd number of values"
    )


def test_foreign_team_own_goal_raises(tmp_path: Path) -> None:
    ev = sbc.own_goal_against(
        "fake-oga-v05", team_id=sbc.AWAY, player_id=2004, location=(5.5, 40.5)
    )
    ev["team"] = sbc.team(303)
    _value_error(lambda: _shot_only(tmp_path, ev), str(sbc.A), "303", "not in the match")


def _corrupt_cases() -> dict[str, Callable[[Path, bytes], None]]:
    events = EVENTS_A

    def put(path: str, raw: bytes, gz_bytes: bytes | None = None) -> Callable[[Path, bytes], None]:
        return lambda checkout, _: sbc.set_cached(checkout, path, raw, gz_bytes=gz_bytes)

    def gz_file(checkout: Path, path: str = events) -> Path:
        return checkout / f"{path}.gz"

    return {
        "truncated": lambda c, raw: gz_file(c).write_bytes(sbc.gz(raw)[: len(sbc.gz(raw)) // 2]),
        "flipped_header": lambda c, raw: gz_file(c).write_bytes(
            bytes([sbc.gz(raw)[0] ^ 1]) + sbc.gz(raw)[1:]
        ),
        "empty": lambda c, raw: gz_file(c).write_bytes(b""),
        "zlib_error": lambda c, raw: gz_file(c).write_bytes(sbc.zlib_error_gz(raw)),
        "size_mismatch": lambda c, raw: gz_file(c).write_bytes(sbc.gz(raw + b" ")),  # valid JSON
        "bad_json": put(events, b"[{not json"),
        "deep_nesting": put(events, b"[" * 100_000 + b"]" * 100_000),
        "nan": put(events, b'[{"id": NaN}]'),
        "invalid_utf8": put(events, b'["\xff\xfe"]'),
        "corrupt_360_gzip": lambda c, raw: gz_file(c, SIXTY_A).write_bytes(
            (gz_file(c, SIXTY_A).read_bytes())[:-12]
        ),
    }


@pytest.mark.parametrize("case", sorted(_corrupt_cases()))
def test_corrupt_gzip_raises_value_error(tmp_path: Path, case: str) -> None:
    files = sbc.repo_files([sbc.match_a()])
    checkout = sbc.write_cache(tmp_path / "cache", files)
    _corrupt_cases()[case](checkout, files[EVENTS_A])
    name = f"{sbc.A}.json.gz"
    _value_error(lambda: statsbomb.load(checkout), name, "delete it and rerun")


def _edit_event(eid: str, edit: Callable[[sbc.Event], None]) -> Callable[[Any], None]:
    def apply(events: list[sbc.Event]) -> None:
        edit(next(e for e in events if e["id"] == eid))

    return apply


@pytest.mark.parametrize(
    "edit",
    [
        _edit_event("fake-shot-x01", lambda e: e.pop("location")),
        _edit_event("fake-shot-x01", lambda e: e["shot"].update(outcome=["Saved"])),
        _edit_event("fake-shot-x01", lambda e: e.update(player=None)),
        _edit_event("fake-shot-x01", lambda e: e["shot"].update(freeze_frame={"player": 1})),
        lambda events: {"events": events},
    ],
    ids=["missing_location", "list_for_dict", "null_player", "dict_for_frame_list", "not_a_list"],
)
def test_missing_field_raises_value_error(tmp_path: Path, edit: Callable[[Any], Any]) -> None:
    files = sbc.repo_files([sbc.one_match(sbc.shot("fake-shot-x01"))])
    sbc.edit_json(files, EVENTS_A, edit)
    checkout = sbc.write_cache(tmp_path / "cache", files)
    _value_error(lambda: statsbomb.load(checkout), str(sbc.A))


def _edit_entry(edit: Callable[[dict[str, Any]], object]) -> Callable[[Any], None]:
    def apply(entries: list[dict[str, Any]]) -> None:
        edit(entries[0])  # in place; a pop's return value must not replace the file

    return apply


# fmt: off
BAD_ENTRIES: dict[str, Callable[[dict[str, Any]], object]] = {
    "null_score": lambda m: m.update(home_score=None),
    "score_40000": lambda m: m.update(home_score=40000),
    "negative_score": lambda m: m.update(away_score=-1),
    "bool_score": lambda m: m.update(home_score=True),
    "float_score": lambda m: m.update(home_score=1.0),
    "date_24-01-05": lambda m: m.update(match_date="24-01-05"),
    "date_2024-02-30": lambda m: m.update(match_date="2024-02-30"),
    "date_trailing_newline": lambda m: m.update(match_date="2024-01-06\n"),
    "gender_x": lambda m: m["home_team"].update(home_team_gender="x"),
    "genders_differ": lambda m: m["away_team"].update(away_team_gender="female"),
    "missing_competition_name": lambda m: m["competition"].pop("competition_name"),
    "null_season_name": lambda m: m["season"].update(season_name=None),
}
# fmt: on


@pytest.mark.parametrize("case", list(BAD_ENTRIES))
def test_bad_match_entry_raises(tmp_path: Path, case: str) -> None:
    files = sbc.repo_files([sbc.one_match(sbc.shot("fake-shot-e01"))])
    sbc.edit_json(files, MATCHES_FILE, _edit_entry(BAD_ENTRIES[case]))
    checkout = sbc.write_cache(tmp_path / "cache", files)
    _value_error(lambda: statsbomb.load(checkout), str(sbc.A))


def test_matches_table(sb_standard: Any) -> None:
    # MS3.matches: the canonical table of football_core.schema, no alias left in the loader
    assert list(schema.MATCH_SCHEMA.items()) == list(sbc.MATCH_DTYPES.items())
    assert not hasattr(statsbomb, "MATCH_SCHEMA")
    matches = sb_standard.matches.sort("match_id")
    assert matches.schema == schema.MATCH_SCHEMA
    assert matches.null_count().sum_horizontal().item() == 0
    assert matches.to_dict(as_series=False) == {
        "match_id": ["sb:900001", "sb:900002", "sb:900003"],
        "provider": ["statsbomb"] * 3,
        "competition_id": ["sb:9001"] * 3,
        "season_id": ["sb:9002"] * 3,
        "match_date": [dt.date(2024, 1, 6), dt.date(2024, 1, 7), dt.date(2024, 1, 13)],
        "home_team_id": ["sb:101"] * 3,
        "away_team_id": ["sb:202"] * 3,
        "competition_name": ["Fixture League"] * 3,
        "season_name": ["2023/2024"] * 3,
        "gender": ["male", "male", "female"],
        "home_score": [3, 0, 0],
        "away_score": [1, 1, 0],
        "has_360": [True, False, False],
        "schema_version": [schema.SCHEMA_VERSION] * 3,
    }
    schema.validate_matches(matches)
    schema.validate_shots(sb_standard.shots, matches=matches)  # the shots agree with it
    assert sb_standard.skipped_360 == ()


# --- load arguments, missing files ------------------------------------------------------------


@pytest.mark.parametrize("case", ["filter", "str", "unknown"])
def test_load_match_ids(tmp_path: Path, case: str) -> None:
    checkout = sbc.write_cache(tmp_path / "cache", sbc.repo_files(sbc.standard_matches()))
    if case == "filter":
        data = statsbomb.load(checkout, match_ids=[sbc.B])
        assert set(data.shots["match_id"]) == {"sb:900002"}
        assert data.shots.height == 5
        assert data.matches["match_id"].to_list() == ["sb:900002"]
        assert set(data.freeze_frames["shot_id"]) <= set(data.shots["shot_id"])
    elif case == "str":
        with pytest.raises(TypeError):
            statsbomb.load(checkout, match_ids=str(sbc.B))  # type: ignore[arg-type]
    else:
        _value_error(lambda: statsbomb.load(checkout, match_ids={sbc.B, 424242}), "424242")


def _drop_tree_entry(checkout: Path, path: str) -> None:
    tree = json.loads((checkout / "tree.json").read_text(encoding="utf-8"))
    del tree[path]
    (checkout / "tree.json").write_text(json.dumps(tree), encoding="utf-8")


@pytest.mark.parametrize(
    ("case", "needle"),
    [
        ("tree.json", "tree.json"),
        ("no_checkout", "tree.json"),
        ("events_file", f"{sbc.A}.json"),
        ("events_not_in_tree", str(sbc.A)),
        ("listed_three_sixty", "three-sixty"),
    ],
)
def test_load_missing_file_raises(tmp_path: Path, case: str, needle: str) -> None:
    checkout = sbc.write_cache(tmp_path / "cache", sbc.repo_files([sbc.match_a()]))
    target = checkout
    if case == "tree.json":
        (checkout / "tree.json").unlink()
    elif case == "no_checkout":
        target = tmp_path / "nowhere" / sbc.COMMIT
    elif case == "events_file":
        (checkout / f"{EVENTS_A}.gz").unlink()
    elif case == "events_not_in_tree":
        _drop_tree_entry(checkout, EVENTS_A)
    else:
        (checkout / f"{SIXTY_A}.gz").unlink()
    with pytest.raises(FileNotFoundError) as info:
        statsbomb.load(target)
    assert needle in str(info.value)


# --- cell offset ------------------------------------------------------------------------------


def test_cell_offset_coarse_and_fine(sb_standard: Any) -> None:
    shots, ff = sb_standard.shots, sb_standard.freeze_frames

    def xy(sid: str, prefix: str = "") -> list[float]:
        r = sbc.row(shots, sid)
        return [r[f"{prefix}x"], r[f"{prefix}y"]]

    assert _same(xy(sbc.PENALTY), [94.0, 34.0])  # fine [108.05, 40.05]: the spot exactly
    assert _same(xy(sbc.COARSE_SHOT), [94.0 + 5.5 / 12, 33.5425])  # coarse [109, 41]
    assert _same(xy(sbc.COARSE_SHOT, "gk_"), list(sbc.metres(119, 40, COARSE)))
    assert _same(xy(sbc.OPEN_SHOT, "gk_"), list(sbc.metres(118.25, 40.15, FINE)))
    keeper = ff.filter((pl.col("shot_id") == f"sb:{sbc.COARSE_SHOT}") & pl.col("is_keeper"))
    assert _same([keeper["x"].item(), keeper["y"].item()], list(sbc.metres(119, 40, COARSE)))
    mate = ff.filter(
        (pl.col("shot_id") == f"sb:{sbc.OPEN_SHOT}") & (pl.col("player_id") == "sb:1010")
    )
    assert _same([mate["x"].item(), mate["y"].item()], list(sbc.metres(108.45, 44.55, FINE)))
    # coarse own goal [5, 30], flipped to the goal entered
    assert _same(xy(sbc.OWN_GOAL_B), [100.875, 68.0 - (43.16 + 11 / 24)])
    # coarse 3-value end [110, 38.1, 1.0]: x - 0.5, y - 0.05 (goal-mouth tenths), z unshifted
    coarse = sbc.row(shots, sbc.COARSE_SHOT)
    assert _same([coarse["end_x"], coarse["end_y"]], [95.375, 35.78425])
    # MS5.goalmouth §3.1: the save stopped at 110 yd is projected to the goal line (MS6 §2.2:
    # and snapped to its 0.1 yd goal-mouth cell)
    assert _same([coarse["goal_y"], coarse["goal_z"], coarse["end_z"]],
                 [sbc.snapped(_goal_line_y(coarse)), YARD, YARD])  # fmt: skip
    # a shot location at x = 120 is shifted like any other
    assert _same(xy(sbc.FINE_GOAL_LINE_SHOT), [99.5 + 5.95 / 6 * 5.5, 34.04575])
    assert _same(xy(sbc.COARSE_GOAL_LINE_SHOT), [99.5 + 5.5 * 5.5 / 6, 34.4575])
    for sid in (sbc.FINE_GOAL_LINE_SHOT, sbc.COARSE_GOAL_LINE_SHOT):
        assert 0.0 <= sbc.row(shots, sid)["angle_rad"] <= math.pi
    # an end_location x of 120 is the goal line on both grids
    assert sbc.row(shots, sbc.OPEN_SHOT)["end_x"] == 105.0
    assert sbc.row(shots, sbc.COARSE_GOAL_LINE_SHOT)["end_x"] == 105.0


@pytest.mark.parametrize(
    ("fidelity", "location", "offset"),
    [
        ("2", (102.35, 35.65), FINE),
        ("1", (102, 36), COARSE),
        ("2", (102, 36), None),  # metadata says fine, every location is an integer
        ("1", (102.35, 35.65), None),
        ("3", (102.35, 35.65), None),
        (None, (102, 36), COARSE),
        (None, (102.35, 35.65), FINE),
        ("2", "own_goal_only", FINE),
    ],
    ids=[
        "agree_fine",
        "agree_coarse",
        "disagree_fine_meta",
        "disagree_coarse_meta",
        "unknown_version",
        "absent_inferred_coarse",
        "absent_inferred_fine",
        "own_goal_only_fine",
    ],
)
def test_cell_offset_metadata(
    tmp_path: Path, fidelity: str | None, location: Any, offset: float | None
) -> None:
    sid = "fake-shot-c01"
    if location == "own_goal_only":
        ev = sbc.own_goal_against(sid, team_id=sbc.AWAY, player_id=2005, location=(5.35, 30.25))
        expected = sbc.flipped(5.35, 30.25, FINE)
    else:
        ev = sbc.shot(sid, location=location, freeze_frame=None)
        expected = sbc.metres(*location, offset) if offset is not None else (0.0, 0.0)
    if offset is None:
        load = lambda: _shot_only(tmp_path, ev, fidelity=fidelity)  # noqa: E731
        _value_error(load, str(sbc.A), "shot_fidelity_version")
        return
    r = sbc.row(_shot_only(tmp_path, ev, fidelity=fidelity).shots, sid)
    assert _same([r["x"], r["y"]], list(expected))


# --- penalties, shoot-outs, own goals ---------------------------------------------------------


def test_penalty_has_no_frame_context(sb_standard: Any) -> None:
    shots, ff = sb_standard.shots, sb_standard.freeze_frames
    for sid in (sbc.PENALTY, sbc.SHOOTOUT_GOAL):  # both have an event freeze frame
        r = sbc.row(shots, sid)
        assert r["is_penalty"] and r["situation"] == "penalty"
        assert r["gk_x"] is None and r["gk_y"] is None
        assert r["visible_area"] is None, sid  # PENALTY also has a 360 frame
        assert ff.filter(pl.col("shot_id") == f"sb:{sid}").height == 0
    for text in shots["raw_qualifiers"]:
        assert "freeze_frame" not in json.loads(text).get("shot", {})


def test_shootout_kept_as_period_5(sb_standard: Any) -> None:
    goal, save = (
        sbc.row(sb_standard.shots, sbc.SHOOTOUT_GOAL),
        sbc.row(sb_standard.shots, sbc.SHOOTOUT_SAVE),
    )
    assert (goal["period"], goal["minute"], goal["outcome"], goal["is_goal"]) == (
        5,
        120,
        "goal",
        True,
    )
    assert (save["period"], save["minute"], save["outcome"]) == (5, 121, "saved")
    assert _same([goal["seconds_in_period"], save["seconds_in_period"]], [30.0, 70.0])
    assert goal["is_penalty"] and save["is_penalty"]
    assert (sb_standard.shots["period"] == 5).sum() == 2


def test_own_goal_row(sb_standard: Any) -> None:
    ev = next(e for e in sbc.full_events(sbc.match_b()) if e["id"] == sbc.OWN_GOAL_B)
    r = sbc.row(sb_standard.shots, sbc.OWN_GOAL_B)
    x, y = sbc.flipped(5, 30, COARSE)
    expected: dict[str, Any] = {
        "team_id": "sb:101",  # the side of the player who put it in
        "opponent_team_id": "sb:202",
        "player_id": "sb:1004",
        "is_home": True,
        "period": 2,
        "x": x,
        "y": y,
        "distance_m": float(pitch.shot_distance(x, y)),
        "angle_rad": float(pitch.opening_angle(x, y)),
        "situation": "open_play",
        "outcome": "goal",
        "is_goal": True,
        "is_own_goal": True,
        "is_penalty": False,
        "on_target": False,
        "raw_type": "Own Goal Against",
        "raw_qualifiers": sbc.raw_qualifiers(ev),
    }
    # fmt: off
    nulls = (
        "end_x", "end_y", "end_z", "goal_y", "goal_z", "gk_x", "gk_y", "visible_area",
        "body_part", "assist_type", "assist_event_id", "technique", "first_time",
        "follows_dribble", "open_goal", "one_on_one", "deflected", "big_chance",
        "provider_xg", "provider_psxg",
    )
    # fmt: on
    expected |= dict.fromkeys(nulls)
    for column, value in expected.items():
        assert _same(r[column], value), (column, r[column], value)
    # the Open Play rule applies: the fine-grid own goal of match A came from a corner
    a = sbc.row(sb_standard.shots, sbc.OWN_GOAL_A)
    assert a["situation"] == "from_corner"
    assert _same([a["x"], a["y"]], list(sbc.flipped(5.35, 30.25, FINE)))
    ids = {f"sb:{sbc.OWN_GOAL_A}", f"sb:{sbc.OWN_GOAL_B}"}
    assert sb_standard.freeze_frames.filter(pl.col("shot_id").is_in(sorted(ids))).height == 0


def test_own_goal_for_ignored(sb_standard: Any) -> None:
    shots = sb_standard.shots
    assert f"sb:{sbc.OWN_GOAL_FOR_A}" not in set(shots["shot_id"])
    assert set(shots["raw_type"]) == {"Shot", "Own Goal Against"}
    assert (shots["raw_type"] == "Own Goal Against").sum() == 2
    assert shots["is_own_goal"].sum() == 2


# --- freeze frames, 360 -----------------------------------------------------------------------


def test_freeze_frame_rows(sb_standard: Any) -> None:
    shots, ff = sb_standard.shots, sb_standard.freeze_frames
    assert ff.height == sbc.FREEZE_FRAME_ROWS
    schema.validate_freeze_frames(ff, shots=shots)
    open_rows = ff.filter(pl.col("shot_id") == f"sb:{sbc.OPEN_SHOT}").sort("player_id")
    expected = [
        ("sb:1001", True, True, sbc.metres(30.05, 40.05, FINE)),
        ("sb:1010", True, False, sbc.metres(108.45, 44.55, FINE)),
        ("sb:2001", False, True, sbc.metres(118.25, 40.15, FINE)),
        ("sb:2004", False, False, sbc.metres(110.15, 38.05, FINE)),
    ]
    assert open_rows["player_id"].to_list() == [e[0] for e in expected]
    assert open_rows["is_teammate"].to_list() == [e[1] for e in expected]
    assert open_rows["is_keeper"].to_list() == [e[2] for e in expected]
    assert _same(
        [list(r) for r in open_rows.select("x", "y").rows()], [list(e[3]) for e in expected]
    )
    keepers = ff.filter(pl.col("is_keeper") & ~pl.col("is_teammate")).select(
        "shot_id", pl.col("x").alias("kx"), pl.col("y").alias("ky")
    )
    joined = shots.select("shot_id", "gk_x", "gk_y").join(
        keepers, on="shot_id", how="full", coalesce=True
    )
    # every opponent-keeper row is the shot's gk_*, and every non-null gk_* has such a row
    assert joined.filter(pl.col("gk_x").is_null() != pl.col("kx").is_null()).height == 0
    paired = joined.drop_nulls()
    assert paired.height > 0
    assert _same(paired["gk_x"].to_list(), paired["kx"].to_list())
    assert _same(paired["gk_y"].to_list(), paired["ky"].to_list())


def test_visible_area_vertices(sb_standard: Any) -> None:
    shots = sb_standard.shots
    area = sbc.row(shots, sbc.OPEN_SHOT)["visible_area"]
    assert _same(area, sbc.polygon(sbc.AREA_A01))  # 5 vertices: the closing one is kept
    shifted = sbc.polygon([v - FINE for v in sbc.AREA_A01])
    assert not _same(area, shifted)  # vertices are not cell-shifted
    assert _same(sbc.row(shots, "fake-shot-a04")["visible_area"], sbc.polygon(sbc.AREA_A04))
    assert sbc.row(shots, "fake-shot-a02")["visible_area"] is None  # no 360 frame for it
    assert shots.filter(pl.col("match_id") == "sb:900002")["visible_area"].is_null().all()


@pytest.mark.parametrize("case", ["bad_json_after_clean_gzip", "no_join"])
def test_unusable_360_skipped(tmp_path: Path, case: str) -> None:
    defect = 900004
    path = f"data/three-sixty/{defect}.json"
    extra = {path: b"[\x00\x00\x00]"} if case == "bad_json_after_clean_gzip" else {}
    broken = sbc.one_match(
        sbc.shot("fake-shot-d01"),
        match_id=defect,
        three_sixty=[
            sbc.visible_area("fake-not-an-event" if extra == {} else "fake-shot-d01", sbc.AREA_A01)
        ],
        extra_files=extra,
    )
    data = _load(tmp_path, sbc.match_a(), broken)
    assert data.skipped_360 == ("sb:900004",)
    has_360 = dict(zip(data.matches["match_id"], data.matches["has_360"], strict=True))
    assert has_360 == {"sb:900001": True, "sb:900004": False}
    assert sbc.row(data.shots, "fake-shot-d01")["visible_area"] is None
    assert sbc.row(data.shots, sbc.OPEN_SHOT)["visible_area"] is not None
