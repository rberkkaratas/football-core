"""Hand-written StatsBomb-shaped fixtures for the C4 loader tests (spec ``runs/MS2/c4`` §7).

Every id, name and number is invented (competition 9001, teams 101 / 202, players 1001-1011
and 2001-2011, event ids ``fake-...``); nothing is copied from an open-data file. The shapes
follow the open-data layout: ``data/matches/<competition>/<season>.json``,
``data/events/<match>.json``, ``data/lineups/<match>.json``, ``data/three-sixty/<match>.json``.

``write_cache`` lays out a checkout the way spec §3 describes it (gzip files plus ``tree.json``
with git blob ids), with the blob id computed here independently of the loader.
``fake_remote`` + ``FakeFetch`` stand in for GitHub, so no test uses the network.

Every match also carries Starting XI and Half Start / Half End events and a lineups file, so
the same fixture parses in kloppy (the dev-only oracle).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from football_core import pitch

Event = dict[str, Any]

COMMIT = "f00d" * 10  # obviously fake 40-hex pin
COMPETITION, SEASON = 9001, 9002
HOME, AWAY = 101, 202
PLAYER_BASE = {HOME: 1000, AWAY: 2000}  # squad numbers 1-11: player id = base + number
HOME_KEEPER, AWAY_KEEPER = 1001, 2001
FINE, COARSE = 0.05, 0.5
YARD = 0.9144
PERIOD_START = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
PERIOD_END = {1: 45, 2: 90, 3: 105, 4: 120, 5: 125}
# StatsBomb vocabulary ids (as in the open-data specification); unknown names get id 999.
# fmt: off
TYPE_IDS = {"Shot": 16, "Pass": 30, "Starting XI": 35, "Half Start": 18, "Half End": 34,
            "Own Goal Against": 20, "Own Goal For": 25}
SHOT_TYPE_IDS = {"Open Play": 87, "Penalty": 88, "Free Kick": 62, "Corner": 61, "Kick Off": 65}
OUTCOME_IDS = {"Blocked": 96, "Goal": 97, "Off T": 98, "Post": 99, "Saved": 100, "Wayward": 101,
               "Saved Off Target": 115, "Saved to Post": 116}
BODY_PART_IDS = {"Head": 37, "Left Foot": 38, "Right Foot": 40, "Other": 70}
TECHNIQUE_IDS = {"Backheel": 89, "Diving Header": 90, "Half Volley": 91, "Lob": 92, "Normal": 93,
                 "Overhead Kick": 94, "Volley": 95}
PLAY_PATTERN_IDS = {"Regular Play": 1, "From Corner": 2, "From Free Kick": 3, "From Throw In": 4,
                    "Other": 5, "From Counter": 6, "From Goal Kick": 7, "From Keeper": 8,
                    "From Kick Off": 9}
# fmt: on
TWO_VALUE_END = ("Blocked", "Wayward")


# --- Coordinates (the §2.3 rule, written with football_core.pitch as the reference map) -------


def metres(x: float, y: float, offset: float) -> tuple[float, float]:
    """A StatsBomb pitch point minus the half-cell ``offset``, in canonical metres."""
    mx, my = pitch.to_canonical([x - offset], [y - offset], "statsbomb")
    return float(mx[0]), float(my[0])


def metres_y(y: float) -> float:
    return metres(60.0, y, 0.0)[1]


def flipped(x: float, y: float, offset: float) -> tuple[float, float]:
    fx, fy = pitch.flip(*metres(x, y, offset))
    return float(fx), float(fy)


def polygon(flat: Sequence[float]) -> list[list[float]]:
    """A 360 ``visible_area`` flat list as metre vertices, unshifted, closing vertex kept."""
    xs, ys = np.asarray(flat[0::2]), np.asarray(flat[1::2])
    mx, my = pitch.to_canonical(xs, ys, "statsbomb")
    return [[float(a), float(b)] for a, b in zip(mx, my, strict=True)]


# --- Events -----------------------------------------------------------------------------------


def named(ids: Mapping[str, int], name: str) -> dict[str, Any]:
    return {"id": ids.get(name, 999), "name": name}


def team(team_id: int) -> dict[str, Any]:
    return {"id": team_id, "name": "Fixture Home FC" if team_id == HOME else "Fixture Away FC"}


def player(player_id: int) -> dict[str, Any]:
    return {"id": player_id, "name": f"Fixture Player {player_id}"}


def clock(period: int, minute: int, second: int, millis: int = 0) -> str:
    total = (minute - PERIOD_START[period]) * 60 + second
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}.{millis:03d}"


def event(
    eid: str,
    type_name: str,
    *,
    team_id: int = HOME,
    period: int = 1,
    minute: int = 10,
    second: int = 0,
    millis: int = 0,
    play_pattern: str = "Regular Play",
    player_id: int | None = None,
    location: Sequence[float] | None = None,
    **extra: Any,
) -> Event:
    ev: Event = {
        "id": eid,
        "index": 0,
        "period": period,
        "timestamp": clock(period, minute, second, millis),
        "minute": minute,
        "second": second,
        "type": named(TYPE_IDS, type_name),
        "possession": 1,
        "possession_team": team(team_id),
        "play_pattern": named(PLAY_PATTERN_IDS, play_pattern),
        "team": team(team_id),
        "duration": 0.0,
    }
    if player_id is not None:
        ev["player"] = player(player_id)
        ev["position"] = {"id": 23, "name": "Center Forward"}
    if location is not None:
        ev["location"] = list(location)
    ev.update(extra)
    return ev


def ff_player(
    player_id: int, location: Sequence[float], *, teammate: bool, keeper: bool = False
) -> dict[str, Any]:
    position = {"id": 1, "name": "Goalkeeper"} if keeper else {"id": 3, "name": "Right Center Back"}
    return {
        "location": list(location),
        "player": player(player_id),
        "position": position,
        "teammate": teammate,
    }


def default_frame(team_id: int, *, coarse: bool = False) -> list[dict[str, Any]]:
    """Opponent keeper, one opponent defender, one teammate (never the default shooter)."""
    opp = AWAY if team_id == HOME else HOME
    keeper, defender, mate = (
        ((119, 40), (112, 44), (100, 30))
        if coarse
        else ((118.25, 40.15), (110.15, 38.05), (108.45, 44.55))
    )
    return [
        ff_player(PLAYER_BASE[opp] + 1, keeper, teammate=False, keeper=True),
        ff_player(PLAYER_BASE[opp] + 3, defender, teammate=False),
        ff_player(PLAYER_BASE[team_id] + 10, mate, teammate=True),
    ]


DEFAULT_END = {
    "Goal": (120.0, 38.35, 0.85),
    "Saved": (118.15, 39.05, 0.55),
    "Saved to Post": (119.95, 43.95, 1.25),
    "Saved Off Target": (119.15, 45.05, 0.65),
    "Post": (120.0, 36.05, 2.35),
    "Off T": (120.0, 30.05, 2.55),
    "Blocked": (110.15, 38.05),
    "Wayward": (115.05, 60.05),
}


def shot(
    eid: str,
    *,
    team_id: int = HOME,
    player_id: int | None = None,
    location: Sequence[float] = (102.35, 35.65),
    outcome: str = "Saved",
    shot_type: str = "Open Play",
    play_pattern: str = "Regular Play",
    body_part: str = "Right Foot",
    technique: str = "Normal",
    end_location: Sequence[float] | None = None,
    freeze_frame: list[dict[str, Any]] | None | str = "default",
    key_pass_id: str | None = None,
    flags: Iterable[str] = (),
    xg: float = 0.12,
    period: int = 1,
    minute: int = 10,
    second: int = 0,
    millis: int = 0,
    **extra: Any,
) -> Event:
    """A Shot event; ``freeze_frame=None`` leaves the frame out, ``"default"`` adds one."""
    body: dict[str, Any] = {
        "statsbomb_xg": xg,
        "end_location": list(DEFAULT_END[outcome] if end_location is None else end_location),
        "type": named(SHOT_TYPE_IDS, shot_type),
        "outcome": named(OUTCOME_IDS, outcome),
        "technique": named(TECHNIQUE_IDS, technique),
        "body_part": named(BODY_PART_IDS, body_part),
    }
    if key_pass_id is not None:
        body["key_pass_id"] = key_pass_id
    for flag in flags:
        body[flag] = True
    if freeze_frame == "default":
        body["freeze_frame"] = default_frame(team_id)
    elif freeze_frame is not None:
        body["freeze_frame"] = freeze_frame
    shooter = player_id if player_id is not None else PLAYER_BASE[team_id] + 9
    return event(
        eid,
        "Shot",
        team_id=team_id,
        period=period,
        minute=minute,
        second=second,
        millis=millis,
        play_pattern=play_pattern,
        player_id=shooter,
        location=location,
        shot=body,
        **extra,
    )


def key_pass(
    eid: str,
    *,
    team_id: int = HOME,
    cross: bool = False,
    cut_back: bool = False,
    through: bool = False,
    minute: int = 10,
) -> Event:
    body: dict[str, Any] = {
        "end_location": [102.35, 35.65],
        "shot_assist": True,
        "recipient": player(PLAYER_BASE[team_id] + 9),
    }
    if cross:
        body["cross"] = True
    if cut_back:
        body["cut_back"] = True
    if through:
        body["technique"] = {"id": 108, "name": "Through Ball"}
    return event(
        eid,
        "Pass",
        team_id=team_id,
        minute=minute,
        player_id=PLAYER_BASE[team_id] + 10,
        location=(100.05, 50.05),
        **{"pass": body},
    )


def own_goal_against(
    eid: str,
    *,
    team_id: int,
    player_id: int,
    location: Sequence[float],
    play_pattern: str = "Regular Play",
    period: int = 2,
    minute: int = 70,
) -> Event:
    """Own Goal Against: ``team_id`` is the conceding side; location in its own frame."""
    return event(
        eid,
        "Own Goal Against",
        team_id=team_id,
        period=period,
        minute=minute,
        play_pattern=play_pattern,
        player_id=player_id,
        location=location,
    )


def own_goal_for(eid: str, *, team_id: int, period: int = 2, minute: int = 70) -> Event:
    return event(
        eid, "Own Goal For", team_id=team_id, period=period, minute=minute, location=(114.65, 49.75)
    )


def visible_area(event_uuid: str, flat: Sequence[float]) -> dict[str, Any]:
    return {
        "event_uuid": event_uuid,
        "visible_area": list(flat),
        "freeze_frame": [
            {"teammate": True, "actor": True, "keeper": False, "location": [100.0, 40.0]}
        ],
    }


# --- Matches ----------------------------------------------------------------------------------


@dataclass
class Match:
    """One fixture match. ``events`` are the core events; the frame events are added on write."""

    match_id: int
    events: list[Event]
    home_score: int = 0
    away_score: int = 0
    fidelity: str | None = "2"  # metadata.shot_fidelity_version; None = key absent
    three_sixty: list[dict[str, Any]] | None = None
    date: str = "2024-01-06"
    gender: str = "male"
    extra_files: dict[str, bytes] = field(default_factory=dict)


def entry(m: Match) -> dict[str, Any]:
    """The matches-file entry."""
    metadata = {"data_version": "1.1.0", "xy_fidelity_version": "2"}
    if m.fidelity is not None:
        metadata["shot_fidelity_version"] = m.fidelity
    return {
        "match_id": m.match_id,
        "match_date": m.date,
        "kick_off": "15:00:00.000",
        "competition": {
            "competition_id": COMPETITION,
            "country_name": "Nowhere",
            "competition_name": "Fixture League",
        },
        "season": {"season_id": SEASON, "season_name": "2023/2024"},
        "home_team": {
            "home_team_id": HOME,
            "home_team_name": "Fixture Home FC",
            "home_team_gender": m.gender,
            "home_team_group": None,
        },
        "away_team": {
            "away_team_id": AWAY,
            "away_team_name": "Fixture Away FC",
            "away_team_gender": m.gender,
            "away_team_group": None,
        },
        "home_score": m.home_score,
        "away_score": m.away_score,
        "match_status": "available",
        "match_status_360": "available" if m.three_sixty is not None else "unscheduled",
        "metadata": metadata,
        "match_week": 1,
    }


def _starting_xi(eid: str, team_id: int) -> Event:
    lineup = [
        {
            "player": player(PLAYER_BASE[team_id] + k),
            "position": {"id": 1, "name": "Goalkeeper"}
            if k == 1
            else {"id": 23, "name": "Center Forward"},
            "jersey_number": k,
        }
        for k in range(1, 12)
    ]
    return event(
        eid, "Starting XI", team_id=team_id, minute=0, tactics={"formation": 442, "lineup": lineup}
    )


def full_events(m: Match) -> list[Event]:
    """Starting XI, then per period Half Start, the core events, Half End; indices set in place."""
    periods = sorted({1, 2} | {e["period"] for e in m.events})
    out = [
        _starting_xi(f"fake-xi-{m.match_id}-h", HOME),
        _starting_xi(f"fake-xi-{m.match_id}-a", AWAY),
    ]
    for p in periods:
        start = PERIOD_START[p]
        out.append(event(f"fake-hs-{m.match_id}-{p}", "Half Start", period=p, minute=start))
        out += [e for e in m.events if e["period"] == p]
        out.append(event(f"fake-he-{m.match_id}-{p}", "Half End", period=p, minute=PERIOD_END[p]))
    for i, e in enumerate(out, start=1):
        e["index"] = i
    return out


def lineups() -> list[dict[str, Any]]:
    return [
        {
            "team_id": t,
            "team_name": team(t)["name"],
            "lineup": [
                {
                    "player_id": PLAYER_BASE[t] + k,
                    "player_name": f"Fixture Player {PLAYER_BASE[t] + k}",
                    "player_nickname": None,
                    "jersey_number": k,
                    "country": {"id": 1, "name": "Nowhere"},
                    "cards": [],
                    "positions": [],
                }
                for k in range(1, 12)
            ],
        }
        for t in (HOME, AWAY)
    ]


def dump(obj: object) -> bytes:
    return json.dumps(obj).encode("utf-8")


def repo_files(matches: Sequence[Match]) -> dict[str, bytes]:
    """Raw repo files (path -> bytes) for the matches, all in one competition-season."""
    files = {
        "data/competitions.json": dump(
            [
                {
                    "competition_id": COMPETITION,
                    "season_id": SEASON,
                    "country_name": "Nowhere",
                    "competition_name": "Fixture League",
                    "competition_gender": "male",
                    "season_name": "2023/2024",
                }
            ]
        ),
        f"data/matches/{COMPETITION}/{SEASON}.json": dump([entry(m) for m in matches]),
    }
    for m in matches:
        files[f"data/events/{m.match_id}.json"] = dump(full_events(m))
        files[f"data/lineups/{m.match_id}.json"] = dump(lineups())
        if m.three_sixty is not None:
            files[f"data/three-sixty/{m.match_id}.json"] = dump(m.three_sixty)
        files.update(m.extra_files)
    return files


def edit_json(files: dict[str, bytes], path: str, edit: Callable[[Any], Any]) -> None:
    """Decode one repo file, apply ``edit`` (in place, or return a replacement), re-encode."""
    obj = json.loads(files[path])
    result = edit(obj)
    files[path] = dump(obj if result is None else result)


# --- Cache (spec §3) --------------------------------------------------------------------------


def blob_sha1(data: bytes) -> str:
    """Git's blob id, written from git's object format (``blob <size>\\0<data>``)."""
    header = f"blob {len(data)}".encode("ascii") + b"\x00"
    return hashlib.sha1(header + data).hexdigest()


def gz(raw: bytes) -> bytes:
    return gzip.compress(raw, compresslevel=6, mtime=0)


def write_cache(cache: Path, files: Mapping[str, bytes], *, commit: str = COMMIT) -> Path:
    """A complete checkout ``<cache>/<commit>/``: one ``.gz`` per file plus ``tree.json``."""
    checkout = cache / commit
    for path, raw in files.items():
        target = checkout / f"{path}.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gz(raw))
    tree = {path: [blob_sha1(raw), len(raw)] for path, raw in sorted(files.items())}
    (checkout / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
    return checkout


def set_cached(checkout: Path, path: str, raw: bytes, *, gz_bytes: bytes | None = None) -> None:
    """Replace one cached file; ``tree.json`` records ``raw``'s blob id and size."""
    (checkout / f"{path}.gz").write_bytes(gz(raw) if gz_bytes is None else gz_bytes)
    tree_path = checkout / "tree.json"
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    tree[path] = [blob_sha1(raw), len(raw)]
    tree_path.write_text(json.dumps(tree), encoding="utf-8")


def zlib_error_gz(raw: bytes) -> bytes:
    """A gzip member with one flipped bit in its deflate stream that makes zlib raise."""
    good = gz(raw)
    for offset in range(10, len(good) - 8):
        for bit in range(8):
            bad = bytearray(good)
            bad[offset] ^= 1 << bit
            try:
                gzip.decompress(bytes(bad))
            except zlib.error:
                return bytes(bad)
            except (OSError, EOFError):
                continue
    raise AssertionError("no single-bit flip raises zlib.error")


def raw_url(path: str, commit: str = COMMIT) -> str:
    return f"https://raw.githubusercontent.com/statsbomb/open-data/{commit}/{path}"


def tree_url(commit: str = COMMIT) -> str:
    return f"https://api.github.com/repos/statsbomb/open-data/git/trees/{commit}?recursive=1"


def tree_listing(
    files: Mapping[str, bytes],
    *,
    commit: str = COMMIT,
    truncated: bool = False,
    extra_blobs: Iterable[str] = (),
) -> bytes:
    """The GitHub git-trees API response for ``files`` plus decoy entries."""
    entries: list[dict[str, Any]] = [
        {"path": "data", "mode": "040000", "type": "tree", "sha": "1" * 40},
        {"path": "data/events", "mode": "040000", "type": "tree", "sha": "2" * 40},
        {"path": "README.md", "mode": "100644", "type": "blob", "sha": "3" * 40, "size": 10},
    ]
    for path in extra_blobs:
        entries.append({"path": path, "mode": "100644", "type": "blob", "sha": "4" * 40, "size": 3})
    for path, raw in sorted(files.items()):
        entries.append(
            {
                "path": path,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha1(raw),
                "size": len(raw),
            }
        )
    return dump(
        {"sha": commit, "url": "https://example.invalid", "tree": entries, "truncated": truncated}
    )


def fake_remote(
    files: Mapping[str, bytes], *, commit: str = COMMIT, **listing: Any
) -> dict[str, bytes]:
    """URL -> body for the tree listing and every raw file."""
    remote = {tree_url(commit): tree_listing(files, commit=commit, **listing)}
    remote.update({raw_url(path, commit): raw for path, raw in files.items()})
    return remote


class FakeFetch:
    """A ``Fetch`` serving bytes from a dict; records every URL; ``fail`` maps URL -> error."""

    def __init__(
        self, remote: Mapping[str, bytes], fail: Mapping[str, BaseException] | None = None
    ) -> None:
        self.remote = dict(remote)
        self.fail = dict(fail or {})
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, url: str) -> bytes:
        with self._lock:
            self.calls.append(url)
        if url in self.fail:
            raise self.fail[url]
        if url not in self.remote:
            raise AssertionError(f"unexpected fetch {url}")
        return self.remote[url]


# --- The standard fixture ---------------------------------------------------------------------

A, B, C = 900001, 900002, 900003  # fine grid with 360 / coarse grid with a shoot-out / 0-0
OPEN_SHOT = "fake-shot-a01"  # the column-mapping shot
OPEN_PASS = "fake-pass-a01"
PENALTY = "fake-shot-a03"
OWN_GOAL_A = "fake-oga-a12"
OWN_GOAL_FOR_A = "fake-ogf-a13"
COARSE_SHOT = "fake-shot-b01"
COARSE_GOAL_LINE_SHOT = "fake-shot-b02"
OWN_GOAL_B = "fake-oga-b03"
SHOOTOUT_GOAL, SHOOTOUT_SAVE = "fake-shot-b04", "fake-shot-b05"
FINE_GOAL_LINE_SHOT = "fake-shot-a11"
AREA_A01 = (70.3, 5.7, 120.0, 5.7, 120.0, 74.1, 70.3, 74.1, 70.3, 5.7)
AREA_A04 = (80.55, 10.25, 119.9, 0.0, 119.9, 80.0, 80.55, 70.25, 80.55, 10.25)
AREA_A03 = (90.0, 10.0, 120.0, 10.0, 120.0, 70.0, 90.0, 10.0)
OPEN_SHOT_FRAME = [
    ff_player(AWAY_KEEPER, (118.25, 40.15), teammate=False, keeper=True),
    ff_player(HOME_KEEPER, (30.05, 40.05), teammate=True, keeper=True),  # shooter's own keeper
    ff_player(2004, (110.15, 38.05), teammate=False),
    ff_player(1010, (108.45, 44.55), teammate=True),
]


def match_a() -> Match:
    """Fine grid, metadata '2', 360 data; home 3 (incl. an away own goal) - away 1; every
    outcome, situation and direct set piece at least once."""
    # fmt: off
    events = [
        shot("fake-shot-a09", player_id=1011, location=(60.05, 40.05), outcome="Saved to Post",
             shot_type="Kick Off", play_pattern="From Kick Off", technique="Volley", minute=0,
             second=15),
        key_pass(OPEN_PASS, minute=23),
        shot(OPEN_SHOT, outcome="Goal", body_part="Left Foot", technique="Half Volley",
             key_pass_id=OPEN_PASS, flags=("first_time",), xg=0.31, minute=23, second=41,
             millis=250, freeze_frame=OPEN_SHOT_FRAME, under_pressure=True),
        shot("fake-shot-a02", team_id=AWAY, shot_type="Free Kick", play_pattern="From Free Kick",
             location=(95.25, 45.45), minute=30),
        shot(PENALTY, player_id=1007, location=(108.05, 40.05), outcome="Goal", shot_type="Penalty",
             freeze_frame=[ff_player(AWAY_KEEPER, (119.95, 40.05), teammate=False, keeper=True)],
             end_location=(120.0, 37.15, 0.35), xg=0.78, minute=40),
        shot("fake-shot-a04", player_id=1008, location=(113.35, 41.05), outcome="Blocked",
             play_pattern="From Corner", body_part="Head", technique="Diving Header", period=2,
             minute=52, second=10),
        shot("fake-shot-a05", team_id=AWAY, location=(99.45, 30.15), outcome="Off T",
             play_pattern="From Counter", period=2, minute=60, second=30),
        shot("fake-shot-a06", team_id=AWAY, player_id=2011, location=(104.55, 55.25),
             outcome="Wayward", play_pattern="From Throw In", body_part="Other", technique="Lob",
             period=2, minute=65),
        shot("fake-shot-a07", player_id=1006, location=(119.95, 79.95), outcome="Post",
             shot_type="Corner", play_pattern="From Corner", period=2, minute=70),
        shot("fake-shot-a08", team_id=AWAY, player_id=2008, location=(97.15, 22.35),
             outcome="Saved Off Target", play_pattern="From Free Kick", technique="Backheel",
             period=2, minute=75),
        shot("fake-shot-a10", team_id=AWAY, location=(114.15, 39.95), outcome="Goal",
             play_pattern="From Goal Kick", body_part="Head", end_location=(120.0, 41.05, 1.95),
             period=2, minute=80),
        shot(FINE_GOAL_LINE_SHOT, team_id=AWAY, player_id=2007, location=(120.0, 40.0),
             outcome="Off T", play_pattern="From Keeper", end_location=(120.0, 47.05, 0.15),
             period=2, minute=85),
        own_goal_against(OWN_GOAL_A, team_id=AWAY, player_id=2005, location=(5.35, 30.25),
                         play_pattern="From Corner", minute=88),
        own_goal_for(OWN_GOAL_FOR_A, team_id=HOME, minute=88),
    ]
    # fmt: on
    three_sixty = [
        visible_area(OPEN_SHOT, AREA_A01),
        visible_area(PENALTY, AREA_A03),
        visible_area("fake-shot-a04", AREA_A04),
        visible_area(OPEN_PASS, AREA_A04),  # a non-shot event: ignored
    ]
    return Match(A, events, home_score=3, away_score=1, three_sixty=three_sixty)


def match_b() -> Match:
    """Coarse grid, metadata '1', no 360; home 0 - away 1 (a home own goal); a shoot-out."""
    frame = default_frame(HOME, coarse=True)
    # fmt: off
    events = [
        shot(COARSE_SHOT, location=(109, 41), end_location=(110, 38.1, 1.0), freeze_frame=frame,
             minute=12),
        shot(COARSE_GOAL_LINE_SHOT, player_id=1008, location=(120, 40), outcome="Off T",
             end_location=(120, 44.0, 0.5), freeze_frame=frame, period=2, minute=50),
        own_goal_against(OWN_GOAL_B, team_id=HOME, player_id=1004, location=(5, 30), minute=70),
        shot(SHOOTOUT_GOAL, player_id=1007, location=(109, 41), outcome="Goal",
             shot_type="Penalty", end_location=(120, 37.2, 0.4),
             freeze_frame=[ff_player(AWAY_KEEPER, (120, 40), teammate=False, keeper=True)],
             period=5, minute=120, second=30),
        shot(SHOOTOUT_SAVE, team_id=AWAY, player_id=2007, location=(109, 41), shot_type="Penalty",
             end_location=(119, 41.2, 0.8), freeze_frame=None, period=5, minute=121, second=10),
    ]
    # fmt: on
    return Match(B, events, home_score=0, away_score=1, fidelity="1", date="2024-01-07")


def match_c() -> Match:
    """No shots, 0-0, women's match."""
    return Match(C, [], date="2024-01-13", gender="female")


def standard_matches() -> list[Match]:
    return [match_a(), match_b(), match_c()]


SHOT_ROWS = 17  # A: 11 Shot + 1 OGA; B: 4 Shot + 1 OGA (Own Goal For ignored)
FREEZE_FRAME_ROWS = 37  # A: 4 + 9 x 3 (penalty dropped); B: 2 x 3 (shoot-out dropped)


# The canonical matches table (MS3.matches spec §2), typed by hand in canonical order so the
# tests check the code against an independent table.
# fmt: off
MATCH_DTYPES = {
    "match_id": pl.String(), "provider": pl.Enum(("statsbomb", "whoscored")),
    "competition_id": pl.String(), "season_id": pl.String(), "match_date": pl.Date(),
    "home_team_id": pl.String(), "away_team_id": pl.String(), "competition_name": pl.String(),
    "season_name": pl.String(), "gender": pl.Enum(("male", "female")),
    "home_score": pl.Int16(), "away_score": pl.Int16(), "has_360": pl.Boolean(),
    "schema_version": pl.String(),
}
# fmt: on


def one_match(*events: Event, match_id: int = A, **kwargs: Any) -> Match:
    return Match(match_id, list(events), **kwargs)


def load_matches(cache: Path, matches: Sequence[Match]) -> Any:
    """Write ``matches`` to a fresh cache under ``cache`` and load it."""
    from football_core.io import statsbomb

    return statsbomb.load(write_cache(cache, repo_files(matches)))


def row(frame: Any, shot_id: str) -> dict[str, Any]:
    """The single output row of one fixture shot (reading results, not building fixtures)."""
    rows = frame.filter(pl.col("shot_id") == f"sb:{shot_id}").to_dicts()
    assert len(rows) == 1, (shot_id, len(rows))
    out: dict[str, Any] = rows[0]
    return out


def raw_qualifiers(ev: Event) -> str:
    """Spec §2.1: the whole event minus ``shot.freeze_frame``, canonical JSON."""
    copy = json.loads(json.dumps(ev))
    if "shot" in copy:
        copy["shot"].pop("freeze_frame", None)
    return json.dumps(copy, sort_keys=True, separators=(",", ":"), allow_nan=False)
