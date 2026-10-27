"""Hand-written WhoScored-shaped fixtures for the C5 loader tests (spec ``runs/MS4/whoscored``
§6).

Every id, name and number is invented: match ids 70001xx-70009xx, team ids 9101-9602, global
event ids 29010001xx-29060002xx, team names from the NATO alphabet. Nothing is copied from an
archive file. The layout follows the archive the spec describes:
``match_ids/<folder>_<season>.csv`` and ``events/<folder>/<season>/<DDMMYYYY>_<id>.csv`` with
``metadata/<DDMMYYYY>_<id>_metadata.json`` and a ``players/`` directory beside the files.

Cell text follows the archive's shapes: ``id`` / ``playerId`` / ``relatedEventId`` as ``"N.0"``,
``eventId`` / ``minute`` / ``teamId`` as ``"N"``, ``second`` as ``"12"`` in some files and
``"12.0"`` in others, qualifiers as a Python literal (single quotes). Every event file has the
real non-shot shape: ``PreMatch`` and ``PostGame`` rows, and non-shot rows with an empty
``playerId`` and ``second``. Global ids never equal event ids. Each team's ``FormationSet``
(MS5.goalmouth) carries q30 / q131 with the keeper in slot 1 (``keeper_of``) and the archive's
``second`` text ``"0.0"``.

``standard()`` is the archive every loader test starts from; tests mutate a fresh copy.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

# --- Vocabulary (spec §2.1, written independently of the module) ---------------------------

QUALIFIER_NAMES = {
    2: "Cross",
    4: "Throughball",
    9: "Penalty",
    15: "Head",
    20: "RightFoot",
    21: "OtherBodyPart",
    22: "RegularPlay",
    23: "FastBreak",
    24: "SetPiece",
    25: "FromCorner",
    26: "DirectFreekick",
    28: "OwnGoal",
    30: "InvolvedPlayers",
    55: "RelatedEventId",
    72: "LeftFoot",
    82: "Blocked",
    89: "OneOnOne",
    101: "SavedOffline",
    108: "Volley",
    131: "TeamPlayerFormation",
    145: "FormationSlot",
    160: "ThrowinSetPiece",
    212: "Length",
    214: "BigChance",
    215: "IndividualPlay",
    263: "DirectCorner",
    328: "FirstTouch",
    11112: "SyntheticWhoScoredTag",  # ids above 10,000 occur (spec §3.1: no upper bound)
}
SHOT_TYPES = ("Goal", "SavedShot", "ShotOnPost", "MissedShots")

# Layout A is the archive's most common header (no isOwnGoal); B reorders it and adds
# isOwnGoal; C reorders it again and lacks isGoal, isOwnGoal and cardType.
LAYOUT_A: tuple[str, ...] = (
    "id", "eventId", "minute", "second", "teamId", "x", "y", "expandedMinute", "period", "type",
    "outcomeType", "qualifiers", "satisfiedEventsTypes", "isTouch", "playerId", "endX", "endY",
    "relatedEventId", "relatedPlayerId", "blockedX", "blockedY", "goalMouthZ", "goalMouthY",
    "isShot", "isGoal", "cardType", "teamName",
)  # fmt: skip
LAYOUT_B: tuple[str, ...] = (
    "teamName", "type", "period", "minute", "second", "expandedMinute", "id", "eventId",
    "teamId", "playerId", "x", "y", "goalMouthY", "goalMouthZ", "blockedX", "blockedY",
    "endX", "endY", "relatedEventId", "relatedPlayerId", "outcomeType", "isTouch", "isShot",
    "isGoal", "isOwnGoal", "cardType", "satisfiedEventsTypes", "qualifiers",
)  # fmt: skip
LAYOUT_C: tuple[str, ...] = (
    "eventId", "id", "teamId", "teamName", "playerId", "period", "minute", "second",
    "expandedMinute", "type", "qualifiers", "x", "y", "blockedY", "blockedX", "goalMouthY",
    "goalMouthZ", "relatedEventId", "relatedPlayerId", "endY", "endX", "outcomeType",
    "isTouch", "isShot", "satisfiedEventsTypes",
)  # fmt: skip
REQUIRED = (
    "id", "eventId", "minute", "second", "teamId", "x", "y", "period", "type", "qualifiers",
    "playerId", "teamName", "relatedEventId", "goalMouthY", "goalMouthZ", "blockedX", "blockedY",
)  # fmt: skip
IDS_COLUMNS = (
    "match_id", "scraped", "competition_key", "competition_type", "source_stage_id",
    "competition_phase", "source_url", "validation_status", "validated_home_team",
    "validated_away_team",
)  # fmt: skip
FOLDER = "<folder>"  # competition_key sentinel: the match's own folder
GARBAGE_CSV = 'id,eventId\n"unclosed quote,12\n\x00\xff'  # opened, it fails the header check
GARBAGE_JSON = "{not json"


def q(qid: int, value: str | None = None) -> dict[str, Any]:
    """One qualifier as WhoScored writes it: ``{'type': {'value', 'displayName'}[, 'value']}``."""
    out: dict[str, Any] = {"type": {"value": qid, "displayName": QUALIFIER_NAMES[qid]}}
    if value is not None:
        out["value"] = value
    return out


def raw_json(quals: object) -> str:
    """The spec §2.2 ``raw_qualifiers`` text."""
    return json.dumps(quals, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


# --- Events and matches ----------------------------------------------------------------------


@dataclass
class Ev:
    """One event row. ``None`` is an empty cell; ``raw`` overrides any written cell text."""

    gid: int  # global Opta id (written "N.0")
    eid: int  # team-scoped eventId
    team: int
    type: str
    period: str = "FirstHalf"
    minute: int = 10
    second: int | None = 12
    x: str | None = "50.0"
    y: str | None = "50.0"
    player: int | None = None  # default: team * 10 + 1 on shots, empty on non-shots
    quals: list[dict[str, Any]] = field(default_factory=list)
    related: int | None = None
    mouth: tuple[str, str] | None = None  # goalMouthY, goalMouthZ
    blocked: tuple[str, str] | None = None  # blockedX, blockedY
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def is_shot(self) -> bool:
        return self.type in SHOT_TYPES

    @property
    def qids(self) -> set[int]:
        return {e["type"]["value"] for e in self.quals}


@dataclass
class Match:
    key: str
    mid: int
    folder: str
    season: str
    date: str  # DDMMYYYY, the file stem's date
    home: int
    away: int
    names: dict[int, str]
    events: list[Ev] = field(default_factory=list)
    status: str | None = "ok"
    scores: tuple[int, int] | None = None  # metadata (home, away); None: no metadata file
    meta_overrides: dict[str, Any] = field(default_factory=dict)
    meta_text: str | None = None
    file_text: str | None = None  # raw event-file text (an unparseable file)
    has_file: bool = True
    layout: tuple[str, ...] = LAYOUT_A
    second_suffix: str = ""  # "" writes "12", ".0" writes "12.0"
    validated: bool = True
    ids_file: str | None = None  # match_ids stem; default "<folder>_<season>"
    competition_key: str | None = FOLDER

    @property
    def stem(self) -> str:
        return f"{self.date}_{self.mid}"

    @property
    def iso(self) -> str:
        return f"{self.date[4:]}-{self.date[2:4]}-{self.date[:2]}"

    @property
    def ids_stem(self) -> str:
        return self.ids_file or f"{self.folder}_{self.season}"

    def event(self, gid: int) -> Ev:
        (found,) = [e for e in self.events if e.gid == gid]
        return found

    def event_path(self, root: Path) -> Path:
        return root / "events" / self.folder / self.season / f"{self.stem}.csv"

    def meta_path(self, root: Path) -> Path:
        return (
            root / "events" / self.folder / self.season / "metadata" / f"{self.stem}_metadata.json"
        )

    def metadata(self) -> dict[str, Any]:
        assert self.scores is not None
        meta: dict[str, Any] = {
            "match_id": str(self.mid),
            "start_time": f"{self.iso}T15:00:00",
            "competition_key": self.folder,
            "competition_type": "league",
            "source_stage_id": "1",
            "competition_phase": "regular",
            "phase_table_scope": "",
            "source_url": "https://example.invalid/match",
            "validation_status": "ok",
            "home_team_id": self.home,
            "home_team_name": self.names[self.home],
            "away_team_id": self.away,
            "away_team_name": self.names[self.away],
            "home_score": self.scores[0],
            "away_score": self.scores[1],
            "validated_home_team": self.names[self.home],
            "validated_away_team": self.names[self.away],
        }
        return meta | self.meta_overrides


@dataclass
class Fixture:
    matches: list[Match]
    orphans: list[tuple[str, str, str]] = field(default_factory=list)  # folder, season, stem
    extra_files: dict[str, str] = field(default_factory=dict)  # path under root -> text

    def match(self, key: str) -> Match:
        (found,) = [m for m in self.matches if m.key == key]
        return found


# --- Writer ----------------------------------------------------------------------------------


def _cells(ev: Ev, m: Match) -> dict[str, str]:
    player = ev.player if ev.player is not None else (ev.team * 10 + 1 if ev.is_shot else None)
    is_goal = ev.type == "Goal"
    cells = {
        "id": f"{ev.gid}.0",
        "eventId": str(ev.eid),
        "minute": str(ev.minute),
        "second": "" if ev.second is None else f"{ev.second}{m.second_suffix}",
        "teamId": str(ev.team),
        "x": ev.x or "",
        "y": ev.y or "",
        "expandedMinute": str(ev.minute),
        "period": ev.period,
        "type": ev.type,
        "outcomeType": "Successful",
        "qualifiers": repr(ev.quals),
        "satisfiedEventsTypes": "[]",
        "isTouch": "True",
        "playerId": "" if player is None else f"{player}.0",
        "endX": "",
        "endY": "",
        "relatedEventId": "" if ev.related is None else f"{ev.related}.0",
        "relatedPlayerId": "",
        "blockedX": ev.blocked[0] if ev.blocked else "",
        "blockedY": ev.blocked[1] if ev.blocked else "",
        "goalMouthY": ev.mouth[0] if ev.mouth else "",
        "goalMouthZ": ev.mouth[1] if ev.mouth else "",
        "isShot": "True" if ev.is_shot else "",
        "isGoal": "True" if is_goal else "",
        "isOwnGoal": "True" if 28 in ev.qids else "",
        "cardType": "",
        "teamName": m.names.get(ev.team, "Zulu Extra"),
    }
    return cells | ev.raw


def event_csv(m: Match) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(m.layout)
    for ev in m.events:
        cells = _cells(ev, m)
        writer.writerow([cells[c] for c in m.layout])
    return buf.getvalue()


def _ids_row(m: Match) -> list[str]:
    key = m.folder if m.competition_key == FOLDER else (m.competition_key or "")
    return [
        str(m.mid),
        "True" if m.has_file else "False",
        key,
        "league",
        "1",
        "",
        "https://example.invalid/match",
        m.status or "",
        m.names[m.home] if m.validated else "",
        m.names[m.away] if m.validated else "",
    ]


def write_archive(root: Path, fx: Fixture) -> Path:
    """Lay out ``fx`` under ``root`` (spec §3 ``load``); returns ``root``."""
    by_file: dict[str, list[list[str]]] = {}
    for m in fx.matches:
        by_file.setdefault(m.ids_stem, []).append(_ids_row(m))
    (root / "match_ids").mkdir(parents=True, exist_ok=True)
    for stem, rows in sorted(by_file.items()):
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(IDS_COLUMNS)
        writer.writerows(rows)
        (root / "match_ids" / f"{stem}.csv").write_text(buf.getvalue(), encoding="utf-8")
    events = root / "events"
    events.mkdir(exist_ok=True)
    (events / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    seasons = {(m.folder, m.season) for m in fx.matches if m.has_file}
    seasons |= {(f, s) for f, s, _ in fx.orphans}
    for folder, season in sorted(seasons):
        d = events / folder / season
        (d / "metadata").mkdir(parents=True, exist_ok=True)
        (d / "players").mkdir(exist_ok=True)
        (d / "players" / "squad.csv").write_text("playerId,name\n1,Nobody\n", encoding="utf-8")
        (d / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    for m in fx.matches:
        if not m.has_file:
            continue
        text = m.file_text if m.file_text is not None else event_csv(m)
        m.event_path(root).write_text(text, encoding="utf-8")
        if m.meta_text is not None:
            m.meta_path(root).write_text(m.meta_text, encoding="utf-8")
        elif m.scores is not None:
            m.meta_path(root).write_text(json.dumps(m.metadata()), encoding="utf-8")
    for folder, season, stem in fx.orphans:
        (events / folder / season / f"{stem}.csv").write_text(GARBAGE_CSV, encoding="utf-8")
    for rel, text in fx.extra_files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class Loaded(NamedTuple):
    root: Path
    archive: Any  # whoscored.Archive


def load_fixture(root: Path, fx: Fixture) -> Loaded:
    from football_core.io import whoscored

    write_archive(root, fx)
    return Loaded(root, whoscored.load(root))


# --- Event templates -------------------------------------------------------------------------


def _frame(
    base: int,
    t1: int,
    t2: int,
    first: Iterable[Ev] = (),
    second: Iterable[Ev] = (),
    *,
    extra_time: Iterable[Ev] = (),
    shootout: Iterable[Ev] = (),
    end_minute: int = 94,
    postgame_minute: int = 95,
) -> list[Ev]:
    """A match's rows in file order: ``t1`` acts first in every non-shot pair."""
    n = iter(range(base + 1, base + 100))

    def pair(eid: int, type_: str, period: str, minute: int) -> list[Ev]:
        return [
            Ev(next(n), eid, t, type_, period, minute, None, None, None, quals=[]) for t in (t1, t2)
        ]

    rows = [
        Ev(next(n), 1, t, "FormationSet", "PreMatch", 0, None, None, None,
           quals=lineup_quals(t), raw={"second": "0.0"})
        for t in (t1, t2)
    ]  # fmt: skip
    rows += pair(2, "Start", "FirstHalf", 0)
    rows += [*first, *pair(3, "End", "FirstHalf", 47), *pair(4, "Start", "SecondHalf", 45)]
    rows += [*second, *pair(5, "End", "SecondHalf", end_minute)]
    if extra_time:
        rows += pair(6, "Start", "FirstPeriodOfExtraTime", 90)
        rows += [*extra_time, *pair(7, "End", "SecondPeriodOfExtraTime", 121)]
    rows += [*shootout, *pair(8, "End", "PostGame", postgame_minute)]
    return rows


def keeper_of(team: int) -> int:
    """The fixture keeper of ``team`` (MS5.goalmouth): slot 1 of its FormationSet, and the
    player of MAIN's Save row."""
    return team * 10 + 1


def lineup_quals(team: int, keeper: int | None = None) -> list[dict[str, Any]]:
    """q30 InvolvedPlayers and q131 TeamPlayerFormation of a FormationSet (MS5.goalmouth §3.2):
    11 starters and 2 substitutes (slot 0). The keeper is listed third, so slot 1 is not the
    first entry; q30 is written with spaces after the commas and q131 without."""
    keeper = keeper_of(team) if keeper is None else keeper
    others = [team * 100 + k for k in range(2, 14)]
    players = [*others[:2], keeper, *others[2:]]
    slots = ["2", "3", "1", *(str(k) for k in range(4, 12)), "0", "0"]
    return [q(30, ", ".join(map(str, players))), q(131, ",".join(slots))]


def _pass(gid: int, eid: int, team: int, *quals: dict[str, Any], minute: int = 30) -> Ev:
    return Ev(gid, eid, team, "Pass", "FirstHalf", minute, 5, "60.0", "50.0", team * 10 + 2,
              quals=[q(212, "12.5"), *quals])  # fmt: skip


# --- The standard archive --------------------------------------------------------------------

# Match ids
MAIN, OG, ET, NAMES, SHORT = 7000101, 7000102, 7000103, 7000104, 7000105
DUP_OK, DUP_WRONG = 7000301, 7000302
PENDING, WRONG2, NULL_FILE, NULL_NOFILE = 7000401, 7000402, 7000403, 7000404
OK_NOFILE = (7000501, 7000502, 7000503, 7000504)
ORPHAN = 7000999

# MAIN (Championship 2025-26, metadata, layout A, second "12"): A acts first but B is home.
A, B = 9101, 9102
_M = 2_901_000_000
SAVED, BLOCKED, CLEARED, POST, OFF, OFF_BXY = (_M + i for i in range(101, 107))
GOAL, CROSS, THROUGH, OTHER_TEAM, NON_PASS, UNRESOLVED = (_M + i for i in range(107, 113))
DFK, DCORNER, PENALTY, NO_BODY = (_M + i for i in range(113, 117))
PASS_B24, CROSS_B26, THROUGH_A34, CROSS_A36, THROUGH_B36 = (_M + i for i in range(201, 206))
TOUCH_B28, RECOVERY_B90, DUPKEY_A50_1, DUPKEY_A50_2 = (_M + i for i in range(206, 210))
MAIN_SHOTS = (SAVED, BLOCKED, CLEARED, POST, OFF, OFF_BXY, GOAL, CROSS, THROUGH, OTHER_TEAM,
              NON_PASS, UNRESOLVED, DFK, DCORNER, PENALTY, NO_BODY)  # fmt: skip
MAIN_SCORE = (1, 2)  # home B: GOAL; away A: THROUGH + PENALTY

# OG (World Cup 2026, metadata, layout B, second "12.0"): home H scores an own goal and a
# goal (1-1), then wins a shoot-out kick that must not count.
H, V = 9201, 9202
_O = 2_902_000_000
OWN_GOAL, H_GOAL, V_SAVED, SO_GOAL, SO_MISS = (_O + i for i in range(101, 106))
OG_PASS = _O + 201
OG_SCORE = (1, 1)

# ET (World Cup 2026, metadata with the 90-minute score, no validated names): 1-1 after 90,
# K scores in extra time.
J, K = 9301, 9302
_E = 2_903_000_000
ET_J_GOAL, ET_K_GOAL, ET_K_ET_GOAL = (_E + i for i in range(101, 104))
ET_SCORE = (1, 1)

# NAMES (Belgian Pro League 2025-26, names route, layout C): L acts first, M is home.
L, MM = 9401, 9402
_N = 2_904_000_000
N_SHOT1, N_SHOT2 = _N + 101, _N + 102

# SHORT (Championship 2026-27, names route): last in-play event at minute 60.
NV, OS = 9501, 9502
SH_SHOT = 2_905_000_101

# DUP: one match, two copies (ok in Championship, wrong_competition in Belgium); same ids.
P, QB = 9601, 9602
DUP_SHOT1, DUP_SHOT2 = 2_906_000_101, 2_906_000_102

TEAM_NAMES = {
    A: "Alpha Town", B: "Bravo City", H: "Hotel United", V: "Victor Rovers",
    J: "Juliet FC", K: "Kilo Athletic", L: "Lima SK", MM: "Mike KV",
    NV: "November Albion", OS: "Oscar Wanderers", P: "Papa Borough", QB: "Quebec Athletic",
}  # fmt: skip
ALL_SHOTS = (*MAIN_SHOTS, OWN_GOAL, H_GOAL, V_SAVED, SO_GOAL, SO_MISS, ET_J_GOAL, ET_K_GOAL,
             ET_K_ET_GOAL, N_SHOT1, N_SHOT2, SH_SHOT, DUP_SHOT1, DUP_SHOT2)  # fmt: skip

# Expected partition and notes (spec §3 ``Archive.counts``, keys in this order).
COUNTS = {
    "wrong_competition": 2,  # DUP_WRONG, WRONG2
    "pending": 1,  # PENDING
    "unvalidated": 3,  # NULL_FILE, NULL_NOFILE, ORPHAN (no match_ids row)
    "no_event_file": 4,  # OK_NOFILE
    "unscored": 3,  # NAMES, SHORT, DUP_OK
    "extra_time": 1,  # ET
    "ended_before_85": 1,  # SHORT
}
KEPT = (MAIN, OG, ET, NAMES, SHORT, DUP_OK)
MATCH_IDS_ROWS = 15


def _names(*teams: int) -> dict[int, str]:
    return {t: TEAM_NAMES[t] for t in teams}


def main_match() -> Match:
    first = [
        Ev(SAVED, 20, B, "SavedShot", "FirstHalf", 5, 12, "83.0", "21.1",
           quals=[q(22), q(20), q(108), q(215)], mouth=("48.0", "10.0"),
           blocked=("98.5", "49.0")),
        Ev(BLOCKED, 30, A, "SavedShot", "FirstHalf", 8, 30, "90.0", "40.0",
           quals=[q(22), q(72), q(82)], mouth=("50.0", "12.0"), blocked=("88.9", "45.0")),
        Ev(POST, 31, A, "ShotOnPost", "FirstHalf", 20, 5, "100.0", "45.2",
           quals=[q(23), q(21)], mouth=("45.2", "38.0")),
        Ev(OFF, 22, B, "MissedShots", "FirstHalf", 25, 40, "75.0", "60.0",
           quals=[q(24), q(20), q(11112)], mouth=("60.0", "15.0")),
        Ev(OFF_BXY, 32, A, "MissedShots", "FirstHalf", 30, 0, "85.0", "30.0",
           quals=[q(160), q(72)], mouth=("40.0", "5.0"), blocked=("90.0", "35.0")),
        _pass(PASS_B24, 24, B, minute=35),
        Ev(GOAL, 23, B, "Goal", "FirstHalf", 35, 12, "91.5", "52.0",
           quals=[q(22), q(20), q(214), q(328), q(89), q(55, "24")], related=24,
           mouth=("50.0", "5.0")),
        Ev(DUPKEY_A50_1, 50, A, "BallRecovery", "FirstHalf", 40, 1, "20.0", "30.0", A * 10 + 3),
        Ev(DUPKEY_A50_2, 50, A, "Save", "FirstHalf", 40, 2, "1.0", "50.0", A * 10 + 1),
        Ev(CLEARED, 21, B, "SavedShot", "FirstHalf", 46, 30, "94.2", "50.0",
           quals=[q(25), q(15), q(82), q(101)], mouth=("49.0", "20.0"),
           blocked=("100.0", "50.0")),
    ]  # fmt: skip
    second = [
        _pass(CROSS_B26, 26, B, q(2), minute=50),
        Ev(CROSS, 25, B, "SavedShot", "SecondHalf", 50, 20, "93.0", "47.0",
           quals=[q(22), q(15), q(55, "26")], related=26, mouth=("51.0", "20.0"),
           blocked=("99.0", "51.0")),
        _pass(THROUGH_A34, 34, A, q(4), minute=54),
        Ev(THROUGH, 33, A, "Goal", "SecondHalf", 55, 0, "89.0", "55.0",
           quals=[q(22), q(20), q(55, "34")], related=34, mouth=("47.0", "8.0")),
        _pass(THROUGH_B36, 36, B, q(4), minute=58),
        _pass(CROSS_A36, 36, A, q(2), minute=60),
        Ev(OTHER_TEAM, 35, A, "SavedShot", "SecondHalf", 60, 10, "87.0", "40.0",
           quals=[q(22), q(20), q(55, "36")], related=36, mouth=("52.0", "9.0"),
           blocked=("98.0", "50.0")),
        Ev(TOUCH_B28, 28, B, "BallTouch", "SecondHalf", 64, 58, "70.0", "35.0", B * 10 + 4),
        Ev(NON_PASS, 27, B, "MissedShots", "SecondHalf", 65, 0, "83.0", "63.2",
           quals=[q(22), q(72), q(55, "28")], related=28, mouth=("30.0", "10.0")),
        Ev(RECOVERY_B90, 90, B, "BallRecovery", "SecondHalf", 70, 1, "30.0", "40.0", B * 10 + 5),
        Ev(UNRESOLVED, 37, A, "MissedShots", "SecondHalf", 70, 30, "84.0", "62.0",
           quals=[q(22), q(20), q(55, "90")], related=90, mouth=("58.0", "25.0")),
        Ev(DFK, 38, A, "SavedShot", "SecondHalf", 75, 45, "80.0", "55.0",
           quals=[q(26), q(20)], mouth=("52.0", "8.0"), blocked=("99.0", "52.0")),
        Ev(DCORNER, 29, B, "MissedShots", "SecondHalf", 78, 0, "99.6", "0.4",
           quals=[q(25), q(263), q(72)], mouth=("70.0", "20.0")),
        Ev(PENALTY, 39, A, "Goal", "SecondHalf", 82, 0, "88.5", "50.0",
           quals=[q(9), q(214), q(20)], mouth=("52.0", "10.0")),
        Ev(NO_BODY, 40, A, "MissedShots", "SecondHalf", 88, 0, "70.0", "50.0",
           quals=[q(22)]),
    ]  # fmt: skip
    return Match("main", MAIN, "Championship", "2025-2026", "16082025", B, A, _names(A, B),
                 _frame(_M, A, B, first, second), scores=MAIN_SCORE)  # fmt: skip


def og_match() -> Match:
    first = [
        _pass(OG_PASS, 41, H, q(2), minute=30),
        Ev(OWN_GOAL, 40, H, "Goal", "FirstHalf", 30, 12, "4.0", "52.0",
           quals=[q(22), q(72), q(28), q(55, "41")], related=41, mouth=("50.0", "5.0")),
    ]  # fmt: skip
    second = [
        Ev(H_GOAL, 42, H, "Goal", "SecondHalf", 70, 5, "92.0", "48.0",
           quals=[q(22), q(20)], mouth=("49.0", "15.0")),
        Ev(V_SAVED, 20, V, "SavedShot", "SecondHalf", 80, 0, "88.0", "45.0",
           quals=[q(22), q(15)], mouth=("50.0", "20.0"), blocked=("99.0", "50.0")),
    ]  # fmt: skip
    shootout = [
        Ev(SO_GOAL, 43, H, "Goal", "PenaltyShootout", 121, 0, "88.5", "50.0",
           quals=[q(9), q(214), q(20)], mouth=("47.0", "10.0")),
        Ev(SO_MISS, 21, V, "MissedShots", "PenaltyShootout", 121, 30, "88.5", "50.0",
           quals=[q(9), q(214), q(72)], mouth=("60.0", "40.0")),
    ]  # fmt: skip
    return Match("og", OG, "World_Cup_2026", "2025-2026", "14062026", H, V, _names(H, V),
                 _frame(_O, H, V, first, second, shootout=shootout, postgame_minute=122),
                 scores=OG_SCORE, layout=LAYOUT_B, second_suffix=".0")  # fmt: skip


def et_match() -> Match:
    first = [Ev(ET_J_GOAL, 20, J, "Goal", "FirstHalf", 20, 0, "90.0", "50.0",
                quals=[q(22), q(20)], mouth=("50.0", "10.0"))]  # fmt: skip
    second = [Ev(ET_K_GOAL, 20, K, "Goal", "SecondHalf", 80, 0, "92.0", "45.0",
                 quals=[q(22), q(72)], mouth=("48.0", "12.0"))]  # fmt: skip
    extra = [Ev(ET_K_ET_GOAL, 21, K, "Goal", "FirstPeriodOfExtraTime", 100, 0, "95.0", "50.0",
                quals=[q(22), q(15)], mouth=("51.0", "30.0"))]  # fmt: skip
    return Match("et", ET, "World_Cup_2026", "2025-2026", "04072026", J, K, _names(J, K),
                 _frame(_E, J, K, first, second, extra_time=extra, postgame_minute=122),
                 scores=ET_SCORE, second_suffix=".0", validated=False)  # fmt: skip


def names_match() -> Match:
    first = [Ev(N_SHOT1, 20, MM, "SavedShot", "FirstHalf", 15, 0, "88.0", "52.0",
                quals=[q(22), q(20)], mouth=("50.0", "10.0"),
                blocked=("99.0", "50.0"))]  # fmt: skip
    second = [Ev(N_SHOT2, 20, L, "MissedShots", "SecondHalf", 60, 0, "80.0", "40.0",
                 quals=[q(22), q(72)], mouth=("40.0", "20.0"))]  # fmt: skip
    return Match("names", NAMES, "Belgium_Pro_League", "2025-2026", "23082025", MM, L,
                 _names(L, MM), _frame(_N, L, MM, first, second), layout=LAYOUT_C,
                 second_suffix=".0")  # fmt: skip


def short_match() -> Match:
    second = [Ev(SH_SHOT, 20, OS, "MissedShots", "SecondHalf", 55, 0, "82.0", "50.0",
                 quals=[q(22), q(20)], mouth=("55.0", "20.0"))]  # fmt: skip
    # The PostGame row sits at minute 90: only in-play periods decide "ended before 85".
    return Match("short", SHORT, "Championship", "2026-2027", "15082026", NV, OS,
                 _names(NV, OS),
                 _frame(2_905_000_000, NV, OS, (), second, end_minute=60, postgame_minute=90),
                 layout=LAYOUT_B)  # fmt: skip


def dup_events() -> list[Ev]:
    first = [Ev(DUP_SHOT1, 20, P, "Goal", "FirstHalf", 10, 0, "90.0", "50.0",
                quals=[q(22), q(20)], mouth=("50.0", "10.0"))]  # fmt: skip
    second = [Ev(DUP_SHOT2, 20, QB, "SavedShot", "SecondHalf", 70, 0, "85.0", "45.0",
                 quals=[q(22), q(72)], mouth=("49.0", "11.0"),
                 blocked=("99.0", "49.0"))]  # fmt: skip
    return _frame(2_906_000_000, P, QB, first, second)


def dup_matches() -> list[Match]:
    """The same match twice: ``ok`` in Championship, ``wrong_competition`` in Belgium (whose
    folder sorts first, so a read-order rule would keep the wrong copy)."""
    ok = Match("dup_ok", DUP_OK, "Championship", "2025-2026", "20092025", P, QB, _names(P, QB),
               dup_events())  # fmt: skip
    wrong = Match("dup_wrong", DUP_WRONG, "Belgium_Pro_League", "2025-2026", "20092025", P, QB,
                  _names(P, QB), dup_events(), status="wrong_competition")  # fmt: skip
    return [ok, wrong]


def _row_only(key: str, mid: int, folder: str, season: str, status: str | None) -> Match:
    return Match(key, mid, folder, season, "01102025", 1, 2, {1: "Xray", 2: "Yankee"},
                 status=status, has_file=False)  # fmt: skip


def dropped_matches() -> list[Match]:
    """Non-``ok`` rows with unparseable files (never opened) and rows without a file."""
    pending = Match("pending", PENDING, "Championship", "2025-2026", "27092025", 1, 2,
                    {1: "Xray", 2: "Yankee"}, status="pending", file_text=GARBAGE_CSV,
                    meta_text=GARBAGE_JSON)  # fmt: skip
    wrong2 = Match("wrong2", WRONG2, "Championship", "2026-2027", "22082026", 1, 2,
                   {1: "Xray", 2: "Yankee"}, status="wrong_competition", file_text=GARBAGE_CSV,
                   meta_text=GARBAGE_JSON)  # fmt: skip
    null_file = Match("null_file", NULL_FILE, "Belgium_Pro_League", "2025-2026", "30082025", 1,
                      2, {1: "Xray", 2: "Yankee"}, status=None, file_text=GARBAGE_CSV,
                      validated=False)  # fmt: skip
    rows = [
        _row_only("null_nofile", NULL_NOFILE, "Belgium_Pro_League", "2025-2026", None),
        _row_only("ok_nofile1", OK_NOFILE[0], "Championship", "2025-2026", "ok"),
        _row_only("ok_nofile2", OK_NOFILE[1], "Championship", "2026-2027", "ok"),
        _row_only("ok_nofile3", OK_NOFILE[2], "World_Cup_2026", "2025-2026", "ok"),
        _row_only("ok_nofile4", OK_NOFILE[3], "Belgium_Pro_League", "2025-2026", "ok"),
    ]
    rows[0].validated = False
    return [pending, wrong2, null_file, *rows]


def standard() -> Fixture:
    """A fresh copy of the standard archive (spec §6 fixture list)."""
    matches = [main_match(), og_match(), et_match(), names_match(), short_match()]
    return Fixture(
        [*matches, *dup_matches(), *dropped_matches()],
        orphans=[("Championship", "2025-2026", f"04102025_{ORPHAN}")],
    )


def mini(layout: Sequence[str] = LAYOUT_A, second_suffix: str = "") -> Fixture:
    """MAIN alone: one kept match with metadata, for layout and coordinate properties."""
    m = main_match()
    m.layout = tuple(layout)
    m.second_suffix = second_suffix
    return Fixture([m])


def expected_files(root: Path) -> set[str]:
    """Spec §1.9 / ``Archive.files``: every match_ids CSV, kept event CSVs and kept metadata."""
    stems = {
        "Championship_2025-2026",
        "Championship_2026-2027",
        "Belgium_Pro_League_2025-2026",
        "World_Cup_2026_2025-2026",
    }
    files = {f"match_ids/{s}.csv" for s in stems}
    kept = [m for m in standard().matches if m.mid in KEPT]
    files |= {m.event_path(root).relative_to(root).as_posix() for m in kept}
    files |= {m.meta_path(root).relative_to(root).as_posix() for m in kept if m.scores}
    return files
