"""WhoScored (Opta) CSV archive: a private loader into the C3 shots and canonical matches tables.

**Private data.** The archive is scraped WhoScored data. It is read in place and never copied,
committed or published: the snapshot lives under the git-ignored ``data/`` folder, its manifest
records counts and a digest but no file list (``list_files=False``), and model cards and docs
carry counts and metrics only. The archive layout::

    <root>/match_ids/<folder>_<season>.csv          one row per match, with validation_status
    <root>/events/<folder>/<season>/DDMMYYYY_<id>.csv
    <root>/events/<folder>/<season>/metadata/DDMMYYYY_<id>_metadata.json

**Keep rule, status first.** A match is kept exactly when its ``match_ids`` status is ``ok``
and it has an event file; every other row is dropped under one reason (``DROP_REASONS``), and
dropped files are never opened. This keeps the duplicated Championship match once, whatever
the read order: its wrong copy carries ``wrong_competition``. Home and away are never
inferred: they come from the metadata team ids, else from the validated team names.

**Scores.** Only metadata carries a score. WhoScored's score is the 90-minute score, so in an
extra-time match it is checked against the goals of periods 1-2 and then nulled (C3 stores
the final score without the shoot-out). ``build_snapshot`` reconciles every scored match.

**Shots.** ``shot_id`` is ``"ws:"`` plus the global Opta ``id`` (``eventId`` is numbered per
team). Raw 0-100 coordinates map to metres through the ``football_core.pitch`` landmarks.
Columns come from Opta qualifier ids, parsed once per shot (and per q55 target) from the
``qualifiers`` Python literal. Own goals (q28) keep the conceding player's team, are flipped
so the goal entered is at x = 105, and have their context nulled. ``big_chance`` (q214) is an
analyst's post-hoc label, kept with the ``restricted`` role: every penalty carries it, so any
ablation drops penalties first. Opta q328 / q89 are not StatsBomb's ``first_time`` /
``one_on_one``: on men's big-five league shots their shares are 0.511 vs 0.315 and 0.019 vs
0.055. Columns Opta cannot fill (keeper position, technique, open goal, deflection, end
height, provider xG) are null. Penalties, shoot-out kicks and own goals are all kept: choosing
populations is the models' job.

**Goal mouth.** ``goalMouthY`` / ``goalMouthZ`` are already the goal-plane point (every save
lies in the frame), so they are mapped, never projected. Opta draws the woodwork as bands
(posts in goalMouthY [44.2, 45.2] and [54.8, 55.8], the bar in goalMouthZ (38, 42]); the
``football_core.pitch`` goal-mouth landmarks map each band onto the 12 cm of woodwork.

**Facing keeper.** ``gk_id`` is the defending team's goalkeeper from the lineup state. Event
files are not in clock order (formation rows can come last), so the lineup and shot rows are
sorted by period, minute, second and ``id`` first. ``FormationSet`` and ``FormationChange``
name the keeper as the one player in formation slot 1 (q30 player ids paired with q131
slots); a ``SubstitutionOn`` into slot 1 (q145) hands the role on. Own goals get null.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, TypedDict

import numpy as np
import polars as pl

from football_core import pitch, schema
from football_core.io import snapshot
from football_core.io.snapshot import BuildRequest, BuildResult, StrPath

Qualifier = dict[str, Any]
Target = tuple[str | None, str | None, str | None]  # global id, type, qualifiers text


@dataclass(frozen=True)
class Competition:
    """One competition folder of the archive.

    Attributes
    ----------
    name : str
        ``competition_name`` (StatsBomb's spelling where the competition is the same).
    gender : {"male", "female"}
        ``gender`` of every match.
    calendar_year : bool
        True when the season is one calendar year (``season_name`` is the second year).
    """

    name: str
    gender: Literal["male", "female"]
    calendar_year: bool


ARCHIVE_URI: Final = "whoscored-csv-archive"
DROP_REASONS: Final = ("wrong_competition", "pending", "unvalidated", "no_event_file")
SCORE_NOTES: Final = ("unscored", "extra_time")
SHORT_NOTE: Final = "ended_before_85"
# fmt: off
PERIODS: Final[Mapping[str, int]] = MappingProxyType({
    "FirstHalf": 1, "SecondHalf": 2, "FirstPeriodOfExtraTime": 3,
    "SecondPeriodOfExtraTime": 4, "PenaltyShootout": 5,
})
NON_PLAY_PERIODS: Final = ("PreMatch", "PostGame")  # legal on non-shot rows only
EXTRA_TIME: Final = ("FirstPeriodOfExtraTime", "SecondPeriodOfExtraTime")
SHORT_MATCH_MINUTE: Final = 85
PERIOD_RANK: Final[Mapping[str, int]] = MappingProxyType({
    "PreMatch": 0, **PERIODS, "PostGame": 6,
})
LINEUP_TYPES: Final = ("FormationSet", "FormationChange", "SubstitutionOn")
INT_TEXT: Final = r"[0-9]+(\.0)?"  # ids, eventId, minute, second (fullmatch)
COORD_TEXT: Final = r"-?[0-9]+(\.[0-9]+)?"  # x, y, goal mouth, blocked point (fullmatch)
SHOT_TYPES: Final[Mapping[str, str]] = MappingProxyType({
    "Goal": "goal", "SavedShot": "saved", "ShotOnPost": "post", "MissedShots": "off_target",
})
BODY_PARTS: Final[Mapping[int, str]] = MappingProxyType(
    {20: "right_foot", 72: "left_foot", 15: "head", 21: "other"}
)
PATTERNS: Final[Mapping[int, str]] = MappingProxyType({
    22: "open_play", 23: "fast_break", 24: "from_free_kick", 25: "from_corner",
    26: "direct_free_kick", 160: "from_throw_in", 9: "penalty",
})
DIRECT_CORNER: Final = 263
BLOCKED: Final = 82
OWN_GOAL: Final = 28
BIG_CHANCE: Final = 214
FIRST_TOUCH: Final = 328
ONE_ON_ONE: Final = 89
RELATED: Final = 55
CROSS: Final = 2
THROUGH_BALL: Final = 4
INVOLVED_PLAYERS: Final = 30
FORMATION_SLOTS: Final = 131
FORMATION_SLOT: Final = 145
KEEPER_SLOT: Final = "1"
STATUSES: Final[Mapping[str, str | None]] = MappingProxyType({
    "ok": None, "wrong_competition": "wrong_competition", "pending": "pending",
    "": "unvalidated",
})
COMPETITIONS: Final[Mapping[str, Competition]] = MappingProxyType({
    folder: Competition(name, "male", folder == "World_Cup_2026") for folder, name in (
        ("Belgium_Pro_League", "Belgian Pro League"), ("Bundesliga", "1. Bundesliga"),
        ("Championship", "Championship"), ("Eredivisie", "Eredivisie"), ("La_Liga", "La Liga"),
        ("Ligue_1", "Ligue 1"), ("Premier_League", "Premier League"),
        ("Primeira_Liga", "Primeira Liga"), ("Serie_A", "Serie A"), ("Super_Lig", "Super Lig"),
        ("World_Cup_2026", "FIFA World Cup"),
    )
})
EVENT_COLUMNS: Final = (
    "id", "eventId", "minute", "second", "teamId", "x", "y", "period", "type", "qualifiers",
    "playerId", "teamName", "relatedEventId", "goalMouthY", "goalMouthZ", "blockedX", "blockedY",
)
MATCH_ID_COLUMNS: Final = (
    "match_id", "validation_status", "competition_key", "validated_home_team",
    "validated_away_team",
)
# fmt: on
_COUNT_KEYS: Final = (*DROP_REASONS, *SCORE_NOTES, SHORT_NOTE)
_OPTIONAL_COORDS: Final = ("goalMouthY", "goalMouthZ", "blockedX", "blockedY")
_EVENT_FILE: Final = re.compile(r"([0-9]{2})([0-9]{2})([0-9]{4})_([0-9]+)\.csv")
_META_FILE: Final = re.compile(r"[0-9]{8}_([0-9]+)_metadata\.json")
_SEASON: Final = re.compile(r"([0-9]{4})-([0-9]{4})")
_NORM: Final = r"\.0$"  # applied only after INT_TEXT has matched

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Archive:
    """What :func:`load` returns.

    Attributes
    ----------
    shots, freeze_frames, matches : polars.DataFrame
        The validated C3 tables; ``freeze_frames`` is always empty (Opta has none).
    counts : mapping of str to int
        Read-only; the keys ``DROP_REASONS``, ``SCORE_NOTES`` and ``SHORT_NOTE`` in that
        order, each present (0 when nothing was counted).
    files : tuple of pathlib.Path
        Every file opened (all ``match_ids`` CSVs, kept event CSVs and their metadata), sorted.
    """

    shots: pl.DataFrame
    freeze_frames: pl.DataFrame
    matches: pl.DataFrame
    counts: Mapping[str, int]
    files: tuple[Path, ...]


@dataclass(frozen=True)
class _Listed:
    source: str  # stem of the match_ids file
    status: str
    competition_key: str | None
    home_name: str | None
    away_name: str | None


@dataclass(frozen=True)
class _EventFile:
    path: Path
    folder: str
    season: str
    match_date: date


class _MatchRow(TypedDict):
    """A kept match's ``matches`` columns before the score (typed, so mypy checks the shots'
    match columns between parsing and ``cast_to_schema``)."""

    match_id: str
    provider: str
    competition_id: str
    season_id: str
    match_date: date
    home_team_id: str
    away_team_id: str
    competition_name: str
    season_name: str
    gender: str
    has_360: bool | None
    schema_version: str


# --- Reading ----------------------------------------------------------------------------------


def _norm(text: str) -> str:
    return text[:-2] if text.endswith(".0") else text


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_constant(name: str) -> object:
    raise ValueError(f"JSON constant {name} is not allowed")


def _read_csv(path: Path, columns: Sequence[str]) -> pl.DataFrame:
    """The named columns as String; the header is read first, so polars never renames one."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f), [])
    except (UnicodeDecodeError, csv.Error) as e:
        raise ValueError(f"cannot read the header of '{path}': {e}") from None
    dupes = sorted({c for c in header if header.count(c) > 1})
    missing = [c for c in columns if c not in header]
    if dupes or missing:
        raise ValueError(f"'{path}': duplicated column(s) {dupes}, missing column(s) {missing}")
    try:
        return pl.read_csv(path, columns=list(columns), infer_schema=False)
    except pl.exceptions.PolarsError as e:
        raise ValueError(f"cannot read '{path}': {e}") from None


def _match_ids(folder: Path) -> tuple[dict[str, _Listed], list[Path]]:
    if not folder.is_dir():
        raise FileNotFoundError(f"no match_ids directory at '{folder}'")
    paths = sorted(p for p in folder.iterdir() if p.suffix == ".csv" and p.is_file())
    if not paths:
        raise FileNotFoundError(f"'{folder}' holds no .csv file")
    listed: dict[str, _Listed] = {}
    for path in paths:
        for mid, status, key, home, away in _read_csv(path, MATCH_ID_COLUMNS).iter_rows():
            if mid is None or re.fullmatch(INT_TEXT, mid) is None:
                raise ValueError(f"'{path}': match_id {mid!r} does not match {INT_TEXT}")
            status = "" if status is None else status
            if status not in STATUSES:
                allowed = ", ".join(repr(s) for s in STATUSES)
                raise ValueError(f"'{path}': match {mid} status {status!r}; allowed: {allowed}")
            if _norm(mid) in listed:
                first = listed[_norm(mid)].source
                raise ValueError(f"match_id {mid} is in two match_ids rows ({first}, {path.stem})")
            listed[_norm(mid)] = _Listed(path.stem, status, key, home, away)
    return listed, paths


def _event_files(events: Path, listed: Mapping[str, _Listed]) -> dict[str, _EventFile]:
    """Every event file by match id, with the location checks (dropped files included)."""
    found: dict[str, _EventFile] = {}
    for comp in sorted(p for p in events.iterdir() if p.is_dir()):
        if comp.name not in COMPETITIONS:
            known = ", ".join(COMPETITIONS)
            raise ValueError(f"unknown competition folder '{comp}'; known: {known}")
        for season in sorted(p for p in comp.iterdir() if p.is_dir()):
            years = _SEASON.fullmatch(season.name)
            if years is None or int(years[2]) != int(years[1]) + 1:
                raise ValueError(f"season folder '{season}' is not YYYY-(YYYY+1)")
            for path in sorted(p for p in season.iterdir() if p.suffix == ".csv" and p.is_file()):
                stem = _EVENT_FILE.fullmatch(path.name)
                if stem is None:
                    raise ValueError(f"event file '{path}' is not named DDMMYYYY_<match id>.csv")
                try:
                    day = date(int(stem[3]), int(stem[2]), int(stem[1]))
                except ValueError:
                    raise ValueError(
                        f"event file '{path}': {path.name[:8]} is not a date"
                    ) from None
                if day.year not in (int(years[1]), int(years[2])):  # season_name comes from it
                    raise ValueError(
                        f"event file '{path}': its date {day} is outside its season folder "
                        f"'{season.name}'"
                    )
                mid = stem[4]
                if mid in found:
                    raise ValueError(
                        f"match {mid} has two event files: '{found[mid].path}', '{path}'"
                    )
                row = listed.get(mid)
                if row is not None and row.source != f"{comp.name}_{season.name}":
                    raise ValueError(f"'{path}': its match_ids row is in {row.source}.csv")
                if row is not None and row.competition_key not in (None, comp.name):
                    raise ValueError(f"'{path}': match_ids competition_key {row.competition_key!r}")
                found[mid] = _EventFile(path, comp.name, season.name, day)
    return found


def _metadata_index(season: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    folder = season / "metadata"
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            name = _META_FILE.fullmatch(path.name)
            if name is not None:
                if name[1] in index:
                    raise ValueError(f"two metadata files for match {name[1]} in '{folder}'")
                index[name[1]] = path
    return index


def _metadata(
    path: Path, mid: str, event: _EventFile, teams: set[str]
) -> tuple[str, str, int | None, int | None]:
    """Home id, away id and the two scores (both None, or both ints in [0, 99])."""
    try:
        meta = json.loads(path.read_bytes(), parse_constant=_reject_constant)
    except (ValueError, RecursionError) as e:
        raise ValueError(f"cannot read metadata '{path}': {e}") from None
    if not isinstance(meta, dict):
        raise ValueError(f"metadata '{path}': the top level is not an object")
    home, away = meta.get("home_team_id"), meta.get("away_team_id")
    start, key = meta.get("start_time"), meta.get("competition_key")
    scores: tuple[Any, Any] = (meta.get("home_score"), meta.get("away_score"))
    if meta.get("match_id") != mid:
        problem = f"match_id {meta.get('match_id')!r} is not the string {mid!r}"
    elif not (_is_int(home) and _is_int(away)) or {str(home), str(away)} != teams:
        problem = f"team ids {home!r}, {away!r} are not the event file's {sorted(teams)}"
    elif not isinstance(start, str) or start[:10] != event.match_date.isoformat():
        problem = f"start_time {start!r} disagrees with the file date {event.match_date}"
    elif key is not None and key != event.folder:
        problem = f"competition_key {key!r} is not the folder {event.folder!r}"
    elif scores != (None, None) and not all(_is_int(s) and 0 <= s <= 99 for s in scores):
        problem = f"scores {scores!r} are not two integers in [0, 99] (or both null)"
    else:
        return str(home), str(away), scores[0], scores[1]
    raise ValueError(f"metadata '{path}': {problem}")


def _check_text(df: pl.DataFrame, column: str, pattern: str, where: str, *, empty: bool) -> None:
    """Every cell of ``column`` fullmatches ``pattern`` (an empty cell passes iff ``empty``)."""
    ok = df[column].str.contains(f"^(?:{pattern})$").fill_null(empty)
    if not ok.all():
        i = int((~ok).arg_true()[0])
        raise ValueError(
            f"{where}: event {df['id'][i]!r}: {column} {df[column][i]!r} does not match {pattern}"
        )


def _plain(value: object) -> bool:
    """Only list, dict with str keys, str, int (not bool), finite float and None."""
    if value is None or isinstance(value, str) or _is_int(value):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_plain(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _plain(v) for k, v in value.items())
    return False


def _is_qualifier(item: object) -> bool:
    kind = item.get("type") if isinstance(item, dict) else None
    return (
        isinstance(kind, dict)
        and _is_int(kind.get("value"))
        and kind["value"] >= 0
        and isinstance(kind.get("displayName"), str)
    )


def _qualifiers(text: str | None, where: str) -> list[Qualifier]:
    try:
        value = ast.literal_eval("" if text is None else text)
        plain = _plain(value)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as e:
        raise ValueError(
            f"{where}: qualifiers are not a Python literal ({type(e).__name__})"
        ) from None
    if not plain or not isinstance(value, list) or not all(_is_qualifier(q) for q in value):
        raise ValueError(
            f"{where}: qualifiers {str(text)[:60]!r} are not a list of "
            "{'type': {'value': int >= 0, 'displayName': str}, ...} of plain values"
        )
    return value


# --- Mapping ----------------------------------------------------------------------------------


def _assist(found: list[Target], where: str) -> tuple[str | None, str | None]:
    """assist_type and assist_event_id from the events a q55 resolves to."""
    if not found:
        return None, None  # cannot tell
    for gid, kind, _ in found:
        if gid is None or re.fullmatch(INT_TEXT, gid) is None or kind is None:
            raise ValueError(f"{where}: q55 target id {gid!r} / type {kind!r} is malformed")
    if len({_norm(str(gid)) for gid, _, _ in found}) > 1:
        raise ValueError(
            f"{where}: q55 resolves to two events: {sorted({str(g) for g, _, _ in found})}"
        )
    gid, kind, text = found[0]
    ids = {q["type"]["value"] for q in _qualifiers(text, f"{where} (q55 target {gid})")}
    if kind != "Pass":
        return "none", None
    assist = "through_ball" if THROUGH_BALL in ids else "cross" if CROSS in ids else "pass"
    return assist, f"ws:{_norm(str(gid))}"


def _lineup_keeper(r: Mapping[str, Any], at: str) -> str | None:
    """The keeper a lineup row names (normalised id), or None when it names none."""
    q = {item["type"]["value"]: item.get("value") for item in _qualifiers(r["qualifiers"], at)}
    if r["type"] == "SubstitutionOn":
        return r["playerId"] if str(q.get(FORMATION_SLOT) or "").strip() == KEEPER_SLOT else None
    players = [p.strip() for p in str(q.get(INVOLVED_PLAYERS) or "").split(",")]
    slots = [p.strip() for p in str(q.get(FORMATION_SLOTS) or "").split(",")]
    keepers = [p for p, slot in zip(players, slots, strict=False) if slot == KEEPER_SLOT]
    if len(players) != len(slots) or len(keepers) != 1:
        raise ValueError(
            f"{at}: q{INVOLVED_PLAYERS} lists {len(players)} player(s) and q{FORMATION_SLOTS} "
            f"{len(slots)} slot(s), {len(keepers)} in slot {KEEPER_SLOT}; expected equal "
            "lengths and exactly one keeper"
        )
    if re.fullmatch(INT_TEXT, keepers[0]) is None:
        raise ValueError(f"{at}: keeper id {keepers[0]!r} does not match {INT_TEXT}")
    return _norm(keepers[0])


def _facing_keepers(df: pl.DataFrame, where: str) -> dict[str, str | None]:
    """Shot id -> the defending team's keeper (None before any lineup), in clock order."""
    rows = df.filter(pl.col("type").is_in([*LINEUP_TYPES, *SHOT_TYPES]))
    lineup = rows.filter(pl.col("type").is_in(LINEUP_TYPES))
    for column in ("id", "minute", "second"):
        _check_text(lineup, column, INT_TEXT, where, empty=False)
    subs = lineup.filter(pl.col("type") == "SubstitutionOn")
    _check_text(subs, "playerId", INT_TEXT, where, empty=False)
    rows = rows.with_columns(
        pl.col("id", "playerId").str.replace(_NORM, ""),
        pl.col("id", "minute", "second").str.replace(_NORM, "").cast(pl.Int64).name.prefix("_"),
        pl.col("period").replace_strict(PERIOD_RANK, default=None).alias("_rank"),
    )
    if (bad := rows.filter(pl.col("_rank").is_null())).height:
        raise ValueError(f"{where}: event {bad['id'][0]!r}: period {bad['period'][0]!r}")
    keeper: dict[str, str] = {}
    faced: dict[str, str | None] = {}
    for r in rows.sort("_rank", "_minute", "_second", "_id").iter_rows(named=True):
        team = r["teamId"]
        if r["type"] in SHOT_TYPES:
            other = next((t for t in keeper if t != team), None)
            faced[r["id"]] = None if other is None else keeper[other]
        elif (named := _lineup_keeper(r, f"{where}: event {r['id']}")) is not None:
            keeper[team] = named
    return faced


def _shot_rows(df: pl.DataFrame, where: str) -> pl.DataFrame:
    """The shot rows, every cell checked, ids normalised, numbers cast."""
    shots = df.filter(pl.col("type").is_in(list(SHOT_TYPES)))
    for column in ("id", "eventId", "playerId", "minute", "second", "relatedEventId"):
        _check_text(shots, column, INT_TEXT, where, empty=column == "relatedEventId")
    for column in ("x", "y", *_OPTIONAL_COORDS):
        _check_text(shots, column, COORD_TEXT, where, empty=column in _OPTIONAL_COORDS)
    bad = shots.filter(~pl.col("period").is_in(list(PERIODS)))
    if bad.height:
        raise ValueError(f"{where}: event {bad['id'][0]!r}: shot period {bad['period'][0]!r}")
    shots = shots.with_columns(
        pl.col("id", "eventId", "playerId", "relatedEventId").str.replace(_NORM, ""),
        pl.col("minute", "second").str.replace(_NORM, "").cast(pl.Int64),
        pl.col("x", "y", *_OPTIONAL_COORDS).cast(pl.Float64),
    )
    for column in ("x", "y", *_OPTIONAL_COORDS):  # on the raw value: pitch would clip
        out = shots.filter(~pl.col(column).is_between(0.0, 100.0).fill_null(True))
        if out.height:
            raise ValueError(
                f"{where}: event {out['id'][0]!r}: {column} {out[column][0]} not in [0, 100]"
            )
    return shots


def _add_shots(
    df: pl.DataFrame, match: _MatchRow, cols: defaultdict[str, list[Any]], where: str
) -> tuple[int, int]:
    """Append the match's shots to ``cols``; return the periods 1-2 goals (home, away)."""
    shots = _shot_rows(df, where)
    keepers = _facing_keepers(df, where)
    refs = shots.select("teamId", pl.col("relatedEventId").alias("eventId")).drop_nulls().unique()
    targets: defaultdict[tuple[str, str], list[Target]] = defaultdict(list)
    linked = df.with_columns(pl.col("eventId").str.replace(_NORM, "")).join(
        refs, on=["teamId", "eventId"], how="semi"
    )
    for team, event, *target in linked.select(
        "teamId", "eventId", "id", "type", "qualifiers"
    ).iter_rows():
        targets[(team, event)].append((target[0], target[1], target[2]))
    home_id, away_id = match["home_team_id"], match["away_team_id"]
    goals = [0, 0]
    for r in shots.iter_rows(named=True):
        at = f"{where}: event {r['id']}"
        parsed = _qualifiers(r["qualifiers"], at)
        q = {item["type"]["value"]: item.get("value") for item in parsed}
        own, blocked, related = OWN_GOAL in q, BLOCKED in q, r["relatedEventId"]
        if own and r["type"] != "Goal":
            raise ValueError(f"{at}: own-goal qualifier q28 on a {r['type']} event")
        body = [BODY_PARTS[k] for k in BODY_PARTS if k in q]
        patterns = [PATTERNS[k] for k in PATTERNS if k in q]
        if len(body) > 1 or (DIRECT_CORNER not in q and len(patterns) != 1):
            raise ValueError(f"{at}: body part qualifiers {body}, pattern qualifiers {patterns}")
        q55 = q.get(RELATED)
        q55_text = q55 if isinstance(q55, str) else str(q55) if _is_int(q55) else None
        if (RELATED in q) != (related is not None) or (
            related is not None
            and (
                q55_text is None
                or re.fullmatch(INT_TEXT, q55_text) is None
                or _norm(q55_text) != related
            )
        ):
            raise ValueError(f"{at}: q55 {q55!r} disagrees with relatedEventId {related!r}")
        team = "ws:" + r["teamId"]
        if own:
            assist: tuple[str | None, str | None] = (None, None)
        elif related is None:
            assist = ("none", None)
        else:
            assist = _assist(targets[(r["teamId"], related)], at)
        situation = "direct_corner" if DIRECT_CORNER in q else patterns[0]
        outcome = "blocked" if blocked else SHOT_TYPES[r["type"]]
        period = PERIODS[r["period"]]
        saved, mouth = r["type"] == "SavedShot", not (blocked or own)
        keeper = None if own else keepers[r["id"]]
        if keeper is None and not own:
            raise ValueError(f"{at}: shot before the opponent's lineup; no keeper to face")
        try:
            raw = json.dumps(parsed, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (ValueError, TypeError, RecursionError) as e:
            raise ValueError(f"{at}: qualifiers cannot be written as JSON: {e}") from None
        values: dict[str, object] = {
            "match_id": match["match_id"],
            "competition_id": match["competition_id"],
            "season_id": match["season_id"],
            "match_date": match["match_date"],
            "shot_id": "ws:" + r["id"],
            "team_id": team,
            "opponent_team_id": away_id if team == home_id else home_id,
            "player_id": "ws:" + r["playerId"],
            "gk_id": None if keeper is None else "ws:" + keeper,
            "is_home": team == home_id,
            "period": period,
            "minute": r["minute"],
            "second": r["second"],
            "seconds_in_period": float(
                (r["minute"] - schema.PERIOD_START_MINUTE[period]) * 60 + r["second"]
            ),
            "rx": r["x"],
            "ry": r["y"],
            "bx": r["blockedX"] if saved else None,
            "by": r["blockedY"] if saved else None,
            "gmy": r["goalMouthY"] if mouth else None,
            "gmz": r["goalMouthZ"] if mouth else None,
            "body_part": None if own or not body else body[0],
            "situation": situation,
            "assist_type": assist[0],
            "assist_event_id": assist[1],
            "outcome": outcome,
            "is_goal": r["type"] == "Goal",
            "is_own_goal": own,
            "is_penalty": situation == "penalty",
            "on_target": outcome in schema.ON_TARGET_OUTCOMES and not own,
            "first_time": None if own else FIRST_TOUCH in q,
            "one_on_one": None if own else ONE_ON_ONE in q,
            "big_chance": None if own else BIG_CHANCE in q,
            "raw_type": r["type"],
            "raw_qualifiers": raw,
        }
        for name, value in values.items():
            cols[name].append(value)
        if r["type"] == "Goal" and period <= 2:
            goals[0 if (team == home_id) != own else 1] += (
                1  # an own goal counts for the other side
            )
    return goals[0], goals[1]


def _floats(values: list[Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.array(values, dtype=np.float64)  # None -> NaN


def _shots_frame(cols: defaultdict[str, list[Any]]) -> pl.DataFrame:
    x, y = pitch.to_canonical(_floats(cols["rx"]), _floats(cols["ry"]), "opta")
    own = np.array(cols["is_own_goal"], dtype=bool)
    flipped = pitch.flip(x, y)  # own goals: recorded in the conceding player's frame
    x, y = np.where(own, flipped[0], x), np.where(own, flipped[1], y)
    end_x, end_y = pitch.to_canonical(_floats(cols["bx"]), _floats(cols["by"]), "opta")
    goal_y, goal_z = pitch.goal_mouth_to_canonical(
        _floats(cols["gmy"]), _floats(cols["gmz"]), "opta"
    )
    derived = {
        "x": pl.Series(x),
        "y": pl.Series(y),
        "end_x": pl.Series(end_x).fill_nan(None),
        "end_y": pl.Series(end_y).fill_nan(None),
        "goal_y": pl.Series(goal_y).fill_nan(None),
        "goal_z": pl.Series(goal_z).fill_nan(None),
        "distance_m": pl.Series(pitch.shot_distance(x, y)),
        "angle_rad": pl.Series(pitch.opening_angle(x, y)),
    }
    constant = {"provider": "whoscored", "schema_version": schema.SCHEMA_VERSION}
    for name in schema.SHOT_SCHEMA:
        if name not in derived:  # the columns Opta cannot fill stay all null
            values = cols[name] if name in cols else [constant.get(name)] * len(own)
            derived[name] = pl.Series(name, values)
    return schema.cast_to_schema(pl.DataFrame([derived[n].alias(n) for n in schema.SHOT_SCHEMA]))


def _load_match(
    mid: str,
    event: _EventFile,
    row: _Listed,
    meta: Path | None,
    cols: defaultdict[str, list[Any]],
    counts: dict[str, int],
) -> dict[str, object]:
    """Read one kept match: append its shots to ``cols`` and return its matches row."""
    where = f"'{event.path}'"
    df = _read_csv(event.path, EVENT_COLUMNS)
    bad = df.filter(~pl.col("period").is_in([*PERIODS, *NON_PLAY_PERIODS]).fill_null(False))
    if bad.height:
        raise ValueError(f"{where}: event {bad['id'][0]!r}: unknown period {bad['period'][0]!r}")
    _check_text(df, "minute", INT_TEXT, where, empty=False)
    _check_text(df, "teamId", INT_TEXT, where, empty=False)
    df = df.with_columns(pl.col("teamId").str.replace(_NORM, ""))
    teams = set(df["teamId"])
    names = df.select("teamId", "teamName").drop_nulls().unique()
    shared = sorted(set(names.filter(pl.col("teamName").is_duplicated())["teamName"]))
    if len(teams) != 2 or shared:
        raise ValueError(f"{where}: team ids {sorted(teams)}; team names on two ids: {shared}")
    if meta is not None:
        home, away, home_score, away_score = _metadata(meta, mid, event, teams)
    else:
        by_name: dict[str, str] = dict(zip(names["teamName"], names["teamId"], strict=True))
        pair = [by_name.get(name or "") for name in (row.home_name, row.away_name)]
        if set(pair) != teams:
            raise ValueError(
                f"{where}: no metadata, and the validated teams {row.home_name!r} / "
                f"{row.away_name!r} are not the file's two team names"
            )
        home, away = str(pair[0]), str(pair[1])
        home_score = away_score = None
    competition = COMPETITIONS[event.folder]
    first, second = event.season.split("-")
    match: _MatchRow = {
        "match_id": f"ws:{mid}",
        "provider": "whoscored",
        "competition_id": f"ws:{event.folder}",
        "season_id": f"ws:{event.season}",
        "match_date": event.match_date,
        "home_team_id": f"ws:{home}",
        "away_team_id": f"ws:{away}",
        "competition_name": competition.name,
        "season_name": second if competition.calendar_year else f"{first}/{second}",
        "gender": competition.gender,
        "has_360": None,
        "schema_version": schema.SCHEMA_VERSION,
    }
    goals = _add_shots(df, match, cols, where)
    if home_score is None:
        counts["unscored"] += 1
    elif not df.filter(pl.col("period").is_in(EXTRA_TIME)).is_empty():
        if (home_score, away_score) != goals:  # the provider score is the 90-minute score
            raise ValueError(
                f"match {mid}: extra-time match whose score {home_score}-{away_score} is not "
                f"its periods 1-2 goals {goals[0]}-{goals[1]}"
            )
        counts["extra_time"] += 1
        home_score = away_score = None
    last = df.filter(pl.col("period").is_in(list(PERIODS)))["minute"].str.replace(_NORM, "")
    last_minute = last.cast(pl.Int64).max()
    if not isinstance(last_minute, int) or last_minute < SHORT_MATCH_MINUTE:  # None: no play
        counts[SHORT_NOTE] += 1
    return {**match, "home_score": home_score, "away_score": away_score}


# --- Public API -------------------------------------------------------------------------------


def load(root: StrPath) -> Archive:
    """Load the archive into the validated C3 ``shots`` and ``matches`` tables.

    Every ``match_ids`` row is counted once: status not ``ok`` drops it under its status
    (``unvalidated`` for an empty one), an ``ok`` row without an event file is
    ``no_event_file``, and the rest are kept. An event file with no ``match_ids`` row is
    ``unvalidated``. Only the kept matches' event CSVs and metadata are opened, one at a time.

    Parameters
    ----------
    root : str or path
        The archive root, holding ``events/`` and ``match_ids/``.

    Returns
    -------
    Archive
        Validated tables (``freeze_frames`` empty), the drop and note counts, the files opened.

    Raises
    ------
    FileNotFoundError
        A missing ``root``, ``events/`` or ``match_ids/``, or a ``match_ids/`` with no CSV.
    ValueError
        Any malformed cell, file name, header, qualifier list, metadata file or status; the
        message names the file (and event id) and the value.
    SchemaValidationError
        The tables break a C3 rule (for example two ``ok`` copies of one match).

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     (Path(tmp) / "events").mkdir()
    ...     (Path(tmp) / "match_ids").mkdir()
    ...     header = ",".join(MATCH_ID_COLUMNS)
    ...     _ = (Path(tmp) / "match_ids/Championship_2025-2026.csv").write_text(
    ...         f"{header}\\n7,pending,,,\\n")
    ...     archive = load(tmp)
    >>> archive.shots.height, dict(archive.counts)["pending"], len(archive.files)
    (0, 1, 1)
    """
    base = Path(root)
    if not (base / "events").is_dir():
        raise FileNotFoundError(f"no events directory at '{base / 'events'}'")
    listed, opened = _match_ids(base / "match_ids")
    events = _event_files(base / "events", listed)
    counts = dict.fromkeys(_COUNT_KEYS, 0)
    kept: list[str] = []
    for mid, row in listed.items():
        reason = STATUSES[row.status]
        if reason is None and mid in events:
            kept.append(mid)
        else:
            counts[reason or "no_event_file"] += 1
    orphans = sum(mid not in listed for mid in events)
    counts["unvalidated"] += orphans
    dropped = sum(counts[r] for r in DROP_REASONS)
    if len(kept) + dropped != len(listed) + orphans:
        raise ValueError(f"internal: kept {len(kept)} + dropped {dropped} != rows + orphans")
    cols: defaultdict[str, list[Any]] = defaultdict(list)
    rows: list[dict[str, object]] = []
    indexes: dict[Path, dict[str, Path]] = {}
    for mid in sorted(kept, key=lambda m: events[m].path):
        event = events[mid]
        if event.path.parent not in indexes:
            indexes[event.path.parent] = _metadata_index(event.path.parent)
        meta = indexes[event.path.parent].get(mid)
        try:
            rows.append(_load_match(mid, event, listed[mid], meta, cols, counts))
        except pl.exceptions.PolarsError as e:
            raise ValueError(f"'{event.path}': {e}") from None
        opened.extend([event.path] if meta is None else [event.path, meta])
    frame = pl.DataFrame({k: pl.Series(k, [r[k] for r in rows]) for k in schema.MATCH_SCHEMA})
    matches = schema.validate_matches(schema.cast_to_schema(frame, "matches"))
    shots = schema.validate_shots(_shots_frame(cols), matches=matches)
    frames = schema.validate_freeze_frames(schema.empty_freeze_frames(), shots=shots)
    return Archive(shots, frames, matches, MappingProxyType(counts), tuple(sorted(opened)))


def build_snapshot(request: BuildRequest) -> BuildResult:
    """The ``whoscored`` snapshot builder: ``load``, reconcile scored matches, log the counts.

    Nothing is written when a scored match does not reconcile its goals with its score. The
    source digests every file opened under the fixed ``ARCHIVE_URI`` and lists none, so the
    manifest names no match and no path.

    Parameters
    ----------
    request : BuildRequest
        Exactly one input, the archive root; no params.

    Returns
    -------
    BuildResult
        Tables ``shots``, ``freeze_frames`` (empty) and ``matches``, a private ``files``
        source, ``snapshot.C3_SORT_BY`` and ``private=True``, so the manifest must be
        git-ignored too.

    Raises
    ------
    ValueError
        Wrong inputs or params, unreconciled goals, or as :func:`load`.
    FileNotFoundError
        As :func:`load`.

    Examples
    --------
    >>> try:
    ...     build_snapshot(BuildRequest())
    ... except ValueError as e:
    ...     print(str(e)[:44])
    the whoscored builder needs exactly one --in
    """
    if len(request.inputs) != 1:
        raise ValueError(
            "the whoscored builder needs exactly one --input: the archive root holding "
            "events/ and match_ids/"
        )
    if request.params:
        raise ValueError(
            f"the whoscored builder takes no params, got: {', '.join(sorted(request.params))}"
        )
    root = request.inputs[0]
    data = load(root)
    scored = data.matches.filter(pl.col("home_score").is_not_null())
    bad = schema.reconcile_goals(data.shots, scored)
    if bad.height:
        rows = "; ".join(str(r) for r in bad.head(5).iter_rows())
        raise ValueError(
            f"{bad.height} match(es) do not reconcile goals with the score "
            f"(match_id, home_score, away_score, home_goals, away_goals): {rows}"
        )
    for reason in DROP_REASONS:
        _log.warning("whoscored: dropped %d match(es): %s", data.counts[reason], reason)
    for note in SCORE_NOTES:
        _log.warning("whoscored: kept %d match(es) without a score: %s", data.counts[note], note)
    _log.warning(
        "whoscored: kept %d match(es) ending before minute %d",
        data.counts[SHORT_NOTE],
        SHORT_MATCH_MINUTE,
    )
    return BuildResult(
        {"shots": data.shots, "freeze_frames": data.freeze_frames, "matches": data.matches},
        snapshot.source_from_files(data.files, root=root, uri=ARCHIVE_URI, list_files=False),
        sort_by=snapshot.C3_SORT_BY,
        private=True,
    )
