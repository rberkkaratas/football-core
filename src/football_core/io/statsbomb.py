"""StatsBomb open data: a pinned, verified cache and a loader into the C3 shot tables.

**Licence.** The data (github.com/statsbomb/open-data) is free for research but must never be
committed or redistributed: the cache lives under the git-ignored ``data/`` folder, and
``sync`` refuses a cache that git would track. Every chart and every published result (model
cards, posts) credits StatsBomb as the data source, with the StatsBomb logo on charts. The logo
is fetched into the same cache by ``fetch_logo`` and never committed.

**Pinned and verified.** ``sync`` downloads one file at a time at ``OPEN_DATA_COMMIT`` (the
repo is 7.4 GB packed; the JSON we need is 1.6 GB stored gzip) and checks each file against the
git blob id that the commit's tree records, so the snapshot's ``git`` source is exact. A
complete cache needs no network, and a rerun is the retry::

    <cache>/<commit>/tree.json              {repo path: [blob sha1, size]}
    <cache>/<commit>/data/events/7298.json.gz

**Cell offset.** A StatsBomb value names the cell that *ends* at it, on a 1 yd grid for old
matches and a 0.1 yd grid for new ones, so the loader subtracts half a cell (``cell_offset``)
before ``football_core.pitch`` maps yards to metres. Two deviations from kloppy, both on
post-shot columns only: an ``end_location`` x of 120 is the goal line and is not shifted, and
the end height ``z`` is never shifted (heights start at 0.0). ``visible_area`` vertices are
continuous, so they are mapped without an offset; the piecewise map moves each vertex exactly
but bends an edge that crosses a landmark, so point-in-polygon work should use StatsBomb units
or accept that small error.

**Populations.** Penalties get no freeze-frame rows and null ``gk_*`` and ``visible_area`` by
design (the frames exist mostly for misses, so they would leak the outcome); model cards
treat penalties as their own population. Shoot-out kicks stay in the table as period 5. Own
goals come from the ``Own Goal Against`` event, flipped so the goal entered is at x = 105;
their locations are as recorded (12 at the pin lie outside the box, 40-85 m from goal), which
matters for shot maps only. ``situation`` comes from ``play_pattern``, which spans the whole
possession, so StatsBomb set-piece shares are not comparable with Opta's phase-only marking.

**Goal mouth.** ``goal_y`` is the recorded goal-plane point when the ball was stopped within one
grid cell of the goal line (raw ``end_x`` ≥ 120 − 2 × ``cell_offset``: 1 yd on coarse matches,
0.1 yd on fine ones, where a point cannot be told from the line; :func:`is_projected`). A shot
stopped earlier (a save, mostly) is projected: ``goal_y`` is where the line from the shot
through the stop point meets the goal line (``pitch.project_to_goal_line``, in metres), snapped
to the centre of its ``MOUTH_CELL`` goal-mouth cell (:func:`snap_goal_y`). The goal-mouth y is
recorded in tenths of a yard in every match, coarse ones included, where even tenths are
over-represented but no match keeps a coarser grid; so a recorded point sits on the 0.1 yd
lattice, and a projected one, continuous otherwise, is put on it too, and the lattice cannot
tell goals from saves. The rule reads the stop point and the grid, never the outcome, so goals
and saves get one rule. ``goal_z`` stays the recorded height.

**Facing keeper.** ``gk_id`` is the defending team's goalkeeper from the lineup state, walked
in event ``index`` order: the ``position.id == 1`` player of the last ``Starting XI`` or
``Tactical Shift``, handed on by a ``Substitution`` of that player. Save and Goal Keeper events
are not used: they exist only for some outcomes. A non-teammate keeper in a shot's freeze frame
must be the same player.

**360.** Visible areas fill ``shots.visible_area``. Eight 360 files at the pin are unusable
(3845506 is not JSON, upstream #50/#83; 3895158, 3895266, 3895309, 3837706, 3877115, 3877170
and 3877194 share no event with their match, #82): they are listed in ``skipped_360`` and the
build logs them. 360 freeze frames have no player ids and are not loaded.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import threading
import urllib.request
import zlib
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from http.client import HTTPException
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

import numpy as np
import numpy.typing as npt
import polars as pl

from football_core import pitch, schema
from football_core.io import snapshot
from football_core.io.snapshot import BuildRequest, BuildResult, StrPath

Fetch = Callable[[str], bytes]
Event = dict[str, Any]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

OPEN_DATA_URL: Final[str] = "https://github.com/statsbomb/open-data"
OPEN_DATA_COMMIT: Final[str] = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
COARSE_OFFSET: Final[float] = 0.5  # half a cell of the 1 yd grid
MOUTH_CELL: Final[float] = 0.1  # yd: the goal-mouth y is recorded in tenths in every match
MOUTH_CELLS: Final[int] = 800  # 80 yd / MOUTH_CELL
SNAP_TOL: Final[float] = 1e-4  # in cells: a recorded value maps back to its cell up to rounding
FINE_OFFSET: Final[float] = MOUTH_CELL / 2  # half a cell of the 0.1 yd grid
DEFAULT_CACHE: Final[str] = "data/raw/statsbomb/open-data"
TREE_PATH: Final = re.compile(
    r"data/(competitions|(matches/[0-9]+/[0-9]+|events/[0-9]+|lineups/[0-9]+|three-sixty/[0-9]+))\.json"
)
LOGO_PATH: Final[str] = "img/SB - Icon Lockup - Colour positive.png"
LOGO_BLOB: Final[str] = "fc75fa646cba5d0fa9dfcf8b183646b5fe32d8cc"  # git blob at OPEN_DATA_COMMIT
_TREE_URL: Final = "https://api.github.com/repos/statsbomb/open-data/git/trees/{}?recursive=1"
_RAW_URL: Final = "https://raw.githubusercontent.com/statsbomb/open-data/{}/{}"
_FIDELITY_OFFSET: Final[Mapping[str, float]] = {"1": COARSE_OFFSET, "2": FINE_OFFSET}
_YARD: Final[float] = 0.9144
_GOAL_LINE_X: Final[float] = 120.0
_FLAGS: Final = ("first_time", "follows_dribble", "open_goal", "one_on_one", "deflected")
_OWN_GOAL_NULLS: Final = (
    *("ex", "ey", "ez", "gy", "gz", "gkx", "gky", "visible_area", "body_part", "technique"),
    *("assist_type", "assist_event_id", "provider_xg", "project", *_FLAGS),
)  # the context of an own goal is not recorded
_DATE: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_TIMESTAMP: Final = re.compile(r"([0-9]{2}):([0-9]{2}):([0-9]{2}(?:\.[0-9]+)?)")

# Provider strings exactly as in the data; new vocabulary is mapped by hand, never guessed.
# fmt: off
_OUTCOMES: Final[Mapping[str, str]] = {
    "Goal": "goal", "Saved": "saved", "Saved to Post": "saved", "Blocked": "blocked",
    "Post": "post", "Off T": "off_target", "Wayward": "off_target",
    "Saved Off Target": "off_target",
}
_SHOT_TYPES: Final[Mapping[str, str | None]] = {  # None: decided by the play pattern
    "Penalty": "penalty", "Free Kick": "direct_free_kick", "Corner": "direct_corner",
    "Kick Off": "kick_off", "Open Play": None,
}
_PLAY_PATTERNS: Final[Mapping[str, str]] = {
    "From Corner": "from_corner", "From Free Kick": "from_free_kick",
    "From Throw In": "from_throw_in", "From Counter": "fast_break", "Regular Play": "open_play",
    "From Goal Kick": "open_play", "From Keeper": "open_play", "From Kick Off": "open_play",
    "Other": "open_play",
}
_BODY_PARTS: Final[Mapping[str, str]] = {
    "Right Foot": "right_foot", "Left Foot": "left_foot", "Head": "head", "Other": "other",
}
_TECHNIQUES: Final[Mapping[str, str]] = {
    name: name.lower().replace(" ", "_")
    for name in ("Normal", "Volley", "Half Volley", "Lob", "Backheel", "Diving Header",
                 "Overhead Kick")
}
# fmt: on

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenData:
    """What ``load`` returns: the validated C3 tables and the skipped 360 files.

    Attributes
    ----------
    shots, freeze_frames : polars.DataFrame
        The C3 tables, already validated.
    matches : polars.DataFrame
        The canonical matches table (``schema.MATCH_COLUMNS``), validated.
    skipped_360 : tuple of str
        ``sb:`` ids of the matches whose 360 file is listed but unusable.
    """

    shots: pl.DataFrame
    freeze_frames: pl.DataFrame
    matches: pl.DataFrame
    skipped_360: tuple[str, ...]


# --- Cache and download -----------------------------------------------------------------------


def _blob_sha1(data: bytes) -> str:
    """Git's blob id: what the tree listing at the pinned commit records for each file."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _http_get(url: str) -> bytes:
    """GET ``url``; every failure is an ``OSError``. No retries: rerunning ``sync`` is one."""
    request = urllib.request.Request(
        url, headers={"Accept-Encoding": "gzip", "User-Agent": "football-core"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body: bytes = response.read()
            encoding = response.headers.get("Content-Encoding")
    except HTTPException as e:  # IncompleteRead and friends are not OSErrors
        raise OSError(f"GET {url} failed: {e!r}") from None
    if encoding == "gzip":
        try:
            body = gzip.decompress(body)
        except (OSError, EOFError, zlib.error) as e:
            raise OSError(f"GET {url}: cannot decompress the gzip body: {e}") from None
    return body


def _read_tree(path: Path) -> dict[str, tuple[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"no tree.json at '{path}'; run statsbomb.sync first")
    try:
        raw = _parse_json(path.read_bytes())
    except (ValueError, RecursionError) as e:
        raise ValueError(f"cannot read '{path}': {e}; delete it and rerun sync") from None
    with _shape(f"'{path}'"):
        # TREE_PATH is the directory filter: every path it accepts stays under the cache.
        return {k: (str(v[0]), int(v[1])) for k, v in raw.items() if TREE_PATH.fullmatch(k)}


def _list_tree(commit: str, fetch: Fetch) -> dict[str, tuple[str, int]]:
    try:
        listing = _parse_json(fetch(_TREE_URL.format(commit)))
    except (ValueError, RecursionError) as e:
        raise ValueError(f"tree listing of {commit} is not JSON: {e}") from None
    with _shape(f"tree listing of {commit}"):
        if listing["truncated"] is not False:
            raise ValueError(f"tree listing of {commit} is truncated; cannot pin every file")
        return {
            e["path"]: (e["sha"], e["size"])
            for e in listing["tree"]
            if e["type"] == "blob" and TREE_PATH.fullmatch(e["path"])
        }


def _download(
    dest: Path, path: str, blob: str, commit: str, fetch: Fetch, stop: threading.Event
) -> None:
    if stop.is_set():  # another file failed: start nothing new
        return
    try:
        raw = fetch(_RAW_URL.format(commit, path))
        if _blob_sha1(raw) != blob:
            raise ValueError(
                f"'{path}' at {commit} has blob id {_blob_sha1(raw)}, tree.json says {blob}; "
                "nothing written"
            )
        target = dest / f"{path}.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.tmp")
        tmp.write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
        os.replace(tmp, target)  # renamed files are complete, so a rerun resumes
    except BaseException:
        stop.set()
        raise


def sync(
    cache: StrPath, *, commit: str = OPEN_DATA_COMMIT, fetch: Fetch | None = None, workers: int = 8
) -> Path:
    """Complete the cache of the open data at ``commit`` and return its checkout directory.

    Only missing files are downloaded, so a complete cache needs no network and a rerun after
    a failure resumes. Each file is checked against its git blob id before it is renamed into
    place.

    Parameters
    ----------
    cache : str or path
        Cache root, normally ``DEFAULT_CACHE``; created if missing. Inside a git work tree it
        must be git-ignored.
    commit : str, default OPEN_DATA_COMMIT
        40 lowercase hex characters.
    fetch : callable, optional
        ``fetch(url) -> bytes`` for the tree listing and the raw files; default HTTP.
    workers : int, default 8
        Parallel downloads.

    Returns
    -------
    pathlib.Path
        ``<cache>/<commit>``, the argument for :func:`load`.

    Raises
    ------
    ValueError
        A malformed commit, a truncated tree listing, a blob id mismatch, or a cache that git
        would not ignore.
    OSError
        Network or filesystem errors (``ELOOP`` for a symlink loop).

    Examples
    --------
    See :func:`load`.
    """
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"commit must be 40 lowercase hex characters, not {commit!r}")
    get = _http_get if fetch is None else fetch
    # Check before resolving: a data/ symlinked out of the repo then fails closed (C6 rule).
    snapshot._check_ignored(Path(os.path.abspath(cache)) / commit / "tree.json")
    dest = snapshot._resolve(Path(cache)) / commit
    tree_file = dest / "tree.json"
    if not tree_file.exists():
        tree = _list_tree(commit, get)
        dest.mkdir(parents=True, exist_ok=True)
        tmp = dest / "tree.json.tmp"
        tmp.write_text(json.dumps({k: list(v) for k, v in tree.items()}, sort_keys=True, indent=0))
        os.replace(tmp, tree_file)
    tree = _read_tree(tree_file)
    missing = [p for p in sorted(tree) if not os.path.exists(dest / f"{p}.gz")]
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download, dest, p, tree[p][0], commit, get, stop) for p in missing]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    return dest


def logo_file(cache: StrPath = DEFAULT_CACHE) -> Path:
    """Where :func:`fetch_logo` puts the StatsBomb logo: ``<cache>/<commit>/LOGO_PATH``.

    Parameters
    ----------
    cache : str or path, default DEFAULT_CACHE
        Cache root; relative paths are relative to the working directory.

    Returns
    -------
    pathlib.Path
        The path, whether or not the file exists.

    Examples
    --------
    >>> logo_file("cache").name
    'SB - Icon Lockup - Colour positive.png'
    """
    return Path(cache) / OPEN_DATA_COMMIT / LOGO_PATH


def fetch_logo(cache: StrPath = DEFAULT_CACHE, *, fetch: Fetch | None = None) -> Path:
    """Download the StatsBomb logo from the open-data repo at ``OPEN_DATA_COMMIT``, once.

    The licence asks for the logo on every chart of the data but grants no right to
    redistribute it, so it is cached beside the data (git-ignored) instead of committed. The
    bytes are checked against the pinned git blob id before they are renamed into place; a
    cached file with the right blob id is returned without a download.

    Parameters
    ----------
    cache : str or path, default DEFAULT_CACHE
        Cache root. Inside a git work tree it must be git-ignored.
    fetch : callable, optional
        ``fetch(url) -> bytes``; default HTTP.

    Returns
    -------
    pathlib.Path
        :func:`logo_file` of ``cache``, resolved; ``football_core.viz`` reads it by default.

    Raises
    ------
    ValueError
        The downloaded bytes do not have blob id ``LOGO_BLOB`` (nothing is written), or git
        would track the cache.
    OSError
        Network or filesystem errors, including a directory at the target path.

    Examples
    --------
    >>> fetch_logo().name  # doctest: +SKIP
    'SB - Icon Lockup - Colour positive.png'
    """
    snapshot._check_ignored(Path(os.path.abspath(logo_file(cache))))
    target = snapshot._resolve(logo_file(cache))
    if target.is_file() and _blob_sha1(target.read_bytes()) == LOGO_BLOB:
        return target
    get = _http_get if fetch is None else fetch
    raw = get(_RAW_URL.format(OPEN_DATA_COMMIT, quote(LOGO_PATH)))
    if _blob_sha1(raw) != LOGO_BLOB:
        raise ValueError(
            f"'{LOGO_PATH}' at {OPEN_DATA_COMMIT} has blob id {_blob_sha1(raw)}, "
            f"expected {LOGO_BLOB}; nothing written"
        )
    # a plain .png, not <path>.gz like the data files: sync and load read only tree.json paths
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, target)
    return target


# --- Reading ----------------------------------------------------------------------------------


def _reject_constant(name: str) -> object:
    raise ValueError(f"JSON constant {name} is not allowed")


def _gunzip(path: Path, size: int) -> bytes:
    """Decompress a cached file; ``size`` is the blob size that tree.json records for it."""
    try:
        body = gzip.decompress(path.read_bytes())
    except (OSError, EOFError, zlib.error) as e:  # BadGzipFile is an OSError
        problem = str(e)
    else:
        if len(body) == size:
            return body
        problem = f"{len(body)} bytes after decompressing, tree.json says {size}"
    raise ValueError(
        f"cannot read cached file '{path}': {problem}; delete it and rerun to download it again"
    )


def _parse_json(body: bytes) -> Any:
    """Raises ValueError (JSONDecodeError, UnicodeDecodeError, NaN/Infinity) or RecursionError."""
    return json.loads(body, parse_constant=_reject_constant)


def _read_json(path: Path, size: int) -> Any:
    body = _gunzip(path, size)
    try:
        return _parse_json(body)
    except (ValueError, RecursionError) as e:
        raise ValueError(
            f"cannot read cached file '{path}': {e}; delete it and rerun to download it again"
        ) from None


def _cached(root: Path, tree: Mapping[str, tuple[str, int]], repo_path: str) -> tuple[Path, int]:
    if repo_path not in tree:
        raise FileNotFoundError(f"'{repo_path}' is not listed in '{root / 'tree.json'}'")
    path = root / f"{repo_path}.gz"
    if not path.is_file():
        raise FileNotFoundError(f"cached file '{path}' is missing; run statsbomb.sync")
    return path, tree[repo_path][1]


@contextmanager
def _shape(where: str) -> Iterator[None]:
    """Turn a lookup on JSON of the wrong shape into a ``ValueError`` naming where it was."""
    try:
        yield
    except (KeyError, TypeError, IndexError, AttributeError) as e:
        raise ValueError(f"{where}: unexpected StatsBomb JSON shape: {e!r}") from None


# --- Mapping ----------------------------------------------------------------------------------


def cell_offset(
    match_id: int, locations: Iterable[Sequence[float]], shot_fidelity: str | None
) -> float:
    """Half a cell of the match's grid, inferred from its Shot and Own Goal Against locations.

    Any non-integer coordinate means the 0.1 yd grid. ``metadata.shot_fidelity_version``
    (``"1"`` coarse, ``"2"`` fine), when present, must agree, which turns a silent 0.45 yd
    shift (a fine match whose few shots all land on integers) into an error.

    Parameters
    ----------
    match_id : int
        For error messages.
    locations : iterable of sequence of float
        ``location`` of every Shot and Own Goal Against event of the match.
    shot_fidelity : str or None
        ``metadata.shot_fidelity_version`` of the matches-file entry, if any.

    Returns
    -------
    float
        ``COARSE_OFFSET`` or ``FINE_OFFSET``.

    Raises
    ------
    ValueError
        An unknown fidelity version, or one that disagrees with the locations.

    Examples
    --------
    >>> cell_offset(1, [[109.0, 41.0], [100.0, 30.0]], None)
    0.5
    >>> cell_offset(1, [[108.1, 40.0]], "2")
    0.05
    """
    values = [float(v) for loc in locations for v in loc[:2]]
    inferred = FINE_OFFSET if any(not v.is_integer() for v in values) else COARSE_OFFSET
    if shot_fidelity is None:
        return inferred
    if shot_fidelity not in _FIDELITY_OFFSET:
        raise ValueError(
            f"match {match_id}: unknown metadata.shot_fidelity_version {shot_fidelity!r}; "
            "allowed: '1', '2'"
        )
    expected = _FIDELITY_OFFSET[shot_fidelity]
    if values and expected != inferred:
        raise ValueError(
            f"match {match_id}: metadata.shot_fidelity_version {shot_fidelity!r} disagrees "
            f"with the shot locations (inferred half-cell {inferred})"
        )
    return expected


def is_projected(end_x: npt.ArrayLike, offset: npt.ArrayLike) -> BoolArray:
    """Whether a goal-mouth stop point is projected: more than one grid cell short of the line.

    A stop point within one cell of the goal line (raw ``end_x`` ≥ 120 − 2 × ``offset``) cannot
    be told from the line, so its recorded goal-mouth y is kept. The loader and the grid census
    (``football_core.goalmouth.grid_census``) both call this, so they count the same rows.

    Parameters
    ----------
    end_x : array_like
        Raw ``end_location[0]`` in StatsBomb yards.
    offset : array_like
        The match's :func:`cell_offset`.

    Returns
    -------
    np.ndarray
        Boolean; NaN ``end_x`` gives False.

    Examples
    --------
    >>> is_projected([118.0, 119.0, 119.8, 119.9, 120.0], [0.5, 0.5, 0.05, 0.05, 0.05]).tolist()
    [True, False, True, False, False]
    """
    ex = np.asarray(end_x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        out: BoolArray = ex < _GOAL_LINE_X - 2 * np.asarray(offset, dtype=np.float64) - 1e-9
    return out


def snap_goal_y(goal_y: npt.ArrayLike) -> FloatArray:
    """Snap stored goal-mouth y values (metres) to the centre of their 0.1 yd goal-mouth cell.

    A StatsBomb value names the cell that ends at it, so a point at ``yd`` goes to cell
    ``k = ceil(yd / MOUTH_CELL − SNAP_TOL)`` (clipped to ``1 … MOUTH_CELLS``) and lands on its
    centre ``(k − 0.5) · MOUTH_CELL``, mapped back to metres. A recorded point is already a cell
    centre and stays where it is. The rule reads the value only, never an outcome.

    Parameters
    ----------
    goal_y : array_like
        ``shots.goal_y`` (pitch-width metres).

    Returns
    -------
    np.ndarray
        Snapped metres; NaN stays NaN, ±inf clip to the end cells.

    Examples
    --------
    >>> yd = [39.97, 40.0, 39.9, 39.95]
    >>> m = pitch.goal_mouth_to_canonical(yd, [0.0] * 4, "statsbomb")[0]
    >>> back = pitch.goal_mouth_y_from_canonical(snap_goal_y(m), "statsbomb")
    >>> [round(float(v), 6) for v in back]
    [39.95, 39.95, 39.85, 39.95]
    """
    yd = pitch.goal_mouth_y_from_canonical(goal_y, "statsbomb")
    k = np.clip(np.ceil(yd / MOUTH_CELL - SNAP_TOL), 1, MOUTH_CELLS)
    snapped = pitch.goal_mouth_to_canonical((k - 0.5) * MOUTH_CELL, np.zeros_like(k), "statsbomb")
    return snapped[0]


def _vocab(table: Mapping[str, Any], value: object, match_id: int, field: str) -> Any:
    if not isinstance(value, str) or value not in table:
        allowed = ", ".join(repr(k) for k in table)
        raise ValueError(f"match {match_id}: unknown {field} {value!r}; allowed: {allowed}")
    return table[value]


def _seconds(timestamp: object, match_id: int) -> float:
    found = _TIMESTAMP.fullmatch(timestamp) if isinstance(timestamp, str) else None
    if found is None:
        raise ValueError(f"match {match_id}: timestamp {timestamp!r} is not HH:MM:SS.fff")
    hours, minutes, seconds = found.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _number(value: object, what: str) -> float:
    """A JSON number as float; ``float()`` alone would take ``true`` or ``"0.3"`` silently."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{what} {value!r} is not numeric")  # _shape adds the match
    return float(value)


def _xy(location: Sequence[object]) -> tuple[float, float]:
    x, y = location[0], location[1]
    if (
        not isinstance(x, int | float)
        or not isinstance(y, int | float)
        or isinstance(x, bool)
        or isinstance(y, bool)
    ):
        raise TypeError(f"location {location!r} is not numeric")  # _shape adds the match
    return float(x), float(y)


def _flag(value: object, mid: int, eid: str, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"match {mid}: event {eid} field {field} is {value!r}, not true or false")
    return value


def _assist(shot: Event, by_id: Mapping[str, Event]) -> tuple[str, str | None]:
    if "key_pass_id" not in shot:
        return "none", None
    key = shot["key_pass_id"]
    pass_ = by_id[key]["pass"]
    if pass_.get("cut_back"):
        kind = "cutback"
    elif (pass_.get("technique") or {}).get("name") == "Through Ball":
        kind = "through_ball"
    elif pass_.get("cross"):
        kind = "cross"
    else:
        kind = "pass"
    return kind, f"sb:{key}"


def _match_entry(entry: Event) -> dict[str, object]:
    """One ``matches`` row from a matches-file entry, checked field by field."""
    mid, home, away = entry["match_id"], entry["home_team"], entry["away_team"]
    fields = {
        "match_id": mid,
        "competition_id": entry["competition"]["competition_id"],
        "season_id": entry["season"]["season_id"],
        "home_team_id": home["home_team_id"],
        "away_team_id": away["away_team_id"],
        "competition_name": entry["competition"]["competition_name"],
        "season_name": entry["season"]["season_name"],
        "gender": home["home_team_gender"],
        "match_date": entry["match_date"],
        "home_score": entry["home_score"],
        "away_score": entry["away_score"],
    }
    for name, value in fields.items():
        if value is None:
            raise ValueError(f"match {mid}: field {name} is missing or null")
    gender, away_gender = fields["gender"], away["away_team_gender"]
    if gender not in ("male", "female") or gender != away_gender:
        raise ValueError(
            f"match {mid}: team genders {gender!r} / {away_gender!r}; expected both 'male' or "
            "both 'female'"
        )
    for name in ("home_score", "away_score"):
        score = fields[name]
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 99:
            raise ValueError(f"match {mid}: {name} {score!r} is not an integer in [0, 99]")
    raw_date = fields["match_date"]
    try:
        if not isinstance(raw_date, str) or _DATE.fullmatch(raw_date) is None:
            raise ValueError
        fields["match_date"] = date.fromisoformat(raw_date)
    except ValueError:
        raise ValueError(f"match {mid}: match_date {raw_date!r} is not a YYYY-MM-DD date") from None
    for name in ("match_id", "competition_id", "season_id", "home_team_id", "away_team_id"):
        fields[name] = f"sb:{fields[name]}"
    return fields


def _visible_areas(path: Path, size: int, event_ids: Collection[str]) -> dict[str, Any] | None:
    """``event_uuid`` -> flat polygon, or None when the file is unusable (upstream defects)."""
    body = _gunzip(path, size)  # a corrupt cache file is an error, not a skip
    try:
        frames = _parse_json(body)
    except (ValueError, RecursionError):
        return None
    areas = {f["event_uuid"]: f["visible_area"] for f in frames}
    return areas if not areas.keys().isdisjoint(event_ids) else None


def _polygon(flat: Sequence[float], match_id: int) -> list[list[float]]:
    if len(flat) % 2:
        raise ValueError(f"match {match_id}: a visible_area has an odd number of values")
    xs, ys = pitch.to_canonical(flat[0::2], flat[1::2], "statsbomb")  # continuous: no offset
    return [[float(x), float(y)] for x, y in zip(xs, ys, strict=True)]


def _facing_keepers(mid: int, events: Sequence[Event], home: int, away: int) -> dict[str, str]:
    """Shot event id -> ``sb:`` id of the defending keeper, from the lineup state."""
    keeper: dict[int, int] = {}
    faced: dict[str, str] = {}
    last = None
    for event in events:
        if last is not None and not event["index"] > last:
            raise ValueError(
                f"match {mid}: event {event['id']} has index {event['index']!r} after "
                f"{last!r}; events must be in index order"
            )
        last, kind = event["index"], event["type"]["name"]
        if kind in ("Starting XI", "Tactical Shift"):
            lineup = event["tactics"]["lineup"]
            ids = [p["player"]["id"] for p in lineup if p["position"]["id"] == 1]
            if len(ids) != 1:
                raise ValueError(
                    f"match {mid}: event {event['id']} ({kind}) lists {len(ids)} goalkeepers; "
                    "expected exactly one"
                )
            keeper[event["team"]["id"]] = ids[0]
        elif kind == "Substitution" and keeper.get(event["team"]["id"]) == event["player"]["id"]:
            keeper[event["team"]["id"]] = event["substitution"]["replacement"]["id"]
        elif kind == "Shot":
            other = away if event["team"]["id"] == home else home
            if other not in keeper:
                raise ValueError(
                    f"match {mid}: shot {event['id']} comes before team {other}'s lineup; "
                    "no keeper to face"
                )
            faced[event["id"]] = f"sb:{keeper[other]}"
    return faced


def _add_match(
    row: Mapping[str, object],
    entry: Event,
    events: list[Event],
    areas: Mapping[str, Any] | None,
    shots: defaultdict[str, list[Any]],
    frames: defaultdict[str, list[Any]],
) -> None:
    """Append one match's shots and freeze-frame players, in shifted StatsBomb yards."""
    mid: int = entry["match_id"]
    home, away = entry["home_team"]["home_team_id"], entry["away_team"]["away_team_id"]
    kept = [e for e in events if e["type"]["name"] in ("Shot", "Own Goal Against")]
    fidelity = (entry.get("metadata") or {}).get("shot_fidelity_version")
    c = cell_offset(mid, [_xy(e["location"]) for e in kept], fidelity)
    by_id = {e["id"]: e for e in events}
    keepers = _facing_keepers(mid, events, home, away)
    for event in kept:
        own = event["type"]["name"] == "Own Goal Against"
        shot: Event = {} if own else event["shot"]
        team = event["team"]["id"]
        if team not in (home, away):
            raise ValueError(f"match {mid}: event {event['id']} has team {team}, not in the match")
        x, y = _xy(event["location"])
        raw = dict(event)
        if not own:
            raw["shot"] = {k: v for k, v in shot.items() if k != "freeze_frame"}
            outcome_name = shot["outcome"]["name"]
            outcome = _vocab(_OUTCOMES, outcome_name, mid, "shot.outcome")
            direct = _vocab(_SHOT_TYPES, shot["type"]["name"], mid, "shot.type")
        else:
            outcome_name, outcome, direct = None, "goal", None
        pattern = _vocab(_PLAY_PATTERNS, event["play_pattern"]["name"], mid, "play_pattern")
        penalty = direct == "penalty"
        for key in ("match_id", "competition_id", "season_id", "match_date"):
            shots[key].append(row[key])
        shots["shot_id"].append(f"sb:{event['id']}")
        shots["player_id"].append(f"sb:{event['player']['id']}")
        shots["gk_id"].append(None if own else keepers[event["id"]])
        shots["team_id"].append(f"sb:{team}")
        shots["opponent_team_id"].append(f"sb:{away if team == home else home}")
        shots["is_home"].append(team == home)
        shots["period"].append(event["period"])
        shots["minute"].append(event["minute"])
        shots["second"].append(event["second"])
        shots["seconds_in_period"].append(_seconds(event["timestamp"], mid))
        shots["sx"].append(x - c)
        shots["sy"].append(y - c)
        shots["situation"].append(direct or pattern)
        shots["outcome"].append(outcome)
        shots["is_goal"].append(outcome == "goal")
        shots["is_own_goal"].append(own)
        shots["is_penalty"].append(penalty)
        shots["on_target"].append(not own and outcome in schema.ON_TARGET_OUTCOMES)
        shots["raw_type"].append(event["type"]["name"])
        shots["raw_qualifiers"].append(
            json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        if own:
            for name in _OWN_GOAL_NULLS:
                shots[name].append(None)
            continue
        end = shot["end_location"]
        if len(end) not in (2, 3):
            raise ValueError(
                f"match {mid}: event {event['id']} end_location {end!r} needs 2 or 3 values"
            )
        ex, ey = _xy(end)
        three = len(end) == 3
        ez = _number(end[2], "end_location z") if three else None
        shots["ex"].append(ex if ex == _GOAL_LINE_X else ex - c)  # the goal line stays put
        shots["ey"].append(ey - (FINE_OFFSET if three else c))  # goal-mouth y is in tenths
        shots["ez"].append(None if ez is None else ez * _YARD)  # heights start at 0.0
        mouth = three and outcome_name != "Blocked"
        shots["gy"].append(ey - FINE_OFFSET if mouth else None)
        shots["gz"].append(ez if mouth else None)
        shots["project"].append(mouth and bool(is_projected(ex, c)))
        keeper = None
        if not penalty:
            for player in shot.get("freeze_frame", ()):
                px, py = _xy(player["location"])
                is_keeper = player["position"]["name"] == "Goalkeeper"
                mate = _flag(player["teammate"], mid, event["id"], "freeze_frame teammate")
                if is_keeper and not mate:
                    seen = f"sb:{player['player']['id']}"
                    if seen != keepers[event["id"]]:
                        raise ValueError(
                            f"match {mid}: event {event['id']} freeze-frame keeper {seen} is not "
                            f"the lineup keeper {keepers[event['id']]}"
                        )
                if is_keeper and not mate and keeper is None:
                    keeper = (px - c, py - c)  # the opponent keeper; never the shooter's own
                frames["shot_id"].append(f"sb:{event['id']}")
                frames["player_id"].append(f"sb:{player['player']['id']}")
                frames["is_teammate"].append(mate)
                frames["is_keeper"].append(is_keeper)
                frames["sx"].append(px - c)
                frames["sy"].append(py - c)
        shots["gkx"].append(None if keeper is None else keeper[0])
        shots["gky"].append(None if keeper is None else keeper[1])
        area = None if penalty or areas is None else areas.get(event["id"])
        shots["visible_area"].append(None if area is None else _polygon(area, mid))
        shots["body_part"].append(_vocab(_BODY_PARTS, shot["body_part"]["name"], mid, "body_part"))
        shots["technique"].append(_vocab(_TECHNIQUES, shot["technique"]["name"], mid, "technique"))
        assist_type, assist_id = _assist(shot, by_id)
        shots["assist_type"].append(assist_type)
        shots["assist_event_id"].append(assist_id)
        shots["provider_xg"].append(_number(shot["statsbomb_xg"], "shot.statsbomb_xg"))
        for flag in _FLAGS:
            shots[flag].append(_flag(shot.get(flag, False), mid, event["id"], f"shot.{flag}"))


def _metres(xs: list[float | None], ys: list[float | None]) -> tuple[pl.Series, pl.Series]:
    """Shifted StatsBomb yards to metres; None stays null."""
    x, y = pitch.to_canonical(
        np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), "statsbomb"
    )
    return pl.Series(x).fill_nan(None), pl.Series(y).fill_nan(None)


def _shots_frame(cols: defaultdict[str, list[Any]]) -> pl.DataFrame:
    x, y = pitch.to_canonical(
        np.array(cols["sx"], np.float64), np.array(cols["sy"], np.float64), "statsbomb"
    )
    own = np.array(cols["is_own_goal"], dtype=bool)
    flipped = pitch.flip(x, y)  # own goals: recorded in the conceding player's frame
    x, y = np.where(own, flipped[0], x), np.where(own, flipped[1], y)
    end_x, end_y = _metres(cols["ex"], cols["ey"])
    goal_y, goal_z = pitch.goal_mouth_to_canonical(
        np.array(cols["gy"], np.float64), np.array(cols["gz"], np.float64), "statsbomb"
    )
    projected = snap_goal_y(pitch.project_to_goal_line(x, y, end_x.to_numpy(), end_y.to_numpy()))
    goal_y = np.where(np.array(cols["project"], dtype=bool), projected, goal_y)  # None: False
    gk_x, gk_y = _metres(cols["gkx"], cols["gky"])
    derived = {
        "x": pl.Series(x),
        "y": pl.Series(y),
        "end_x": end_x,
        "end_y": end_y,
        "end_z": pl.Series(np.array(cols["ez"], np.float64)).fill_nan(None),
        "goal_y": pl.Series(goal_y).fill_nan(None),
        "goal_z": pl.Series(goal_z).fill_nan(None),
        "gk_x": gk_x,
        "gk_y": gk_y,
        "distance_m": pl.Series(pitch.shot_distance(x, y)),
        "angle_rad": pl.Series(pitch.opening_angle(x, y)),
    }
    constant = {"provider": "statsbomb", "schema_version": schema.SCHEMA_VERSION}
    for name in schema.SHOT_SCHEMA:
        if name not in derived:
            values = [constant.get(name)] * len(own) if name not in cols else cols[name]
            derived[name] = pl.Series(name, values)  # big_chance, provider_psxg: all null
    frame = pl.DataFrame({name: derived[name] for name in schema.SHOT_SCHEMA})
    return schema.cast_to_schema(frame, "shots")


def _frames_frame(cols: defaultdict[str, list[Any]]) -> pl.DataFrame:
    x, y = _metres(cols["sx"], cols["sy"])
    plain = {k: pl.Series(k, cols[k]) for k in ("shot_id", "player_id", "is_teammate", "is_keeper")}
    return schema.cast_to_schema(pl.DataFrame({**plain, "x": x, "y": y}), "freeze_frames")


def load(checkout: StrPath, *, match_ids: Collection[int] | None = None) -> OpenData:
    """Load a synced checkout into the validated C3 ``matches``, ``shots`` and ``freeze_frames``.

    Matches are read one at a time, in ``(competition_id, season_id, match_id)`` order.
    Shots are every Shot event (all periods, shoot-outs as period 5) plus every Own Goal
    Against event; freeze frames are the players of each non-penalty shot's event frame.

    Parameters
    ----------
    checkout : str or path
        The directory :func:`sync` returns.
    match_ids : collection of int, optional
        Raw StatsBomb match ids to keep (they name the cache files); default all.

    Returns
    -------
    OpenData
        Validated tables and the ``sb:`` ids of the matches whose 360 file was unusable.

    Raises
    ------
    TypeError
        ``match_ids`` is a ``str``.
    ValueError
        An unknown match id, an unreadable cache file, JSON of an unexpected shape, an
        unknown provider string, a fidelity disagreement or a bad matches-file entry; the
        message names the match, the field and the value.
    FileNotFoundError
        No ``tree.json``, or a listed match or 360 file missing from the tree or the disk.

    Examples
    --------
    >>> import json, tempfile
    >>> match = {"match_id": 1, "match_date": "2024-05-01", "home_score": 0, "away_score": 0,
    ...          "competition": {"competition_id": 9, "competition_name": "Demo"},
    ...          "season": {"season_id": 7, "season_name": "2024"},
    ...          "home_team": {"home_team_id": 1, "home_team_gender": "female"},
    ...          "away_team": {"away_team_id": 2, "away_team_gender": "female"}}
    >>> files = {"data/matches/9/7.json": json.dumps([match]).encode(),
    ...          "data/events/1.json": b"[]"}
    >>> tree = {"truncated": False, "tree": [{"path": p, "type": "blob", "size": len(b),
    ...                                       "sha": _blob_sha1(b)} for p, b in files.items()]}
    >>> def fetch(url):
    ...     return json.dumps(tree).encode() if "api.github" in url else files[url.split("/", 6)[6]]
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     data = load(sync(tmp, commit="0" * 40, fetch=fetch))
    >>> data.matches["match_id"].to_list(), data.shots.height, data.skipped_360
    (['sb:1'], 0, ())
    """
    if isinstance(match_ids, str):
        raise TypeError("match_ids must be a collection of int match ids, not a str")
    root = Path(checkout)
    tree = _read_tree(root / "tree.json")
    entries: list[Event] = []
    for repo_path in sorted(p for p in tree if p.startswith("data/matches/")):
        path, size = _cached(root, tree, repo_path)
        with _shape(f"'{path}'"):
            entries.extend(_read_json(path, size))
    with _shape("matches files"):
        entries.sort(
            key=lambda e: (
                e["competition"]["competition_id"],
                e["season"]["season_id"],
                e["match_id"],
            )
        )
        if match_ids is not None:
            wanted = set(match_ids)
            unknown = wanted - {e["match_id"] for e in entries}
            if unknown:
                raise ValueError(f"unknown match id(s): {', '.join(map(str, sorted(unknown)))}")
            entries = [e for e in entries if e["match_id"] in wanted]
    shots: defaultdict[str, list[Any]] = defaultdict(list)
    frames: defaultdict[str, list[Any]] = defaultdict(list)
    matches: defaultdict[str, list[object]] = defaultdict(list)
    skipped: list[str] = []
    for entry in entries:
        mid = entry["match_id"]
        with _shape(f"match {mid}"):
            row = _match_entry(entry)
            path, size = _cached(root, tree, f"data/events/{mid}.json")
            events: list[Event] = _read_json(path, size)
            areas = None
            if f"data/three-sixty/{mid}.json" in tree:
                path, size = _cached(root, tree, f"data/three-sixty/{mid}.json")
                areas = _visible_areas(path, size, {e["id"] for e in events})
                if areas is None:
                    skipped.append(f"sb:{mid}")
            row["has_360"] = areas is not None
            _add_match(row, entry, events, areas, shots, frames)
        for key, value in row.items():
            matches[key].append(value)
    n = len(matches["match_id"])
    matches["provider"] = ["statsbomb"] * n
    matches["schema_version"] = [schema.SCHEMA_VERSION] * n
    frame = pl.DataFrame({k: matches[k] for k in schema.MATCH_SCHEMA})
    match_table = schema.validate_matches(schema.cast_to_schema(frame, "matches"))
    shot_table = schema.validate_shots(_shots_frame(shots), matches=match_table)
    frame_table = schema.validate_freeze_frames(_frames_frame(frames), shots=shot_table)
    return OpenData(shot_table, frame_table, match_table, tuple(skipped))


reconcile_goals = schema.reconcile_goals  # moved to schema (C5); kept as an alias


def build_snapshot(request: BuildRequest) -> BuildResult:
    """The ``statsbomb`` snapshot builder: ``sync``, ``load``, then reconcile every match.

    No snapshot is written when a match does not reconcile with its score.

    Parameters
    ----------
    request : BuildRequest
        Exactly one input, the cache root (e.g. ``data/raw/statsbomb/open-data``); the only
        param is ``commit`` (default ``OPEN_DATA_COMMIT``).

    Returns
    -------
    BuildResult
        Tables ``shots``, ``freeze_frames`` and ``matches`` with a ``git`` source.

    Raises
    ------
    ValueError
        Wrong inputs or params, unreconciled goals, or as :func:`sync` and :func:`load`.
    OSError
        As :func:`sync` and :func:`load`.

    Examples
    --------
    >>> try:
    ...     build_snapshot(BuildRequest())
    ... except ValueError as e:
    ...     print(str(e)[:44])
    the statsbomb builder needs exactly one --in
    """
    if len(request.inputs) != 1:
        raise ValueError(
            "the statsbomb builder needs exactly one --input: the cache directory, "
            f"e.g. {DEFAULT_CACHE}"
        )
    unknown = sorted(set(request.params) - {"commit"})
    if unknown:
        raise ValueError(
            f"the statsbomb builder takes only the param 'commit', got: {', '.join(unknown)}"
        )
    commit = request.params.get("commit", OPEN_DATA_COMMIT)
    data = load(sync(request.inputs[0], commit=commit))
    bad = schema.reconcile_goals(data.shots, data.matches)
    if bad.height:
        rows = "; ".join(str(r) for r in bad.head(5).iter_rows())
        raise ValueError(
            f"{bad.height} match(es) do not reconcile goals with the score "
            f"(match_id, home_score, away_score, home_goals, away_goals): {rows}"
        )
    if data.skipped_360:
        _log.warning(
            "statsbomb: %d match(es) without usable 360 data: %s",
            len(data.skipped_360),
            ", ".join(data.skipped_360),
        )
    return BuildResult(
        {"shots": data.shots, "freeze_frames": data.freeze_frames, "matches": data.matches},
        snapshot.source_from_git(OPEN_DATA_URL, commit),
        sort_by=snapshot.C3_SORT_BY,
    )
