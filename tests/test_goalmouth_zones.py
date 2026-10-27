"""Opta goal-mouth y direction against the zone qualifiers (spec ``runs/MS5/goalmouth`` §5.1,
§10.4): ``goalmouth.OPTA_ZONES``, ``zone_sides`` and ``confirms_direction``.

``raw_qualifiers`` is written in the stored WhoScored shape (``[{"type": {"value": id,
"displayName": name}}, ...]``, spec MS4 §2.2). Points are pitch-width metres; the goal-frame y is
``34 - goal_y``, so a point left of the centre seen by the shooter has ``goal_y > 34`` (raw
goalMouthY > 50).
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any

import polars as pl
import pytest

from football_core import pitch

PROVIDER = pl.Enum(("statsbomb", "whoscored"))
NAMES = {
    20: "RightFoot", 22: "RegularPlay", 73: "MissLeft", 75: "MissRight", 76: "LowLeft",
    77: "HighLeft", 78: "HighCentre", 80: "LowRight", 102: "GoalMouthY",
}  # fmt: skip
ZONE_COLUMNS = ("zone", "named_side", "shots", "shooter_left", "shooter_right", "centre")


def _gm() -> Any:
    from football_core import goalmouth

    return goalmouth


def _raw(*qids: int) -> str:
    quals: list[dict[str, Any]] = [
        {"type": {"value": q, "displayName": NAMES[q]}} for q in qids if q != 102
    ]
    if 102 in qids:
        quals.append({"type": {"value": 102, "displayName": NAMES[102]}, "value": "48.6"})
    return json.dumps(quals, separators=(",", ":"))


def _opta_y(raw_y: float) -> float:
    """A raw goalMouthY in stored metres (the loader's map), for realistic fixture points."""
    gy, _ = pitch.goal_mouth_to_canonical([raw_y], [10.0], "opta")
    return float(gy[0])


def _frame(rows: list[tuple[str, float | None, float | None, str]]) -> pl.DataFrame:
    p, y, z, raw = (list(c) for c in zip(*rows, strict=True)) if rows else ([], [], [], [])
    return pl.DataFrame(
        [
            pl.Series("provider", p, dtype=PROVIDER, strict=True),
            pl.Series("goal_y", y, dtype=pl.Float64(), strict=True),
            pl.Series("goal_z", z, dtype=pl.Float64(), strict=True),
            pl.Series("raw_qualifiers", raw, dtype=pl.String(), strict=True),
        ]
    )


# fmt: off
ROWS = [
    ("whoscored", 31.0, 1.0, _raw(80, 20)),        # LowRight, goal-frame y +3: shooter's right
    ("whoscored", 36.0, 1.0, _raw(22, 76, 102)),   # LowLeft, y -2: left (q102 is no zone)
    ("whoscored", 33.0, 2.0, _raw(78)),            # HighCentre names no side: no row
    ("whoscored", _opta_y(50.0), 0.5, _raw(76)),   # raw 50 -> 34 up to rounding: centre
    ("whoscored", 40.0, 1.0, _raw(73)),            # MissLeft, y -6: left (quoted apart)
    ("whoscored", 35.0, 0.5, _raw(76, 20)),        # LowLeft, y -1: left
    ("whoscored", None, None, _raw(76)),           # no goal-mouth point: not counted
]
EXPECTED = [
    (73, "left", 1, 1, 0, 0),
    (76, "left", 3, 2, 0, 1),
    (80, "right", 1, 0, 1, 0),
]
# fmt: on


def test_opta_zones_table() -> None:
    zones = _gm().OPTA_ZONES
    assert isinstance(zones, MappingProxyType)
    assert set(zones) == {73, 75, 76, 77, 80, 81, 83, 84, 86, 87}  # 74/78/79/85 name no side
    assert 82 not in zones  # Blocked


def test_zone_sides() -> None:
    out = _gm().zone_sides(_frame(ROWS))
    assert out.columns == list(ZONE_COLUMNS)
    assert out.schema["named_side"] == pl.String()
    assert all(out.schema[c] == pl.Int64() for c in ZONE_COLUMNS if c != "named_side")
    assert out.rows() == EXPECTED


def test_zone_sides_ignores_statsbomb() -> None:
    """§5.1: non-whoscored rows are ignored, even with a zone-like qualifier list."""
    sb = ("statsbomb", 31.0, 1.0, _raw(76))  # a left zone on the shooter's right
    base = _gm().zone_sides(_frame(ROWS))
    assert _gm().zone_sides(_frame([*ROWS, sb])).rows() == base.rows()
    alone = _gm().zone_sides(_frame([sb]))
    assert alone.height == 0 and alone.columns == list(ZONE_COLUMNS)


@pytest.mark.parametrize("column", ["provider", "goal_y", "goal_z", "raw_qualifiers"])
def test_zone_sides_missing_column(column: str) -> None:
    with pytest.raises(ValueError, match=column):
        _gm().zone_sides(_frame(ROWS).drop(column))


# extra rows added to ROWS, zones argument (None = default 76-87), expected answer
# fmt: off
DIRECTION_CASES = [
    ("confirmed", [], None, True),
    ("lowleft_at_raw_47", [("whoscored", _opta_y(47.0), 1.0, _raw(76))], None, False),
    ("wrong_side_73_outside_default", [("whoscored", 28.0, 1.0, _raw(73))], None, True),
    ("wrong_side_73_when_asked", [("whoscored", 28.0, 1.0, _raw(73))], range(73, 88), False),
    ("left_zones_only", [], (76, 77), False),
]
# fmt: on


@pytest.mark.parametrize(
    ("extra", "zones", "expected"),
    [c[1:] for c in DIRECTION_CASES],
    ids=[c[0] for c in DIRECTION_CASES],
)
def test_confirms_direction(extra: list[Any], zones: Any, expected: bool) -> None:
    # Precondition: raw 47 lies on the shooter's right (goal-frame y > 0), as does 28 m.
    assert 34.0 - _opta_y(47.0) > 0
    table = _gm().zone_sides(_frame([*ROWS, *extra]))
    kwargs = {} if zones is None else {"zones": zones}
    assert _gm().confirms_direction(table, **kwargs) is expected


def test_confirms_direction_needs_both_sides() -> None:
    """§5.1: no right row (zones 76-87) -> False, although every left row is consistent."""
    left_only = [r for r in ROWS if '"value":80' not in r[3]]
    table = _gm().zone_sides(_frame(left_only))
    assert 80 not in table["zone"].to_list() and 76 in table["zone"].to_list()
    assert _gm().confirms_direction(table) is False


@pytest.mark.parametrize(
    "text",
    ["not json", "null", '{"type": {"value": 76}}', "[76]", '[{"kind": {"value": 76}}]',
     '[{"type": 76}]'],
    ids=["not_json", "null", "object", "list_of_ints", "no_type", "type_not_object"],
)  # fmt: skip
def test_zone_sides_bad_raw_qualifiers(text: str) -> None:
    """Build ratify 9: a ``raw_qualifiers`` that is not a JSON list of ``{"type": {"value":
    ...}}`` raises ``ValueError`` (a JSON decode error is one too), never a bare TypeError."""
    with pytest.raises(ValueError):
        _gm().zone_sides(_frame([*ROWS, ("whoscored", 31.0, 1.0, text)]))
