"""Schema 1.3: the ``prev_action`` and ``rebound`` shot columns (spec ``runs/PR1/xg-context``
§2.1). Expected names, levels and roles are typed from the spec, not read from the module."""

from __future__ import annotations

import polars as pl

from football_core import schema

PREV_ACTIONS = ("pass", "take_on", "regain", "shot", "other",
                "opp_clearance", "opp_block", "opp_keeper", "opp_other")  # fmt: skip


def test_context_columns() -> None:
    """§2.1: both columns right after ``one_on_one``, ``pre_shot``, nullable (null: not derived
    or own goal); ``prev_action`` an Enum of the nine levels in this physical order (appending
    one later is a MAJOR bump); both described as keeper-derived, xG only (MS6 E5)."""
    assert schema.PREV_ACTIONS == PREV_ACTIONS
    assert pl.Enum(PREV_ACTIONS) == schema.PREV_ACTION
    names = list(schema.SHOT_SCHEMA.names())
    i = names.index("one_on_one")
    assert names[i + 1 : i + 3] == ["prev_action", "rebound"]
    specs = {s.name: s for s in schema.SHOT_COLUMNS}
    assert specs["prev_action"].dtype == pl.Enum(PREV_ACTIONS)
    assert specs["rebound"].dtype == pl.Boolean()
    for name in ("prev_action", "rebound"):
        spec = specs[name]
        assert (spec.role, spec.nullable, spec.is_id, spec.unique) == ("pre_shot", True, False,
                                                                       False)  # fmt: skip
        assert "MS6 E5" in spec.description, name
        assert "own goal" in spec.description, name
    assert "3 s" in specs["rebound"].description
    assert "q94" in specs["prev_action"].description
    assert "prev_action" in schema.column_names("shots", roles="pre_shot")
    assert "rebound" in schema.column_names("shots", roles="pre_shot")


def test_schema_version_is_1_3() -> None:
    """§2: adding two nullable columns is a MINOR bump, 1.2 → 1.3."""
    assert schema.SCHEMA_VERSION == "1.3"
