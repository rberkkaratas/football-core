"""Schema 1.2: the ``gk_id`` shot column and its rule (spec ``runs/MS5/goalmouth`` §4, §10.6)."""

from __future__ import annotations

import polars as pl
import pytest
import schema_cases as sc

from football_core import schema

RULE = "gk_id_matches_own_goal"
RULE_COLUMNS = ("shot_id", "gk_id", "is_own_goal", "player_id")


def test_gk_id_spec() -> None:
    assert schema.SCHEMA_VERSION == "1.3"  # PR1.xg-context: + prev_action, rebound
    names = [c.name for c in schema.SHOT_COLUMNS]
    assert names.index("gk_id") == names.index("player_id") + 1
    (spec,) = [c for c in schema.SHOT_COLUMNS if c.name == "gk_id"]
    assert spec.dtype == pl.String()
    assert (spec.role, spec.nullable, spec.is_id, spec.unique) == ("identity", True, True, False)
    assert spec.description == (
        "Goalkeeper of the defending team on the pitch at the strike (lineup state); "
        "null for own goals."
    )
    (rule,) = [r for r in schema.SHOT_RULES if r.name == RULE]
    assert rule.columns == RULE_COLUMNS


def test_gk_id_valid_fixture(valid_shots: pl.DataFrame) -> None:
    """Precondition for the cases below: the fixture keeps one own goal with a null gk_id and
    four shots with a keeper that is not the shooter."""
    rows = valid_shots.select("gk_id", "is_own_goal", "player_id").rows()
    assert [(g is None, og) for g, og, _ in rows] == [
        (False, False), (False, False), (False, False), (True, True), (False, False),
    ]  # fmt: skip
    assert all(g != p for g, _, p in rows if g is not None)
    schema.validate_shots(valid_shots)


# row, cells: each breaks gk_id_matches_own_goal (or, for a wrong prefix, ids_match_provider)
CASES = [
    ("null_on_a_shot", 0, {"gk_id": None}, RULE),
    ("set_on_an_own_goal", 3, {"gk_id": "ws:91101"}, RULE),
    ("equal_to_the_shooter", 0, {"gk_id": sc.SHOOTER_1}, RULE),
    ("other_provider_prefix", 2, {"gk_id": "sb:91201"}, "ids_match_provider"),
]


@pytest.mark.parametrize(
    ("row", "cells", "check"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_gk_id_rule(valid_shots: pl.DataFrame, row: int, cells: dict, check: str) -> None:
    with pytest.raises(schema.SchemaValidationError) as info:
        schema.validate_shots(sc.with_values(valid_shots, row, **cells))
    err = info.value
    assert {f.check for f in err.failures} == {check}, str(err)
    (failure,) = err.failures
    assert failure.rows == (row,)
    assert check in str(err)
    if check == RULE:
        assert failure.columns == RULE_COLUMNS
