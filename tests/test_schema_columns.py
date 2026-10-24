"""Column specs, dtypes, vocabularies and leakage roles (C3; spec §1, §2.1, §2.2)."""

from __future__ import annotations

import dataclasses
import math

import polars as pl
import pytest
import schema_cases as sc

from football_core import schema

# --- Constants and vocabularies -------------------------------------------------------------


def test_schema_version_and_constants() -> None:
    assert schema.SCHEMA_VERSION == "1.0"
    assert schema.TABLES == ("shots", "freeze_frames")
    assert schema.ID_PATTERN == r"^(sb|ws):\S+$"
    assert dict(schema.ID_PREFIXES) == {"statsbomb": "sb:", "whoscored": "ws:"}
    assert frozenset({"goal", "saved"}) == schema.ON_TARGET_OUTCOMES
    assert dict(schema.PERIOD_START_MINUTE) == {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
    assert schema.DISTANCE_TOLERANCE == 1e-9
    assert schema.ANGLE_TOLERANCE == 1e-9
    assert schema.MAX_EXAMPLES == 5


def test_id_prefixes_are_read_only() -> None:
    with pytest.raises(TypeError):
        schema.ID_PREFIXES["opta"] = "op:"  # type: ignore[index]


@pytest.mark.parametrize("name", sorted(sc.VOCABULARIES))
def test_vocabularies_are_pinned(name: str) -> None:
    # Tuple order is the Enum physical order: reordering is a MAJOR bump (spec §5).
    assert getattr(schema, name) == sc.VOCABULARIES[name]


@pytest.mark.parametrize(("enum_name", "vocab_name"), sorted(sc.ENUM_CONSTANTS.items()))
def test_enum_dtypes_follow_vocabularies(enum_name: str, vocab_name: str) -> None:
    assert getattr(schema, enum_name) == pl.Enum(sc.VOCABULARIES[vocab_name])


def test_roles_constant_holds_the_six_base_roles() -> None:
    # A seventh role is tolerated only as the answer to open issue 1 (end_* stop point).
    assert set(sc.BASE_ROLES) <= set(schema.ROLES)
    assert len(schema.ROLES) in (6, 7)
    used = {spec.role for spec in schema.SHOT_COLUMNS + schema.FREEZE_FRAME_COLUMNS}
    assert set(schema.ROLES) - set(sc.BASE_ROLES) <= used  # no unused extra role


# --- Dtypes, order, nullability, bounds ----------------------------------------------------


def test_shot_schema_is_pinned() -> None:
    assert pl.Schema(sc.SHOT_DTYPES) == schema.SHOT_SCHEMA
    assert tuple(schema.SHOT_SCHEMA.names()) == sc.SHOT_NAMES
    assert len(sc.SHOT_NAMES) == 47


def test_freeze_frame_schema_is_pinned() -> None:
    assert pl.Schema(sc.FREEZE_FRAME_DTYPES) == schema.FREEZE_FRAME_SCHEMA
    assert tuple(schema.FREEZE_FRAME_SCHEMA.names()) == sc.FREEZE_FRAME_NAMES


@pytest.mark.parametrize(
    ("specs", "expected"),
    [
        (lambda: schema.SHOT_COLUMNS, sc.SHOT_COLUMNS),
        (lambda: schema.FREEZE_FRAME_COLUMNS, sc.FREEZE_FRAME_COLUMNS),
    ],
    ids=["shots", "freeze_frames"],
)
def test_nullability_is_pinned(specs, expected) -> None:
    got = {spec.name: spec.nullable for spec in specs()}
    assert got == {name: nullable for name, _, nullable in expected}


def test_tristate_columns_are_nullable() -> None:
    # Spec §2.0: provider-only or unrecorded attributes are null, never faked.
    nullable = {spec.name for spec in schema.SHOT_COLUMNS if spec.nullable}
    required_nullable = {
        "first_time",
        "follows_dribble",
        "open_goal",
        "one_on_one",
        "big_chance",
        "deflected",
        "technique",
        "assist_type",
        "gk_x",
        "gk_y",
        "goal_y",
        "goal_z",
        "provider_xg",
        "provider_psxg",
    }
    assert required_nullable <= nullable


def test_column_dtypes_are_instances() -> None:
    for spec in schema.SHOT_COLUMNS + schema.FREEZE_FRAME_COLUMNS:
        assert isinstance(spec.dtype, pl.DataType), spec.name


@pytest.mark.parametrize(
    ("specs", "bounds"),
    [
        (lambda: schema.SHOT_COLUMNS, sc.SHOT_BOUNDS),
        (lambda: schema.FREEZE_FRAME_COLUMNS, sc.FREEZE_FRAME_BOUNDS),
    ],
    ids=["shots", "freeze_frames"],
)
def test_bounds_are_pinned(specs, bounds) -> None:
    got = {
        spec.name: (spec.ge, spec.le)
        for spec in specs()
        if spec.ge is not None or spec.le is not None
    }
    assert set(got) == set(bounds)
    for name, (ge, le) in bounds.items():
        got_ge, got_le = got[name]
        assert got_ge == ge, name
        assert (got_le is None and le is None) or math.isclose(got_le, le), name


def test_angle_upper_bound_is_exactly_pi() -> None:
    # opening_angle returns exactly np.pi on the goal line between the posts.
    (angle,) = [spec for spec in schema.SHOT_COLUMNS if spec.name == "angle_rad"]
    assert angle.le == math.pi


def test_period_vocabulary() -> None:
    (period,) = [spec for spec in schema.SHOT_COLUMNS if spec.name == "period"]
    assert tuple(period.allowed) == (1, 2, 3, 4, 5)


@pytest.mark.parametrize(
    ("specs", "ids", "unique"),
    [
        (lambda: schema.SHOT_COLUMNS, sc.SHOT_ID_COLUMNS, {"shot_id"}),
        (lambda: schema.FREEZE_FRAME_COLUMNS, sc.FREEZE_FRAME_ID_COLUMNS, set()),
    ],
    ids=["shots", "freeze_frames"],
)
def test_id_and_unique_flags(specs, ids, unique) -> None:
    assert {spec.name for spec in specs() if spec.is_id} == ids
    assert {spec.name for spec in specs() if spec.unique} == unique


def test_column_spec_is_frozen() -> None:
    spec = schema.SHOT_COLUMNS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.role = "pre_shot"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("specs", "polars_schema", "model", "table"),
    [
        (
            lambda: schema.SHOT_COLUMNS,
            lambda: schema.SHOT_SCHEMA,
            lambda: schema.SHOT_MODEL,
            "shots",
        ),
        (
            lambda: schema.FREEZE_FRAME_COLUMNS,
            lambda: schema.FREEZE_FRAME_SCHEMA,
            lambda: schema.FREEZE_FRAME_MODEL,
            "freeze_frames",
        ),
    ],
    ids=["shots", "freeze_frames"],
)
def test_schemas_match_column_specs(specs, polars_schema, model, table) -> None:
    names = tuple(spec.name for spec in specs())
    assert tuple(polars_schema().names()) == names
    assert schema.column_names(table) == names
    assert list(model().columns) == list(names)
    assert model().strict is True
    for spec in specs():
        column = model().columns[spec.name]
        assert polars_schema()[spec.name] == spec.dtype
        assert column.nullable == spec.nullable, spec.name
        assert column.unique == spec.unique, spec.name


def test_models_do_not_coerce() -> None:
    # Spec §0.6: validate_* never casts.
    for model in (schema.SHOT_MODEL, schema.FREEZE_FRAME_MODEL):
        assert not model.coerce
        assert not any(column.coerce for column in model.columns.values())


def test_every_column_has_role_and_description() -> None:
    for spec in schema.SHOT_COLUMNS + schema.FREEZE_FRAME_COLUMNS:
        assert spec.role in schema.ROLES, spec.name
        assert isinstance(spec.description, str) and spec.description.strip(), spec.name


@pytest.mark.parametrize(
    ("model", "specs"),
    [
        (lambda: schema.SHOT_MODEL, lambda: schema.SHOT_COLUMNS),
        (lambda: schema.FREEZE_FRAME_MODEL, lambda: schema.FREEZE_FRAME_COLUMNS),
    ],
    ids=["shots", "freeze_frames"],
)
def test_pandera_metadata_carries_role(model, specs) -> None:
    for spec in specs():
        assert model().columns[spec.name].metadata == {"role": spec.role}, spec.name


# --- Roles: the M1.0 / P2 allow-list guard --------------------------------------------------


def _role(name: str) -> str:
    (spec,) = [spec for spec in schema.SHOT_COLUMNS if spec.name == name]
    return spec.role


@pytest.mark.parametrize("role", sorted(sc.ROLE_SETS_FIXED))
def test_role_sets_are_pinned(role: str) -> None:
    assert schema.column_names("shots", roles=role) == sc.ROLE_SETS_FIXED[role]


def test_remaining_columns_are_post_shot_or_outcome_family() -> None:
    fixed = {name for names in sc.ROLE_SETS_FIXED.values() for name in names}
    rest = [name for name in sc.SHOT_NAMES if name not in fixed]
    assert set(rest) == {*sc.STOP_POINT_COLUMNS, *sc.P2_POST_SHOT, *sc.OUTCOME_LABELS}
    assert all(_role(name) == "outcome" for name in sc.OUTCOME_LABELS)
    assert all(_role(name) == "post_shot" for name in sc.P2_POST_SHOT)


def test_stop_point_columns_are_never_a_feature_role() -> None:
    """Open issue 1 (football r3 M1), resolution (a).

    end_x / end_y / end_z hold where the keeper or a blocker stopped the ball. On P2's
    on-target population their value (and, for WhoScored, their nullness alone) separates
    goal from saved, so a role that P2 may use as a feature makes the P3 goals-prevented
    measure circular. They must carry a role that is never a model feature: ``outcome``
    (widened) or one new role holding exactly these three columns.
    """
    roles = {_role(name) for name in sc.STOP_POINT_COLUMNS}
    assert len(roles) == 1, roles
    (role,) = roles
    assert role not in sc.FEATURE_ROLES
    if role != "outcome":
        assert role not in sc.BASE_ROLES
        assert schema.column_names("shots", roles=role) == sc.STOP_POINT_COLUMNS


def test_post_shot_is_exactly_the_p2_set() -> None:
    # football-shot-models methodology: P2 = pre-shot + goal-mouth (goal_y, goal_z); plus
    # deflected. Anything else after the strike must not be reachable through post_shot.
    assert schema.column_names("shots", roles="post_shot") == sc.P2_POST_SHOT


def test_feature_roles_never_reach_the_stop_point() -> None:
    features = schema.column_names("shots", roles=sc.FEATURE_ROLES)
    assert set(features).isdisjoint(sc.STOP_POINT_COLUMNS)
    assert set(features).isdisjoint(sc.OUTCOME_LABELS)


def test_no_leakage_column_is_pre_shot() -> None:
    pre_shot = set(schema.column_names("shots", roles="pre_shot"))
    leaky = {
        "end_x",
        "end_y",
        "end_z",
        "goal_y",
        "goal_z",
        "outcome",
        "is_goal",
        "is_own_goal",
        "on_target",
        "deflected",
        "big_chance",
        "provider_xg",
        "provider_psxg",
        "assist_event_id",
        "raw_type",
        "raw_qualifiers",
    }
    assert pre_shot.isdisjoint(leaky)


def test_role_counts_add_up() -> None:
    counts = {role: len(schema.column_names("shots", roles=role)) for role in schema.ROLES}
    assert sum(counts.values()) == 47
    assert counts["identity"] == 9
    assert counts["pre_shot"] == 21
    assert counts["restricted"] == 1
    assert counts["lineage"] == 6


def test_freeze_frame_roles() -> None:
    got = {spec.name: spec.role for spec in schema.FREEZE_FRAME_COLUMNS}
    assert got == sc.FREEZE_FRAME_ROLES


# --- column_names ---------------------------------------------------------------------------

PRE_SHOT = sc.ROLE_SETS_FIXED["pre_shot"]


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        (("shots",), {"roles": "pre_shot"}, PRE_SHOT),
        (
            ("shots",),
            {"roles": ["pre_shot", "restricted"]},
            tuple(n for n in sc.SHOT_NAMES if n in {*PRE_SHOT, "big_chance"}),
        ),
        (
            ("shots",),
            {"roles": ("restricted", "pre_shot")},  # union keeps canonical order
            tuple(n for n in sc.SHOT_NAMES if n in {*PRE_SHOT, "big_chance"}),
        ),
        (("shots",), {"roles": frozenset({"lineage"})}, sc.ROLE_SETS_FIXED["lineage"]),
        (("shots",), {"roles": ()}, ()),
        (("shots",), {"roles": []}, ()),
        (("shots",), {"roles": None}, sc.SHOT_NAMES),
        ((), {}, sc.SHOT_NAMES),
        (("freeze_frames",), {}, sc.FREEZE_FRAME_NAMES),
        (("freeze_frames",), {"roles": "pre_shot"}, ("is_teammate", "is_keeper", "x", "y")),
        (("freeze_frames",), {"roles": "lineage"}, ()),
    ],
    ids=[
        "single_role_string_not_characters",
        "list_union",
        "union_canonical_order",
        "frozenset",
        "empty_tuple",
        "empty_list",
        "none_is_all",
        "default_table_is_shots",
        "freeze_frames_all",
        "freeze_frames_pre_shot",
        "freeze_frames_role_without_columns",
    ],
)
def test_column_names_role_filters(args, kwargs, expected) -> None:
    got = schema.column_names(*args, **kwargs)
    assert isinstance(got, tuple)
    assert got == expected


@pytest.mark.parametrize(
    ("args", "kwargs", "bad"),
    [
        (("passes",), {}, "passes"),
        (("shots",), {"roles": "preshot"}, "preshot"),
        (("shots",), {"roles": ["pre_shot", "label"]}, "label"),
        (("freeze_frames",), {"roles": "Pre_Shot"}, "Pre_Shot"),
    ],
    ids=["table", "role", "role_in_collection", "role_case"],
)
def test_column_names_rejects_unknown_role_and_table(args, kwargs, bad) -> None:
    with pytest.raises(ValueError) as info:
        schema.column_names(*args, **kwargs)
    message = str(info.value)
    assert repr(bad) in message or bad in message
    allowed = schema.TABLES if bad == "passes" else schema.ROLES
    assert all(name in message for name in allowed)
