"""Docs and packaging stay true to the schema (C3; spec §7 "docs stay true", §8, §9)."""

from __future__ import annotations

import inspect
import re
import tomllib
from pathlib import Path

import polars as pl
import pytest
import schema_cases as sc

from football_core import schema

ROOT = Path(__file__).resolve().parents[1]
DOCS_PAGE = ROOT / "docs" / "api" / "schema.md"
PUBLIC_FUNCTIONS = (
    "column_names",
    "empty_shots",
    "empty_freeze_frames",
    "cast_to_schema",
    "validate_shots",
    "validate_freeze_frames",
    "empty_matches",
    "validate_matches",
)


def _clean(cell: str) -> str:
    return re.sub(r"[`*]", "", cell).strip()


def _cells(line: str) -> list[str]:
    return [_clean(c) for c in line.strip().strip("|").split("|")]


def _column_tables(text: str) -> list[list[dict[str, str]]]:
    """Markdown tables whose header has Column/Name, Dtype, Null and Role cells."""
    tables: list[list[dict[str, str]]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        header = [c.lower() for c in _cells(lines[i])] if lines[i].lstrip().startswith("|") else []
        key = "column" if "column" in header else "name" if "name" in header else None
        if key and {"dtype", "null", "role"} <= set(header):
            rows = []
            i += 2  # skip the |---| separator
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                values = _cells(lines[i])
                row = dict(zip(header, values, strict=False))
                rows.append({"name": row[key], **row})
                i += 1
            tables.append(rows)
        i += 1
    return tables


def _dtype_aliases(dtype: pl.DataType) -> set[str]:
    aliases = {str(dtype)}
    if isinstance(dtype, pl.Enum):
        cats = tuple(dtype.categories.to_list())
        for vocab, values in sc.VOCABULARIES.items():
            if values == cats:
                aliases.add(f"Enum({vocab})")
    if dtype == sc.POLYGON:
        aliases.add("List(Array(Float64, 2))")
    return aliases


@pytest.fixture(scope="module")
def tables() -> list[list[dict[str, str]]]:
    assert DOCS_PAGE.exists(), "docs/api/schema.md is missing"
    found = _column_tables(DOCS_PAGE.read_text(encoding="utf-8"))
    expected = "3 column tables (shots, freeze_frames, matches)"
    assert len(found) == 3, f"expected {expected}, got {len(found)}"
    return found


@pytest.mark.parametrize(
    ("index", "specs"),
    [
        (0, lambda: schema.SHOT_COLUMNS),
        (1, lambda: schema.FREEZE_FRAME_COLUMNS),
        (2, lambda: schema.MATCH_COLUMNS),
    ],
    ids=["shots", "freeze_frames", "matches"],
)
def test_docs_tables_match_column_specs(tables, index: int, specs) -> None:
    rows = tables[index]
    names = [row["name"] for row in rows]
    assert names == [spec.name for spec in specs()], "every column exactly once, in order"
    for row, spec in zip(rows, specs(), strict=True):
        assert row["dtype"] in _dtype_aliases(spec.dtype), (spec.name, row["dtype"])
        assert row["null"].lower() == ("yes" if spec.nullable else "no"), spec.name
        assert row["role"] == spec.role, spec.name


def test_docs_page_renders_the_module() -> None:
    text = DOCS_PAGE.read_text(encoding="utf-8")
    assert "::: football_core.schema" in text
    for role in schema.ROLES:
        assert role in text, f"role {role!r} is not explained on the docs page"


def test_docs_page_is_in_the_nav() -> None:
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(r"-\s*schema:\s*api/schema\.md", nav)


@pytest.mark.parametrize("name", PUBLIC_FUNCTIONS)
def test_public_functions_have_doctest_examples(name: str) -> None:
    doc = inspect.getdoc(getattr(schema, name)) or ""
    assert "Examples" in doc and ">>>" in doc, name


def test_module_docstring_explains_roles() -> None:
    doc = schema.__doc__ or ""
    for role in schema.ROLES:
        assert role in doc, role


def test_runtime_dependencies_declared() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = [
        re.split(r"[<>=!~ ;]", d, maxsplit=1)[0].lower() for d in project["project"]["dependencies"]
    ]
    assert "polars" in deps
    assert "pandera[polars]" in deps
    # Reference libraries are test oracles only (playbook).
    assert not {"mplsoccer", "kloppy", "socceraction", "pandas"} & set(deps)


def test_schema_module_is_not_reexported() -> None:
    import football_core

    assert not hasattr(football_core, "validate_shots")


def _section(text: str, heading: str) -> str:
    start = text.index(f"\n## {heading}\n")
    end = min(i for i in (text.find("\n## ", start + 1), text.find("\n::: ", start)) if i > 0)
    return text[start:end]


def test_docs_enum_dtypes_name_the_vocabulary_constant(tables) -> None:
    # MS3.matches build ratify 7: an Enum column's dtype cell is `Enum(<schema constant>)`, the
    # public vocabulary name (Enum(PROVIDERS), Enum(GENDERS)); not the spec's lower-case
    # spelling and not the polars repr, which the column-spec test also accepts.
    specs = (schema.SHOT_COLUMNS, schema.FREEZE_FRAME_COLUMNS, schema.MATCH_COLUMNS)
    seen = 0
    for rows, columns in zip(tables, specs, strict=True):
        for row, spec in zip(rows, columns, strict=True):
            if not isinstance(spec.dtype, pl.Enum):
                continue
            seen += 1
            found = re.fullmatch(r"Enum\(([A-Z_]+)\)", row["dtype"])
            assert found, (spec.name, row["dtype"])
            vocab = getattr(schema, found.group(1), None)
            assert vocab == tuple(spec.dtype.categories.to_list()), (spec.name, row["dtype"])
    assert seen >= 8  # 6 shots enums + provider and gender in matches


def test_docs_versioning_states_the_current_version() -> None:
    # MS3.matches build ratify 8: the Versioning section and the error example carry the
    # current SCHEMA_VERSION and the rule that adding a canonical table is a MINOR bump.
    text = DOCS_PAGE.read_text(encoding="utf-8")
    versioning = _section(text, "Versioning")
    assert f'`SCHEMA_VERSION = "{schema.SCHEMA_VERSION}"`' in versioning
    assert "Adding a canonical table is a MINOR bump" in versioning
    errors = _section(text, "Validation and errors")
    assert f"(schema {schema.SCHEMA_VERSION})" in errors
    assert "validate_matches" in errors
