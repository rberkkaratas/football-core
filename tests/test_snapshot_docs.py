"""Docs, packaging and docstrings stay true to the snapshot module (C6 spec §1.2, §7)."""

from __future__ import annotations

import importlib.resources
import inspect
import json
import re
from pathlib import Path

import snapshot_cases as sc

from football_core import cli
from football_core.io import snapshot

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PAGE = ROOT / "docs" / "api" / "snapshot.md"
CLI_PAGE = ROOT / "docs" / "api" / "cli.md"
PUBLIC_FUNCTIONS = (
    "write_snapshot",
    "verify_snapshot",
    "load_snapshot",
    "read_manifest",
    "source_from_files",
    "source_from_git",
    "source_from_sql",
    "content_digest",
    "git_revision",
    "environment",
    "register_builder",
    "get_builder",
    "builder_names",
    "snapshot_from_builder",
    "build_from_parquet",
)


def test_api_pages_and_nav() -> None:
    assert "::: football_core.io.snapshot" in SNAPSHOT_PAGE.read_text(encoding="utf-8")
    assert "::: football_core.cli" in CLI_PAGE.read_text(encoding="utf-8")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(r"^\s*- io\.snapshot: api/snapshot\.md\s*$", nav, re.MULTILINE)
    assert re.search(r"^\s*- cli: api/cli\.md\s*$", nav, re.MULTILINE)


def _example_manifest(text: str) -> dict[str, object]:
    blocks = re.findall(r"```json\n(.*?)```", text, re.DOTALL)
    candidates = [b for b in blocks if '"manifest_version"' in b]
    assert candidates, "no example manifest in docs/api/snapshot.md"
    raw = candidates[0]
    raw = raw.replace('"<64 hex>"', '"' + "0" * 64 + '"')
    raw = re.sub(r'"<40 hex>\|null"', "null", raw)
    loaded: dict[str, object] = json.loads(raw)
    return loaded


def test_docs_manifest_example_parses() -> None:
    m = _example_manifest(SNAPSHOT_PAGE.read_text(encoding="utf-8"))
    files = m["files"]
    assert isinstance(files, list)
    m["data_digest"] = sc.sha256sum_digest((f["path"], f["sha256"]) for f in files)
    source = m["source"]
    assert isinstance(source, dict)
    if source.get("files"):
        source["revision"] = sc.sha256sum_digest((f["path"], f["sha256"]) for f in source["files"])
    parsed = snapshot.Manifest.from_json(sc.dumps(m))
    assert parsed.name == m["name"]


def test_py_typed_marker_present() -> None:
    assert (importlib.resources.files("football_core") / "py.typed").is_file()


def test_data_policy_describes_layout() -> None:
    policy = (ROOT / "docs" / "data-policy.md").read_text(encoding="utf-8")
    assert "manifests/<name>.json" in policy
    assert "list_files=False" in policy


def test_io_package_has_no_loader_imports() -> None:
    """``football_core/io/__init__.py`` is a docstring only (spec §1.1)."""
    import football_core.io as io_package

    source = inspect.getsource(io_package)
    assert "import" not in source.replace("from __future__ import annotations", "")
    assert io_package.__doc__


def test_public_functions_have_doctests() -> None:
    for name in PUBLIC_FUNCTIONS:
        doc = inspect.getdoc(getattr(snapshot, name)) or ""
        assert ">>>" in doc, f"{name} has no doctest example"
        assert "Parameters" in doc or not inspect.signature(getattr(snapshot, name)).parameters
    assert ">>>" in (inspect.getdoc(cli.main) or "")


def test_module_docstrings() -> None:
    assert snapshot.__doc__ and len(snapshot.__doc__.strip()) > 80
    assert cli.__doc__
