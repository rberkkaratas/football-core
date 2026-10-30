"""football_docs on football-core's own site (DR1 docs-sites spec §6, the core rows).

The shot-models suite tests the results block, number tokens, theme variables and the chart
pages; these are the core rows it cannot see: the role extension on a tiny package (links,
plain code, doctest regions, load order), the built-site role check, the pinned code pattern
list, and core's own pages (no internal code, sourced numbers, nav order, relative links,
generated theme files, text contrast, site config). Written at the DR1 gate, where the package's
QA had covered shot-models only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import griffe
import pytest
import yaml

from football_core import viz
from football_docs import pages, roles, theme

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"


def role(name: str, target: str) -> str:
    return f":{name}:`{target}`"


# --- roles -------------------------------------------------------------------------------------

PKG = {
    "__init__.py": "'''Package.'''\n",
    "mod.py": (
        "'''Module.'''\nLIMIT = 3\ndef fit():\n    '''Fit.'''\ndef _helper():\n    '''Private.'''\n"
    ),
}


def test_extension_on_tiny_package() -> None:
    """A documented function links; an undocumented constant, a private function and a missing
    name are plain code; a doctest region is left as it is."""
    doc = (
        f"See {role('func', 'fit')}, {role('data', 'LIMIT')}, {role('func', '_helper')} and "
        f"{role('class', 'Missing')}.\n\n>>> {role('func', 'fit')}\n"
    )
    pkg_files = {**PKG, "mod.py": PKG["mod.py"].replace("'''Module.'''", repr(doc), 1)}
    ext = griffe.load_extensions(roles.SphinxRoles())
    with griffe.temporary_visited_package("tiny", pkg_files, extensions=ext) as pkg:
        value = pkg["mod"].docstring.value
    assert value.splitlines()[0] == ("See [`fit`][tiny.mod.fit], `LIMIT`, `_helper` and `Missing`.")
    assert value.splitlines()[2] == f">>> {role('func', 'fit')}"


def test_extension_load_order() -> None:
    """A role into a package loaded later is plain code; the reverse direction links."""
    ext = griffe.load_extensions(roles.SphinxRoles())
    collection = griffe.ModulesCollection()
    first = {"__init__.py": f"'''See {role('func', 'second.fit')}.'''\ndef go():\n    '''Go.'''\n"}
    second = {"__init__.py": f"'''See {role('func', 'first.go')}.'''\ndef fit():\n    '''Fit.'''\n"}
    with (
        griffe.temporary_visited_package(
            "first", first, extensions=ext, modules_collection=collection
        ) as a,
        griffe.temporary_visited_package(
            "second", second, extensions=ext, modules_collection=collection
        ) as b,
    ):
        assert a.docstring.value == "See `second.fit`."
        assert b.docstring.value == "See [`first.go`][first.go]."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"a {role('func', '~pkg.mod.fit')} b", "a [`fit`][pkg.mod.fit] b"),
        (role("py:meth", "pkg.Row.cells"), "[`pkg.Row.cells`][pkg.Row.cells]"),
        (role("class", ".Row"), "[`Row`][.Row]"),
        (role("data", "pkg.\n    LIMIT"), "[`pkg.LIMIT`][pkg.LIMIT]"),
        ("no role here, `code` and :not:`a role`", "no role here, `code` and :not:`a role`"),
    ],
)
def test_convert_labels(text: str, expected: str) -> None:
    assert roles.convert(text, lambda t: t) == expected


# --- pages: codes and the built site -----------------------------------------------------------

EXAMPLES = {
    "ruling": ["PM E2 (b)", "PM 2026-10-28 (a)", "PM X5"],
    "milestone": ["MS6", "PR1", "CH1", "DR1"],
    "component": ["MS6 P3.1", "M1.4", "C3.1", "S0.2", "C7", "M0"],
    "review": ["E1", "R14", "A5", "X5", "D2", "G1", "S3"],
    "run": ["run8"],
    "package": ["package 2", "Package 1"],
    "rule": ["Rule 1", "rule 3"],
}
# Breadth is intentional: legitimate tokens such as E0, F1, R2, T1, S3 and "2 PM" also match,
# and no public page needs one.
NEAR_MISSES = ["q214", "#E4002B", "1.2", "2015/16", "PMF", "CV", "AUC", "xG", "rerun8", "MS"]


def test_codes_table() -> None:
    assert set(pages.CODES) == set(EXAMPLES)
    for kind, examples in EXAMPLES.items():
        for example in examples:
            assert re.search(pages.CODES[kind], example), (kind, example)
            assert pages.find_codes(f"Text {example} text."), example
    for near in NEAR_MISSES:
        assert pages.find_codes(f"Text {near} text.") == [], near
    stale = pages.STALE[0].upper()
    assert [c for _, c, _ in pages.find_codes(f"x {stale} y")] == [stale]


def test_raw_roles_site(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="zensical build"):
        pages.raw_roles(tmp_path)
    (tmp_path / "index.html").write_text('<a href="api/x/#pkg.fit"><code>fit</code></a>')
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "x.html").write_text(f"<p>{role('func', 'fit')} {role('class', 'Row')}</p>")
    (tmp_path / "search.json").write_text(f'{{"text": "{role("meth", "a.b")}"}}')
    (tmp_path / "notes.txt").write_text(role("func", "ignored"))
    assert pages.raw_roles(tmp_path) == [("api/x.html", 2), ("search.json", 1)]
    assert pages.main(["site", str(tmp_path)]) == 1
    (tmp_path / "api" / "x.html").write_text("<p>clean</p>")
    (tmp_path / "search.json").write_text("{}")
    assert pages.main(["site", str(tmp_path)]) == 0


# --- core's own pages --------------------------------------------------------------------------


def _page_files() -> list[Path]:
    readmes = [REPO / "README.md", *sorted(REPO.glob("[!.]*/README*.md"))]
    return readmes + sorted(DOCS.rglob("*.md"))


def test_core_pages_and_docstrings_clean() -> None:
    """No code or stale phrase on any README or docs page (generated blocks skipped), in
    ``viz.py`` docstrings (render-only for the model repos' cache) or in any ``football_docs``
    docstring, doctests included."""
    files = _page_files()
    assert {REPO / "README.md", DOCS / "index.md", DOCS / "roadmap.md"} <= set(files)
    hits = [
        (p.relative_to(REPO).as_posix(), *hit)
        for p in files
        for hit in pages.find_codes(pages.strip_generated(p.read_text(encoding="utf-8")))
    ]
    src = REPO / "src"
    hits += [("viz.py", *h) for h in pages.docstring_codes(src / "football_core" / "viz.py")]
    for module in sorted((src / "football_docs").glob("*.py")):
        hits += [(module.name, *h) for h in pages.docstring_codes(module, doctests=True)]
    assert hits == []


def test_core_home_numbers_sourced() -> None:
    home = (DOCS / "index.md").read_text(encoding="utf-8")
    sources = [(DOCS / f).read_text(encoding="utf-8") for f in ("roadmap.md", "conventions.md")]
    assert pages.unsourced_numbers(home, *sources) == []
    assert pages.unsourced_numbers(home + "\n123,457 shots", *sources) == ["123,457"]


def test_core_links_relative() -> None:
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r'\]\(/|src="/|href="/', text), path


def _config() -> Any:
    return yaml.safe_load((REPO / "mkdocs.yml").read_text(encoding="utf-8"))


def test_core_nav_order() -> None:
    nav = _config()["nav"]
    assert isinstance(nav, list)
    assert [next(iter(item)) for item in nav] == [
        "Home",
        "Conventions",
        "Data policy",
        "Modules",
        "Roadmap",
        "API reference",
    ]
    api = nav[-1]["API reference"]
    assert [next(iter(item)) for item in api] == [
        "pitch",
        "goalmouth",
        "schema",
        "io.snapshot",
        "io.statsbomb",
        "io.whoscored",
        "viz",
        "cli",
        "docs.roles",
        "docs.pages",
        "docs.theme",
        "docs.meta",
    ]
    for item in api:
        assert (DOCS / next(iter(item.values()))).is_file(), item


def test_core_site_config() -> None:
    config = _config()
    assert config["site_author"] == "R. Berk Karataş"
    th = config["theme"]
    assert isinstance(th, dict)
    assert th["font"]["text"] == "Inter"
    assert th["logo"] == th["favicon"] == "assets/logo.svg"
    assert [(p["primary"], p["accent"]) for p in th["palette"]] == [("custom", "custom")] * 2
    assert [p["scheme"] for p in th["palette"]] == ["default", "slate"]
    assert "stylesheets/theme.css" in config["extra_css"]
    plugin = next(p for p in config["plugins"] if isinstance(p, dict) and "mkdocstrings" in p)
    options = plugin["mkdocstrings"]["handlers"]["python"]["options"]
    assert options["extensions"] == ["football_docs.roles"]
    assert options["show_source"] is False


# --- theme -------------------------------------------------------------------------------------


def test_committed_theme_files_current(capsys: pytest.CaptureFixture[str]) -> None:
    assert theme.main([str(DOCS), "--check"]) == 0, capsys.readouterr().out


def test_css_tokens_match_themes() -> None:
    css = theme.css()
    for scheme, name in theme.SCHEMES:
        block = re.search(rf'\[data-md-color-scheme="{scheme}"\] \{{(.*?)\}}', css, re.S)
        assert block is not None
        decls = dict(re.findall(r"(--[\w-]+): (#[0-9a-f]+);", block.group(1)))
        expected = {v: theme.value(viz.THEMES[name], t) for v, t in theme.VARS + theme.OWN_VARS}
        assert decls == expected


@pytest.mark.parametrize("name", ["light", "dark"])
def test_text_pairs_contrast(name: str) -> None:
    """Every text colour the site maps (ink, muted, title, links) is at least 4.5:1 on the page
    background; links and the muted footer text also on the panel."""
    t = viz.THEMES[name]
    vars_ = dict(theme.VARS)
    link = theme.value(t, vars_["--md-typeset-a-color"])
    muted = theme.value(t, vars_["--md-default-fg-color--light"])
    pairs = [
        (t.ink, t.background),
        (muted, t.background),
        (t.title, t.background),
        (link, t.background),
        (link, t.panel),
        (muted, t.panel),
    ]
    for fg, bg in pairs:
        assert viz.contrast_ratio(fg, bg) >= 4.5, (name, fg, bg)
