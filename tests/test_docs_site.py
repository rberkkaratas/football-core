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
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
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
        "'''Module.'''\nfrom extpkg import sub\nimport polars as pl\nLIMIT = 3\n"
        "def fit():\n    '''Fit.'''\ndef _helper():\n    '''Private.'''\n"
        "class Base:\n    '''Base.'''\n    def helper(self):\n        '''Help.'''\n"
        "class Row(Base):\n    '''Row.'''\n    def cells(self):\n        '''Cells.'''\n"
        "    def total(self):\n        '''Total.'''\n"
    ),
}


def _tiny(**docs: str) -> dict[str, str]:
    """The tiny package with the docstring of each named ``mod.py`` object replaced."""
    text = PKG["mod.py"]
    for name, doc in docs.items():
        old = f"'''{name.title()}.'''" if name != "mod" else "'''Module.'''"
        assert text.count(old) == 1, name  # precondition: exactly one docstring replaced
        text = text.replace(old, repr(doc))
    return {**PKG, "mod.py": text}


def test_extension_on_tiny_package() -> None:
    """A documented function links; an undocumented constant, a private function, a missing
    name and a name through an import alias of a package that is not loaded (``from extpkg
    import sub``, ``import polars as pl``: griffe raises on those paths) are plain code; a
    doctest region is left as it is, and prose after it is converted again."""
    doc = (
        f"See {role('func', 'fit')}, {role('data', 'LIMIT')}, {role('func', '_helper')} and "
        f"{role('class', 'Missing')}.\n"
        f"Aliases {role('func', 'sub.fn')} and {role('class', 'pl.DataFrame')}.\n\n"
        f">>> {role('func', 'fit')}\n\nAgain {role('func', 'fit')}.\n"
    )
    ext = griffe.load_extensions(roles.SphinxRoles())
    with griffe.temporary_visited_package("tiny", _tiny(mod=doc), extensions=ext) as pkg:
        lines = pkg["mod"].docstring.value.splitlines()
    assert lines[0] == "See [`fit`][tiny.mod.fit], `LIMIT`, `_helper` and `Missing`."
    assert lines[1] == "Aliases `sub.fn` and `pl.DataFrame`."
    assert lines[3] == f">>> {role('func', 'fit')}"
    assert lines[5] == "Again [`fit`][tiny.mod.fit]."


def test_extension_class_members() -> None:
    """A method role inside a class links to its sibling and to an inherited method (the
    owner-class candidate: griffe's own scope lookup misses inherited members)."""
    doc = f"Uses {role('meth', 'cells')} and {role('meth', 'helper')}."
    ext = griffe.load_extensions(roles.SphinxRoles())
    with griffe.temporary_visited_package("tiny", _tiny(row=doc, total=doc), extensions=ext) as pkg:
        values = {pkg[p].docstring.value for p in ("mod.Row", "mod.Row.total")}
    assert values == {"Uses [`cells`][tiny.mod.Row.cells] and [`helper`][tiny.mod.Base.helper]."}


def test_resolver_external_prefix() -> None:
    """An external prefix links its own package and members only, by whole dotted segment."""
    with griffe.temporary_visited_package("tiny", PKG) as pkg:
        find = roles.resolver(pkg, ["other"])
        found = [find(t) for t in ("other", "other.x", "otherpkg.x", "othe")]
    assert found == ["other", "other.x", None, None]


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
        # a doctest region ends at the next blank line; the prose after it is converted
        (
            f">>> {role('func', 'f')}\n\nSee {role('func', 'f')}.",
            f">>> {role('func', 'f')}\n\nSee [`f`][f].",
        ),
    ],
)
def test_convert_labels(text: str, expected: str) -> None:
    assert roles.convert(text, lambda t: t) == expected


# --- pages: codes and the built site -----------------------------------------------------------

EXAMPLES = {
    "ruling": ["PM E2 (b)", "PM 2026-10-28 (a)", "PM X5"],
    "milestone": ["MS6", "PR1", "CH1", "DR1"],
    "component": ["MS6 P3.1", "M1.4", "C3.1", "S0.2", "C7", "M0"],
    "review": ["E1", "R14", "A5", "X5", "D2", "G1", "S3", "RI1", "RI12"],
    "run": ["run8"],
    "package": ["package 2", "Package 1"],
    "rule": ["Rule 1", "rule 3"],
    "section": ["§1", "§ 2", "§4.3"],
}
# Breadth is intentional: legitimate tokens such as E0, F1, R2, T1, S3 and "2 PM" also match,
# and no public page needs one.
NEAR_MISSES = ["q214", "#E4002B", "1.2", "2015/16", "PMF", "CV", "AUC", "xG", "rerun8", "MS", "RIP"]
NEAR_MISSES += ["§", "RI", "TRI1"]


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


def test_pages_standard_library_only() -> None:
    """``football_docs.pages`` imports with griffe and yaml blocked (no ``docs`` group); the
    role pattern of ``roles`` and the site check share its ``ROLE_NAMES``."""
    block = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('griffe', 'yaml', 'mkdocs'):\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import football_docs.pages\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", block], capture_output=True, text=True)
    assert (out.returncode, out.stdout) == (0, "ok\n"), out.stderr
    assert roles.ROLE.pattern.count("|".join(pages.ROLE_NAMES)) == 1


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


# --- pages: the results block ------------------------------------------------------------------

README = """\
<!-- m:summary:start -->
| Model | Data | CV |
|---|---|---|
| **GBM** | A | 0.26 |
| GBM | B | 0.27 |
| LR | A | 0.30 |

GBM: lower is better. GBM: same rows twice. LR is the baseline.
<!-- m:summary:end -->
"""
GBM_A = pages.Pick("xG", "m", ("GBM", "A"), ("CV",))


def _pyproject(**entry: object) -> str:
    """A ``pyproject.toml`` with one results pick (``GBM_A`` with ``entry`` applied)."""
    fields = {"result": "xG", "block": "m", "row": ["GBM", "A"], "columns": ["CV"], **entry}
    lines = [f"{k} = {v!r}".replace("'", '"') for k, v in fields.items() if v is not None]
    return "[[tool.football_docs.results]]\n" + "\n".join(lines) + "\n"


def test_results_block_picks_by_name() -> None:
    """Precondition for the raises table: the README fixture, the pick and the TOML writer work;
    a note shared by two picks is printed once, notes in pick order."""
    assert pages.picks(_pyproject()) == (GBM_A,)
    lr = pages.Pick("LR", "m", ("LR",), ("CV", "Data"), ("LR is", "GBM: lower"))
    block = pages.results_block(README, [GBM_A, lr, lr])
    lr_row = "| LR | CV: 0.30<br>Data: A |"
    assert block.splitlines()[3:6] == ["| xG | CV: 0.26 |", lr_row, lr_row]
    assert block.split("\n\n")[1:3] == ["LR is the baseline.", "GBM: lower is better."]
    assert block.count("LR is the baseline.") == 1


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: pages.results_block(README, []), "at least one pick"),
        (lambda: pages.results_block(README, [replace(GBM_A, block="x")]), "no generated block"),
        (lambda: pages.results_block(README, [replace(GBM_A, row=("XGB",))]), "matches 0 rows"),
        (lambda: pages.results_block(README, [replace(GBM_A, row=("GBM",))]), "matches 2 rows"),
        (lambda: pages.results_block(README, [replace(GBM_A, columns=("AUC",))]), "no column"),
        (lambda: pages.results_block(README, [replace(GBM_A, notes=("Nope",))]), "0 sentences"),
        (lambda: pages.results_block(README, [replace(GBM_A, notes=("GBM:",))]), "2 sentences"),
        (lambda: pages.results_block(README, [replace(GBM_A, result="xG 0.99")]), "not in the"),
        (lambda: pages.unsourced_numbers("12 shots"), "at least one source"),
        (lambda: pages.picks("[project]\nname = 'x'\n"), "no \\[\\[tool.football_docs"),
        (lambda: pages.picks(_pyproject(row=[])), "needs result, block, row and columns"),
        (lambda: pages.picks(_pyproject(columns=[])), "needs result, block, row and columns"),
        (lambda: pages.picks(_pyproject(result=None)), "needs result, block, row and columns"),
        (lambda: pages.picks(_pyproject(row="GBM")), "row must be a list of strings"),
        (lambda: theme.value(viz.THEMES["light"], "family"), "not a colour field"),
    ],
    ids=[
        "no-pick",
        "no-block",
        "row-0",
        "row-2",
        "no-column",
        "note-0",
        "note-2",
        "label-number",
        "no-source",
        "no-entry",
        "empty-row",
        "empty-columns",
        "no-result",
        "row-str",
        "theme-not-colour",
    ],
)
def test_documented_raises(call: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_results_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``results <page>`` exits 1 on a stale block and 0 on a current one; ``--write`` rewrites
    only the block; an error (a page without a block) is one stderr line, exit 1."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(_pyproject(), encoding="utf-8")
    stale = "# Home\n\n<!-- results:start -->\n| old |\n<!-- results:end -->\n\nEnd.\n"
    (tmp_path / "index.md").write_text(stale, encoding="utf-8")
    assert pages.main(["results", "index.md"]) == 1
    assert pages.main(["results", "index.md", "--write"]) == 0
    text = (tmp_path / "index.md").read_text(encoding="utf-8")
    assert text == "# Home\n\n" + pages.results_block(README, [GBM_A]) + "\n\nEnd.\n"
    assert pages.main(["results", "index.md"]) == 0
    (tmp_path / "index.md").write_text("# Home\n", encoding="utf-8")
    assert pages.main(["results", "index.md"]) == 1


# --- core's own pages --------------------------------------------------------------------------


def _page_files() -> list[Path]:
    readmes = [REPO / "README.md", *sorted(REPO.glob("[!.]*/README*.md"))]
    return readmes + sorted(DOCS.rglob("*.md"))


def test_core_pages_and_docstrings_clean() -> None:
    """No code or stale phrase on any README or docs page (no page has a generated block), in
    ``viz.py`` docstrings (render-only for the model repos' cache) or in any ``football_docs``
    docstring, doctests included."""
    files = _page_files()
    assert {REPO / "README.md", DOCS / "index.md", DOCS / "roadmap.md"} <= set(files)
    # no page of core has a generated block, so the scan below exempts nothing
    assert [(p.name, b) for p in files for b, _ in pages.generated_blocks(p.read_text())] == []
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


def test_core_pages_state_loader_rules() -> None:
    """The home page's reconcile claim excludes shoot-outs, as the roadmap row does (StatsBomb
    logs shoot-out kicks as goal Shot events); the viz page and module docstring say a StatsBomb
    stop point's ``goal_y`` is already projected and snapped, and only ``goal_z`` is the stop."""
    home = " ".join((DOCS / "index.md").read_text(encoding="utf-8").split())
    assert "with the recorded score in every match (shoot-outs excluded" in home
    roadmap = (DOCS / "roadmap.md").read_text(encoding="utf-8")
    assert "goals reconcile with the score in every match (shoot-outs excluded)" in roadmap
    for text in ((DOCS / "api" / "viz.md").read_text(encoding="utf-8"), viz.__doc__ or ""):
        flat = " ".join(text.split()).replace("``", "`")
        assert "already projected `goal_y` to the goal line" in flat
        assert (
            "0.1 yd lattice" in flat and "`goal_z` is the height where the ball was stopped" in flat
        )
        assert "not a projection" not in flat and "follow-up" not in flat


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
    assert [(p["primary"], p["accent"]) for p in th["palette"]] == [("custom", "custom")] * 3
    # automatic entry first (follows the device on every visit), then the two toggled schemes
    assert [(p["media"], p.get("scheme")) for p in th["palette"]] == [
        ("(prefers-color-scheme)", None),
        ("(prefers-color-scheme: light)", "default"),
        ("(prefers-color-scheme: dark)", "slate"),
    ]
    assert "stylesheets/theme.css" in config["extra_css"]
    plugin = next(p for p in config["plugins"] if isinstance(p, dict) and "mkdocstrings" in p)
    options = plugin["mkdocstrings"]["handlers"]["python"]["options"]
    assert options["extensions"] == ["football_docs.roles"]
    assert options["show_source"] is False


# --- theme -------------------------------------------------------------------------------------


def test_committed_theme_files_current(capsys: pytest.CaptureFixture[str]) -> None:
    assert theme.main([str(DOCS), "--check"]) == 0, capsys.readouterr().out


def test_docs_target_checks_theme() -> None:
    """``make docs`` (what the Pages workflow deploys from) checks the theme files are current,
    after the strict build and the raw-role site check."""
    m = re.search(r"^docs:[^\n]*\n((?:\t[^\n]*\n)+)", (REPO / "Makefile").read_text(), re.M)
    assert m is not None
    steps = [line.strip() for line in m.group(1).splitlines()]
    assert steps[-3:] == [
        "uv run zensical build --strict",
        "uv run python -m football_docs.pages site site",
        "uv run python -m football_docs.theme docs --check",
    ]


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
