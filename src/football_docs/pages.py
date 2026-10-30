"""What a public docs page must pass, and the home page's results block.

A public page carries no internal process code (review rulings, milestone and package ids,
run names, spec sections): :data:`CODES` is the one pattern list both sites' tests and the
maintenance list use, :func:`find_codes` scans a page and :func:`docstring_codes` the
docstrings of a module.
Generated blocks (between ``<!-- name:start -->`` and ``<!-- name:end -->`` markers) are
skipped: they are card output, checked byte for byte elsewhere.

**Numbers are sourced, never typed.** :func:`unsourced_numbers` lists the number tokens of a
page that occur in none of its sources (the README's card blocks, the charts' titles), and
:func:`results_block` copies the home page's headline results by code from the README's
generated card blocks, choosing rows and columns by name (``[[tool.football_docs.results]]`` in
``pyproject.toml``, read by :func:`picks`).

**The built site.** :func:`raw_roles` counts Sphinx role text left in the built HTML and search
index, so ``make docs`` fails when a docstring role was not rendered. :data:`ROLE_NAMES`, the
role names it counts, lives here (and :mod:`football_docs.roles` imports it) so that this module
stays standard library only.

``python -m football_docs.pages site <dir>`` and ``python -m football_docs.pages results <page>
[--write]`` run the site check and the results block from the command line. Standard library
only.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

ROLE_NAMES: Final = ("func", "class", "meth", "data", "attr", "mod", "obj", "exc", "const")
CODES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ruling": r"\bPM\b",
        "milestone": r"\b(?:MS|PR|CH|DR)\d+\b",
        "component": r"\b[MPCS]\d+(?:\.\d+)+\b|\b[CM]\d+\b",
        "review": r"\b[ERAXDFGTS]\d{1,2}\b|\bRI\d+\b",
        "run": r"\brun\d+\b",
        "package": r"\b[Pp]ackage \d+\b",
        "rule": r"\b[Rr]ule \d+\b",
        "section": r"§\s?\d",
    }
)
STALE: Final[tuple[str, ...]] = (
    "no chart of real data is committed",
    "the charts are git-ignored",
)
GENERATED: Final[re.Pattern[str]] = re.compile(
    r"<!-- (?P<name>[a-z0-9-]+(?::summary)?):start -->(?P<inner>.*?)<!-- (?P=name):end -->", re.S
)
NUMBER: Final[str] = r"\d[\d,]*(?:\.\d+)?"
RESULTS: Final[str] = "results"
_CODE: Final = re.compile("|".join(f"(?:{p})" for p in CODES.values()))
_STALE: Final = re.compile("|".join(re.escape(p) for p in STALE), re.I)
_TOKEN: Final = re.compile(rf"(?<![\w.,])(?:{NUMBER})(?<!,)(?!\w|[.,]\d)")  # "1,118," -> 1,118
_RAW_ROLE: Final = re.compile(":(?:py:)?(?:" + "|".join(ROLE_NAMES) + "):")
_DOCTEST: Final = re.compile(r"\s*>>>")
_FRONT_MATTER: Final = re.compile(r"\A---\n.*?\n---\n", re.S)
_TARGET: Final = re.compile(r"\]\([^)]*\)")
_ATTRS: Final = re.compile(r"\{[^{}\n]*\}")

_OWNERS: Final = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

Hit = tuple[int, str, str]


def strip_generated(text: str) -> str:
    """``text`` with every generated block emptied: the marker lines stay, the lines between
    them become blank, so line numbers are kept.

    Examples
    --------
    >>> strip_generated("a\\n<!-- x:summary:start -->\\nb\\n<!-- x:summary:end -->\\nc")
    'a\\n<!-- x:summary:start -->\\n\\n<!-- x:summary:end -->\\nc'
    """

    def empty(m: re.Match[str]) -> str:
        start, end = f"<!-- {m['name']}:start -->", f"<!-- {m['name']}:end -->"
        return start + "\n" * m.group(0).count("\n") + end

    return GENERATED.sub(empty, text)


def generated_blocks(text: str) -> list[tuple[str, str]]:
    """``(name, inner text)`` of every generated block of ``text``, in text order, the marker
    lines excluded: what :func:`strip_generated` empties, so a page's test can pin which blocks
    it exempts.

    Examples
    --------
    >>> generated_blocks("a\\n<!-- m:summary:start -->\\nb\\n<!-- m:summary:end -->\\n")
    [('m:summary', '\\nb\\n')]
    """
    return [(m["name"], m["inner"]) for m in GENERATED.finditer(text)]


def find_codes(text: str) -> list[Hit]:
    """``(line, code, stripped line)`` for every internal code (:data:`CODES`) and every
    :data:`STALE` phrase of ``text``, lines counted from 1; pass :func:`strip_generated` text to
    skip the generated blocks.

    Examples
    --------
    >>> code = "".join(["M", "S", "6"])
    >>> [(n, c == code) for n, c, _ in find_codes(f"Fine 2015/16 and q214.\\nSee {code}.")]
    [(2, True)]
    """
    hits: list[Hit] = []
    for number, line in enumerate(text.splitlines(), start=1):
        found = [(m.start(), m.group(0)) for m in _CODE.finditer(line)]
        found += [(m.start(), m.group(0)) for m in _STALE.finditer(line)]
        hits += [(number, code, line.strip()) for _, code in sorted(found)]
    return hits


def _prose_lines(lines: list[str], doctests: bool) -> list[tuple[int, str]]:
    kept: list[tuple[int, str]] = []
    in_doctest = False
    for i, line in enumerate(lines):
        if not in_doctest and _DOCTEST.match(line):
            in_doctest = True
        elif in_doctest and not line.strip():
            in_doctest = False
        if doctests or not in_doctest:
            kept.append((i, line))
    return kept


def _docstrings(tree: ast.Module) -> list[ast.Constant]:
    found: list[ast.Constant] = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body):
            first = i == 0 and isinstance(node, _OWNERS)
            after_assign = i > 0 and isinstance(body[i - 1], ast.Assign | ast.AnnAssign)
            if (
                (first or after_assign)
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                found.append(stmt.value)
    return found


def docstring_codes(path: str | Path, *, doctests: bool = False) -> list[Hit]:
    """:func:`find_codes` over every module, class, function and attribute docstring of the
    Python file ``path``, with file line numbers; doctest regions (a ``>>>`` line through the
    next blank line) are skipped unless ``doctests``.

    Raises
    ------
    SyntaxError, OSError
        ``path`` does not parse or cannot be read (the message names it).

    Examples
    --------
    >>> import tempfile
    >>> code, quotes = "R" + "14", '"' * 3
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     p = Path(tmp) / "m.py"
    ...     _ = p.write_text(f"{quotes}Doc.\\n\\nSee {code}.\\n\\n>>> {code}\\n{quotes}\\n")
    ...     [n for n, c, _ in docstring_codes(p)], len(docstring_codes(p, doctests=True))
    ([3], 2)
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hits: list[Hit] = []
    for node in _docstrings(tree):
        lines = str(node.value).splitlines()
        for i, line in _prose_lines(lines, doctests):
            hits += [(node.lineno + i, code, text) for _, code, text in find_codes(line)]
    return sorted(hits)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def unsourced_numbers(text: str, *sources: str) -> list[str]:
    """The number tokens of ``text`` that occur as a token in none of ``sources``, in text order.

    Generated blocks, front matter, link and image targets (``](…)``) and attribute lists
    (``{…}``) are removed first; alt text and link text stay and are checked like any text.

    Raises
    ------
    ValueError
        No source given.

    Examples
    --------
    >>> unsourced_numbers("![Saves 54 of 57](x-540.png){ width=540 } and 913 goals", "57 913")
    ['54']
    """
    if not sources:
        raise ValueError("unsourced_numbers needs at least one source text")
    body = _FRONT_MATTER.sub("", strip_generated(text))
    body = _ATTRS.sub("", _TARGET.sub("]", body))
    known = {t for source in sources for t in _tokens(source)}
    return [t for t in _tokens(body) if t not in known]


def raw_roles(site: str | Path) -> list[tuple[str, int]]:
    """``(file, count)`` of Sphinx role text (the func, class, meth and data roles, and the
    other :data:`ROLE_NAMES`) per built ``.html`` / ``.json`` file under ``site``, relative
    paths, files without any omitted.

    Raises
    ------
    ValueError
        ``site/index.html`` does not exist (run ``zensical build`` first).

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     _ = (Path(tmp) / "index.html").write_text(":".join(["", "func", "<code>x</code>"]))
    ...     raw_roles(tmp)
    [('index.html', 1)]
    """
    site = Path(site)
    if not (site / "index.html").is_file():
        raise ValueError(f"no built site at {site} (no index.html); run zensical build first")
    found: list[tuple[str, int]] = []
    for path in sorted(site.rglob("*")):
        if path.suffix in (".html", ".json") and path.is_file():
            count = len(_RAW_ROLE.findall(path.read_text(encoding="utf-8")))
            if count:
                found.append((path.relative_to(site).as_posix(), count))
    return found


# --- The results block -------------------------------------------------------------------------


@dataclass(frozen=True)
class Pick:
    """One headline result: a row of a README card block (``block``) chosen by the cell values
    in ``row``, the ``columns`` copied from it, and the ``notes`` (sentence prefixes) of the
    same block printed under the table."""

    result: str
    block: str
    row: tuple[str, ...]
    columns: tuple[str, ...]
    notes: tuple[str, ...] = ()


def _strings(entry: Mapping[str, object], key: str, i: int) -> tuple[str, ...]:
    value = entry.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"results pick {i}: {key} must be a list of strings, got {value!r}")
    return tuple(str(v) for v in value)


def picks(pyproject: str) -> tuple[Pick, ...]:
    """The ``[[tool.football_docs.results]]`` entries of the ``pyproject.toml`` text.

    Raises
    ------
    ValueError
        No entry; an entry without a string ``result`` and ``block`` or a non-empty ``row``
        and ``columns`` list (``notes`` may be left out).

    Examples
    --------
    >>> picks('[[tool.football_docs.results]]\\nresult = "xG"\\nblock = "models"\\n'
    ...       'row = ["GBM"]\\ncolumns = ["CV log loss"]\\n')
    (Pick(result='xG', block='models', row=('GBM',), columns=('CV log loss',), notes=()),)
    """
    tool = tomllib.loads(pyproject).get("tool", {})
    entries = tool.get("football_docs", {}).get(RESULTS, [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("pyproject.toml has no [[tool.football_docs.results]] entry")
    out: list[Pick] = []
    for i, entry in enumerate(entries, start=1):
        result, block = entry.get("result"), entry.get("block")
        row, columns = _strings(entry, "row", i), _strings(entry, "columns", i)
        if not (isinstance(result, str) and isinstance(block, str) and row and columns):
            raise ValueError(f"results pick {i} needs result, block, row and columns: {entry!r}")
        out.append(Pick(result, block, row, columns, _strings(entry, "notes", i)))
    return tuple(out)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _block(readme: str, name: str) -> str:
    for m in GENERATED.finditer(readme):
        if m["name"] == f"{name}:summary":
            return m.group(0)
    raise ValueError(f"README has no generated block <!-- {name}:summary:start -->")


def _measured(block: str, pick: Pick) -> list[str]:
    table = [line for line in block.splitlines() if line.lstrip().startswith("|")]
    if len(table) < 3:
        raise ValueError(f"block {pick.block}: no table")
    header = _cells(table[0])
    rows = [_cells(line) for line in table[2:]]
    hits = [r for r in rows if set(pick.row) <= {c.replace("**", "").strip() for c in r}]
    if len(hits) != 1:
        raise ValueError(f"block {pick.block}: row {list(pick.row)} matches {len(hits)} rows")
    measured: list[str] = []
    for column in pick.columns:
        if column not in header:
            raise ValueError(f"block {pick.block}: no column {column!r} in {header}")
        measured.append(f"{column}: {hits[0][header.index(column)]}")
    return measured


def _note(block: str, name: str, prefix: str) -> str:
    found: list[str] = []
    for line in block.splitlines():
        if line.lstrip().startswith(("|", "<!--")):
            continue
        for start in [0] + [m.end() for m in re.finditer(r"\.\s+", line)]:
            if line.startswith(prefix, start):
                end = re.search(r"\.(?=\s|$)", line[start:])
                found.append(line[start : start + end.end()] if end else line[start:])
    if len(found) != 1:
        raise ValueError(f"block {name}: note {prefix!r} matches {len(found)} sentences")
    return found[0].strip()


def results_block(readme: str, picks: Sequence[Pick]) -> str:
    """The home page's ``results`` block: one table row per pick, then its notes.

    Each measured cell is copied verbatim from the pick's README block, ``column: cell`` joined
    by ``<br>``; each note is the block's unique sentence that starts with the given prefix,
    printed once, in pick order.

    Raises
    ------
    ValueError
        No pick; the block is missing; the row matches no row or several; a column is not in
        the block's header; a note prefix matches no sentence or several; a number in a
        ``result`` label does not occur in ``readme``.

    Examples
    --------
    >>> readme = ("<!-- m:summary:start -->\\n| Model | CV |\\n|---|---|\\n| **GBM** | 0.26 |\\n\\n"
    ...           "GBM: lower is better. Done.\\n<!-- m:summary:end -->")
    >>> print(results_block(readme, [Pick("xG", "m", ("GBM",), ("CV",), ("GBM:",))]))
    <!-- results:start -->
    | Result | Measured |
    |---|---|
    | xG | CV: 0.26 |
    <BLANKLINE>
    GBM: lower is better.
    <BLANKLINE>
    <!-- results:end -->
    """
    if not picks:
        raise ValueError("results_block needs at least one pick")
    rows: list[str] = []
    notes: list[str] = []
    for pick in picks:
        if bad := unsourced_numbers(pick.result, readme):
            raise ValueError(f"result label {pick.result!r}: numbers {bad} are not in the README")
        block = _block(readme, pick.block)
        rows.append(f"| {pick.result} | {'<br>'.join(_measured(block, pick))} |")
        for prefix in pick.notes:
            note = _note(block, pick.block, prefix)
            if note not in notes:
                notes.append(note)
    lines = [f"<!-- {RESULTS}:start -->", "| Result | Measured |", "|---|---|", *rows, ""]
    for note in notes:
        lines += [note, ""]
    return "\n".join([*lines, f"<!-- {RESULTS}:end -->"])


def _page_block(page: str) -> re.Match[str]:
    for m in GENERATED.finditer(page):
        if m["name"] == RESULTS:
            return m
    raise ValueError(f"the page has no <!-- {RESULTS}:start --> ... <!-- {RESULTS}:end --> block")


# --- Command line ------------------------------------------------------------------------------


def _main(args: list[str]) -> int:
    if len(args) == 2 and args[0] == "site":
        found = raw_roles(args[1])
        for name, count in found:
            print(f"raw role: {name}: {count}", file=sys.stderr)
        print(f"site: {sum(c for _, c in found)} raw roles in {len(found)} files")
        return 1 if found else 0
    if 2 <= len(args) <= 3 and args[0] == RESULTS and args[2:] in ([], ["--write"]):
        page = Path(args[1])
        text = page.read_text(encoding="utf-8")
        old = _page_block(text)
        readme = Path("README.md").read_text(encoding="utf-8")
        new = results_block(readme, picks(Path("pyproject.toml").read_text(encoding="utf-8")))
        if args[2:]:
            page.write_text(text[: old.start()] + new + text[old.end() :], encoding="utf-8")
            print(f"results: wrote {page}")
            return 0
        print(f"results: {page} is {'current' if old.group(0) == new else 'stale'}")
        return 0 if old.group(0) == new else 1
    raise ValueError("usage: python -m football_docs.pages site <dir> | results <page> [--write]")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: ``site <dir>`` (exit 1 on any raw role) or ``results <page>
    [--write]`` (exit 1 when the page's block is stale); an error is one stderr line, exit 1.

    Examples
    --------
    >>> main(["site", "no-such-site"])
    1
    """
    try:
        return _main(list(sys.argv[1:] if argv is None else argv))
    except (ValueError, OSError, tomllib.TOMLDecodeError) as e:
        print(f"football_docs.pages: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
