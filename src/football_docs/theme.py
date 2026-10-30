"""The docs-site theme, generated from the chart themes so that site and charts read as one.

:func:`css` maps the tokens of ``football_core.viz.THEMES`` onto the CSS variables Zensical's
modern theme reads (:data:`VARS`, one mapping for both schemes: ``"light"`` → the ``default``
scheme, ``"dark"`` → ``slate``) and adds a few fixed rules: a flat header with title-coloured
text, slate headings, links in ink with an accent underline (red stays the charts' meaning
colour), compact tables (tabular figures, one-line headers, a shadow at an edge that scrolls),
model-card pages without the right-hand contents and on a wider grid (a page whose ``h1`` id
starts ``model-card``, i.e. "# Model card: ..."), the home page's chart grid and the chart image
frame.
:func:`logo_svg` draws the header logo and favicon, a goal in the light accent.

``python -m football_docs.theme <docs_dir> [--check]`` writes (or compares) the two generated
files every site commits, ``stylesheets/theme.css`` and ``assets/logo.svg``; ``make
docs-theme`` runs it and each repo's freshness test runs the check. Needs the ``viz`` extra.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from football_core import viz

# (CSS variable, Theme field[+alpha hex]); every name is one Zensical's modern CSS reads
VARS: Final[tuple[tuple[str, str], ...]] = (
    ("--md-default-fg-color", "ink"),
    ("--md-default-fg-color--light", "muted"),
    ("--md-default-fg-color--lighter", "box_line"),
    ("--md-default-fg-color--lightest", "line"),
    ("--md-default-bg-color", "background"),
    ("--md-default-bg-color--light", "background+e6"),
    ("--md-typeset-table-color", "line"),
    ("--md-typeset-table-color--light", "panel"),
    ("--md-accent-fg-color", "accent"),
    ("--md-accent-fg-color--transparent", "accent+1a"),
    ("--md-accent-bg-color", "background"),
    ("--md-typeset-a-color", "ink"),
    ("--md-code-bg-color", "panel"),
    ("--md-code-bg-color--light", "panel+b3"),
    ("--md-code-bg-color--lighter", "panel+4d"),
    ("--md-code-fg-color", "ink"),
    ("--md-primary-fg-color", "title"),
    ("--md-primary-bg-color", "background"),
)
OWN_VARS: Final[tuple[tuple[str, str], ...]] = (("--fd-title", "title"), ("--fd-panel", "panel"))
SCHEMES: Final[tuple[tuple[str, str], ...]] = (("default", "light"), ("slate", "dark"))
RULES: Final[str] = """\
.md-header {
  color: var(--fd-title);
  border-bottom: .05rem solid var(--md-default-fg-color--lightest);
}
.md-typeset h1, .md-typeset h2, .md-typeset h3 { color: var(--fd-title); }
.md-typeset h1, .md-typeset h2 { font-weight: 700; }
.md-typeset h3 { font-weight: 500; }
.md-typeset a { text-decoration-color: var(--md-accent-fg-color); }
.md-footer-meta { background-color: var(--fd-panel); }
.md-typeset table { font-variant-numeric: tabular-nums; }
.md-typeset table:not([class]) { font-size: .6rem; background-color: transparent; }
.md-typeset table:not([class]) th { white-space: nowrap; min-width: 0; }
.md-typeset table:not([class]) th, .md-typeset table:not([class]) td { padding: .45em .8em; }
.md-typeset__scrollwrap {
  background:
    linear-gradient(to right, var(--md-default-bg-color) 30%, #0000)
      left / 2rem 100% no-repeat local,
    linear-gradient(to left, var(--md-default-bg-color) 30%, #0000)
      right / 2rem 100% no-repeat local,
    linear-gradient(to right, var(--md-default-fg-color--lighter), #0000)
      left / .6rem 100% no-repeat scroll,
    linear-gradient(to left, var(--md-default-fg-color--lighter), #0000)
      right / .6rem 100% no-repeat scroll;
}
@media screen and (min-width: 60em) {
  .md-main:has(h1[id^="model-card"]) .md-sidebar--secondary { display: none; }
  .md-main__inner:has(h1[id^="model-card"]) { max-width: 72rem; }
}
.fd-home-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 26rem), 1fr));
  gap: .8rem;
  align-items: start;
}
.md-typeset img[src*="assets/charts/"] {
  max-height: 90vh;
  width: auto;
  max-width: 100%;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: .4rem;
}
"""
FILES: Final[tuple[str, str]] = ("stylesheets/theme.css", "assets/logo.svg")


def value(theme: viz.Theme, token: str) -> str:
    """The CSS value of a :data:`VARS` token: a ``Theme`` colour, ``+hh`` appending an alpha.

    Raises
    ------
    ValueError
        ``token`` names no colour field of ``Theme``.

    Examples
    --------
    >>> value(viz.THEMES["light"], "accent+1a")
    '#e4002b1a'
    """
    field, _, alpha = token.partition("+")
    colour = getattr(theme, field, None)
    if not (isinstance(colour, str) and colour.startswith("#")):
        raise ValueError(f"theme token {token!r}: {field!r} is not a colour field of Theme")
    return colour + alpha


def css() -> str:
    """The site stylesheet: one variable block per scheme from ``viz.THEMES``, then the rules.

    Examples
    --------
    >>> print(css().splitlines()[1])
      --md-default-fg-color: #334155;
    """
    blocks: list[str] = []
    for scheme, name in SCHEMES:
        theme = viz.THEMES[name]
        decls = [f"  {var}: {value(theme, token)};" for var, token in VARS + OWN_VARS]
        blocks.append(f'[data-md-color-scheme="{scheme}"] {{\n' + "\n".join(decls) + "\n}\n")
    return "".join(blocks) + RULES


def logo_svg() -> str:
    """A 24 x 24 goal (posts, bar and a ground line past both posts) in the light accent.

    Examples
    --------
    >>> 'stroke="#e4002b"' in logo_svg()
    True
    """
    colour = viz.THEMES["light"].accent
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">'
        f'<path d="M4.5 18.5V7.5h15v11M0 18.5h24" fill="none" stroke="{colour}" stroke-width="3" '
        'stroke-linejoin="miter"/></svg>\n'
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Write (or with ``--check`` compare) ``theme.css`` and ``logo.svg`` under a docs folder;
    exit 1 when a committed copy differs or on an error (one stderr line).

    Examples
    --------
    >>> import contextlib, io, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
    ...     codes = main([tmp, "--check"]), main([tmp]), main([tmp, "--check"])
    >>> codes
    (1, 0, 0)
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in (1, 2) or args[1:] not in ([], ["--check"]):
        print("usage: python -m football_docs.theme <docs_dir> [--check]", file=sys.stderr)
        return 1
    docs, check = Path(args[0]), args[1:] == ["--check"]
    stale = 0
    for name, text in zip(FILES, (css(), logo_svg()), strict=True):
        path = docs / name
        try:
            current = path.read_text(encoding="utf-8") if path.is_file() else None
            if check:
                stale += current != text
                print(f"theme: {path} {'current' if current == text else 'stale'}")
            elif current != text:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                print(f"theme: wrote {path}")
        except OSError as e:
            print(f"football_docs.theme: error: {e}", file=sys.stderr)
            return 1
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
