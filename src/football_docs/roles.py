"""Sphinx cross-reference roles in numpy docstrings, rendered at docs build time.

The docstrings of the football-* packages name other objects with Sphinx roles (the func,
class, meth and data roles, among others). mkdocstrings does not read them, so an API page would
show the raw role text. :class:`SphinxRoles` is a griffe extension that rewrites each role in
memory while mkdocstrings loads a package: a target that resolves to an object the site renders
becomes a cross-reference link, anything else becomes plain code. The source files are never
touched, which matters in the model repos, where docstrings sit inside modules that the
prediction cache hashes byte for byte.

Load it from ``mkdocs.yml`` (python handler options)::

    extensions: [football_docs.roles]

**Load order.** The resolver sees only the packages already in the modules collection when
``on_package`` runs. A role into a package that mkdocstrings loads later renders as plain code,
while the reverse direction links. The result is deterministic and never a broken link (a link
to a missing anchor would fail ``zensical build --strict``).

Needs ``griffe`` (the ``docs`` dependency group, through mkdocstrings).
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Sequence
from typing import Any, Final

import griffe

from football_docs.pages import ROLE_NAMES

ROLE: Final[re.Pattern[str]] = re.compile(
    ":(?:py:)?(?:" + "|".join(ROLE_NAMES) + "):`(?P<tilde>~)?(?P<target>[^`]+)`"
)
_DOCTEST: Final = re.compile(r"\s*>>>")

Resolver = Callable[[str], "str | None"]


def _label(target: str, tilde: bool) -> str:
    return target.rsplit(".", 1)[-1] if tilde else target.lstrip(".")


def _convert_prose(text: str, resolve: Resolver) -> str:
    def sub(m: re.Match[str]) -> str:
        target = "".join(m["target"].split())
        label = _label(target, m["tilde"] is not None)
        path = resolve(target)
        return f"`{label}`" if path is None else f"[`{label}`][{path}]"

    return ROLE.sub(sub, text)


def convert(text: str, resolve: Resolver) -> str:
    """Every role in ``text`` as a link (``resolve`` gives a path) or as plain code.

    A role becomes the reference link ``[`label`][path]`` when ``resolve(target)`` returns a
    path, else the label in backticks. Whitespace inside the target (a wrapped target) is removed
    first; the label is the last dotted part when the target starts with a tilde, else the
    target with leading dots stripped. Doctest regions (a ``>>>`` line through the next blank
    line) are returned unchanged, and text without roles comes back identical.

    Examples
    --------
    >>> role = ":".join(["", "func", "`~pkg.mod.fit`"])
    >>> convert(f"Calls {role}.", lambda t: t)
    'Calls [`fit`][pkg.mod.fit].'
    >>> convert(":".join(["", "class", "`.Row`"]), lambda t: None)
    '`Row`'
    >>> convert(":".join(["", "data", "`pkg.\\n    LIMIT`"]), lambda t: None)
    '`pkg.LIMIT`'
    """
    out: list[str] = []
    prose: list[str] = []
    in_doctest = False
    for line in text.splitlines(keepends=True):
        if not in_doctest and _DOCTEST.match(line):
            out.append(_convert_prose("".join(prose), resolve))
            prose, in_doctest = [], True
        elif in_doctest and not line.strip():
            in_doctest = False
        (out if in_doctest else prose).append(line)
    out.append(_convert_prose("".join(prose), resolve))
    return "".join(out)


def _rendered(found: griffe.Object | griffe.Alias) -> str | None:
    """The path of the object mkdocstrings renders an anchor for, or ``None``."""
    try:
        target = found.final_target if isinstance(found, griffe.Alias) else found
    except griffe.AliasResolutionError:
        return None
    parts = target.path.split(".")
    if any(p.startswith("_") and not p.startswith("__") for p in parts):
        return None
    if not target.is_module and not target.docstring:
        return None  # show_if_no_docstring is false: no anchor
    return target.path


def _candidates(obj: griffe.Object, target: str) -> list[str]:
    module, package = obj.module.path, obj.package.path
    if target.startswith("."):
        return [module + target, package + target]
    head, _, rest = target.partition(".")
    found: list[str] = []
    with contextlib.suppress(griffe.NameResolutionError):
        found.append(obj.resolve(head) + (f".{rest}" if rest else ""))
    found += [target, f"{module}.{target}", f"{package}.{target}"]
    owner = obj if obj.is_class else obj.parent
    if owner is not None and owner.is_class:
        found.append(f"{owner.path}.{target}")
    return found


def resolver(obj: griffe.Object, external: Sequence[str] = ()) -> Resolver:
    """The resolver :class:`SphinxRoles` binds to ``obj``: first hit in its modules collection.

    A target under one of the ``external`` package prefixes always gets a link (the inventory
    of another site validates it); anything that does not resolve to a rendered object gives
    ``None``, including a path through an import alias of a package that is not loaded.

    Examples
    --------
    >>> import griffe
    >>> with griffe.temporary_visited_package(
    ...     "pkg", {"__init__.py": "'''P.'''\\ndef fit():\\n    '''Fit.'''\\n"}
    ... ) as pkg:
    ...     find = resolver(pkg)
    ...     find("fit"), find("missing"), resolver(pkg, ["other"])("other.x")
    ('pkg.fit', None, 'other.x')
    """
    collection = obj.modules_collection

    def resolve(target: str) -> str | None:
        if any(target == p or target.startswith(f"{p}.") for p in external):
            return target
        for path in _candidates(obj, target):
            try:
                found = collection[path]
            except (KeyError, ValueError, griffe.AliasResolutionError, griffe.CyclicAliasError):
                continue  # an alias of an unloaded package (``import polars as pl``) is a miss
            return _rendered(found)
        return None

    return resolve


def _walk(obj: griffe.Object, external: Sequence[str]) -> None:
    if obj.docstring is not None:
        obj.docstring.value = convert(obj.docstring.value, resolver(obj, external))
    for member in obj.members.values():
        if not isinstance(member, griffe.Alias):
            _walk(member, external)


class SphinxRoles(griffe.Extension):
    """Rewrite the roles of every docstring of a package as it loads (see :func:`convert`).

    ``external``: package prefixes whose targets always link, for a site that lists another
    site's inventory (empty until the sites have URLs).

    Examples
    --------
    >>> SphinxRoles(external="football_core")
    Traceback (most recent call last):
    ...
    TypeError: external must be a sequence of package names, not a str
    """

    def __init__(self, external: Sequence[str] = ()) -> None:
        super().__init__()
        if isinstance(external, str):
            raise TypeError("external must be a sequence of package names, not a str")
        self.external = tuple(external)

    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        """Rewrite the docstrings of ``pkg`` and its members (aliases are skipped)."""
        _walk(pkg, self.external)
