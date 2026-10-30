"""The repository metadata check: author, CI and Pages wiring, one rule for five repos.

Every football-* repo names the same author in its ``LICENSE``, ``mkdocs.yml`` and
``pyproject.toml``, runs the same fast checks in ``.github/workflows/check.yml`` and, when it
publishes a site, builds and deploys it from ``.github/workflows/docs.yml``. Writing those rules
once here, and calling ``problems`` from a one-line test in each repo, keeps the five repos from
drifting apart: a repo that complies gets an empty list, any other gets one fixed-text problem
per violation.

**The core checkout is a stopgap, and this check is its seam.** While a consumer reads
football-core from the path ``../football-core`` (``[tool.uv.sources]``), each installing job
checks core out beside the repo. The expected number of those checkout steps follows the
source: one for a path source, none otherwise. Switching the source to a git dependency
therefore makes the repo's test fail until the step is deleted, and deleting the step early
fails it too, so the switch and the deletion are one change.

The workflows are read with ``yaml.safe_load``, which drops comments, so the stopgap note
itself is checked by the gate, not here. A missing, unparsable or non-mapping file is one
problem and is skipped by the later rules; other malformed shapes are left to actionlint.

Needs ``yaml`` (PyYAML, a transitive ``docs``-group dependency through mkdocs).

Examples
--------
>>> problems(Path("no-such-repo"), pages=False)[0]
'LICENSE: missing'
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Final

import yaml

AUTHOR: Final[str] = "R. Berk Karataş"
OLD_SPELLINGS: Final[tuple[str, ...]] = ("R.Berk", "Karatas")
AUTHOR_FILES: Final[tuple[str, ...]] = ("LICENSE", "mkdocs.yml", "pyproject.toml")
CORE: Final[str] = "football-core"
CHECK: Final[str] = ".github/workflows/check.yml"
PAGES: Final[str] = ".github/workflows/docs.yml"
SYNC: Final[str] = "uv sync --locked --all-groups"
PIN: Final[re.Pattern[str]] = re.compile(r"[\w.-]+/[\w.-]+@v\d+\.\d+\.\d+")

TARGETS: Final[dict[str, str]] = {CHECK: "make check", PAGES: "make docs"}

Job = tuple[str, dict[str, Any], list[dict[str, Any]]]  # (job id, job, steps)


def _author(root: Path) -> list[str]:
    out: list[str] = []
    for f in AUTHOR_FILES:
        text = (root / f).read_text(encoding="utf-8") if (root / f).is_file() else None
        if text is None:
            out.append(f"{f}: missing")
            continue
        out += [] if AUTHOR in text else [f"{f}: no '{AUTHOR}'"]
        out += [f"{f}: old spelling '{s}'" for s in OLD_SPELLINGS if s in text]
    return out


def _project(root: Path, out: list[str]) -> tuple[str | None, bool]:
    """The ``[project] name`` and whether core is a path source (``(None, False)`` if unread)."""
    if not (root / "pyproject.toml").is_file():
        return None, False
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        out.append("pyproject.toml: invalid TOML")
        return None, False
    name = data["project"].get("name") if isinstance(data.get("project"), dict) else None
    if not isinstance(name, str):
        out.append("pyproject.toml: no [project] name")
        return None, False
    core = data.get("tool", {}).get("uv", {}).get("sources", {}).get(CORE)
    return name, isinstance(core, dict) and "path" in core


def _load(root: Path, rel: str, out: list[str]) -> dict[Any, Any] | None:
    try:
        doc = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
    except yaml.YAMLError:
        out.append(f"{rel}: invalid YAML")
        return None
    if not isinstance(doc, dict):
        out.append(f"{rel}: not a mapping")
    return doc if isinstance(doc, dict) else None


def _on(doc: dict[Any, Any]) -> Any:
    return doc.get("on", doc.get(True))  # PyYAML reads the bare key on as True


def _run(step: dict[str, Any]) -> str:
    return step["run"].strip() if isinstance(step.get("run"), str) else ""


def _uses(step: dict[str, Any], prefix: str, **inputs: object) -> bool:
    """The step uses the action ``prefix`` with these ``with`` values (None: input absent)."""
    given = step.get("with") or {}
    return str(step.get("uses", "")).startswith(prefix) and all(
        given.get(k) == v for k, v in inputs.items()
    )


def _jobs(doc: dict[Any, Any], *, installing: bool = False) -> list[Job]:
    jobs = [(j, job, job.get("steps") or []) for j, job in (doc.get("jobs") or {}).items()]
    return [t for t in jobs if not installing or SYNC in map(_run, t[2])]


def _ran_after_sync(steps: list[dict[str, Any]], target: str) -> bool:
    runs = [_run(s) for s in steps]
    return target in runs[runs.index(SYNC) + 1 :]


def _is_core_checkout(step: dict[str, Any]) -> bool:
    repo = (step.get("with") or {}).get("repository")
    return _uses(step, "actions/checkout@", path=CORE) and str(repo).endswith(f"/{CORE}")


def _job_rows(rel: str, doc: dict[Any, Any], name: str) -> list[str]:
    jobs, target = _jobs(doc, installing=True), TARGETS[rel]
    out = (
        []
        if any(_ran_after_sync(s, target) for *_, s in jobs)
        else [f"{rel}: no job runs '{target}' after '{SYNC}'"]
    )
    out += [
        f"{rel}: job {j}: first step must check out this repo at path '{name}'"
        for j, _, steps in jobs
        if not _uses(steps[0], "actions/checkout@", path=name, repository=None)
    ]
    return out + [
        f"{rel}: job {j}: defaults.run.working-directory must be '{name}'"
        for j, job, _ in jobs
        if (job.get("defaults") or {}).get("run", {}).get("working-directory") != name
    ]


def _core_rows(rel: str, doc: dict[Any, Any], expected: int) -> list[str]:
    counts = [(j, sum(map(_is_core_checkout, s))) for j, _, s in _jobs(doc, installing=True)]
    return [
        f"{rel}: job {j}: {n} {CORE} checkout steps, expected {expected}"
        f" ({'add' if n < expected else 'delete'} the stopgap step)"
        for j, n in counts
        if n != expected
    ]


def _pages_rows(doc: dict[Any, Any], name: str) -> list[str]:
    on = _on(doc)
    push = on.get("push") if isinstance(on, dict) else None
    branches = push.get("branches") if isinstance(push, dict) else None
    out = [] if branches == ["main"] else [f"{PAGES}: on.push.branches must be ['main']"]
    site = f"{name}/site"
    upload = "actions/upload-pages-artifact@"
    ups = [
        j for j, _, s in _jobs(doc, installing=True) if any(_uses(x, upload, path=site) for x in s)
    ]
    deploys = [
        (j, job) for j, job, s in _jobs(doc) if any(_uses(x, "actions/deploy-pages@") for x in s)
    ]
    if not ups:
        out.append(f"{PAGES}: no job uploads '{site}'")
    if not deploys:
        out.append(f"{PAGES}: no job uses actions/deploy-pages")
    for d, job in deploys if ups else []:
        needs = job.get("needs")
        if ups[0] not in ([needs] if isinstance(needs, str) else needs or []):
            out.append(f"{PAGES}: job {d}: needs must name '{ups[0]}'")
    for d, job in deploys:
        perms = job.get("permissions")
        if not isinstance(perms, dict) or {perms.get("pages"), perms.get("id-token")} != {"write"}:
            out.append(f"{PAGES}: job {d}: permissions need pages: write and id-token: write")
    return out


def problems(root: Path, *, pages: bool) -> list[str]:
    """The ways the repo at ``root`` breaks the author, CI and Pages rules (``[]`` if none).

    The author must appear, in its U+015F spelling and with no old spelling, in each of
    ``LICENSE``, ``mkdocs.yml`` and ``pyproject.toml``. ``check.yml`` must run on push and pull
    request; every job that installs with ``uv sync --locked --all-groups`` must check this repo
    out first at the path named by ``[project] name``, work in that directory, and check core out
    beside it exactly when core is a path source; one such job must run ``make check`` after
    the sync. Every action is pinned to a full ``vX.Y.Z`` tag. With ``pages``, ``docs.yml``
    must exist, run on pushes to ``main``, run ``make docs`` after the sync, upload
    ``<name>/site``, and deploy it from a job that needs the uploading job and may write pages
    and id tokens; without ``pages`` it must not exist.

    Parameters
    ----------
    root
        The repository directory.
    pages
        Whether the repo publishes a documentation site (required: each repo states it).

    Returns
    -------
    list of str
        One ``"<file>: <problem>"`` string per violation, author problems first, then the
        loading problems, then the workflow rules; ``check.yml`` before ``docs.yml``.

    Examples
    --------
    >>> problems(Path("no-such-repo"), pages=True)[-2:]
    ['.github/workflows/check.yml: missing', '.github/workflows/docs.yml: missing']
    """
    out = _author(root)
    name, path_source = _project(root, out)
    docs: dict[str, dict[Any, Any]] = {}
    for rel, wanted in ((CHECK, True), (PAGES, pages)):
        if (root / rel).is_file() != wanted:
            out.append(f"{rel}: missing" if wanted else f"{rel}: unexpected (pages=False)")
        elif wanted and (doc := _load(root, rel, out)) is not None:
            docs[rel] = doc
    if CHECK in docs:
        on = _on(docs[CHECK])
        events = {on} if isinstance(on, str) else set(on) if isinstance(on, list | dict) else set()
        out += [f"{CHECK}: on lacks {e}" for e in ("push", "pull_request") if e not in events]
    if name is not None:
        for rel, doc in docs.items():
            out += _job_rows(rel, doc, name)
        for rel, doc in docs.items():
            out += _core_rows(rel, doc, 1 if path_source else 0)
    for rel, doc in docs.items():
        uses = [str(x["uses"]) for *_, steps in _jobs(doc) for x in steps if "uses" in x]
        out += [f"{rel}: unpinned '{u}'" for u in uses if not PIN.fullmatch(u)]
    if PAGES in docs and name is not None:
        out += _pages_rows(docs[PAGES], name)
    return out
