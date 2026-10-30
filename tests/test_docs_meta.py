"""The repository metadata check ``football_docs.meta.problems`` (DR1 repo-meta spec §4, §5).

One rule for the five repos: the author string in LICENSE, mkdocs.yml and pyproject.toml, and the
CI (check.yml) and Pages (docs.yml) wiring, with the football-core checkout beside a consumer as
the seam of the git-dependency switch. The fixture is a hand-written compliant consumer
``football-demo-models``, its workflows shaped like the shot-models drafts; every case changes one
thing and asserts the exact problem list.
"""

from __future__ import annotations

import doctest
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

import football_docs
from football_docs import meta, pages, roles

REPO = Path(__file__).resolve().parents[1]
D = "football-demo-models"
C = ".github/workflows/check.yml"
P = ".github/workflows/docs.yml"
SYNC = "uv sync --locked --all-groups"
AUTHOR = "R. Berk Karataş"  # s-cedilla, U+015F

NO_CHECK_TARGET = f"{C}: no job runs 'make check' after '{SYNC}'"
NO_DOCS_TARGET = f"{P}: no job runs 'make docs' after '{SYNC}'"


def first_step(f: str, job: str, name: str = D) -> str:
    return f"{f}: job {job}: first step must check out this repo at path '{name}'"


def workdir(f: str, job: str, name: str = D) -> str:
    return f"{f}: job {job}: defaults.run.working-directory must be '{name}'"


def core_count(f: str, job: str, n: int, e: int) -> str:
    verb = "add" if n < e else "delete"
    return (
        f"{f}: job {job}: {n} football-core checkout steps, expected {e} ({verb} the stopgap step)"
    )


# --- the compliant consumer, written by hand -----------------------------------------------

LICENSE = f"""MIT License

Copyright (c) 2026 {AUTHOR}

Permission is hereby granted, free of charge, to any person obtaining a copy of this software.
"""

MKDOCS = f"""site_name: {D}
site_author: {AUTHOR}
copyright: © 2026 {AUTHOR} · MIT licence
"""

PATH_SOURCE = 'football-core = { path = "../football-core", editable = true }'
GIT_SOURCE = 'football-core = { git = "https://example.invalid/football-core" }'
NAME_LINE = f'name = "{D}"'

PYPROJECT = f"""[project]
{NAME_LINE}
version = "0.1.0"
authors = [{{ name = "{AUTHOR}", email = "demo@example.invalid" }}]
requires-python = ">=3.12"
dependencies = ["football-core"]

[tool.uv.sources]
{PATH_SOURCE}
"""

ON_CHECK = """on:
  push:
    branches: ["**"]
  pull_request:
"""

OWN = f"""      - uses: actions/checkout@v7.0.1
        with:
          path: {D}
          persist-credentials: false
"""

STOPGAP = """      # Stopgap: core is checked out beside this repo; delete when the source is git.
      - uses: actions/checkout@v7.0.1
        with:
          repository: ${{ github.repository_owner }}/football-core
          path: football-core
          persist-credentials: false
"""

SETUP = f"""      - uses: astral-sh/setup-uv@v10.3.0
        with:
          version: "0.11.16"
          python-version: "3.12.13"
          working-directory: {D}
"""

DEFAULTS = f"""    defaults:
      run:
        working-directory: {D}
"""

SYNC_THEN_CHECK = f"""      - run: {SYNC}
      - run: make check
"""

CHECK_YML = (
    f"""name: check

{ON_CHECK}
permissions:
  contents: read

jobs:
  check:
    runs-on: ubuntu-24.04
    timeout-minutes: 30
{DEFAULTS}    steps:
"""
    + OWN
    + STOPGAP
    + SETUP
    + "      - run: uvx --from actionlint-py==1.7.12.25 --with shellcheck-py==0.11.0.1 actionlint\n"
    + SYNC_THEN_CHECK
)

BRANCHES_MAIN = "    branches: [main]\n"
UPLOAD_PATH = f"path: {D}/site"
NEEDS = "    needs: build\n"
DEPLOY_PERMS = """    permissions:
      pages: write
      id-token: write
"""

DEPLOY = (
    """
  deploy:
"""
    + NEEDS
    + "    runs-on: ubuntu-24.04\n"
    + DEPLOY_PERMS
    + """    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v5.0.1
"""
)

DOCS_YML = (
    """name: docs

on:
  push:
"""
    + BRANCHES_MAIN
    + """  workflow_dispatch:

permissions:
  contents: read

jobs:
  build:
    runs-on: ubuntu-24.04
"""
    + DEFAULTS
    + "    steps:\n"
    + OWN
    + STOPGAP
    + SETUP
    + f"""      - run: {SYNC}
      - run: make docs
      - uses: actions/upload-pages-artifact@v5.0.0
        with:
          {UPLOAD_PATH}
"""
    + DEPLOY
)

FILES = {
    "LICENSE": LICENSE,
    "mkdocs.yml": MKDOCS,
    "pyproject.toml": PYPROJECT,
    C: CHECK_YML,
    P: DOCS_YML,
}


def write_consumer(root: Path) -> Path:
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def edit(root: Path, rel: str, old: str, new: str) -> None:
    """Replace every occurrence of ``old`` (which must be there: the near miss's precondition)."""
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture precondition: {old!r} not in {rel}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def edit_first(root: Path, rel: str, old: str, new: str) -> None:
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert old in text, f"fixture precondition: {old!r} not in {rel}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


Edit = tuple[str, str, str]  # (file, old, new); new == "" deletes


def apply(root: Path, edits: Sequence[Edit]) -> None:
    for rel, old, new in edits:
        edit(root, rel, old, new)


@pytest.fixture
def consumer(tmp_path: Path) -> Path:
    root = write_consumer(tmp_path / D)
    assert meta.problems(root, pages=True) == []  # precondition of every near miss
    return root


# --- 1. the five repos: core itself ---------------------------------------------------------


def test_core_repo_meta() -> None:
    """Rule 1-6 on core itself, plus the author bytes read by an independent route."""
    assert meta.AUTHOR == AUTHOR
    assert meta.AUTHOR.encode() == b"R. Berk Karata\xc5\x9f"
    assert meta.problems(REPO, pages=True) == []
    for name in ("LICENSE", "mkdocs.yml", "pyproject.toml"):
        raw = (REPO / name).read_bytes()
        assert b"R. Berk Karata\xc5\x9f" in raw, name
        assert b"Karatas" not in raw and b"R.Berk" not in raw, name


# --- 2. author ------------------------------------------------------------------------------

AUTHOR_FILES = ("LICENSE", "mkdocs.yml", "pyproject.toml")


@pytest.mark.parametrize("rel", AUTHOR_FILES)
@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (
            "R.Berk Karatas",
            ["no 'R. Berk Karataş'", "old spelling 'R.Berk'", "old spelling 'Karatas'"],
        ),
        ("R. Berk Karatas", ["no 'R. Berk Karataş'", "old spelling 'Karatas'"]),
        # NFD: s + combining cedilla U+0327 looks right but holds the ASCII 'Karatas'
        ("R. Berk Karataş", ["no 'R. Berk Karataş'", "old spelling 'Karatas'"]),
        # s with comma below (U+0219), the Romanian letter: not the author, not an old spelling
        ("R. Berk Karataș", ["no 'R. Berk Karataş'"]),
        (None, ["missing"]),
    ],
    ids=["old", "ascii", "nfd", "comma-below", "missing"],
)
def test_author(consumer: Path, rel: str, variant: str | None, expected: list[str]) -> None:
    if variant is None:
        (consumer / rel).unlink()
    else:
        edit(consumer, rel, AUTHOR, variant)
    # a missing pyproject leaves R undefined: rules 3, 4 and 6 are skipped, nothing else reported
    assert meta.problems(consumer, pages=True) == [f"{rel}: {m}" for m in expected]


@settings(deadline=None, max_examples=60)
@given(
    parts=st.lists(
        st.sampled_from([AUTHOR, "R.Berk", "Karatas", "Karata", "R. Berk", "ş", "s", " ", "x"]),
        max_size=6,
    )
)
def test_author_rule_property(parts: list[str]) -> None:
    """Any LICENSE text: one problem if the author is absent, one per old spelling present."""
    text = "".join(parts)
    expected = [] if AUTHOR in text else ["LICENSE: no 'R. Berk Karataş'"]
    expected += [f"LICENSE: old spelling '{s}'" for s in ("R.Berk", "Karatas") if s in text]
    with tempfile.TemporaryDirectory() as tmp:
        root = write_consumer(Path(tmp))
        (root / "LICENSE").write_text(text, encoding="utf-8")
        assert meta.problems(root, pages=True) == expected


# --- 3. check triggers ----------------------------------------------------------------------

LACKS_PUSH = f"{C}: on lacks push"
LACKS_PR = f"{C}: on lacks pull_request"


@pytest.mark.parametrize(
    ("on", "expected"),
    [
        ('on:\n  push:\n    branches: ["**"]\n', [LACKS_PR]),
        ("on:\n  pull_request:\n", [LACKS_PUSH]),
        ("on: push\n", [LACKS_PR]),
        ("on: [push]\n", [LACKS_PR]),
        ("on: [pull_request, push]\n", []),
        ('"on": [push]\n', [LACKS_PR]),  # a quoted key stays the string "on"
        ("on: pull_request_target\n", [LACKS_PUSH, LACKS_PR]),  # substring near miss
        ("on:\n  push_tag:\n  pull_request_review:\n", [LACKS_PUSH, LACKS_PR]),
        (None, [f"{C}: missing"]),
    ],
    ids=[
        "no-pr",
        "no-push",
        "str",
        "list",
        "list-both",
        "quoted-on",
        "pr-target",
        "mapping-near",
        "deleted",
    ],
)
def test_check_triggers(consumer: Path, on: str | None, expected: list[str]) -> None:
    if on is None:
        (consumer / C).unlink()
    else:
        edit(consumer, C, ON_CHECK, on)
    assert meta.problems(consumer, pages=True) == expected


EVENTS = ("push", "pull_request", "pull_request_target", "workflow_dispatch", "merge_group")


@st.composite
def on_forms(draw: st.DrawFn) -> tuple[str, set[str]]:
    form = draw(st.sampled_from(["str", "list", "mapping"]))
    if form == "str":
        event = draw(st.sampled_from(EVENTS))
        return f"on: {event}\n", {event}
    events = draw(st.lists(st.sampled_from(EVENTS), unique=True, max_size=len(EVENTS)))
    if form == "list":
        return f"on: [{', '.join(events)}]\n", set(events)
    if not events:
        return "on: {}\n", set()
    return "on:\n" + "".join(f"  {e}:\n" for e in events), set(events)


@settings(deadline=None, max_examples=60)
@given(form=on_forms())
def test_trigger_normalisation_property(form: tuple[str, set[str]]) -> None:
    """str, list or mapping `on`: membership by event name, push before pull_request."""
    on, events = form
    expected = [f"{C}: on lacks {e}" for e in ("push", "pull_request") if e not in events]
    with tempfile.TemporaryDirectory() as tmp:
        root = write_consumer(Path(tmp))
        edit(root, C, ON_CHECK, on)
        assert meta.problems(root, pages=True) == expected


# --- 4. check steps (rule 3) ----------------------------------------------------------------

SWAPPED = f"      - run: make check\n      - run: {SYNC}\n"
BLOCKS = f"""      - run: |
          {SYNC}
      - run: |
          make check
"""
EXTRA_JOB = """
  notify:
    runs-on: ubuntu-24.04
    steps:
      - run: echo done
"""
OTHER = "football-demo-model"


@pytest.mark.parametrize(
    ("edits", "expected"),
    [
        ([(C, "- run: make check", "- run: make test")], [NO_CHECK_TARGET]),
        ([(C, SYNC_THEN_CHECK, SWAPPED)], [NO_CHECK_TARGET]),
        # not an installing job any more, so its job rows (3, 4) do not apply
        ([(C, f"- run: {SYNC}", "- run: uv sync --all-groups")], [NO_CHECK_TARGET]),
        ([(C, f"          path: {D}\n", "          path: src\n")], [first_step(C, "check")]),
        ([(C, OWN + STOPGAP + SETUP, SETUP + OWN + STOPGAP)], [first_step(C, "check")]),
        (
            [
                (
                    C,
                    f"          path: {D}\n",
                    f"          repository: someone/{D}\n          path: {D}\n",
                )
            ],
            [first_step(C, "check")],
        ),
        (
            [(C, DEFAULTS, DEFAULTS.replace(f"working-directory: {D}", "working-directory: ."))],
            [workdir(C, "check")],
        ),
        ([(C, SYNC_THEN_CHECK, BLOCKS)], []),
        # a job with SYNC but no target fails the file row and keeps its own rows
        (
            [(C, "- run: make check", "- run: make test"), (C, f"          path: {D}\n", "")],
            [NO_CHECK_TARGET, first_step(C, "check")],
        ),
        # job rows apply to installing jobs only
        ([(C, SYNC_THEN_CHECK, SYNC_THEN_CHECK + EXTRA_JOB)], []),
        (
            [("pyproject.toml", NAME_LINE, f'name = "{OTHER}"')],
            [
                first_step(C, "check", OTHER),
                workdir(C, "check", OTHER),
                first_step(P, "build", OTHER),
                workdir(P, "build", OTHER),
                f"{P}: no job uploads '{OTHER}/site'",
            ],
        ),
    ],
    ids=[
        "make-test",
        "check-before-sync",
        "unlocked-sync",
        "own-path",
        "setup-uv-first",
        "own-repository",
        "workdir",
        "run-blocks",
        "no-target-and-path",
        "extra-job",
        "name-mismatch",
    ],
)
def test_check_steps(consumer: Path, edits: list[Edit], expected: list[str]) -> None:
    apply(consumer, edits)
    assert meta.problems(consumer, pages=True) == expected


# Two installing jobs in one file (the fixture census found none elsewhere): one job running the
# target satisfies the file row, and the job rows and core rows apply to each installing job.
LINT_JOB = (
    """
  lint:
    runs-on: ubuntu-24.04
"""
    + DEFAULTS
    + "    steps:\n"
    + OWN
    + STOPGAP
    + SETUP
    + f"      - run: {SYNC}\n      - run: make lint\n"
)
CHECK_WORKDIR = "    timeout-minutes: 30\n" + DEFAULTS
LINT_HEAD = "    runs-on: ubuntu-24.04\n" + DEFAULTS + "    steps:\n" + OWN + STOPGAP


@pytest.mark.parametrize(
    ("edits", "expected"),
    [
        ([], []),
        # rows outer, jobs inner (file order): every first-step row before any workdir row
        (
            [
                (
                    C,
                    CHECK_WORKDIR,
                    CHECK_WORKDIR.replace(f"working-directory: {D}", "working-directory: ."),
                ),
                (
                    C,
                    LINT_HEAD,
                    "    runs-on: ubuntu-24.04\n" + DEFAULTS + "    steps:\n" + SETUP + OWN,
                ),
            ],
            [first_step(C, "lint"), workdir(C, "check"), core_count(C, "lint", 0, 1)],
        ),
        # the target moved to the lint job alone still satisfies the file row
        (
            [
                (C, "- run: make check", "- run: make test"),
                (C, "- run: make lint", "- run: make check"),
            ],
            [],
        ),
        ([(C, "- run: make check", "- run: make test")], [NO_CHECK_TARGET]),
    ],
    ids=["compliant", "row-order", "target-in-second-job", "no-target"],
)
def test_two_installing_jobs(consumer: Path, edits: list[Edit], expected: list[str]) -> None:
    edit(consumer, C, SYNC_THEN_CHECK, SYNC_THEN_CHECK + LINT_JOB)
    apply(consumer, edits)
    doc = meta._load(consumer, C, [])
    assert doc is not None and [j for j, *_ in meta._jobs(doc, installing=True)] == [
        "check",
        "lint",
    ]
    assert meta.problems(consumer, pages=True) == expected


# --- 5. core beside (rule 4, the seam) ------------------------------------------------------

FORK = STOPGAP.replace("}}/football-core\n", "}}/football-core-fork\n")
CORE_ELSEWHERE = STOPGAP.replace("path: football-core\n", "path: core\n")


@pytest.mark.parametrize(
    ("edits", "expected"),
    [
        ([(C, STOPGAP, "")], [core_count(C, "check", 0, 1)]),
        (
            [("pyproject.toml", PATH_SOURCE, GIT_SOURCE)],
            [core_count(C, "check", 1, 0), core_count(P, "build", 1, 0)],
        ),
        ([("pyproject.toml", PATH_SOURCE, GIT_SOURCE), (C, STOPGAP, ""), (P, STOPGAP, "")], []),
        ([(C, STOPGAP, STOPGAP + STOPGAP)], [core_count(C, "check", 2, 1)]),
        # no [tool.uv.sources] at all (core itself): expected 0
        (
            [("pyproject.toml", "[tool.uv.sources]\n" + PATH_SOURCE + "\n", "")],
            [core_count(C, "check", 1, 0), core_count(P, "build", 1, 0)],
        ),
        # composite key: repository must end in /football-core AND path must be football-core
        ([(C, STOPGAP, FORK)], [core_count(C, "check", 0, 1)]),
        ([(C, STOPGAP, CORE_ELSEWHERE)], [core_count(C, "check", 0, 1)]),
    ],
    ids=[
        "deleted-early",
        "git-kept",
        "git-deleted",
        "duplicated",
        "no-sources",
        "fork",
        "other-path",
    ],
)
def test_core_checkout_iff_path_source(
    consumer: Path, edits: list[Edit], expected: list[str]
) -> None:
    apply(consumer, edits)
    assert meta.problems(consumer, pages=True) == expected


@settings(deadline=None, max_examples=40)
@given(
    n_check=st.integers(0, 3),
    n_docs=st.integers(0, 3),
    path_source=st.booleans(),
)
def test_core_checkout_count_property(n_check: int, n_docs: int, path_source: bool) -> None:
    """Expected count is 1 with a path source, else 0; each file reports its own mismatch."""
    e = 1 if path_source else 0
    expected = [core_count(C, "check", n_check, e)] if n_check != e else []
    expected += [core_count(P, "build", n_docs, e)] if n_docs != e else []
    with tempfile.TemporaryDirectory() as tmp:
        root = write_consumer(Path(tmp))
        edit(root, C, STOPGAP, STOPGAP * n_check)
        edit(root, P, STOPGAP, STOPGAP * n_docs)
        if not path_source:
            edit(root, "pyproject.toml", PATH_SOURCE, GIT_SOURCE)
        assert meta.problems(root, pages=True) == expected


# --- 6. pins --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "pinned"),
    [
        ("v7", False),
        ("main", False),
        ("v7.0", False),
        ("v7.0.1.2", False),
        ("7.0.1", False),
        ("v7.0.1-rc.1", False),
        ("v12.10.0", True),
    ],
)
def test_unpinned_action(consumer: Path, tag: str, pinned: bool) -> None:
    uses = f"actions/checkout@{tag}"
    edit_first(consumer, C, "actions/checkout@v7.0.1", uses)
    expected = [] if pinned else [f"{C}: unpinned '{uses}'"]
    assert meta.problems(consumer, pages=True) == expected


# --- 7. pages -------------------------------------------------------------------------------

PERMS = f"{P}: job deploy: permissions need pages: write and id-token: write"


@pytest.mark.parametrize(
    ("pages_flag", "edits", "expected"),
    [
        (True, [(P, DOCS_YML, None)], [f"{P}: missing"]),
        (False, [(P, DOCS_YML, None)], []),  # a skeleton without a site
        (False, [], [f"{P}: unexpected (pages=False)"]),
        (True, [(P, UPLOAD_PATH, "path: site")], [f"{P}: no job uploads '{D}/site'"]),
        (True, [(P, "- run: make docs", "- run: make html")], [NO_DOCS_TARGET]),
        (True, [(P, DEPLOY, "")], [f"{P}: no job uses actions/deploy-pages"]),
        (True, [(P, NEEDS, "")], [f"{P}: job deploy: needs must name 'build'"]),
        (True, [(P, NEEDS, "    needs: [check]\n")], [f"{P}: job deploy: needs must name 'build'"]),
        (True, [(P, NEEDS, "    needs: [lint, build]\n")], []),
        (True, [(P, "      id-token: write\n", "")], [PERMS]),
        (True, [(P, "      pages: write\n", "")], [PERMS]),
        (True, [(P, "      pages: write\n", "      pages: read\n")], [PERMS]),
        (
            True,
            [(P, BRANCHES_MAIN, "    branches: [master]\n")],
            [f"{P}: on.push.branches must be ['main']"],
        ),
        (
            True,
            [(P, BRANCHES_MAIN, "    branches: [main, dev]\n")],
            [f"{P}: on.push.branches must be ['main']"],
        ),
    ],
    ids=[
        "deleted",
        "deleted-no-pages",
        "unexpected",
        "upload-path",
        "make-html",
        "no-deploy",
        "no-needs",
        "needs-other",
        "needs-list",
        "no-id-token",
        "no-pages-write",
        "pages-read",
        "master",
        "main-and-dev",
    ],
)
def test_pages(
    consumer: Path,
    pages_flag: bool,
    edits: list[tuple[str, str, str | None]],
    expected: list[str],
) -> None:
    for rel, old, new in edits:
        if new is None:
            (consumer / rel).unlink()
        else:
            edit(consumer, rel, old, new)
    assert meta.problems(consumer, pages=pages_flag) == expected


# --- 8. unreadable input: fixed-text problems, no traceback ---------------------------------


def _replace(rel: str, text: str) -> Callable[[Path], None]:
    def write(root: Path) -> None:
        (root / rel).write_text(text, encoding="utf-8")

    return write


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (_replace(C, "name: check\non: [push\n"), [f"{C}: invalid YAML"]),
        (_replace(C, "- push\n- pull_request\n"), [f"{C}: not a mapping"]),
        (_replace(C, "check\n"), [f"{C}: not a mapping"]),
        (_replace(C, ""), [f"{C}: not a mapping"]),
        (_replace(P, "jobs: {build: [\n"), [f"{P}: invalid YAML"]),
        (_replace(P, "- build\n"), [f"{P}: not a mapping"]),
        # a workflow with no jobs key has no jobs
        (_replace(C, ON_CHECK), [NO_CHECK_TARGET]),
        (
            lambda r: edit(r, "pyproject.toml", "[tool.uv.sources]", "[tool.uv.sources"),
            ["pyproject.toml: invalid TOML"],
        ),
        (
            lambda r: edit(r, "pyproject.toml", NAME_LINE + "\n", ""),
            ["pyproject.toml: no [project] name"],
        ),
        (
            lambda r: edit(r, "pyproject.toml", NAME_LINE, "name = 3"),
            ["pyproject.toml: no [project] name"],
        ),
    ],
    ids=[
        "check-yaml",
        "check-list",
        "check-scalar",
        "check-empty",
        "docs-yaml",
        "docs-list",
        "no-jobs",
        "toml",
        "no-name",
        "name-int",
    ],
)
def test_unreadable(consumer: Path, change: Callable[[Path], None], expected: list[str]) -> None:
    change(consumer)
    assert meta.problems(consumer, pages=True) == expected


# --- order across phases --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pages_flag", "expected"),
    [
        (
            False,
            ["LICENSE: missing", "mkdocs.yml: missing", "pyproject.toml: missing", f"{C}: missing"],
        ),
        (
            True,
            [
                "LICENSE: missing",
                "mkdocs.yml: missing",
                "pyproject.toml: missing",
                f"{C}: missing",
                f"{P}: missing",
            ],
        ),
    ],
)
def test_empty_root(tmp_path: Path, pages_flag: bool, expected: list[str]) -> None:
    assert meta.problems(tmp_path / "no-such-repo", pages=pages_flag) == expected


def test_problem_order(consumer: Path) -> None:
    """Rule 1, load, then rules 2-6; check.yml before docs.yml; steps in file order."""
    (consumer / "mkdocs.yml").unlink()  # rule 1
    edit(consumer, C, "  pull_request:\n", "")  # rule 2
    edit(consumer, C, DEFAULTS, DEFAULTS.replace(f"working-directory: {D}", "working-directory: ."))
    edit(consumer, C, STOPGAP, "")  # rule 4
    edit(consumer, P, "astral-sh/setup-uv@v10.3.0", "astral-sh/setup-uv@v10")  # rule 5, docs
    edit(consumer, C, "actions/checkout@v7.0.1", "actions/checkout@main")  # rule 5, check
    edit(consumer, P, BRANCHES_MAIN, "    branches: [master]\n")  # rule 6
    edit(consumer, P, NEEDS, "")  # rule 6, later row
    assert meta.problems(consumer, pages=True) == [
        "mkdocs.yml: missing",
        LACKS_PR,
        workdir(C, "check"),
        core_count(C, "check", 0, 1),
        f"{C}: unpinned 'actions/checkout@main'",
        f"{P}: unpinned 'astral-sh/setup-uv@v10'",
        f"{P}: on.push.branches must be ['main']",
        f"{P}: job deploy: needs must name 'build'",
    ]


# --- docs wiring of the new module ----------------------------------------------------------


def test_meta_docs_wiring() -> None:
    """API page and nav after docs.theme, modules row, package docstring, clean module docstring
    with its one doctest."""
    assert "::: football_docs.meta" in (REPO / "docs/api/docs_meta.md").read_text(encoding="utf-8")
    nav = [line.strip() for line in (REPO / "mkdocs.yml").read_text(encoding="utf-8").splitlines()]
    theme = nav.index("- docs.theme: api/docs_theme.md")
    assert nav[theme + 1] == "- docs.meta: api/docs_meta.md"
    modules = (REPO / "docs/modules.md").read_text(encoding="utf-8")
    row = next(line for line in modules.splitlines() if "`football_docs.roles`" in line)
    assert "`.meta`" in row and "repository metadata check: author, CI and Pages wiring" in row
    package = football_docs.__doc__ or ""
    assert "Four modules" in package and "football_docs.meta" in package
    assert "``meta`` imports ``yaml``" in package
    module = meta.__doc__ or ""
    assert module.strip()
    assert not any(f":{role}:`" in module for role in roles.ROLE_NAMES)
    assert pages.docstring_codes(meta.__file__) == []
    examples = [ex.source for test in doctest.DocTestFinder().find(meta) for ex in test.examples]
    assert any('problems(Path("no-such-repo"), pages=False)' in src for src in examples)


# --- build decisions R1-R8 (DR1 repo-meta build.md, Architect: ratify) ----------------------

PERMS2 = f"{P}: job deploy2: permissions need pages: write and id-token: write"
BUILD_BLOCK = DOCS_YML[DOCS_YML.index("  build:\n") : DOCS_YML.index(DEPLOY)]
BARE_DEPLOY = DEPLOY.replace(NEEDS, "").replace(DEPLOY_PERMS, "")


@pytest.mark.parametrize(
    "text",
    ["jobs: {build: [\n", DOCS_YML.replace("@v5.0.1", "@v5").replace(UPLOAD_PATH, "path: x")],
    ids=["invalid-yaml", "non-compliant"],
)
def test_r1_unexpected_docs_is_not_loaded(consumer: Path, text: str) -> None:
    """R1: with pages=False a present docs.yml gives only 'unexpected'; rules 3-6 skip it."""
    (consumer / P).write_text(text, encoding="utf-8")
    assert meta.problems(consumer, pages=False) == [f"{P}: unexpected (pages=False)"]


def test_r2_load_phase_file_order(consumer: Path) -> None:
    """R2: pyproject rows, then check.yml, then docs.yml; file order beats the table's row order
    (the table lists 'invalid YAML' before 'not a mapping')."""
    edit(consumer, "pyproject.toml", "[tool.uv.sources]", "[tool.uv.sources")
    (consumer / C).write_text("- push\n", encoding="utf-8")
    (consumer / P).write_text("jobs: {build: [\n", encoding="utf-8")
    assert meta.problems(consumer, pages=True) == [
        "pyproject.toml: invalid TOML",
        f"{C}: not a mapping",
        f"{P}: invalid YAML",
    ]


def test_r3_rule6_rows_outer_jobs_inner(consumer: Path) -> None:
    """R3: with two deploy jobs, every needs row comes before any permissions row."""
    edit(consumer, P, DEPLOY, BARE_DEPLOY + BARE_DEPLOY.replace("  deploy:\n", "  deploy2:\n"))
    assert meta.problems(consumer, pages=True) == [
        f"{P}: job deploy: needs must name 'build'",
        f"{P}: job deploy2: needs must name 'build'",
        PERMS,
        PERMS2,
    ]


def test_r4_target_after_first_sync(consumer: Path) -> None:
    """R4: 'after its SYNC step' is after the first SYNC; a later second SYNC does not undo it."""
    edit(consumer, C, SYNC_THEN_CHECK, SYNC_THEN_CHECK + f"      - run: {SYNC}\n")
    assert meta.problems(consumer, pages=True) == []


def test_r5_job_level_uses_not_judged(consumer: Path) -> None:
    """R5: a reusable-workflow job (job-level uses) is not judged by the pin rule."""
    reuse = "\n  reuse:\n    uses: someone/shared/.github/workflows/lint.yml@main\n"
    edit(consumer, C, SYNC_THEN_CHECK, SYNC_THEN_CHECK + reuse)
    assert meta.problems(consumer, pages=True) == []


@pytest.mark.parametrize("needs", ["build", "[build2, build]"], ids=["first", "both"])
def test_r6_needs_names_first_uploading_job(consumer: Path, needs: str) -> None:
    """R6: with two uploading jobs the needs row names the first in file order."""
    edit(consumer, P, DEPLOY, BUILD_BLOCK.replace("  build:\n", "  build2:\n") + DEPLOY)
    assert meta.problems(consumer, pages=True) == []
    edit(consumer, P, NEEDS, "    needs: build2\n")
    assert meta.problems(consumer, pages=True) == [f"{P}: job deploy: needs must name 'build'"]
    edit(consumer, P, "    needs: build2\n", f"    needs: {needs}\n")
    assert meta.problems(consumer, pages=True) == []


def test_r7_null_repository_is_absent(consumer: Path) -> None:
    """R7: own checkout with `repository:` null counts as no repository."""
    edit_first(
        consumer, C, f"          path: {D}\n", f"          repository:\n          path: {D}\n"
    )
    doc = meta._load(consumer, C, [])
    assert doc is not None and "repository" in doc["jobs"]["check"]["steps"][0]["with"]
    assert meta.problems(consumer, pages=True) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("uses: astral-sh/setup-uv@v10.3.0", "uses: 7", [f"{C}: unpinned '7'"]),
        ("- run: make check", "- run: [make, check]", [NO_CHECK_TARGET]),
    ],
    ids=["uses-int", "run-list"],
)
def test_r8_non_string_uses_and_run(
    consumer: Path, old: str, new: str, expected: list[str]
) -> None:
    """R8: a non-string uses is judged as str(uses); a non-string run is no command."""
    edit(consumer, C, old, new)
    assert meta.problems(consumer, pages=True) == expected
