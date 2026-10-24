"""The ``fc`` command line: a thin layer over ``football_core.io.snapshot``.

``fc snapshot`` builds and writes a snapshot, ``fc verify`` checks snapshots against their
manifests, ``fc builders`` lists the builders. Exit codes: 0 success, 1 a snapshot or
verification problem, 2 a usage error or a manifest that cannot be read. ``fc`` is also a
shell builtin in zsh and bash, so run it as ``uv run fc``, or use the alias ``football-core``
in an activated shell.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from collections.abc import Sequence
from typing import Final

from football_core import schema
from football_core.io import snapshot

PARAM_KEY_PATTERN: Final[str] = r"^[a-z][a-z0-9_-]*$"


def _version() -> str:
    try:
        return importlib.metadata.version("football-core")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fc", description="Snapshots and manifests for the football-* repos."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s (football-core) {_version()}"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    snap = commands.add_parser("snapshot", help="build and write a snapshot plus its manifest")
    snap.add_argument("name", metavar="NAME", help="snapshot name, e.g. statsbomb-open-v1")
    snap.add_argument("--builder", default="parquet", help="builder name (default: parquet)")
    snap.add_argument(
        "--input", action="append", default=[], metavar="PATH", help="builder input (repeatable)"
    )
    snap.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE", help="builder parameter"
    )
    snap.add_argument("--root", default=".", metavar="DIR", help="consuming repo (default: .)")
    snap.add_argument("--overwrite", action="store_true", help="replace different data")

    verify = commands.add_parser("verify", help="re-hash snapshots against their manifests")
    verify.add_argument("manifests", nargs="+", metavar="MANIFEST")
    verify.add_argument("--data-dir", metavar="DIR", help="data directory (one manifest only)")

    commands.add_parser("builders", help="list the available builders")
    return parser


def _error(message: object, code: int) -> int:
    print(f"fc: error: {message}", file=sys.stderr)
    return code


def _params(raw: Sequence[str]) -> dict[str, str] | str:
    """KEY=VALUE pairs split at the first ``=``, or the usage error message."""
    params: dict[str, str] = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or re.fullmatch(PARAM_KEY_PATTERN, key) is None:
            return f"bad --param {item!r}: expected KEY=VALUE with KEY matching {PARAM_KEY_PATTERN}"
        if key in params:
            return f"--param key {key!r} is given twice"
        params[key] = value
    return params


def _snapshot(args: argparse.Namespace) -> int:
    params = _params(args.param)
    if isinstance(params, str):
        return _error(params, 2)
    try:
        snapshot.get_builder(args.builder)
    except ValueError as e:
        return _error(e, 2)
    try:
        result = snapshot.snapshot_from_builder(
            args.name,
            args.builder,
            inputs=args.input,
            params=params,
            root=args.root,
            overwrite=args.overwrite,
        )
    except (
        schema.SchemaValidationError,
        snapshot.SnapshotExistsError,
        snapshot.ManifestError,
        ValueError,
        OSError,
    ) as e:
        return _error(e, 1)
    status: str = result.status
    if result.status == "unchanged" and result.manifest_written:
        status = "unchanged (manifest updated)"
    tables = result.manifest.tables
    rows = sum(t.rows for t in tables)
    print(
        f"{status} snapshot '{result.manifest.name}': {len(tables)} table(s), {rows} row(s); "
        f"data {result.data_dir}; manifest {result.manifest_path}"
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    if args.data_dir is not None and len(args.manifests) != 1:
        return _error("--data-dir needs exactly one manifest", 2)
    code = 0
    for manifest in args.manifests:
        try:
            report = snapshot.verify_snapshot(manifest, data_dir=args.data_dir)
        except snapshot.ManifestError as e:
            code = max(code, _error(e, 2))
            continue
        if report.ok:
            print(f"ok: snapshot '{report.name}' ({manifest}): {report.checked} file(s) match")
            continue
        for problem in report.problems:
            print(problem, file=sys.stderr)
        print(
            f"verify failed: snapshot '{report.name}' ({manifest}): "
            f"{len(report.problems)} problem(s)",
            file=sys.stderr,
        )
        code = max(code, 1)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``fc`` and return its exit code (never calls ``sys.exit`` itself).

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name; default ``sys.argv[1:]``.

    Returns
    -------
    int
        0 success; 1 a snapshot error or a verification problem; 2 a usage error, an unknown
        builder or a manifest that cannot be read.

    Examples
    --------
    >>> main(["verify", "no/such/manifest.json"])  # the message goes to stderr
    2
    """
    parser = _parser()
    try:
        args = parser.parse_args(None if argv is None else list(argv))
    except SystemExit as e:  # argparse exits on --help, --version and usage errors
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
    if args.command == "snapshot":
        return _snapshot(args)
    if args.command == "verify":
        return _verify(args)
    for name in snapshot.builder_names():
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
