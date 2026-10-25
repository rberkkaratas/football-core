"""``statsbomb.sync``: the per-file download at the pinned commit into the git-ignored cache
(spec ``runs/MS2/c4/spec.md`` §1, §3). ``fetch`` is always faked; nothing uses the network."""

from __future__ import annotations

import email.message
import gzip
import json
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import snapshot_cases as snc
import statsbomb_cases as sbc
from hypothesis import example, given, settings
from hypothesis import strategies as st

from football_core.io import statsbomb


def _files() -> dict[str, bytes]:
    return sbc.repo_files(sbc.standard_matches())


def _leftovers(root: Path) -> list[Path]:
    return sorted(root.rglob("*.tmp"))


def test_sync_downloads_then_offline(tmp_path: Path) -> None:
    files, cache = _files(), tmp_path / "cache"
    fetch = sbc.FakeFetch(sbc.fake_remote(files))
    checkout = statsbomb.sync(cache, commit=sbc.COMMIT, fetch=fetch, workers=3)
    assert checkout == (cache / sbc.COMMIT).resolve()
    assert fetch.calls[0] == sbc.tree_url()
    assert sorted(fetch.calls[1:]) == sorted(sbc.raw_url(p) for p in files)
    tree = json.loads((checkout / "tree.json").read_text(encoding="utf-8"))
    assert tree == {p: [sbc.blob_sha1(raw), len(raw)] for p, raw in files.items()}
    for path, raw in files.items():
        assert (checkout / f"{path}.gz").read_bytes() == sbc.gz(raw), path
    assert _leftovers(cache) == []
    assert statsbomb.load(checkout).shots.height == sbc.SHOT_ROWS

    offline = sbc.FakeFetch({})  # any call fails the test
    assert statsbomb.sync(cache, commit=sbc.COMMIT, fetch=offline) == checkout
    assert offline.calls == []

    # a crash left a stale .tmp beside a complete file and one in place of a missing file
    beside = checkout / f"data/events/{sbc.A}.json.gz.tmp"
    beside.write_bytes(b"stale")
    missing = f"data/events/{sbc.B}.json"
    (checkout / f"{missing}.gz").unlink()
    (checkout / f"{missing}.gz.tmp").write_bytes(b"stale")
    assert statsbomb.load(checkout, match_ids=[sbc.A]).shots.height == 12  # .tmp never read
    resume = sbc.FakeFetch(sbc.fake_remote(files))
    statsbomb.sync(cache, commit=sbc.COMMIT, fetch=resume)
    assert resume.calls == [sbc.raw_url(missing)]
    assert (checkout / f"{missing}.gz").read_bytes() == sbc.gz(files[missing])
    assert not (checkout / f"{missing}.gz.tmp").exists()
    assert statsbomb.load(checkout).shots.height == sbc.SHOT_ROWS


def test_sync_blob_mismatch_writes_nothing(tmp_path: Path) -> None:
    files = _files()
    path = f"data/events/{sbc.A}.json"
    remote = sbc.fake_remote(files)
    remote[sbc.raw_url(path)] = files[path] + b" "  # not the blob the tree lists
    with pytest.raises(ValueError, match=f"events/{sbc.A}.json"):
        statsbomb.sync(
            tmp_path / "cache", commit=sbc.COMMIT, fetch=sbc.FakeFetch(remote), workers=1
        )
    checkout = tmp_path / "cache" / sbc.COMMIT
    assert not (checkout / f"{path}.gz").exists()
    assert not (checkout / f"{path}.gz.tmp").exists()


DECOYS = (
    "data/events/../../../evil.json",
    "data/events/1/../../../evil.json",
    "../data/events/1.json",
    "/data/events/1.json",
    "data/events/1.json.bak",
    "data/events/abc.json",
    "data/events/1.JSON",
    "data/events/１.json",  # a fullwidth digit is not [0-9]
    "data/events/1.json\n",
    "data/matches/1/2/3.json",
    "data/three-sixty/1/2.json",
    "data/360/1.json",
)


@pytest.mark.parametrize("case", ["truncated", "unsafe_path_skipped"])
def test_sync_tree_listing(tmp_path: Path, case: str) -> None:
    files, cache = _files(), tmp_path / "cache"
    if case == "truncated":
        fetch = sbc.FakeFetch(sbc.fake_remote(files, truncated=True))
        with pytest.raises(ValueError, match="truncated"):
            statsbomb.sync(cache, commit=sbc.COMMIT, fetch=fetch)
        assert fetch.calls == [sbc.tree_url()]
        assert not (cache / sbc.COMMIT / "tree.json").exists()
        return
    fetch = sbc.FakeFetch(sbc.fake_remote(files, extra_blobs=DECOYS))  # decoys are not served
    checkout = statsbomb.sync(cache, commit=sbc.COMMIT, fetch=fetch, workers=1)
    assert set(json.loads((checkout / "tree.json").read_text(encoding="utf-8"))) == set(files)
    assert len(fetch.calls) == 1 + len(files)
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {f"cache/{sbc.COMMIT}/tree.json"} | {
        f"cache/{sbc.COMMIT}/{p}.gz" for p in files
    }


def test_sync_fetch_oserror_propagates(tmp_path: Path) -> None:
    files, cache = _files(), tmp_path / "cache"
    paths = sorted(files)
    boom = sbc.raw_url(paths[1])
    fetch = sbc.FakeFetch(sbc.fake_remote(files), fail={boom: OSError("boom: connection reset")})
    with pytest.raises(OSError, match="boom"):
        statsbomb.sync(cache, commit=sbc.COMMIT, fetch=fetch, workers=1)
    raw_calls = fetch.calls[1:]
    assert raw_calls[:2] == [sbc.raw_url(paths[0]), boom]
    assert len(raw_calls) <= 3  # stops after the failure (one task may already be in flight)
    checkout = cache / sbc.COMMIT
    assert not (checkout / f"{paths[1]}.gz").exists()
    assert _leftovers(cache) == []
    # renamed files are complete, so a rerun is the retry and fetches only what is missing
    todo = [p for p in paths if not (checkout / f"{p}.gz").exists()]
    retry = sbc.FakeFetch(sbc.fake_remote(files))
    statsbomb.sync(cache, commit=sbc.COMMIT, fetch=retry, workers=2)
    assert sorted(retry.calls) == sorted(sbc.raw_url(p) for p in todo)
    assert all((checkout / f"{p}.gz").read_bytes() == sbc.gz(files[p]) for p in paths)


@pytest.mark.parametrize(
    "commit",
    [
        "",
        "abc",
        "F00D" * 10,
        "g" * 40,
        sbc.COMMIT + "0",
        sbc.COMMIT[:-1],
        sbc.COMMIT + "\n",
        "../" + "0" * 37,
    ],
    ids=["empty", "short", "uppercase", "not_hex", "41", "39", "newline", "dotdot"],
)
def test_sync_bad_commit(tmp_path: Path, commit: str) -> None:
    fetch = sbc.FakeFetch({})
    with pytest.raises(ValueError, match="commit"):
        statsbomb.sync(tmp_path / "cache", commit=commit, fetch=fetch)
    assert fetch.calls == []
    assert not (tmp_path / "cache").exists()


@snc.needs_git
def test_sync_refuses_unignored_cache(tmp_path: Path) -> None:
    repo = snc.git_repo(tmp_path / "repo", ignore="*.log\n")
    fetch = sbc.FakeFetch(sbc.fake_remote(_files()))
    with pytest.raises(ValueError, match="ignore"):
        statsbomb.sync(repo / "data" / "raw" / "open-data", commit=sbc.COMMIT, fetch=fetch)
    assert fetch.calls == []
    assert not (repo / "data").exists()  # refused before any write
    ignored = snc.git_repo(tmp_path / "ignored")  # data/* is ignored
    checkout = statsbomb.sync(
        ignored / "data" / "raw" / "open-data", commit=sbc.COMMIT, fetch=fetch
    )
    assert (checkout / "tree.json").is_file()


@snc.needs_git
@settings(max_examples=25, deadline=None)
@given(data=st.binary(max_size=4096))
@example(data=b"")
@example(data=b"blob 3\x00abc")
def test_property_blob_sha1_matches_git(data: bytes) -> None:
    """The blob check ``sync`` applies is git's own object id for those bytes."""
    git = subprocess.run(
        ["git", "hash-object", "--stdin"],
        input=data,
        capture_output=True,
        check=True,
        env=snc.git_env(),
    )
    sha = git.stdout.decode("ascii").rstrip("\n")
    assert sbc.blob_sha1(data) == sha  # the fixture helper agrees with git
    files = {"data/competitions.json": data}
    with tempfile.TemporaryDirectory() as tmp:
        checkout = statsbomb.sync(
            Path(tmp), commit=sbc.COMMIT, fetch=sbc.FakeFetch(sbc.fake_remote(files))
        )
        assert json.loads((checkout / "tree.json").read_text(encoding="utf-8")) == {
            "data/competitions.json": [sha, len(data)]
        }
        assert gzip.decompress((checkout / "data/competitions.json.gz").read_bytes()) == data


class _Response:
    def __init__(self, body: bytes, encoding: str | None) -> None:
        self._body = body
        self.headers = email.message.Message()
        if encoding is not None:
            self.headers["Content-Encoding"] = encoding
        self.status = self.code = 200

    def read(self, amt: int | None = None) -> bytes:
        body, self._body = self._body, b""
        return body

    def info(self) -> email.message.Message:
        return self.headers

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name, default)

    def close(self) -> None:
        pass

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@pytest.mark.parametrize("case", ["gzip_body", "identity_body", "404", "bad_gzip_body"])
def test_http_get(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """``fetch=None`` uses ``_http_get`` (``urllib.request.urlopen`` patched here)."""
    files = {"data/competitions.json": b"[]"}
    remote = sbc.fake_remote(files)
    seen: list[tuple[Any, Any]] = []

    def urlopen(request: Any, data: Any = None, timeout: Any = None, **_: Any) -> _Response:
        seen.append((request, timeout))
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        if case == "404":
            raise urllib.error.HTTPError(url, 404, "Not Found", email.message.Message(), None)
        if case == "bad_gzip_body":
            return _Response(gzip.compress(remote[url])[:-12], "gzip")  # EOFError when decoded
        if case == "gzip_body":
            return _Response(gzip.compress(remote[url]), "gzip")
        return _Response(remote[url], None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    cache = tmp_path / "cache"
    if case in ("404", "bad_gzip_body"):
        with pytest.raises(OSError):
            statsbomb.sync(cache, commit=sbc.COMMIT)
        assert not (cache / sbc.COMMIT / "tree.json").exists()
        return
    checkout = statsbomb.sync(cache, commit=sbc.COMMIT, workers=1)
    assert gzip.decompress((checkout / "data/competitions.json.gz").read_bytes()) == b"[]"
    assert len(seen) == 2
    for request, timeout in seen:
        assert isinstance(request, urllib.request.Request)
        assert request.get_header("Accept-encoding") == "gzip"
        assert request.get_header("User-agent") == "football-core"
        assert timeout == 60
