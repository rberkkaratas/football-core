"""``statsbomb.fetch_logo`` / ``logo_file``: the StatsBomb logo is downloaded pinned and
blob-verified into the git-ignored cache and never committed (C7 spec §1, §3). ``fetch`` is
always faked; nothing uses the network."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import snapshot_cases as snc
import statsbomb_cases as sbc
import viz_cases as vc

from football_core.io import statsbomb

REPO = Path(__file__).resolve().parents[1]
LOGO_PATH = "img/SB - Icon Lockup - Colour positive.png"
QUOTED = "img/SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png"


def _logo_url() -> str:
    return f"https://raw.githubusercontent.com/statsbomb/open-data/{statsbomb.OPEN_DATA_COMMIT}/{QUOTED}"


@pytest.fixture
def fake_logo(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Synthetic PNG bytes standing in for the logo; ``LOGO_BLOB`` patched to their blob id."""
    raw = vc.write_logo(tmp_path_factory.mktemp("fake_logo") / "fake.png").read_bytes()
    monkeypatch.setattr(statsbomb, "LOGO_BLOB", sbc.blob_sha1(raw))
    return raw


def test_fetch_logo_downloads_verifies_and_caches(tmp_path: Path, fake_logo: bytes) -> None:
    assert statsbomb.LOGO_PATH == LOGO_PATH
    repo = snc.git_repo(tmp_path / "repo")
    cache = repo / "data" / "raw" / "statsbomb" / "open-data"
    assert statsbomb.logo_file(cache) == cache / statsbomb.OPEN_DATA_COMMIT / LOGO_PATH
    fetch = sbc.FakeFetch({_logo_url(): fake_logo})

    path = statsbomb.fetch_logo(cache, fetch=fetch)
    assert path == (cache / statsbomb.OPEN_DATA_COMMIT / LOGO_PATH).resolve()
    assert path.read_bytes() == fake_logo
    assert fetch.calls == [_logo_url()]  # one call, the path URL-quoted (it has spaces)

    assert statsbomb.fetch_logo(cache, fetch=fetch) == path
    assert len(fetch.calls) == 1  # cached and verified: no second download

    path.write_bytes(b"corrupt")
    assert statsbomb.fetch_logo(cache, fetch=fetch) == path
    assert len(fetch.calls) == 2
    assert path.read_bytes() == fake_logo
    assert sorted(cache.rglob("*.tmp")) == []
    assert snc.git(repo, "status", "--porcelain", "--untracked-files=all") == ""


def test_fetch_logo_rejects_wrong_bytes(tmp_path: Path, fake_logo: bytes) -> None:
    repo = snc.git_repo(tmp_path / "repo")
    cache = repo / "data" / "cache"
    fetch = sbc.FakeFetch({_logo_url(): fake_logo + b"x"})
    with pytest.raises(ValueError, match=statsbomb.LOGO_BLOB):
        statsbomb.fetch_logo(cache, fetch=fetch)
    target = statsbomb.logo_file(cache)
    assert not target.exists()
    assert not target.with_name(f"{target.name}.tmp").exists()


def test_fetch_logo_refuses_unignored_cache(tmp_path: Path, fake_logo: bytes) -> None:
    repo = snc.git_repo(tmp_path / "repo")  # ignores data/* only
    cache = repo / "cache"
    fetch = sbc.FakeFetch({_logo_url(): fake_logo})
    with pytest.raises(ValueError, match="git-ignored"):
        statsbomb.fetch_logo(cache, fetch=fetch)
    assert fetch.calls == []  # refused before any download
    assert not cache.exists() or sorted(p for p in cache.rglob("*") if p.is_file()) == []


@snc.needs_git
def test_fetch_logo_symlinked_data_fails_closed(tmp_path: Path, fake_logo: bytes) -> None:
    """As ``sync``: a ``data/`` symlinked out of the repo cannot be checked; nothing is fetched."""
    repo = snc.git_repo(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "data").symlink_to(outside, target_is_directory=True)
    fetch = sbc.FakeFetch({_logo_url(): fake_logo})
    with pytest.raises(ValueError, match="cannot check whether"):
        statsbomb.fetch_logo(repo / "data" / "raw" / "statsbomb" / "open-data", fetch=fetch)
    assert fetch.calls == []
    assert list(outside.iterdir()) == []


def test_logo_not_tracked() -> None:
    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", "-C", str(REPO), *args],
            capture_output=True, encoding="utf-8", env=snc.git_env(), check=False,
        )  # fmt: skip
        if out.returncode == 128 and "not a git repository" in out.stderr:
            pytest.skip("football-core is not a git checkout here")
        return out.stdout

    staged = git("ls-files", "-s", "-z").split("\0")
    assert not [e for e in staged if statsbomb.LOGO_BLOB in e]
    paths = [e.split("\t", 1)[1] for e in staged if "\t" in e]
    stray = [
        p for p in paths
        if p.lower().endswith(".png")
        and not p.startswith(("tests/baseline_images/", "docs/"))
    ]  # fmt: skip
    assert stray == []
    # and the default download target is ignored by this repo's .gitignore
    ignored = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", "--", str(statsbomb.logo_file())],
        env=snc.git_env(), check=False,
    )  # fmt: skip
    assert ignored.returncode == 0
