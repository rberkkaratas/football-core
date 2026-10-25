"""Automatic attribution: the StatsBomb logo and text, the WhoScored text, the default logo path
and its failures (C7 spec §1, §2 "Attribution is automatic", "Logo read")."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import viz_cases as vc

from football_core import viz
from football_core.io import statsbomb

CHARTS = ("shot_map", "goal_mouth")
SB, WS = "Data: StatsBomb", "Data: Opta via WhoScored"


@pytest.mark.parametrize("chart", CHARTS)
def test_statsbomb_gets_logo_and_text(chart: str, synthetic_logo: Path) -> None:
    ax = vc.render(chart, vc.edge_frame("statsbomb"), logo=synthetic_logo)
    assert vc.texts(ax, viz.TEXT_GID) == [SB]
    assert len(vc.logos(ax)) == 1
    assert viz.ATTRIBUTION["statsbomb"] == SB


@pytest.mark.parametrize("chart", CHARTS)
def test_whoscored_gets_text_only(chart: str, tmp_path: Path) -> None:
    # the logo is never read for a frame without StatsBomb rows, so a missing path is fine
    ax = vc.render(chart, vc.edge_frame("whoscored"), logo=tmp_path / "missing.png")
    assert vc.texts(ax, viz.TEXT_GID) == [WS]
    assert vc.logos(ax) == []
    assert viz.ATTRIBUTION["whoscored"] == WS


@pytest.mark.parametrize("chart", CHARTS)
def test_mixed_providers_get_both(chart: str, synthetic_logo: Path) -> None:
    # whoscored on the first row: the text still follows schema.PROVIDERS order
    providers = ["whoscored"] + ["statsbomb"] * 8
    ax = vc.render(chart, vc.edge_frame(providers), logo=synthetic_logo)
    assert vc.texts(ax, viz.TEXT_GID) == [f"{SB} · {WS}"]
    assert len(vc.logos(ax)) == 1
    direct = vc.blank_axes()
    viz.add_attribution(direct, "whoscored", "statsbomb", logo=synthetic_logo)
    assert vc.texts(direct, viz.TEXT_GID) == [f"{SB} · {WS}"]
    assert len(vc.logos(direct)) == 1


def test_default_logo_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    default = statsbomb.logo_file()
    assert (
        default == Path(statsbomb.DEFAULT_CACHE) / statsbomb.OPEN_DATA_COMMIT / statsbomb.LOGO_PATH
    )
    assert Path(viz.DEFAULT_LOGO) == default  # viz spells it out so it needs no loader import
    vc.write_logo(default)
    for chart in CHARTS:
        ax = vc.render(chart, vc.edge_frame("statsbomb"))  # no logo=
        assert len(vc.logos(ax)) == 1, chart
    default.unlink()
    for chart in CHARTS:
        with pytest.raises(FileNotFoundError) as info:
            vc.render(chart, vc.edge_frame("statsbomb"))
        assert str(default) in str(info.value), chart


def _bad_logo(kind: str, tmp_path: Path) -> Path:
    path = tmp_path / f"{kind}.png"
    if kind == "directory":
        path.mkdir()
    elif kind == "dangling-symlink":
        os.symlink(tmp_path / "nowhere.png", path)
    return path  # "missing": nothing at the path


@pytest.mark.parametrize("chart", CHARTS)
@pytest.mark.parametrize("kind", ["missing", "directory", "dangling-symlink"])
def test_missing_logo_raises(chart: str, kind: str, tmp_path: Path) -> None:
    logo = _bad_logo(kind, tmp_path)
    ax = vc.blank_axes()
    with pytest.raises(FileNotFoundError) as info:
        vc.render(chart, vc.edge_frame("statsbomb"), ax=ax, logo=logo)
    assert str(logo) in str(info.value)
    assert "fetch_logo" in str(info.value)
    assert vc.is_blank(ax), "a bad logo must raise before anything is drawn"
    # add_attribution reads the logo first: no half-made attribution on failure
    direct = vc.blank_axes()
    with pytest.raises(FileNotFoundError):
        viz.add_attribution(direct, "statsbomb", logo=logo)
    assert vc.is_blank(direct)


@pytest.mark.parametrize("chart", CHARTS)
@pytest.mark.parametrize("content", [b"not a png", b""], ids=["text", "empty"])
def test_unreadable_logo_raises(chart: str, content: bytes, tmp_path: Path) -> None:
    logo = tmp_path / "logo.png"
    logo.write_bytes(content)
    ax = vc.blank_axes()
    with pytest.raises(ValueError, match="cannot read the logo") as info:
        vc.render(chart, vc.edge_frame("statsbomb"), ax=ax, logo=logo)
    assert str(logo) in str(info.value)
    assert vc.is_blank(ax)


@pytest.mark.parametrize("providers", [("opta",), ("statsbomb", "wyscout")])
def test_unknown_provider_raises(providers: tuple[str, ...], synthetic_logo: Path) -> None:
    ax = vc.blank_axes()
    with pytest.raises(ValueError) as info:
        viz.add_attribution(ax, *providers, logo=synthetic_logo)
    message = str(info.value)
    assert providers[-1] in message  # the bad one
    assert "statsbomb" in message  # and the allowed ones
    assert "whoscored" in message
    assert vc.is_blank(ax)


def test_attribution_not_repeated(synthetic_logo: Path) -> None:
    ax = vc.render("shot_map", vc.edge_frame("statsbomb"), logo=synthetic_logo)
    viz.add_attribution(ax, "statsbomb", logo=synthetic_logo)
    viz.add_attribution(ax, "whoscored")
    vc.render("goal_mouth", vc.edge_frame("statsbomb"), ax=ax, logo=synthetic_logo)
    assert vc.texts(ax, viz.TEXT_GID) == [SB]
    assert len(vc.logos(ax)) == 1
    # no providers: nothing at all
    blank = vc.blank_axes()
    viz.add_attribution(blank)
    assert vc.is_blank(blank)
