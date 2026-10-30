"""The shared layout: sizes at a fixed dpi, bands, measured wrapping, the stat strip, footer,
legend and axes helpers (spec ``runs/CH1/style-shots`` §2.3)."""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Any

import canvas_cases as cc
import matplotlib.figure
import pytest
from matplotlib.colors import to_hex
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from football_core import viz

CONTENT = 1080 - 2 * 64  # 952 px
TEXT_BLOCK = CONTENT - 170 - 20  # 762 px beside the logo


def test_png_pixel_size(tmp_path: Path) -> None:
    """Every theme and size saves at its exact pixel size and ``DPI``, byte-identical twice."""
    assert viz.SIZES == {"portrait": (1080, 1350), "square": (1080, 1080)}
    assert (viz.DPI, viz.MARGIN_PX, viz.GAP_PX) == (150, 64, 20)
    for theme, size in itertools.product(viz.THEMES, viz.SIZES):
        c = cc.make(theme, size)
        assert c.fig.dpi == viz.DPI
        first = viz.save(c, tmp_path / f"{theme}-{size}.png")
        assert isinstance(first, Path) and first.is_file()
        with Image.open(first) as img:
            assert img.size == viz.SIZES[size], (theme, size)
            assert round(img.info["dpi"][0]) == viz.DPI
        again = viz.save(cc.make(theme, size), tmp_path / f"{theme}-{size}-again.png")
        assert first.read_bytes() == again.read_bytes(), (theme, size)


@pytest.mark.parametrize("n_stats", [0, 5])
@pytest.mark.parametrize("size", ["portrait", "square"])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_band_order_and_check_clean(
    theme: str, size: str, n_stats: int, synthetic_logo: Path
) -> None:
    """Worst-case strings: title above subtitle above the strip above the body; the footer below
    it; no text inside the empty body or outside the side margins; ``check == []``."""
    c = cc.make(
        theme, size, logo=synthetic_logo, provider="statsbomb", title=cc.TITLE,
        subtitle=cc.SUBTITLE, stats=cc.FIVE_STATS[:n_stats], model=cc.MODEL, note=cc.NOTE,
    )  # fmt: skip
    r = cc.renderer(c)
    body = cc.body_px(c)
    assert body.height >= viz.MIN_BODY_PX
    title = cc.styled(c, "title", cc.TITLE_SIZES)
    subtitle = cc.styled(c, "subtitle")
    assert cc.words(title, r) == cc.TITLE
    assert cc.words(subtitle, r) == cc.SUBTITLE
    above, below = [], []
    for t in cc.texts(c):
        e = t.get_window_extent(r)
        assert e.x0 >= viz.MARGIN_PX - 0.5 and e.x1 <= 1080 - viz.MARGIN_PX + 0.5, t.get_text()
        if e.y0 >= body.y1 - 0.5:
            above.append(t)
        elif e.y1 <= body.y0 + 0.5:
            below.append(t)
        else:
            raise AssertionError(f"text inside the empty body: {t.get_text()!r}")
    assert cc.union(title, r).y0 >= cc.union(subtitle, r).y1 - 0.5
    rest = [t for t in above if t not in title and t not in subtitle]
    if n_stats:
        values = cc.styled(c, "stat_value")
        assert sorted(t.get_text() for t in values) == sorted(s.value for s in cc.FIVE_STATS)
        assert cc.union(subtitle, r).y0 >= cc.union(rest, r).y1 - 0.5
    else:
        assert rest == []
    below_text = " ".join(t.get_text() for t in below)
    assert viz.PREPARED_BY in [t.get_text() for t in below]
    assert "Data: StatsBomb" in below_text
    assert viz.check(c) == []


def test_title_steps_down_then_raises() -> None:
    """A title that fits 2 lines at 21 pt stays at 21; one that needs 3 at 21 and 2 at 19 drops
    to ``TITLE_STEP_PT``; one that needs 3 at 19 drops to ``TITLE_MIN_PT`` (ratified D1/R1: the
    Leicester box-share title is 2,171 px at 21 pt); one that needs 3 even there raises."""
    probe = cc.make()
    fits = cc.words_to_width(probe, "title", 1.3 * CONTENT, 1.6 * CONTENT)
    step = cc.words_to_width(probe, "title", 2 * CONTENT + 81, 2 * CONTENT + 111)
    never = cc.words_to_width(probe, "title", 3.6 * CONTENT, 4.2 * CONTENT)
    # 3 lines at 19 pt (> 2 x 952 px there) and 2 at 17 pt: the last step
    below_step = cc.words_to_width(probe, "title", 2130.0, 2200.0)
    assert probe.measure(below_step, "title")[0] * viz.TITLE_STEP_PT / 21.0 > 2 * CONTENT
    for text, pt in ((fits, 21.0), (step, viz.TITLE_STEP_PT), (below_step, viz.TITLE_MIN_PT)):
        c = cc.make(title=text)
        r = cc.renderer(c)
        ts = cc.styled(c, "title", cc.TITLE_SIZES)
        assert {float(t.get_fontsize()) for t in ts} == {pt}
        assert cc.lines_of(ts) == 2
        assert cc.words(ts, r) == text
    with pytest.raises(ValueError, match="title"):
        cc.make(title=never)


def test_subtitle_two_lines_then_raises() -> None:
    probe = cc.make()
    two = cc.words_to_width(probe, "subtitle", 1.4 * CONTENT, 1.6 * CONTENT)
    c = cc.make(subtitle=two)
    ts = cc.styled(c, "subtitle")
    assert cc.lines_of(ts) == 2 and cc.words(ts, cc.renderer(c)) == two
    with pytest.raises(ValueError, match="subtitle"):
        cc.make(subtitle=cc.words_to_width(probe, "subtitle", 3.3 * CONTENT, 3.8 * CONTENT))


def _overlap(a: Any, b: Any) -> bool:
    return min(a.x1, b.x1) - max(a.x0, b.x0) > 0 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0


def test_strip_columns_from_content() -> None:
    """A label wider than an equal share fits because its column widens (equal columns would
    push it past the margin or onto its neighbour); minimums over 952 px raise, naming the
    widest stat."""
    probe = cc.make()
    long = cc.words_to_width(probe, "stat_label", 0.70 * CONTENT, 0.74 * CONTENT)
    assert probe.measure(long, "stat_label")[0] > CONTENT / 2 + 100  # precondition
    c = cc.make(stats=(viz.Stat("12", long), viz.Stat("3.4", "xG")))
    r = cc.renderer(c)
    body = cc.body_px(c)
    heads = cc.styled(c, "title", cc.TITLE_SIZES) + cc.styled(c, "subtitle")
    strip = [t for t in cc.texts(c) if t not in heads and t.get_window_extent(r).y0 >= body.y1]
    assert len(strip) >= 4  # precondition: 2 values and 2 labels
    assert any(" ".join(t.get_text().split()) == long for t in strip)
    for t in strip:
        e = t.get_window_extent(r)
        assert e.x0 >= viz.MARGIN_PX - 0.5 and e.x1 <= 1080 - viz.MARGIN_PX + 0.5, t.get_text()
    for a, b in itertools.combinations(strip, 2):
        assert not _overlap(a.get_window_extent(r), b.get_window_extent(r))
    assert viz.check(c) == []
    labels = [cc.words_to_width(probe, "stat_label", w, w + 40) for w in (230, 230, 230, 230)]
    widest = "Widest " + cc.words_to_width(probe, "stat_label", 260, 300)
    with pytest.raises(ValueError) as info:
        cc.make(stats=[viz.Stat("1", lab) for lab in labels] + [viz.Stat("2", widest)])
    assert widest in str(info.value)


def test_footer_wraps_then_raises(synthetic_logo: Path) -> None:
    """The note wraps into the 762 px text block beside the logo, <= 2 lines; a note or an
    attribution line over its limit raises, naming the part."""
    probe = cc.make()
    note = "Not shown: " + cc.words_to_width(probe, "caption", 1.3 * TEXT_BLOCK, 1.5 * TEXT_BLOCK)
    c = cc.make(provider="statsbomb", logo=synthetic_logo, note=note, model="xG: own model")
    r = cc.renderer(c)
    body = cc.body_px(c)
    caption = [t for t in cc.styled(c, "caption") if t.get_window_extent(r).y1 <= body.y0 + 0.5]
    assert cc.words(caption, r) == note and cc.lines_of(caption) == 2
    for t in caption:  # inside the text block, left of the logo
        assert t.get_window_extent(r).x1 <= viz.MARGIN_PX + TEXT_BLOCK + 0.5
    assert viz.check(c) == []
    long_note = "Not shown: " + cc.words_to_width(probe, "caption", 2.6 * TEXT_BLOCK, 3e3)
    with pytest.raises(ValueError, match="note|not-shown"):
        cc.make(provider="statsbomb", logo=synthetic_logo, note=long_note)
    long_model = cc.words_to_width(probe, "footer", 2.6 * TEXT_BLOCK, 3 * TEXT_BLOCK)
    with pytest.raises(ValueError, match="model|attribution"):
        cc.make(provider="statsbomb", logo=synthetic_logo, model=long_model)


@pytest.mark.parametrize("case", ["theme", "size", "provider", "six-stats", "body"])
def test_canvas_raises(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs: dict[str, Any] = {}
    bad = {"theme": "sepia", "size": "landscape", "provider": "opta"}.get(case)
    if case == "theme":
        kwargs["theme"] = "sepia"
    elif case == "size":
        kwargs["size"] = "landscape"
    elif case == "provider":
        kwargs["provider"] = "opta"
    elif case == "six-stats":
        kwargs["stats"] = [viz.Stat(str(i), f"S{i}") for i in range(6)]
    else:
        monkeypatch.setattr(viz, "MIN_BODY_PX", 1100)  # read when called; square has 1080 px
        kwargs["size"] = "square"
    with pytest.raises(ValueError) as info:
        cc.make(**kwargs)
    if bad is not None:
        assert bad in str(info.value)


def test_footer_without_provider_raises() -> None:
    """Ratified R3: a footer naming no provider is an uncredited chart and raises."""
    with pytest.raises(ValueError, match="providers"):
        viz.canvas("light", "square", title="T", subtitle="S", footer=viz.Footer(()))


@pytest.mark.parametrize("provider", ["statsbomb", "whoscored"])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_footer_attribution_prepared_by(theme: str, provider: str, synthetic_logo: Path) -> None:
    """ATTRIBUTION · model, then "Prepared by R. Berk Karataş" (U+015F) in footer_strong; the
    logo for StatsBomb only, right-aligned at ``LOGO_PX`` wide; its plate only in dark."""
    assert viz.PREPARED_BY == "Prepared by R. Berk Karataş"
    c = cc.make(theme, provider=provider, logo=synthetic_logo, model="xG: own model")
    r = cc.renderer(c)
    texts = [t.get_text() for t in cc.texts(c)]
    assert f"{viz.ATTRIBUTION[provider]} · xG: own model" in texts
    strong = cc.styled(c, "footer_strong")
    assert [t.get_text() for t in strong] == [viz.PREPARED_BY]
    assert str(strong[0].get_weight()).lower() in {"semibold", "600", "demibold"}
    logos = cc.by_gid(c, viz.LOGO_GID)
    t = viz.THEMES[theme]
    plates = [
        p for p in c.fig.findobj(FancyBboxPatch)
        if t.logo_plate is not None and to_hex(p.get_facecolor()) == t.logo_plate
    ]  # fmt: skip
    if provider == "whoscored":
        assert logos == [] and plates == []
        return
    assert len(logos) == 1
    e = logos[0].get_window_extent(r)
    assert e.x1 == pytest.approx(1080 - viz.MARGIN_PX, abs=1.0)
    assert e.width == pytest.approx(viz.LOGO_PX, abs=1.0)
    assert len(plates) == (1 if theme == "dark" else 0)


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_missing_logo_raises_blank(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """StatsBomb without a logo file raises ``FileNotFoundError`` before any artist is drawn."""
    logo = tmp_path / "logo.png"
    if kind == "directory":
        logo.mkdir()
    made: list[matplotlib.figure.Figure] = []
    init = matplotlib.figure.Figure.__init__

    def spy(self: matplotlib.figure.Figure, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(matplotlib.figure.Figure, "__init__", spy)
    with pytest.raises(FileNotFoundError) as info:
        cc.make(provider="statsbomb", logo=logo)
    assert str(logo) in str(info.value)
    for fig in made:
        assert not (fig.texts or fig.axes or fig.artists or fig.patches or fig.images)


def _keys(labels: list[str]) -> list[viz.Key]:
    return [viz.Key("goal" if i == 0 else "shot", lab, 64.0) for i, lab in enumerate(labels)]


def test_legend_under_axes_and_raises() -> None:
    """Keys sit directly under ``ax``, left-aligned with it, inside its width: one row when they
    fit, two when they do not, labels split in two lines when one-line labels need three;
    three rows needed even then raises."""
    c = cc.make()
    probe_w = c.measure
    ax = c.axes((0.0, 0.35, 1.0, 0.65), gid="map")
    c.legend(_keys(["Goal", "No goal", "0.1 xG", "0.3 xG"]), ax=ax)
    r = cc.renderer(c)
    a = ax.get_window_extent(r)
    marks = cc.by_gid(c, "mark-key")
    assert marks, "precondition: key handles carry gid mark-key"
    centres = [p for m in marks for p in cc.marker_centres(m).tolist()]
    assert len({round(y, 1) for _, y in centres}) == 1
    assert all(y < a.y0 for _, y in centres)
    radius = 64.0**0.5 / 2 * viz.DPI / 72  # Key.size is matplotlib's s in pt²
    assert min(x for x, _ in centres) - radius == pytest.approx(a.x0, abs=3.0)
    for t in cc.styled(c, "legend"):
        e = t.get_window_extent(r)
        assert e.y1 <= a.y0 + 0.5 and e.x0 >= a.x0 - 0.5 and e.x1 <= a.x1 + 0.5, t.get_text()
    assert viz.check(c) == []

    c = cc.make()
    narrow = c.axes((0.0, 0.35, 0.6, 0.65), gid="map")
    width = narrow.get_window_extent(cc.renderer(c)).width
    labels = [cc.words_to_width(c, "legend", 115, 135) + f" {i}" for i in range(5)]
    assert sum(probe_w(lab, "legend")[0] for lab in labels) > width  # precondition: one row fails
    c.legend(_keys(labels), ax=narrow)
    r = cc.renderer(c)
    centres = [p for m in cc.by_gid(c, "mark-key") for p in cc.marker_centres(m).tolist()]
    assert len({round(y, 1) for _, y in centres}) == 2
    assert viz.check(c) == []

    # one-line labels need 4 rows; split in two lines (the widest first) they take 2 (ratified
    # R5; integrate census: no core test reached the split)
    c = cc.make()
    narrow = c.axes((0.0, 0.35, 0.5, 0.65), gid="map")
    labels = [cc.words_to_width(c, "legend", 200, 230) + f" {i}" for i in range(4)]
    width = c.legend_width(narrow)
    key_w = min(probe_w(lab, "legend")[0] for lab in labels) + 64.0**0.5 * viz.DPI / 72
    assert 2 * (key_w + viz.KEY_GAP_PX) + viz.KEY_SPACING_PX > width  # precondition: 1 a row
    c.legend(_keys(labels), ax=narrow)
    r = cc.renderer(c)
    centres = [p for m in cc.by_gid(c, "mark-key") for p in cc.marker_centres(m).tolist()]
    assert len({round(y, 1) for _, y in centres}) == 2
    drawn = cc.styled(c, "legend")
    assert any("\n" in t.get_text() for t in drawn)
    assert sorted(" ".join(t.get_text().split()) for t in drawn) == sorted(labels)
    assert viz.check(c) == []

    c = cc.make()
    narrow = c.axes((0.0, 0.35, 0.5, 0.65), gid="map")
    labels = [cc.words_to_width(c, "legend", 150, 190) + f" {i}" for i in range(10)]
    with pytest.raises(ValueError):
        c.legend(_keys(labels), ax=narrow)


def test_axes_and_measure() -> None:
    """``axes`` places a rect given as a fraction of the body, axis off; ``panel=True`` adds the
    rounded panel patch and prefixes the gid; ``measure`` grows with the style's size."""
    c = cc.make("dark", "square")
    body = cc.body_px(c)
    before = len(c.fig.findobj(FancyBboxPatch))
    ax = c.axes((0.5, 0.0, 0.5, 1.0), gid="map")
    assert ax.get_gid() == "map" and not ax.axison
    e = ax.get_window_extent(cc.renderer(c))
    assert e.x0 == pytest.approx(body.x0 + 0.5 * body.width, abs=1.0)
    assert e.x1 == pytest.approx(body.x1, abs=1.0)
    assert len(c.fig.findobj(FancyBboxPatch)) == before
    panel = c.axes((0.0, 0.0, 0.45, 0.5), gid="zones", panel=True)
    assert panel.get_gid() == "panel-zones" and not panel.axison
    patches = [
        p for p in c.fig.findobj(FancyBboxPatch)
        if to_hex(p.get_facecolor()) == viz.THEMES["dark"].panel
    ]  # fmt: skip
    assert len(patches) == 1
    big, small = c.measure("Goals", "title"), c.measure("Goals", "caption")
    assert big[0] > small[0] > 0 and big[1] > small[1] > 0


# ---- run8-ch1 assessment: limits reached at their boundary, plain text, providers ---------------


def test_subtitle_three_lines_raise() -> None:
    """Assessment item 12 (C4): a subtitle that fits 3 lines but not 2 raises (the 2-line rule
    at its boundary, not only at 4+ lines)."""
    probe = cc.make()
    three = cc.words_to_width(probe, "subtitle", 2.2 * CONTENT, 2.8 * CONTENT)
    assert viz._wrap(probe, three, "subtitle", CONTENT, 3) is not None  # precondition
    with pytest.raises(ValueError, match="subtitle needs more than 2 line"):
        cc.make(subtitle=three)


def test_inter_title_never_takes_the_fallback_step() -> None:
    """Assessment item 12 (M17): ``TITLE_FALLBACK_PT`` is for the fallback family only; an Inter
    title that fits 2 lines only at 15 pt raises instead of shrinking."""
    probe = cc.make()
    assert cc.theme_of(probe).family == viz.FONT_FAMILY  # precondition: Inter registered
    text = cc.words_to_width(probe, "title", 2400.0, 2470.0)
    assert viz._wrap(probe, text, "title", CONTENT, 2, viz.TITLE_MIN_PT) is None  # precondition
    assert viz._wrap(probe, text, "title", CONTENT, 2, viz.TITLE_FALLBACK_PT) is not None
    with pytest.raises(ValueError, match="title needs more than 2 lines"):
        cc.make(title=text)


def test_title_text_is_plain() -> None:
    """Assessment item 15: a title is drawn as typed, never as mathtext ("$5 to $10" keeps its
    dollars and spaces), and an embedded newline counts against the 2-line limit."""
    c = cc.make(title="Cost $5 to $10")
    title = cc.by_gid(c, "title")[0]
    drawn = title.get_window_extent(cc.renderer(c)).width
    assert drawn == pytest.approx(c.measure("Cost $5 to $10", "title")[0], abs=0.5)
    assert drawn > c.measure("Cost 5 to 10", "title")[0]  # the $ glyphs are drawn
    assert " ".join(title.get_text().split()) == "Cost $5 to $10"
    assert cc.lines_of(cc.styled(cc.make(title="A\nB"), "title", cc.TITLE_SIZES)) == 2
    for text, part in (("A\nB\nC\nD", "title"), ("S\nT\nU", "subtitle")):
        with pytest.raises(ValueError, match=part):
            cc.make(**{part: text})


@pytest.mark.parametrize("providers", [("whoscored",), ("statsbomb", "whoscored")])
def test_canvas_carries_providers(providers: tuple[str, ...], synthetic_logo: Path) -> None:
    """Assessment item 6: the canvas records the providers its footer credits, so a writer reads
    the data rule from them rather than from the drawn text."""
    c = viz.canvas("light", "square", title="T", subtitle="S", logo=synthetic_logo,
                   footer=viz.Footer(providers))  # fmt: skip
    assert c.providers == providers


def test_fallback_credit_takes_a_third_line(synthetic_logo: Path) -> None:
    """run8-ch1 fix pass: the wider fallback family (never written: the chart writers refuse it)
    gets one more attribution line, as it gets ``TITLE_FALLBACK_PT``; Inter keeps 2 lines."""
    fallback = dataclasses.replace(viz.THEMES["light"], family=viz.FALLBACK_FAMILY)
    probe = cc.make(fallback)
    model = cc.words_to_width(probe, "footer", 2.2 * TEXT_BLOCK, 2.5 * TEXT_BLOCK)
    assert viz._wrap(probe, f"Data: StatsBomb · {model}", "footer", TEXT_BLOCK, 3)  # precondition
    c = cc.make(fallback, provider="statsbomb", logo=synthetic_logo, model=model)
    assert cc.lines_of(cc.by_gid(c, viz.TEXT_GID)) == 3 and viz.check(c) == []
    inter = cc.make()
    long_model = cc.words_to_width(inter, "footer", 2.2 * TEXT_BLOCK, 2.5 * TEXT_BLOCK)
    with pytest.raises(ValueError, match="attribution"):
        cc.make(provider="statsbomb", logo=synthetic_logo, model=long_model)
