"""Theme tokens, text styles, WCAG ratios and the unchanged half-pitch defaults (spec
``runs/CH1/style-shots`` §2.2, §2.4, §2.5, §2.6)."""

from __future__ import annotations

import dataclasses
import re

import canvas_cases as cc
import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from matplotlib.colors import to_hex

from football_core import viz

# spec §2.2, token -> (light, dark); Berk's deck values where WCAG allows them
TOKENS = {
    "background": ("#f8fafc", "#0f172a"),
    "panel": ("#eef2f6", "#1a2436"),
    "title": ("#475569", "#f1f5f9"),
    "ink": ("#334155", "#e2e8f0"),
    "muted": ("#5b6b80", "#a3b1c4"),
    "line": ("#cbd5e1", "#334155"),
    "box_line": ("#a3afc0", "#475569"),
    "band": ("#d8e0e9", "#2b3a52"),
    "accent": ("#e4002b", "#ff4d6a"),
    "mark": ("#475569", "#94a3b8"),
    "neutral": ("#64748b", "#94a3b8"),
}
TEXT_TOKENS = ("title", "ink", "muted")
SURFACES = ("background", "panel")
MARK_TOKENS = ("accent", "neutral", "ink")
HEX = re.compile(r"#[0-9a-f]{6}")


def test_themes_share_tokens() -> None:
    """One immutable mapping, two frozen themes with the same named tokens and the spec values."""
    assert set(viz.THEMES) == {"light", "dark"}
    light, dark = viz.THEMES["light"], viz.THEMES["dark"]
    assert isinstance(light, viz.Theme) and isinstance(dark, viz.Theme)
    names = [f.name for f in dataclasses.fields(viz.Theme)]
    assert set(TOKENS) | {"mark_alpha", "logo_plate", "family"} <= set(names)
    for token, (lv, dv) in TOKENS.items():
        assert (getattr(light, token), getattr(dark, token)) == (lv, dv), token
        assert HEX.fullmatch(getattr(light, token)) and HEX.fullmatch(getattr(dark, token))
    assert light.mark_alpha == dark.mark_alpha == 0.7
    assert (light.logo_plate, dark.logo_plate) == (None, "#f8fafc")
    assert light.family == dark.family == viz.FAMILY
    with pytest.raises(dataclasses.FrozenInstanceError):
        light.accent = "#000000"  # type: ignore[misc]
    with pytest.raises(TypeError):
        viz.THEMES["sepia"] = light  # type: ignore[index]


def test_token_pairs_meet_wcag() -> None:
    """Text tokens >= 4.5 on background and panel; marks (accent, neutral, ink, mark at its
    alpha) >= 3.0 on background, panel and band; both themes, recomputed independently."""
    for name, t in viz.THEMES.items():
        for surface in SURFACES:
            bg = getattr(t, surface)
            for token in TEXT_TOKENS:
                assert cc.wcag(getattr(t, token), bg) >= 4.5, (name, token, surface)
                assert viz.contrast_ratio(getattr(t, token), bg) >= 4.5, (name, token, surface)
        for surface in (*SURFACES, "band"):
            bg = getattr(t, surface)
            for token in MARK_TOKENS:
                assert cc.wcag(getattr(t, token), bg) >= 3.0, (name, token, surface)
            ring = cc.composite(t.mark, t.mark_alpha, bg)
            assert cc.wcag(ring, bg) >= 3.0, (name, "mark@alpha", surface)
            assert viz.contrast_ratio(bg, ring) >= 3.0, (name, "mark@alpha", surface)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("#000000", "#ffffff", 21.0),
        ("#ffffff", "#000000", 21.0),
        ("#ffffff", "#ffffff", 1.0),
        ("#767676", "#ffffff", 4.54),
        ("#777777", "#ffffff", 4.48),
        ("#000000", (1.0, 1.0, 1.0), 21.0),
        ("#e4002b", "#f8fafc", 4.63),
    ],
)
def test_contrast_ratio_known_values(
    a: str, b: str | tuple[float, float, float], expected: float
) -> None:
    """WCAG 2.x: published pairs (black/white 21, #767676 just passes 4.5 on white, #777777
    just fails), a tuple background in 0..1 RGB, Berk's accent on his deck background."""
    assert viz.contrast_ratio(a, b) == pytest.approx(expected, abs=0.006)


@settings(deadline=None, max_examples=60)
@given(a=st.tuples(*[st.integers(0, 255)] * 3), b=st.tuples(*[st.integers(0, 255)] * 3))
def test_contrast_ratio_property(a: tuple[int, int, int], b: tuple[int, int, int]) -> None:
    """Symmetric, within [1, 21], equal to the written-out formula."""
    ha, hb = ("#{:02x}{:02x}{:02x}".format(*v) for v in (a, b))
    r = viz.contrast_ratio(ha, hb)
    assert 1.0 - 1e-12 <= r <= 21.0 + 1e-12
    assert r == pytest.approx(viz.contrast_ratio(hb, ha), rel=1e-12)
    assert r == pytest.approx(cc.wcag(ha, hb), rel=1e-9)


def test_text_styles() -> None:
    """Shared styles: data-bearing text >= 11.5 pt, only the footer at 9.5; the title step."""
    s = viz.TEXT_STYLES
    expected = {
        "title": (21, "bold", "title"),
        "subtitle": (13, "regular", "muted"),
        "stat_value": (20, "semibold", "ink"),
        "stat_label": (11.5, "medium", "muted"),
        "stat_note": (11.5, "regular", "muted"),
        "heading": (13, "semibold", "ink"),
        "label": (12, "medium", "ink"),
        "caption": (11.5, "regular", "muted"),
        "legend": (11.5, "regular", "ink"),
        "footer": (9.5, "regular", "muted"),
        "footer_strong": (9.5, "semibold", "muted"),
    }
    assert set(s) == set(expected)
    for name, (size, weight, token) in expected.items():
        got_size, got_weight, got_token = s[name]
        assert (float(got_size), got_token) == (float(size), token), name
        assert str(got_weight).lower() == weight, name
        assert got_token in {f.name for f in dataclasses.fields(viz.Theme)}
    assert all(size >= 11.5 for name, (size, _, _) in s.items() if not name.startswith("footer"))
    assert viz.TITLE_STEP_PT == 19.0
    assert viz.TITLE_MIN_PT == 17.0  # D1/R1 ratified: the last step before a title raises
    assert viz.TITLE_FALLBACK_PT == 15.0  # gate G1: fallback family only


def test_half_pitch_defaults_unchanged() -> None:
    """MS2 defaults stay (colour, width); the new keywords colour the boxes apart (§2.4)."""
    ax = viz.draw_half_pitch()
    for line in ax.lines:
        assert to_hex(line.get_color()) == to_hex(viz.LINE_COLOUR), line.get_gid()
        assert line.get_linewidth() == viz.PITCH_LW, line.get_gid()
    ax = viz.draw_half_pitch(colour="#cbd5e1", lw=0.8, box_colour="#a3afc0")
    boxes = {"pitch-box", "pitch-six-yard-box"}
    seen = set()
    for line in ax.lines:
        gid = line.get_gid()
        seen.add(gid)
        want = "#a3afc0" if gid in boxes else "#cbd5e1"
        assert to_hex(line.get_color()) == want, gid
        assert line.get_linewidth() == 0.8, gid
    assert boxes <= seen and "pitch-outline" in seen
    ax = viz.draw_half_pitch(colour="#cbd5e1")  # box_colour None -> colour
    assert {to_hex(ln.get_color()) for ln in ax.lines} == {"#cbd5e1"}
