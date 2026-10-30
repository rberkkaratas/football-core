"""Vendored Inter: digests, licence, default use, registration and the tested fallback (spec
``runs/CH1/style-shots`` §2.1, §4 "Fonts licensed and shipped").

``fontManager`` is process-global, so every registration case runs in a fresh subprocess.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import canvas_cases as cc
import pytest
from matplotlib import font_manager

from football_core import viz

DIGESTS = {
    "Inter-Regular.ttf": "64f8be6e55c37e32ef03da99714bf3aa58b8f2099bfe4f759a7578e3b8291123",
    "Inter-Medium.ttf": "1bda81124d6ae26ed16a7201e2bd93766af5a3b14faf79eea14d191ebbd41146",
    "Inter-SemiBold.ttf": "0dc98e8aa59585394880f25ab89e6d915ad5134522e961b046ca51fad3a18255",
    "Inter-Bold.ttf": "0cb1bc1335372d9e3a0cf6f5311c7cce87af90d2a777fdeec18be605a2a70bc1",
    "LICENSE.txt": "262481e844521b326f5ecd053e59b98c8b2da78c8ee1bdbb6e8174305e54935a",
}
WEIGHT_FILES = ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf")


def test_vendored_digests_and_licence() -> None:
    """The 4 TTFs and the OFL licence sit in package data with the spec's digests; the folder
    holds no ``.py`` (a ``fonts/__init__.py`` would enter the cache hash, p2)."""
    assert dict(viz.FONT_SHA256) == DIGESTS
    folder = Path(viz.FONT_DIR)
    assert folder == Path(viz.__file__).parent / "fonts"
    for name, digest in DIGESTS.items():
        assert cc.sha256(folder / name) == digest, name
    licence = (folder / "LICENSE.txt").read_bytes()
    assert len(licence) == 4380
    text = licence.decode("utf-8")
    assert "SIL OPEN FONT LICENSE" in text.upper() and "Version 1.1" in text
    assert list(folder.glob("*.py")) == []
    assert (viz.FONT_FAMILY, viz.FALLBACK_FAMILY) == ("Inter", "DejaVu Sans")


def test_default_family_is_inter(synthetic_logo: Path) -> None:
    """At import Inter registers; every Text of a themed canvas resolves to a file under
    ``FONT_DIR``, the title to the Bold file."""
    assert viz.FAMILY == viz.FONT_FAMILY
    assert viz.THEMES["light"].family == viz.THEMES["dark"].family == "Inter"
    c = cc.make(
        "dark", "square", logo=synthetic_logo, provider="statsbomb", stats=cc.FIVE_STATS,
        model="xG: own model", note="Not shown: 1 penalty (1 scored)",
    )  # fmt: skip
    found = cc.texts(c)
    assert len(found) >= 10  # precondition: title, subtitle, strip, footer
    folder = Path(viz.FONT_DIR).resolve()
    for t in found:
        path = font_manager.findfont(t.get_fontproperties(), fallback_to_default=False)
        assert Path(path).resolve().parent == folder, (t.get_text(), path)
    titles = cc.styled(c, "title", cc.TITLE_SIZES)
    assert titles, "precondition: the title is found by its style"
    for title in titles:
        path = font_manager.findfont(title.get_fontproperties(), fallback_to_default=False)
        assert Path(path).name == "Inter-Bold.ttf"


_SCRIPT = textwrap.dedent(
    """
    import json, sys, warnings
    from football_core import viz
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        family = viz.register_fonts(sys.argv[1])
    print(json.dumps({"family": family,
                      "warnings": [str(w.message) for w in caught
                                   if issubclass(w.category, UserWarning)]}))
    """
)


def _register(directory: Path) -> dict[str, object]:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(directory)],
        capture_output=True, text=True, env=env, timeout=120, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "Traceback" not in done.stderr
    result: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
    return result


def _font_dir(tmp_path: Path, case: str) -> tuple[Path, str | None]:
    """A font folder for ``case`` and the file name the warning must name (None: no warning)."""
    d = tmp_path / case
    d.mkdir()
    src = Path(viz.FONT_DIR)
    if case == "empty":
        return d, "Inter-Regular.ttf"
    for name in (*WEIGHT_FILES, "LICENSE.txt"):
        shutil.copyfile(src / name, d / name)
    if case == "complete":
        return d, None
    if case == "one-missing":
        (d / "Inter-Medium.ttf").unlink()
        return d, "Inter-Medium.ttf"
    if case == "unreadable":
        (d / "Inter-Bold.ttf").write_bytes(b"not a font file")
        return d, "Inter-Bold.ttf"
    if case == "other-family":  # a weight's file replaced by another family's TTF
        dejavu = Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))
        shutil.copyfile(dejavu, d / "Inter-SemiBold.ttf")
        return d, "Inter-SemiBold.ttf"
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["complete", "empty", "one-missing", "unreadable", "other-family"])
def test_register_fonts_falls_back(case: str, tmp_path: Path) -> None:
    """All four weights resolving to their own file -> ``"Inter"``, no warning. Otherwise one
    ``UserWarning`` naming the first missing, unreadable or mis-resolving file, and the
    fallback family (fresh subprocess each)."""
    directory, named = _font_dir(tmp_path, case)
    result = _register(directory)
    if named is None:
        assert result == {"family": "Inter", "warnings": []}
        return
    assert result["family"] == "DejaVu Sans"
    warnings = result["warnings"]
    assert isinstance(warnings, list) and len(warnings) == 1, warnings
    assert named in warnings[0], warnings


def test_fallback_canvas_passes_check(synthetic_logo: Path) -> None:
    """With the fallback family and the longest measured strings, every theme and size still
    lays out and passes every machine check."""
    for name in viz.THEMES:
        theme = dataclasses.replace(viz.THEMES[name], family=viz.FALLBACK_FAMILY)
        for size in viz.SIZES:
            c = cc.make(
                theme, size, logo=synthetic_logo, provider="statsbomb", title=cc.TITLE,
                subtitle=cc.SUBTITLE, stats=cc.FIVE_STATS,
                model="xG: own model, not trained on Women's Super League 2018/2019",
                note=cc.NOTE,
            )  # fmt: skip
            assert {t.get_fontfamily()[0] for t in cc.texts(c)} == {"DejaVu Sans"}
            assert viz.check(c) == [], (name, size)
            assert any("Data: StatsBomb" in t.get_text() for t in cc.texts(c))


def test_import_leaves_rc_defaults() -> None:
    """Importing ``football_core.viz`` (which registers the fonts) changes no ``rcParams``."""
    script = textwrap.dedent(
        """
        import json
        import matplotlib
        before = {k: repr(v) for k, v in matplotlib.rcParams.items()}
        import football_core.viz
        after = {k: repr(v) for k, v in matplotlib.rcParams.items()}
        print(json.dumps(sorted(k for k in before if before[k] != after.get(k))))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []
