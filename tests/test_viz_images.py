"""Image tests: both charts from the edge fixture against reviewed baselines at ``IMAGE_TOL``, and
byte-identical renders (C7 spec §1 "Deterministic backend", §7).

Baselines live in ``tests/baseline_images/viz/`` and contain only the synthetic fixture and the
synthetic logo, never StatsBomb data or the StatsBomb logo. To regenerate them after an intended
change: ``FC_UPDATE_BASELINES=1 uv run pytest tests/test_viz_images.py``, then look at the PNGs.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import matplotlib.style
import pytest
import viz_cases as vc


def test_shot_map_image(synthetic_logo: Path, tmp_path: Path) -> None:
    ax = vc.render("shot_map", vc.edge_frame("statsbomb"), logo=synthetic_logo)
    vc.check_image(ax, "shot_map", tmp_path)


def test_goal_mouth_image(synthetic_logo: Path, tmp_path: Path) -> None:
    ax = vc.render("goal_mouth", vc.edge_frame("statsbomb"), logo=synthetic_logo)
    vc.check_image(ax, "goal_mouth", tmp_path)


def _png_sha256(chart: str, logo: Path) -> str:
    ax = vc.render(chart, vc.edge_frame("statsbomb"), logo=logo)
    buf = io.BytesIO()
    with matplotlib.style.context("default"):
        ax.figure.savefig(buf, format="png", dpi=100, metadata={"Software": None})
    return hashlib.sha256(buf.getvalue()).hexdigest()


@pytest.mark.parametrize("chart", ["shot_map", "goal_mouth"])
def test_render_is_deterministic(chart: str, synthetic_logo: Path) -> None:
    assert _png_sha256(chart, synthetic_logo) == _png_sha256(chart, synthetic_logo)
