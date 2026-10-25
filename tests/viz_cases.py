"""Synthetic shot frames and artist lookups for the ``football_core.viz`` tests (C7 spec
``runs/MS2/c7/spec.md`` §7).

Every frame is built column by column with the C3 dtypes and made-up values on valid football
rows; nothing is copied from a provider file. Artists are found by the gids of spec §2.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.style
import numpy as np
import polars as pl
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.testing.compare import compare_images

from football_core import schema, viz

IMAGE_TOL = 0.5  # above same-machine noise (0), below one goal's xG 0.45 -> 0.50 (RMS 1.14)
BASELINE_DIR = Path(__file__).parent / "baseline_images" / "viz"
CHARTS: dict[str, Callable[..., Axes]] = {"shot_map": viz.shot_map, "goal_mouth": viz.goal_mouth}
COORDS = {"shot_map": ("y", "x"), "goal_mouth": ("goal_y", "goal_z")}  # (horizontal, vertical)
MARKER_GIDS = (viz.GOALS_GID, viz.SHOTS_GID)


# --- Frames -----------------------------------------------------------------------------------


def frame(
    *,
    x: Sequence[float | None],
    y: Sequence[float | None],
    xg: Sequence[float | None],
    is_goal: Sequence[bool | None] | None = None,
    is_own_goal: Sequence[bool | None] | None = None,
    period: Sequence[int | None] | None = None,
    goal_y: Sequence[float | None] | None = None,
    goal_z: Sequence[float | None] | None = None,
    provider: str | Sequence[str | None] = "whoscored",
    xg_name: str = "xg",
) -> pl.DataFrame:
    """A shot frame with C3 dtypes. Defaults: open-play saves in period 1 at goal-mouth (34, 1)."""
    n = len(x)
    providers = [provider] * n if isinstance(provider, str) else list(provider)
    return pl.DataFrame(
        {
            "provider": pl.Series(providers, dtype=schema.PROVIDER),
            "x": pl.Series(list(x), dtype=pl.Float64()),
            "y": pl.Series(list(y), dtype=pl.Float64()),
            "is_goal": pl.Series(list(is_goal or [False] * n), dtype=pl.Boolean()),
            "is_own_goal": pl.Series(list(is_own_goal or [False] * n), dtype=pl.Boolean()),
            "period": pl.Series(list(period or [1] * n), dtype=pl.Int8()),
            "goal_y": pl.Series(list(goal_y or [34.0] * n), dtype=pl.Float64()),
            "goal_z": pl.Series(list(goal_z or [1.0] * n), dtype=pl.Float64()),
            xg_name: pl.Series(list(xg), dtype=pl.Float64()),
        }
    )


# The edge fixture of spec §7 (the p9 frame). Row: what it is -> shot map / goal mouth.
#   0 goal at the penalty spot              drawn / drawn
#   1 blocked shot, no goal-mouth point     drawn / "without a goal-mouth point"
#   2 off target, wide of the left post     drawn / drawn (inside the pad)
#   3 off target, over the bar              drawn / drawn (inside the pad)
#   4 goal just under the bar               drawn / drawn
#   5 shot from the own half, far wide      "outside the view" / "outside the view"
#   6 own goal (null goal-mouth point, xg)  "own goal(s)" / "own goal(s)"
#   7 shoot-out kick (period 5), goal       "shoot-out kick(s)" / "shoot-out kick(s)"
#   8 save wide of the right post, z = 0    drawn / drawn (not hidden)
EDGE_SHOT_MAP_DRAWN = (0, 1, 2, 3, 4, 8)
EDGE_GOAL_MOUTH_DRAWN = (0, 2, 3, 4, 8)
EDGE_NOTES = {
    "shot_map": "not shown: 1 own goal(s), 1 shoot-out kick(s), 1 outside the view",
    "goal_mouth": (
        "not shown: 1 own goal(s), 1 shoot-out kick(s), 1 without a goal-mouth point, "
        "1 outside the view"
    ),
}
EDGE_DRAWN = {"shot_map": EDGE_SHOT_MAP_DRAWN, "goal_mouth": EDGE_GOAL_MOUTH_DRAWN}


def edge_frame(provider: str | Sequence[str] = "statsbomb") -> pl.DataFrame:
    return frame(
        provider=provider,
        x=[94.0, 99.0, 88.0, 80.0, 102.5, 40.0, 100.0, 94.0, 97.0],
        y=[34.0, 30.0, 45.0, 20.0, 36.0, 34.0, 32.0, 34.0, 28.0],
        is_goal=[True, False, False, False, True, False, True, True, False],
        is_own_goal=[False, False, False, False, False, False, True, False, False],
        period=[1, 1, 2, 2, 2, 1, 2, 5, 1],
        goal_y=[31.0, None, 39.0, 36.5, 34.0, 50.0, None, 35.0, 29.5],
        goal_z=[0.5, None, 2.0, 3.0, 2.3, 1.0, None, 0.0, 0.0],
        xg=[0.76, 0.12, 0.04, 0.02, 0.45, 0.03, None, 0.76, 0.05],
    )


def set_cell(df: pl.DataFrame, col: str, row: int, value: Any) -> pl.DataFrame:
    """``df`` with one cell replaced, keeping the column's dtype."""
    values = df[col].to_list()
    values[row] = value
    return df.with_columns(pl.Series(col, values, dtype=df.schema[col]))


# --- Logos ------------------------------------------------------------------------------------


def write_logo(path: Path, height: int = 10, width: int = 60) -> Path:
    """A synthetic two-colour RGBA block (left red, right blue); never the StatsBomb logo."""
    img = np.zeros((height, width, 4))
    img[..., 3] = 1.0
    img[:, : width // 2, 0] = 1.0
    img[:, width // 2 :, 2] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, img)
    return path


# --- Rendering and lookups --------------------------------------------------------------------


def render(chart: str, shots: pl.DataFrame, **kwargs: Any) -> Axes:
    """Draw ``chart`` under matplotlib's default style (ignores the user's matplotlibrc)."""
    kwargs.setdefault("xg", "xg")
    with matplotlib.style.context("default"):
        return CHARTS[chart](shots, **kwargs)


def blank_axes() -> Axes:
    fig = Figure()
    FigureCanvasAgg(fig)
    return fig.add_axes((0.1, 0.1, 0.8, 0.8))


def is_blank(ax: Axes) -> bool:
    return (
        not ax.lines
        and not ax.collections
        and not ax.texts
        and not ax.artists
        and not ax.images
        and ax.get_legend() is None
    )


def collection(ax: Axes, gid: str) -> Any:
    found = [c for c in ax.collections if c.get_gid() == gid]
    assert len(found) <= 1, f"{len(found)} collections with gid {gid!r}"
    return found[0] if found else None


def offsets(ax: Axes, gid: str) -> np.ndarray:
    c = collection(ax, gid)
    return np.zeros((0, 2)) if c is None else np.asarray(c.get_offsets(), dtype=float)


def sizes(ax: Axes, gid: str) -> np.ndarray:
    c = collection(ax, gid)
    return np.zeros(0) if c is None else np.asarray(c.get_sizes(), dtype=float)


def n_drawn(ax: Axes) -> int:
    return sum(len(offsets(ax, gid)) for gid in MARKER_GIDS)


def texts(ax: Axes, gid: str) -> list[str]:
    return [t.get_text() for t in ax.texts if t.get_gid() == gid]


def note(ax: Axes) -> str | None:
    found = texts(ax, viz.NOTE_GID)
    assert len(found) <= 1, found
    return found[0] if found else None


def note_count(text: str | None) -> int:
    """The total of the counts in a "not shown: ..." note (0 when there is no note)."""
    if text is None:
        return 0
    assert text.startswith("not shown: "), text
    return sum(int(n) for n in re.findall(r"(?:^not shown: |, )([0-9]+) ", text))


def logos(ax: Axes) -> list[Any]:
    return [a for a in ax.artists if a.get_gid() == viz.LOGO_GID]


def lines(ax: Axes) -> dict[str, Any]:
    return {ln.get_gid(): ln for ln in ax.lines if ln.get_gid() is not None}


def expected_points(shots: pl.DataFrame, chart: str, rows: Sequence[int]) -> np.ndarray:
    h, v = COORDS[chart]
    picked = shots[list(rows)]
    return np.column_stack([picked[h].to_numpy(), picked[v].to_numpy()]).astype(float)


def same_points(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Equal as multisets of (h, v) points (order is tested separately)."""
    a = sorted(map(tuple, np.round(actual, 9).tolist()))
    e = sorted(map(tuple, np.round(expected, 9).tolist()))
    return a == e


def check_image(ax: Axes, name: str, tmp_path: Path) -> None:
    """Compare ``ax``'s figure with ``baseline_images/viz/<name>.png`` at ``IMAGE_TOL``.

    With ``FC_UPDATE_BASELINES=1`` the baseline is written instead and the test skips.
    """
    actual = tmp_path / f"{name}.png"
    with matplotlib.style.context("default"):
        ax.figure.savefig(actual, dpi=100, metadata={"Software": None})
    baseline = BASELINE_DIR / f"{name}.png"
    if os.environ.get("FC_UPDATE_BASELINES") == "1":
        baseline.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual, baseline)
        pytest.skip(f"baseline written; review it: {baseline}")
    assert baseline.is_file(), (
        f"no baseline at {baseline}; run with FC_UPDATE_BASELINES=1 and review the PNG"
    )
    result = compare_images(str(baseline), str(actual), tol=IMAGE_TOL)
    assert result is None, result
