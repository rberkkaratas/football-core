"""Agreement with mplsoccer's Standardizer, an independent landmark-anchored implementation."""

import numpy as np
import pytest
from mplsoccer import Standardizer

from football_core.pitch import FRAMES, to_canonical


@pytest.mark.parametrize("provider", sorted(FRAMES))
def test_matches_mplsoccer_standardizer(provider):
    frame = FRAMES[provider]
    rng = np.random.default_rng(7)
    # Strictly inside the pitch: mplsoccer clips at the sidelines too, but differently at edges
    px = rng.uniform(min(frame.x) + 0.01, max(frame.x) - 0.01, 5_000)
    py = rng.uniform(min(frame.y) + 0.01, max(frame.y) - 0.01, 5_000)

    oracle = Standardizer(pitch_from=provider, pitch_to="custom", length_to=105, width_to=68)
    ox, oy = oracle.transform(px, py)
    x, y = to_canonical(px, py, provider)

    np.testing.assert_allclose(x, ox, atol=1e-6)
    np.testing.assert_allclose(y, oy, atol=1e-6)
