"""Run with:  cd pc && python -m pytest tests"""

import numpy as np
import pandas as pd

from cobb import c_curve, fit_c, fit_poly
from piezo import locate


def test_turn_maps():
    assert locate.turn_map("none") == {d: d for d in locate.DISCS}
    # bri 2026-11-23: head's up side to the phantom's right
    assert locate.turn_map("cw90") == {"dr": "dl", "dl": "ul", "ul": "ur", "ur": "dr"}
    assert locate.turn_map("180") == {"ur": "dl", "ul": "dr", "dl": "ur", "dr": "ul"}
    cw, ccw = locate.turn_map("cw90"), locate.turn_map("ccw90")
    assert all(ccw[cw[d]] == d for d in locate.DISCS)


def _fake(n=120, seed=0):
    """Stamps whose right/left share follows x, plus a common gain that share must cancel."""
    rng = np.random.default_rng(seed)
    freqs = np.arange(50, 200, 14.0)
    x = rng.choice([-10, -5, 0, 5, 10], n).astype(float)
    rows = []
    for i in range(n):
        gain = rng.normal(0, 0.3)            # pressing harder: all discs together
        for ch, side in [("ur", 1), ("dr", 1), ("dl", -1), ("ul", -1)]:
            for f in freqs:
                la = 2 + gain - 0.02 * side * x[i] + rng.normal(0, 0.03)
                rows.append({"trial_id": f"t{i}", "channel": ch, "freq_hz": f, "amp": 10 ** la})
    return x, pd.DataFrame(rows)


def test_share_cancels_gain_and_ridge_recovers_x():
    x, feats = _fake()
    ids = [f"t{i}" for i in range(len(x))]
    X, names, chans, freqs = locate.share_matrix(feats, ids)
    assert np.allclose(X.reshape(len(x), len(chans), -1).sum(axis=1), 0)
    groups = np.arange(len(x)) % 5
    pred = locate.grouped_cv(X, x, groups)
    assert np.mean(np.abs(pred - x)) < 1.0


def test_cobb_fits_recover_angle():
    y = np.arange(0, 121, 20.0)
    for cobb in (10, 30, 50):
        x = c_curve(cobb, y, 0, 120)
        assert abs(fit_c(y, x)[0] - cobb) < 0.01
        assert abs(fit_poly(y, x, 4)[0] - cobb) < 2.5
    assert fit_c(y, np.zeros_like(y))[0] < 1e-6


def test_nearest_vertebra():
    lvl, dy = locate.nearest_vertebra([0, 20, 40, 80, 100, 120])
    assert list(lvl) == [1, 2, 2, 4, 4, 5]
    assert list(dy) == [0, 10, -10, 5, -15, 5]
