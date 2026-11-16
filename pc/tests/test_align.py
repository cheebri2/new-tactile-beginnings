"""Alignment template and prediction on a toy dataset."""

import numpy as np
import pandas as pd

from piezo import align


def toy(seed=0):
    rng = np.random.default_rng(seed)
    freqs = [100.0, 200.0, 300.0]
    trials, feats = [], []
    k = 0
    for x in [0, 5, -5, 10, -10]:
        for kind in ["blank"] * 3 + ["object"] * 3:
            tid = f"t{k:03d}"
            trials.append(dict(trial_id=tid, time=f"2026-01-01T00:{k // 60:02d}:{k % 60:02d}",
                               test_type=kind, x_mm=float(x) if kind == "object" else 0.0, session="s"))
            for ch in ["ur", "ul", "dl", "dr"]:
                for i, f in enumerate(freqs):
                    shift = 0.0 if kind == "blank" else 0.03 * x * (i + 1) * (1 if ch == "dl" else -1)
                    feats.append(dict(trial_id=tid, channel=ch, freq_hz=f,
                                      amp=10 ** (1.5 + shift + rng.normal(0, 0.01)),
                                      phase_deg=20 * shift + rng.normal(0, 1)))
            k += 1
    return pd.DataFrame(trials), pd.DataFrame(feats)


def test_template_separates_toy_offsets():
    trials, feats = toy()
    tpl = align.build_template(trials, feats)
    assert set(tpl["positions"]) == {"0", "5", "-5", "10", "-10"}
    assert max(tpl["training_distance"]["0"]) < tpl["bound"]
    assert min(tpl["training_distance"]["10"]) > tpl["bound"]

    base = align.Baseline(tpl["channels"], tpl["freqs"])
    base.add(feats[feats.trial_id == "t000"])
    pred = align.predict(*base.change(feats[feats.trial_id == "t003"]), tpl)  # a centred trial
    assert pred["aligned"] and pred["nearest_x"] == 0
    pred = align.predict(*base.change(feats[feats.trial_id == "t021"]), tpl)  # x = 10
    assert not pred["aligned"] and pred["nearest_x"] == 10 and pred["est_x"] > 5
