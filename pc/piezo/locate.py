"""Locate the spinous process from one stamp (one sweep with the head pressed on the phantom).

What the model predicts
    x_mm of the stamp: how far the head centre is right (+) or left (-) of the spine's centreline.
    The centreline under the head is then at  head_x - predicted x_mm.

Features (per stamp)
    "share" log amplitude: log10(amp) of each disc minus the mean log10(amp) of the discs used, at
    every sweep frequency. Pressing harder or a looser seat scales all discs together, and this
    cancels it, so no blank is needed. On the 2026-11-22 batch this beat subtracting the preceding
    blank (single blanks moved by 25-100 % between reseats, more than the vertebra effect) and
    phase features (they mostly tracked a slow drift over the afternoon).

Model
    Ridge regression (standardised features, closed form, numpy only). The ridge strength is picked
    by cross-validation inside the training rows. Validation is grouped: every stamp of one group
    (default: one y row of the raster, i.e. one ~40 min block of the session) is held out together,
    so drift and reseat effects shared within a block cannot leak into the score.

Head turned on the phantom
    Discs are named by where they sit on the sensor face (ur, ul, dl, dr). When the head is turned,
    `HEAD_TURNS` renames them to where they sit on the phantom (phantom up = toward the head of the
    spine). 'cw90' is bri's 2026-11-22 batch: the head's up side points to the phantom's right, so
    recorded dr -> dl, dl -> ul, ul -> ur, ur -> dr. A faulty disc stays faulty after turning, so
    bad discs are always given by their recorded name.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DISCS = ["ur", "ul", "dl", "dr"]
# disc position on the sensor face, (right, up)
_POS = {"ur": (1, 1), "ul": (-1, 1), "dl": (-1, -1), "dr": (1, -1)}


def turn_map(turn: str) -> dict:
    """recorded disc name -> phantom position name for a head turned clockwise (seen from above)
    by 0/90/180/270 degrees. Accepts 'none', 'cw90', 'ccw90', '180' or a number of degrees."""
    deg = {"none": 0, "": 0, "cw90": 90, "ccw90": 270, "cw270": 270, "180": 180}.get(str(turn))
    if deg is None:
        deg = float(turn)
    if deg % 90:
        raise ValueError(f"head turn must be a multiple of 90 degrees, got {turn!r}")
    k = int(deg // 90) % 4
    out = {}
    for name, (x, y) in _POS.items():
        for _ in range(k):          # clockwise 90: (right, up) -> (up, -right)
            x, y = y, -x
        out[name] = next(n for n, p in _POS.items() if p == (x, y))
    return out


HEAD_TURNS = {t: turn_map(t) for t in ("none", "cw90", "180", "ccw90")}
assert HEAD_TURNS["cw90"] == {"dr": "dl", "dl": "ul", "ul": "ur", "ur": "dr"}  # bri, 2026-11-23


def load(data_dir, turn="none", bad_discs=()):
    """(trials, feats) with feats.channel renamed to phantom positions and bad discs dropped.
    bad_discs are recorded names (the physical disc), e.g. ('dl',)."""
    data_dir = Path(data_dir)
    trials = pd.read_csv(data_dir / "trials.csv")
    feats = pd.read_csv(data_dir / "features.csv")
    feats = feats[~feats.channel.isin(list(bad_discs))].copy()
    feats["recorded_channel"] = feats["channel"]
    feats["channel"] = feats["channel"].map(turn_map(turn))
    return trials, feats


def share_matrix(feats: pd.DataFrame, trial_ids, channels=None, freqs=None, f_min=50, f_max=400):
    """Rows = trials, columns s_<ch>_<f>: log10 amp minus the mean over `channels` at f.
    Returns (matrix, column names, channels, freqs)."""
    f = feats[(feats.freq_hz >= f_min) & (feats.freq_hz <= f_max)]
    channels = sorted(f.channel.unique()) if channels is None else list(channels)
    freqs = sorted(f.freq_hz.unique()) if freqs is None else [float(v) for v in freqs]
    la = np.log10(f.amp.clip(lower=1e-3))
    p = f.assign(la=la).pivot_table(index="trial_id", columns=["channel", "freq_hz"], values="la")
    missing = [t for t in trial_ids if t not in p.index]
    if missing:
        raise ValueError(f"{len(missing)} trials have no features, e.g. {missing[0]}")
    cube = np.stack([p.loc[list(trial_ids), (ch, slice(None))][[(ch, q) for q in freqs]].to_numpy()
                     for ch in channels], axis=1)               # trial x channel x freq
    cube = cube - cube.mean(axis=1, keepdims=True)
    names = [f"s_{ch}_{q:g}" for ch in channels for q in freqs]
    return cube.reshape(len(trial_ids), -1), names, channels, freqs


# ------------------------------------------------------------------ ridge


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return {"mu": mu, "sd": sd, "w": w, "b": float(y.mean()), "alpha": float(alpha)}


def ridge_predict(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"]


ALPHAS = (0.3, 1, 3, 10, 30, 100, 300, 1000)


def pick_alpha(X, y, groups, alphas=ALPHAS):
    """Ridge strength with the lowest grouped-CV mean absolute error."""
    best = None
    for a in alphas:
        pred = grouped_cv(X, y, groups, alpha=a)
        mae = float(np.mean(np.abs(pred - y)))
        if best is None or mae < best[1]:
            best = (a, mae)
    return best[0]


def grouped_cv(X, y, groups, alpha=None):
    """Out-of-group predictions. alpha=None picks it inside each training fold (nested CV)."""
    groups = np.asarray(groups)
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        tr = groups != g
        a = alpha if alpha is not None else pick_alpha(X[tr], y[tr], groups[tr])
        pred[~tr] = ridge_predict(ridge_fit(X[tr], y[tr], a), X[~tr])
    return pred


# ------------------------------------------------------------------ model file


def save_model(path, m, channels, freqs, meta):
    out = dict(meta, channels=list(channels), freqs=[float(v) for v in freqs],
               mu=m["mu"].tolist(), sd=m["sd"].tolist(), w=m["w"].tolist(), b=m["b"], alpha=m["alpha"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1))


def load_model(path):
    d = json.loads(Path(path).read_text())
    for k in ("mu", "sd", "w"):
        d[k] = np.asarray(d[k])
    return d
