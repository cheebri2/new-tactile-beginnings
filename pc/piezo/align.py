"""Blank-calibrated alignment check: is the vertebra centred under the sensor head?

The method (bri's plan, 2026-11-16):
1. Calibrate: record a blank (no vertebra) baseline in the current seating/pressure.
2. Express each new sweep as the change from that baseline: log10 amplitude change and
   phase change, per disc and frequency.
3. Compare that change with the change a centred vertebra produced in training data
   (the template). The distance is an RMS of z-scores, where each feature is scaled by how
   much it varies between repeats AND between blank baselines (so normal drift counts as
   noise, not as misalignment).
4. Distance <= bound -> "aligned". The template also stores the mean change at each trained
   offset, so the closest offset gives a rough x estimate.

Only the up-left and down-left discs are used: down-right was faulty in the 2026-11-15 data
(biased to ~550 counts, copying down-left) and up-right barely changes with x.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CHANNELS = ("ul", "dl")
DEFAULT_BOUND = 1.4


def _circ_mean_deg(a, axis=0):
    return np.degrees(np.angle(np.nanmean(np.exp(1j * np.radians(a)), axis=axis)))


def _wrap(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def response(feats: pd.DataFrame, channels, freqs):
    """(log10 amp [ch, f], phase deg [ch, f]) from one trial's feature rows."""
    f = feats.set_index(["channel", "freq_hz"])
    idx = pd.MultiIndex.from_product([list(channels), [float(v) for v in freqs]])
    missing = idx.difference(f.index)
    if len(missing):
        raise ValueError(f"trial is missing {len(missing)} channel/frequency points, e.g. {missing[0]}")
    amp = f.loc[idx, "amp"].to_numpy().reshape(len(channels), len(freqs))
    ph = f.loc[idx, "phase_deg"].to_numpy().reshape(len(channels), len(freqs))
    if np.isnan(ph).any():
        raise ValueError("no phase vs drive in this trial (needs the one-board firmware with tone events)")
    return np.log10(np.maximum(amp, 1e-6)), ph


class Baseline:
    """Average of one or more blank sweeps taken in the current seating."""

    def __init__(self, channels, freqs):
        self.channels, self.freqs = list(channels), [float(v) for v in freqs]
        self.la, self.ph, self.trial_ids = [], [], []

    def add(self, feats: pd.DataFrame, trial_id=""):
        la, ph = response(feats, self.channels, self.freqs)
        self.la.append(la)
        self.ph.append(ph)
        self.trial_ids.append(trial_id)

    def __len__(self):
        return len(self.la)

    @property
    def mean(self):
        return np.mean(self.la, axis=0), _circ_mean_deg(np.array(self.ph), axis=0)

    def spread(self):
        """Median SD of log amp across the baseline sweeps (0 if only one)."""
        return float(np.median(np.std(self.la, axis=0, ddof=1))) if len(self) > 1 else 0.0

    def change(self, feats: pd.DataFrame):
        """Change from this baseline: (d log10 amp, d phase deg), each [ch, f]."""
        la, ph = response(feats, self.channels, self.freqs)
        bla, bph = self.mean
        return la - bla, _wrap(ph - bph)


def build_template(trials: pd.DataFrame, feats: pd.DataFrame, channels=DEFAULT_CHANNELS,
                   bound=DEFAULT_BOUND, source=""):
    """Learn the centred template from trials recorded as blank blocks each followed by an
    object block (x = offset). Each object trial is compared with the blank block just
    before it."""
    t = trials.sort_values("time").reset_index(drop=True).copy()
    t["is_blank"] = t["test_type"].astype(str).eq("blank")
    t["block"] = (t["is_blank"] != t["is_blank"].shift()).cumsum()
    first = feats[feats.trial_id == t.trial_id.iloc[0]]
    freqs = sorted(first.freq_hz.unique())
    chans = list(channels)

    per_trial = {tid: g for tid, g in feats.groupby("trial_id")}
    dla, dph, xs, blank_la, blank_ph = [], [], [], [], []
    base = None
    for _, r in t.iterrows():
        g = per_trial.get(r.trial_id)
        if g is None:
            continue
        if r.is_blank:
            if base is None or base_block != r.block:
                if base is not None and len(base):
                    m = base.mean
                    blank_la.append(m[0])
                    blank_ph.append(m[1])
                base, base_block = Baseline(chans, freqs), r.block
            base.add(g, r.trial_id)
            continue
        if base is None:
            continue
        a, p = base.change(g)
        dla.append(a)
        dph.append(p)
        xs.append(float(r.x_mm))
    if base is not None and len(base):
        m = base.mean
        blank_la.append(m[0])
        blank_ph.append(m[1])
    if not xs:
        raise ValueError("no object trials with a blank block before them")
    dla, dph, xs = np.array(dla), np.array(dph), np.array(xs)
    positions = sorted(set(xs))
    if 0.0 not in positions:
        raise ValueError("no centred (x = 0) trials to build the template from")

    # repeat spread inside each position (pooled), and drift between blank blocks
    def pooled_sd(v, circular=False):
        num, den = 0.0, 0
        for x in positions:
            s = v[xs == x]
            c = _wrap(s - _circ_mean_deg(s)) if circular else s - s.mean(axis=0)
            num = num + (c ** 2).sum(axis=0)
            den += len(s) - 1
        return np.sqrt(num / max(den, 1))

    w_la, w_ph = pooled_sd(dla), pooled_sd(dph, circular=True)
    bl, bp = np.array(blank_la), np.array(blank_ph)
    d_la = bl.std(axis=0, ddof=1) if len(bl) > 1 else np.zeros_like(w_la)
    d_ph = (np.sqrt((_wrap(bp - _circ_mean_deg(bp)) ** 2).sum(axis=0) / (len(bp) - 1))
            if len(bp) > 1 else np.zeros_like(w_ph))
    floor_la, floor_ph = 0.01, 1.0  # never let one feature dominate on a lucky tiny SD
    s_la = np.maximum(np.hypot(w_la, d_la), floor_la)
    s_ph = np.maximum(np.hypot(w_ph, d_ph), floor_ph)

    means = {}
    for x in positions:
        means[f"{x:g}"] = {"dla": dla[xs == x].mean(axis=0).tolist(),
                           "dph": _circ_mean_deg(dph[xs == x]).tolist(),
                           "n": int((xs == x).sum())}
    tpl = {"version": 1, "source": source, "channels": chans, "freqs": freqs,
           "bound": bound, "scale_dla": s_la.tolist(), "scale_dph": s_ph.tolist(),
           "positions": means, "n_blank_blocks": int(len(bl))}
    # how the training trials score against their own template (in-sample, optimistic)
    tpl["training_distance"] = {
        f"{x:g}": [round(distance(a, p, tpl, f"{0.0:g}"), 3) for a, p in zip(dla[xs == x], dph[xs == x])]
        for x in positions}
    return tpl


def distance(dla, dph, tpl, position="0"):
    """RMS z-score distance between one change and the template mean at `position`."""
    m = tpl["positions"][position]
    za = (np.asarray(dla) - np.array(m["dla"])) / np.array(tpl["scale_dla"])
    zp = _wrap(np.asarray(dph) - np.array(m["dph"])) / np.array(tpl["scale_dph"])
    return float(np.sqrt(np.mean(np.concatenate([za.ravel(), zp.ravel()]) ** 2)))


def predict(dla, dph, tpl, bound=None):
    """Classify one change. Returns a dict with distance, aligned, nearest_x, est_x and
    the distance to every trained offset."""
    bound = tpl["bound"] if bound is None else bound
    d0 = distance(dla, dph, tpl, "0")
    dists = {float(k): distance(dla, dph, tpl, k) for k in tpl["positions"]}
    order = sorted(dists, key=dists.get)
    # rough x: weight each trained offset by how well it matches (softmin over distance^2)
    xs = np.array(order)
    d = np.array([dists[x] for x in order])
    w = np.exp(-(d ** 2 - d.min() ** 2) * 2.0)
    est = float((w * xs).sum() / w.sum())
    if d[-1] - d[0] < 0.3:
        side = "unclear (matches every trained offset about equally)"
    else:
        side = "centred" if abs(est) < 2.5 else ("right (+x)" if est > 0 else "left (-x)")
    return {"distance": d0, "bound": bound, "aligned": bool(d0 <= bound), "nearest_x": order[0],
            "est_x": est, "side": side, "dists": dists}


def save_template(tpl, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(tpl, indent=1))


def load_template(path):
    return json.loads(Path(path).read_text())
