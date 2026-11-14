"""Fake captures for trying the pipeline without hardware (capture.py --simulate).

The model is made up: a smooth skin response per channel, an object that adds a
resonance bump weighted toward the channels it sits under, and a random per-trial
coupling gain on each disc to mimic reseating. It only exercises the code.
"""

import numpy as np

from .schedule import Sweep

SENSOR_XY = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * 15.0  # ur ul dl dr, mm
REST = np.array([1376, 1693, 1628, 1102])


def response(f, meta, rng, placement_sd=0.15):
    f = np.asarray(f, float)
    base = 40 * np.exp(-((f - 180) / 120) ** 2)[:, None] * np.array([1.0, 0.9, 0.8, 0.5])
    delay_ms = np.array([0.4, 0.5, 0.7, 0.6])
    h = base * np.exp(-2j * np.pi * f[:, None] * delay_ms / 1000)
    if meta.get("object", "none") not in ("none", "", None):
        xy = np.array([float(meta.get("x_mm") or 0), float(meta.get("y_mm") or 0)])
        depth = float(meta.get("depth_mm") or 15)
        w = np.exp(-np.sum((SENSOR_XY - xy) ** 2, axis=1) / (2 * 20.0 ** 2)) * 15 / depth
        bump = 25 * np.exp(-((f - 230) / 30) ** 2)
        h = h + bump[:, None] * w * np.exp(1j * 0.6)
    gain = np.exp(rng.normal(0, placement_sd, 4))
    return h * gain


def capture(meta, sweep: Sweep, fs=2000.0, pre_s=1.5, post_s=1.0, seed=None, noise=2.0):
    rng = np.random.default_rng(seed)
    n = int((pre_s + sweep.duration_s + post_s) * fs)
    t_s = np.arange(n) / fs + rng.normal(0, 5e-6, n)
    t0 = pre_s + rng.uniform(-0.3, 0.3)
    h = response(sweep.freqs, meta, rng)
    x = rng.normal(0, noise, (n, 4)) + 3 * np.sin(2 * np.pi * 1.7 * t_s)[:, None]
    offset_us = 5_000_000
    events = [{"event": "sweep_start", "t_us": int(t0 * 1e6) + offset_us, "freq_hz": 0.0}]
    for k, f in enumerate(sweep.freqs):
        start = t0 + sweep.tone_starts_s[k]
        events.append({"event": "tone_start", "t_us": int(round(start * 1e6)) + offset_us, "freq_hz": float(f)})
        start = round(start * 1e6) / 1e6
        sel = (t_s >= start) & (t_s < start + sweep.tone_ms / 1000)
        tt = t_s[sel] - start
        x[sel] += np.real(h[k] * np.exp(1j * 2 * np.pi * f * tt)[:, None])
    adc = np.clip(np.round(REST + x), 0, 4095).astype(np.uint16)
    t_us = np.round(t_s * 1e6).astype(np.int64) + offset_us
    events.append({"event": "sweep_end", "t_us": int(t_us[-1]) - int(post_s * 1e6 / 2), "freq_hz": 0.0})
    return t_us, adc, events
