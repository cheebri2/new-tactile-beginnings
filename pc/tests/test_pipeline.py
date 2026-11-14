"""Run with:  cd pc && python -m pytest tests"""

import numpy as np

from piezo import stream
from piezo.features import extract
from piezo.schedule import Sweep


def synthetic(sweep, amps, lags_deg, t0=1.234, fs=2000.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int((t0 + sweep.duration_s + 1.0) * fs)
    t = np.arange(n) / fs
    x = rng.normal(0, noise, (n, 4)) + np.array([1400, 1700, 1600, 1100])
    for k, f in enumerate(sweep.freqs):
        s = t0 + sweep.tone_starts_s[k]
        sel = (t >= s) & (t < s + sweep.tone_ms / 1000)
        x[sel] += amps * np.cos(2 * np.pi * f * (t[sel, None] - s) - np.radians(lags_deg))
    return (t * 1e6).astype(np.int64) + 123, np.round(x).astype(np.uint16), t0


def test_binary_round_trip():
    t = np.arange(1000, dtype=np.int64) * 500 + (2**32 - 100_000)  # crosses the micros() wrap
    adc = np.random.default_rng(1).integers(0, 4096, (1000, 4)).astype(np.uint16)
    buf = b"noise before" + stream.encode_blocks(t, adc) + b"\xA5\x5Agarbage"
    t2, adc2, st = stream.parse_blocks(buf)
    assert np.array_equal(adc2, adc)
    assert np.array_equal(t2 - t2[0], t - t[0])
    assert st["lost_blocks"] == 0 and st["blocks"] == 20


def test_dropped_block_is_counted():
    t = np.arange(500, dtype=np.int64) * 500
    adc = np.zeros((500, 4), np.uint16)
    buf = stream.encode_blocks(t, adc)
    block = len(buf) // 10
    _, adc2, st = stream.parse_blocks(buf[:3 * block] + buf[4 * block:])
    assert st["lost_blocks"] == 1 and len(adc2) == 450


def test_extract_recovers_amplitude_phase_and_timing():
    sweep = Sweep(f_step=40)
    amps = np.array([30.0, 20.0, 10.0, 5.0])
    lags = np.array([0.0, 30.0, -60.0, 120.0])
    t_us, adc, t0 = synthetic(sweep, amps, lags)
    rows, info = extract(t_us, adc, sweep)
    assert abs(info["sweep_start_s"] - t0) < 0.003
    assert info["align_quality"] > 5
    for r in rows:
        i = ["ur", "ul", "dl", "dr"].index(r["channel"])
        assert abs(r["amp"] / amps[i] - 1) < 0.05, r
        d = (r["phase_deg"] - lags[i] + 180) % 360 - 180
        assert abs(d) < 5, r
        assert r["snr_db"] > 15
