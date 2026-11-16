"""Turn one raw capture into per-frequency amplitude and phase for each piezo.

Steps:
1. Remove each channel's resting level (differs per board).
2. Take each tone's start time from the board's events. The tone and the samples use the
   same micros() clock, so the drive phase is known at every sample.
3. For each step, least-squares fit a sine at that exact frequency over the steady part of
   the tone (plus a slow drift term, so pressure drift does not leak in). Amplitude is in ADC
   counts (peak). Phase is the lag behind the drive command (includes the amp and transducer
   delay, which is the same for every trial) and also the lag behind the up-right disc.
4. The same fit on the silent lead-in gives the noise floor at every frequency (SNR).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import CHANNELS, DISABLED_CHANNELS, REF_CHANNEL
from .schedule import Sweep

HIGHPASS_HZ = 10.0


def highpass(x, fs, cutoff=HIGHPASS_HZ):
    sos = butter(2, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x - np.median(x, axis=0), axis=0)


def find_sweep(t_s, x, sweep: Sweep, step_s=0.001):
    """Return (sweep start time in seconds, quality).

    Scores every candidate start by how much energy sits at each step's frequency inside
    that step's window (each step normalised so all frequencies count equally). quality is
    the best score over the median score; well above 3 means a clean lock.
    """
    # Full tone length, so the score peaks sharply at the true start.
    a = sweep.tone_starts_s
    b = a + sweep.tone_ms / 1000.0
    taus = np.arange(t_s[0], t_s[-1] - sweep.duration_s + 1e-9, step_s)
    if len(taus) == 0:
        raise ValueError(f"capture is {t_s[-1] - t_s[0]:.1f} s, shorter than the "
                         f"{sweep.duration_s:.1f} s sweep")
    total = np.zeros(len(taus))
    for k, f in enumerate(sweep.freqs):
        z = x * np.exp(-2j * np.pi * f * t_s)[:, None]
        c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(z, axis=0)])
        i0 = np.searchsorted(t_s, taus + a[k])
        i1 = np.searchsorted(t_s, taus + b[k])
        s = (np.abs(c[i1] - c[i0]) ** 2).sum(axis=1)
        total += s / max(s.max(), 1e-12)
    best = int(np.argmax(total))
    quality = total[best] / max(np.median(total), 1e-12)
    return float(taus[best]), float(quality)


def fit_sine(t, y, f):
    """Least-squares y = A cos(2 pi f t - phi) + slow quadratic drift.
    Returns complex A e^{i phi} and the rms residual."""
    tc = t - t.mean()
    w = 2 * np.pi * f * t
    m = np.column_stack([np.cos(w), np.sin(w), np.ones_like(t), tc, tc ** 2])
    coef, *_ = np.linalg.lstsq(m, y, rcond=None)
    resid = y - m @ coef
    return complex(coef[0], coef[1]), float(np.sqrt(np.mean(resid ** 2)))


def tone_times_from_events(events, sweep: Sweep, t_first_us):
    """(sweep start, [tone start per frequency]) in seconds from the board's events, or None."""
    if not events:
        return None
    starts = [e for e in events if e["event"] == "sweep_start"]
    tones = [e for e in events if e["event"] == "tone_start"]
    if not starts or len(tones) < len(sweep.freqs):
        return None
    t0 = starts[-1]["t_us"]
    tones = [e for e in tones if e["t_us"] >= t0][:len(sweep.freqs)]
    if len(tones) < len(sweep.freqs):
        return None
    got = np.array([e["freq_hz"] for e in tones])
    if np.max(np.abs(got - sweep.freqs)) > 0.01:
        raise ValueError(f"board played {got[:3]}... but the sweep settings say {sweep.freqs[:3]}...")
    return (t0 - t_first_us) / 1e6, np.array([(e["t_us"] - t_first_us) / 1e6 for e in tones])


def extract(t_us, adc, sweep: Sweep, adc_read_us=0.0, events=None):
    """Features for one capture.

    Tone timing comes from the board's events (same clock as the samples), which also makes
    the phase relative to the drive measurable. Without events (old two-board data) the
    sweep is found by its timing pattern and only the phase between discs is available.

    Returns (rows, info): rows is a list of dicts, one per channel x frequency (channel,
    freq_hz, amp, phase_deg [lag vs drive], phase_rel_deg [lag vs the up-right disc], re, im
    [complex response vs drive], noise_amp, snr_db, resid_rms); info has fs_hz,
    sweep_start_s, timing ("events" or "pattern") and align_quality (pattern only).
    """
    t_us = np.asarray(t_us, dtype=np.int64)
    t_s = (t_us - t_us[0]) / 1e6
    fs = 1.0 / np.median(np.diff(t_s))
    raw = np.asarray(adc, dtype=np.float64)
    raw = raw - np.median(raw, axis=0)

    timed = tone_times_from_events(events, sweep, int(t_us[0]))
    if timed is not None:
        t0, tone_starts = timed
        if tone_starts[-1] + sweep.tone_ms / 1000 > t_s[-1]:
            raise ValueError("capture ended before the last tone finished")
        timing, quality = "events", np.nan
    else:
        t0, quality = find_sweep(t_s, highpass(raw, fs), sweep)
        tone_starts = t0 + sweep.tone_starts_s
        timing = "pattern"
    settle, keep_until = sweep.settle_ms / 1000, (sweep.tone_ms - sweep.tail_ms) / 1000
    na, nb = sweep.noise_window_s
    nsel = (t_s >= t0 + na) & (t_s < t0 + nb)
    ref = CHANNELS.index(REF_CHANNEL)

    rows = []
    for k, f in enumerate(sweep.freqs):
        tone_t0 = tone_starts[k]
        sel = (t_s >= tone_t0 + settle) & (t_s < tone_t0 + keep_until)
        fits = []
        for ch in range(len(CHANNELS)):
            # channel ch was read ch * adc_read_us after the frame timestamp
            tc = t_s + ch * adc_read_us / 1e6
            z, resid = fit_sine(tc[sel] - tone_t0, raw[sel, ch], f)
            zn, _ = fit_sine(tc[nsel], raw[nsel, ch], f) if nsel.sum() > 20 else (np.nan, 0)
            fits.append((z, resid, abs(zn)))
        zref = fits[ref][0]
        for ch, (z, resid, noise) in enumerate(fits):
            if CHANNELS[ch] in DISABLED_CHANNELS:
                continue
            amp = abs(z)
            rel = np.angle(z * np.conj(zref)) if abs(zref) > 0 else np.nan
            rows.append({
                "channel": CHANNELS[ch],
                "freq_hz": float(f),
                "amp": amp,
                "phase_deg": float(np.degrees(np.angle(z))) if timing == "events" else np.nan,
                "phase_rel_deg": float(np.degrees(rel)),
                "re": z.real if timing == "events" else np.nan,
                "im": z.imag if timing == "events" else np.nan,
                "noise_amp": noise,
                "snr_db": float(20 * np.log10(amp / noise)) if noise > 0 else np.nan,
                "resid_rms": resid,
                "n_samples": int(sel.sum()),
            })
    return rows, {"fs_hz": float(fs), "sweep_start_s": float(t0), "timing": timing,
                  "align_quality": quality}
