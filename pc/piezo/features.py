"""Turn one raw capture into per-frequency amplitude and phase for each piezo.

Steps:
1. Remove each channel's resting level (differs per board).
2. Find where the sweep sits in the recording: the drive board has no wire to the sensing
   board, so the known step pattern (on high-passed data) (which frequency when) is matched against the data.
3. For each step, least-squares fit a sine at that exact frequency over the steady part of
   the tone (plus a slow drift term, so hand/pressure drift does not leak in). Amplitude is in ADC counts (peak). Phase is the lag relative to the reference
   channel, because the absolute drive phase is not available without a shared wire.
4. The same fit on the silent lead-in gives the noise floor at every frequency (SNR).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import CHANNELS, REF_CHANNEL
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


def extract(t_us, adc, sweep: Sweep, adc_read_us=0.0):
    """Features for one capture.

    Returns (rows, info): rows is a list of dicts, one per channel x frequency
    (channel, freq_hz, amp, phase_deg, re, im, noise_amp, snr_db, resid_rms);
    info has fs_hz, sweep_start_s, align_quality.
    """
    t_us = np.asarray(t_us, dtype=np.float64)
    t_s = (t_us - t_us[0]) / 1e6
    fs = 1.0 / np.median(np.diff(t_s))
    raw = np.asarray(adc, dtype=np.float64)
    raw = raw - np.median(raw, axis=0)
    t0, quality = find_sweep(t_s, highpass(raw, fs), sweep)
    a, b = sweep.windows_s
    na, nb = sweep.noise_window_s
    ref = CHANNELS.index(REF_CHANNEL)

    rows = []
    for k, f in enumerate(sweep.freqs):
        sel = (t_s >= t0 + a[k]) & (t_s < t0 + b[k])
        nsel = (t_s >= t0 + na) & (t_s < t0 + nb)
        tone_t0 = t0 + sweep.tone_starts_s[k]
        fits = []
        for ch in range(len(CHANNELS)):
            # channel ch was read ch * adc_read_us after the frame timestamp
            tc = t_s + ch * adc_read_us / 1e6
            z, resid = fit_sine(tc[sel] - tone_t0, raw[sel, ch], f)
            zn, _ = fit_sine(tc[nsel], raw[nsel, ch], f) if nsel.sum() > 20 else (np.nan, 0)
            fits.append((z, resid, abs(zn)))
        zref = fits[ref][0]
        rot = np.conj(zref) / abs(zref) if abs(zref) > 0 else 1.0
        for ch, (z, resid, noise) in enumerate(fits):
            zr = z * rot  # phase relative to the reference channel
            amp = abs(z)
            rows.append({
                "channel": CHANNELS[ch],
                "freq_hz": float(f),
                "amp": amp,
                "phase_deg": float(np.degrees(np.angle(zr))),
                "re": zr.real,
                "im": zr.imag,
                "noise_amp": noise,
                "snr_db": float(20 * np.log10(amp / noise)) if noise > 0 else np.nan,
                "resid_rms": resid,
                "n_samples": int(sel.sum()),
            })
    return rows, {"fs_hz": float(fs), "sweep_start_s": t0, "align_quality": quality}
