# Piezo sweep pipeline

Four piezo discs on silicone over a 3D-printed vertebra. A transducer plays a stepped sine
sweep (20–300 Hz); the laptop records every disc's raw signal and extracts amplitude and phase
at each frequency. Trials are tagged with test metadata so conditions (blank, object, offset,
depth, …) can be averaged and compared, and so the data can feed a model later.

```
firmware/sense   sensing ESP32: streams raw ADC samples (GPIO 34/35/32/33 = ur/ul/dl/dr)
firmware/drive   drive ESP32: sine sweep on DAC GPIO25 -> amp -> transducer
pc/              capture.py, process.py, analyze.py (+ piezo/ library, tests/)
```

## Hardware

- **Sensing ESP32**: piezo channels unchanged: up_right 34, up_left 35, down_left 32, down_right 33.
- **Drive ESP32**: GPIO25 (DAC) → amp input, GND → amp GND. The DAC sine sits on ~1.65 V, so the amp
  input must be AC coupled (add a 1–10 µF series cap if it is not). No wire between the two ESP32s:
  the laptop finds the sweep in the piezo data by its timing pattern.
- Both boards on laptop USB is the normal setup. Without the drive board on USB, leave out
  `--drive-port` and press the drive board's BOOT button when the script asks.

## Setup

1. Flash each board: open `firmware/sense` (then `firmware/drive`) in PlatformIO and Upload.
2. `cd pc && pip install -r requirements.txt`

## Run a test

Every trial runs one full sweep (~21 s). Lift and reseat the sensor head between repeats; the
spread across reseats is what tells you whether a difference is real.

```
cd pc
python capture.py --port COM3 --drive-port COM4 --test blank --depth 15 --load 200 --repeats 5
python capture.py --port COM3 --drive-port COM4 --test object --object vertebra --depth 15 --load 200 --repeats 5
python capture.py --port COM3 --drive-port COM4 --test offset --object vertebra --x 10 --depth 15 --load 200 --repeats 5
```

Metadata flags: `--test` (blank, object, offset, depth, rotation, load, noise, other), `--object`,
`--x`/`--y` (mm, + = right/up), `--depth` (skin mm), `--rotation` (deg), `--load` (g), `--session`
(defaults to today), `--notes`. Sweep flags: `--f-start --f-stop --f-step --tone-ms --amp`.
Try it without hardware: `python capture.py --simulate --test object --object vertebra --repeats 3`.

After each trial it prints the sweep lock quality and SNR per disc, and asks whether to keep it.

## Compare conditions

```
python analyze.py                                     # all trials
python analyze.py --session 2026-11-14 --norm share   # one session, level-independent
python analyze.py --where "depth_mm == 15" --conditions "blank d=15 load=200" "object vertebra x=0 y=0 d=15 load=200"
```

`--norm`: `none` (ADC counts), `share` (each disc's fraction of the four, cancels overall
coupling/preload), `blank` (divided by the same session's blank). Output in `results/…/`:

| file | shows |
|---|---|
| `amplitude.png` | mean ± SD per condition for each disc, every repeat as a thin line |
| `phase.png` | phase vs the up-right disc, mean ± SD |
| `repeatability.png` | between-condition / within-condition spread per frequency (below 1 = repeats dominate) |
| `pairs.png` | effect size d for each pair of conditions, disc × frequency |
| `trials_pca.png` | each trial on two principal components, with a leave-one-out accuracy |
| `report.md` | headline numbers; `summary.csv`, `separation.csv`, `pairs.csv` hold the full tables |

## Data (in `pc/data/`)

| file | contents |
|---|---|
| `raw/<trial>.csv.gz` + `.json` | raw samples (t_us, ur, ul, dl, dr) and all metadata. Source of truth |
| `trials.csv` | one row per trial: metadata, condition label, lock quality, dropped data |
| `features.csv` | one row per trial × disc × frequency: amp, phase_deg, re, im, noise_amp, snr_db |
| `dataset_wide.csv` | one row per trial, every feature a column, ready for ML |

Amplitude is the peak sine amplitude in ADC counts, fitted at the exact step frequency on the
steady part of each tone. Phase is relative to the up-right disc (there is no shared wire, so
the absolute drive phase is unknown). Noise is the same fit on the silent lead-in. After changing
the feature code or fixing metadata in a `.json`, rebuild the tables with `python process.py`.

Tests: `cd pc && python -m pytest tests`.
