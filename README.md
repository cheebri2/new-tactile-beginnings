# Piezo sweep pipeline

Four piezo discs on silicone over a 3D-printed vertebra. A transducer plays a stepped sine
sweep (20–300 Hz); the laptop records every disc's raw signal and extracts amplitude and phase
at each frequency. Trials are tagged with test metadata so conditions (blank, object, offset,
depth, …) can be averaged and compared, and so the data can feed a model later.

```
firmware/esp32   one ESP32: samples the 4 piezos and plays the sine sweep
pc/              capture.py, process.py, analyze.py (+ piezo/ library, tests/)
```

## Hardware (one ESP32)

- Piezos: up_right GPIO34, up_left GPIO35, down_left GPIO32, down_right GPIO33.
- Drive: GPIO26 (DAC2) → 100 nF → PAM8403 left input (`ACT_INPUT` on system_board). GPIO25 is
  the battery sense divider on that board and is not driven.
- Board connectors → firmware channels: PZT1 (J1, GPIO32) = down_left, PZT2 (J2, GPIO33) =
  down_right, PZT3 (J3, GPIO34) = up_right, PZT4 (J4, GPIO35) = up_left.
- The tone runs on core 0 and the sampling on core 1, both on the same `micros()` clock. The
  board reports when each tone starts, so the laptop knows the drive phase at every sample.

## Setup

1. Flash the board: open `firmware/esp32` in PlatformIO and Upload.
2. `cd pc && pip install -r requirements.txt`

## Run a test

Every trial runs one full sweep (~21 s). Lift and reseat the sensor head between repeats; the
spread across reseats is what tells you whether a difference is real.

```
cd pc
python capture.py --port COM3 --test blank --depth 15 --load 200 --repeats 5
python capture.py --port COM3 --test object --object vertebra --depth 15 --load 200 --repeats 5
python capture.py --port COM3 --test offset --object vertebra --x 10 --depth 15 --load 200 --repeats 5
```

Metadata flags: `--test` (blank, object, offset, depth, rotation, load, noise, other), `--object`,
`--x`/`--y` (mm, + = right/up), `--depth` (skin mm), `--rotation` (deg), `--load` (g), `--session`
(defaults to today), `--notes`. Sweep flags: `--f-start --f-stop --f-step --tone-ms --amp`.
Try it without hardware: `python capture.py --simulate --test object --object vertebra --repeats 3`.

After each trial it prints the SNR and peak amplitude per disc, and asks whether to keep it.

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
| `phase.png` | phase lag behind the drive signal, mean ± SD |
| `repeatability.png` | between-condition / within-condition spread per frequency (below 1 = repeats dominate) |
| `pairs.png` | effect size d for each pair of conditions, disc × frequency |
| `trials_pca.png` | each trial on two principal components, with a leave-one-out accuracy |
| `report.md` | headline numbers; `summary.csv`, `separation.csv`, `pairs.csv` hold the full tables |

## Data (in `pc/data/`)

| file | contents |
|---|---|
| `raw/<trial>.csv.gz` + `.json` | raw samples (t_us, ur, ul, dl, dr) and all metadata. Source of truth |
| `trials.csv` | one row per trial: metadata, condition label, dropped data |
| `features.csv` | one row per trial × disc × frequency: amp, phase_deg, phase_rel_deg, re, im, noise_amp, snr_db |
| `dataset_wide.csv` | one row per trial, every feature a column, ready for ML |

Amplitude is the peak sine amplitude in ADC counts, fitted at the exact step frequency on the
steady part of each tone. `phase_deg` is the lag behind the drive command (it includes the amp
and transducer delay, which is the same every trial); `phase_rel_deg` is the lag behind the
up-right disc; `re`/`im` are the complex response vs the drive. Noise is the same fit on the silent lead-in. After changing
the feature code or fixing metadata in a `.json`, rebuild the tables with `python process.py`.

Tests: `cd pc && python -m pytest tests`.
