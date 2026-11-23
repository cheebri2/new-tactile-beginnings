# Piezo sweep pipeline

Four piezo discs on silicone over a 3D-printed vertebra. A transducer plays a stepped sine
sweep (50–400 Hz); the laptop records every disc's raw signal and extracts amplitude and phase
at each frequency. Trials are tagged with test metadata so conditions (blank, object, offset,
depth, …) can be averaged and compared, and so the data can feed a model later.

```
firmware/esp32   one ESP32: samples the 4 piezos, plays the sine sweep, tracks the head (IMU), logs to flash
pc/              capture.py, process.py, analyze.py, flash_dump.py (+ piezo/ library, tests/)
```

## Hardware (system_board: one ESP32)

- Piezos: up_right GPIO35, up_left GPIO32, down_left GPIO33, down_right GPIO34.
- Amp (PAM8403D): input ← GPIO26 (DAC2, ACT_INPUT, 100 nF coupling); SHDN ← GPIO4. The amp is
  switched on only while a sweep or tone plays (`amp auto`, default); `amp on` / `amp off` override.
  SHDN also has a 10k pull-up to VDC, so if the ESP32 is not driving GPIO4 the amp is on.
- IMU (BNO055, UART mode: PS1 high, PS0 low): COM0 → RX2 (GPIO16), COM1 ← TX2 (GPIO17), 115200 baud,
  read at 100 Hz. Runs in IMU fusion mode (accelerometer + gyro, no magnetometer) because the
  transducer and speaker magnets sit right next to it.
- Flash (W25Q32JVSS): CS GPIO5, CLK GPIO18, MISO GPIO19, MOSI GPIO23. Backup log of the IMU track.
- GPIO25 reads the battery (V_Sense, 100k/100k); INFO reports it as `vbat`.
- The tone runs on core 0 and the sampling on core 1, both on the same `micros()` clock. The
  board reports when each tone starts, so the laptop knows the drive phase at every sample.

## Setup

1. Flash the board: open `firmware/esp32` in PlatformIO and Upload.
2. `cd pc && pip install -r requirements.txt`

## Run a test

Every trial runs one full sweep (~19 s). Lift and reseat the sensor head between repeats; the
spread across reseats is what tells you whether a difference is real.

```
cd pc
python capture.py --port COM3 --test blank --depth 15 --repeats 5
python capture.py --port COM3 --test object --object vertebra --depth 15 --repeats 5
python capture.py --port COM3 --test offset --object vertebra --x 10 --depth 15 --repeats 5
```

Metadata flags: `--test` (blank, object, offset, depth, rotation, noise, other), `--object`,
`--x`/`--y` (mm, + = right/up), `--depth` (skin mm), `--rotation` (deg), `--session`
(defaults to today), `--notes`. Sweep flags: `--f-start --f-stop --f-step --tone-ms --amp`.
Try it without hardware: `python capture.py --simulate --test object --object vertebra --repeats 3`.

After each trial it prints the SNR and peak amplitude per disc, and asks whether to keep it.

## Phantom tests: IMU angle and x/y

For every test except `blank`, `capture.py` uses the BNO055 on the board:

1. At the start it asks you to put the sensor head on the **lower-left corner** of the phantom,
   square to its edges (board x axis pointing along the phantom's right), and hold it still. The
   board then sets that spot as (0, 0) and the current pose as zero angle. `--no-zero` keeps the
   corner from an earlier run (as long as the board has not been power-cycled).
2. Move the head to the next spot in **one quick slide** (about 0.3–3 s), then hold it still and
   press Enter. Before each sweep it prints where the IMU thinks the head is.
3. Each trial saves the head's pose during the sweep: `imu_x_mm`, `imu_y_mm`, `imu_yaw_deg`,
   `imu_pitch_deg`, `imu_roll_deg` and checks (`imu_move_during_sweep_mm`, `imu_turn_during_sweep_deg`,
   calibration levels) in `trials.csv`, and the full 100 Hz track in `raw/<trial>_imu.csv.gz`.
   The `--x`/`--y` you type are still the truth labels; the IMU estimate sits next to them.

```
python capture.py --port COM3 --test object --object vertebra --depth 15 --repeats 5      # zeroes first
python capture.py --port COM3 --test object --object vertebra --depth 15 --no-zero        # same corner
python capture.py --port COM3 --test blank --depth 15 --repeats 5                          # no IMU
python capture.py ... --imu-axes "-y,x"     # if the board sits rotated 90 deg on the head
```

**How good is x/y?** An accelerometer alone cannot give position: double-integrating a typical
0.05 m/s² bias gives 2.5 cm of error after 1 s and 2.5 m after 10 s. The board only integrates
while the head is sliding, forces the velocity to zero whenever the head is still or a sweep is
playing, removes the leftover velocity of each slide as a linear drift, and re-learns the bias
while still (`firmware/esp32/src/tracker.h`; same algorithm in `pc/piezo/imu.py`). On simulated
moves with BNO055-like noise and bias (`tests/test_imu.py`) the error after each slide is about
1 mm for 0.5–1 s slides, up to ~13 mm for 2 s slides and up to ~70 mm for slow 4 s creeps. The real
sensor adds fusion lag and tilt errors, so plan on **centimetre-level** accuracy, check it once by
sliding a known distance (e.g. 50 mm along a ruler), and don't use it for millimetre offsets. The
angles are much better (about a degree; yaw drifts slowly in IMU mode, so re-zero every session).

Flash log: the board also writes the track (25 Hz), zeroes, sweep starts/ends and trial marks to
its flash. `python flash_dump.py --port COM3` reads it to CSV; `--erase` wipes it before a session.

If `imu zero` fails, the error says why: `no IMU data arriving` (wiring/UART), or `not still`
with the measured `acc_dev`/`gyro_dev` against their limits (0.25 m/s², 3 deg/s). The board
learns the sensor's resting offsets once the readings stop changing, so an uncalibrated
BNO055 that reads, say, 0.4 m/s² at rest still zeroes after about a second. `imu` prints the
same numbers live.

Board commands for checking by hand (serial monitor at 921600): `i`, `imu`, `imu zero`,
`amp on|off|auto`, `flash info`, `test` (150 Hz tone).

## Live alignment test (blank calibration + centred template)

```
cd pc
python align_test.py --port COM3                               # baseline, then one test per placement
python align_test.py --port COM3 --plan -10 -5 0 5 10 --reps 2 # shuffled placements, tells you where to put it
python align_test.py --replay data                             # re-run the method on the calibration runs
python build_template.py --out models/centred_template.json    # rebuild from training/data
```

1. **Baseline**: 3 sweeps on blank silicone (`--baseline-sweeps`), same pressure as the test. `b` redoes it.
2. **Test**: put the head over the vertebra, type the true x (mm), Enter. It prints ALIGNED / NOT ALIGNED,
   the distance from the centred template vs the bound (`--bound`, default 1.4), and the closest trained
   offset as a rough x.
3. Every sweep is saved to `training/data/` like `capture.py` (with the prediction in the trial's JSON),
   plus one line per test in `training/data/predictions.csv`, so each run is new training data.

Method (`piezo/align.py`): change from the blank in log10 amplitude and phase for up-left and down-left
(down-right is disabled, up-right is insensitive to x), 50-400 Hz. Distance = RMS z-score vs the mean
centred change, each feature scaled by repeat spread plus blank-to-blank drift. The shipped template,
`models/centred_template_2026-11-15.json`, comes from one session (25 vertebra trials at x = -10..+10 mm,
depth 15 mm). Replaying that session gives 23/25 correct, but that is in-sample; +5 mm sits on the bound.

## Locate the spine and estimate a Cobb angle (stamps)

A "stamp" is one sweep with the head pressed on the phantom at a typed x/y (`capture.py --test object
--object healthy_vertebra --x .. --y ..`). The locator learns `x_mm` (how far the head is right of the
spine) from the discs' amplitude shares; the spine under the head is then at head x minus the prediction.

```
cd pc
python locator.py train --data final_data --turn cw90 --bad-discs dl   # fit + held-out score -> models/locator.json
python locator.py predict --data <new stamps>                          # spine_x_mm per stamp
python cobb.py estimate training/results/locator_predict/predictions.csv
python cobb.py feasibility training/results/locator/predictions.csv     # Cobb error with the real locator errors
```

- `--turn cw90`: the head was turned so its up side points to the phantom's right (2026-11-22 batch):
  recorded dr -> dl, dl -> ul, ul -> ur, ur -> dr. Discs are renamed to phantom positions before
  training. Use `none` for square data.
- `--bad-discs`: recorded names of discs to leave out. `dl` (GPIO33) was intermittent on 2026-11-22.
- Features: log amplitude of each disc minus the mean of the discs, 50-190 Hz. No blank needed.
- Validation holds out one whole y row at a time (`--group`), never single stamps.
- `--vertebrae 0 30 60 85 125` (default): stamp y at which each of the phantom's 5 processes sits at the
  sensor centre. Adds `level` and `dy_mm` to predictions; `cobb.py` places each vertebra at its own y.
- Positions come from the typed x/y. The IMU x/y did not track the typed positions on 2026-11-22, so
  don't use `--position imu` until that is fixed.

Results and what data to take next: `analysis/stamps_2026-11-22/README.md`.

## Compare conditions

```
python analyze.py                                     # all trials
python analyze.py --session 2026-11-14 --norm share   # one session, level-independent
python analyze.py --where "depth_mm == 15" --conditions "blank d=15" "object vertebra x=0 y=0 d=15"
```

`--norm`: `none` (ADC counts), `share` (each disc's fraction of the four, cancels overall
coupling/preload), `blank` (divided by the same session's blank). Output in `training/results/…/`
(or `results/…/` for `--data data`):

| file | shows |
|---|---|
| `amplitude.png` | mean ± SD per condition for each disc, every repeat as a thin line |
| `phase.png` | phase lag behind the drive signal, mean ± SD |
| `repeatability.png` | between-condition / within-condition spread per frequency (below 1 = repeats dominate) |
| `pairs.png` | effect size d for each pair of conditions, disc × frequency |
| `trials_pca.png` | each trial on two principal components, with a leave-one-out accuracy |
| `report.md` | headline numbers; `summary.csv`, `separation.csv`, `pairs.csv` hold the full tables |

## Data (in `pc/training/data/`)

New captures go to `pc/training/data/`. The calibration runs stay in `pc/data/`; reach them with
`--data data` on `capture.py`, `process.py` or `analyze.py`.

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
