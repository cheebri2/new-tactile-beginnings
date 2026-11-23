# Healthy-phantom stamps, 2026-11-22: locator and Cobb feasibility

Data: `pc/final_data` on bri's laptop (not in the repo). 177 vertebra stamps on the healthy phantom,
x = -10, -5, 0, +5, +10 mm (head right of spine) at y = 0, 20, ... 120 mm, 5 reseats each, depth
15 mm, plus 60 blanks (one before every block). One session. Head turned 90° clockwise (its up side
to the phantom's right): recorded dr -> dl, dl -> ul, ul -> ur, ur -> dr. The `.json` files say
`rotation_deg 0`; the turn is applied with `--turn cw90` instead of editing the data.

Reproduce: `cd pc && python locator.py train --data final_data --turn cw90 --bad-discs dl` then
`python cobb.py feasibility training/results/locator/predictions.csv`.

## What the data showed

- **The remap.** bri's two descriptions (up side to the right, and the four-way rename) agree with
  each other: both are a clockwise quarter turn. The data can't confirm it on its own, because the
  model learns each disc separately and scores the same either way. The remap matters when this
  model is used on data taken square, or when results are named by phantom side.
- **Recorded dl (GPIO33, phantom up-left after the turn) is intermittent.** Its noise in silence
  jumps between ~0.2 and ~15 counts from trial to trial, 12 % of its points are under 10 dB SNR, and
  in some trials it reads 0. Left out. The other three are healthy (median SNR 43-53 dB).
- **Blanks don't help the model.** Single blanks moved by 25-100 % between reseats, more than the
  vertebra effect, and subtracting the preceding blank made the locator worse (MAE 5.4 vs 3.9 mm).
  Each disc's share of the total cancels pressure/seating and works better.
- **The y effect in the raw features is drift.** Many features change steadily with y, but y rows
  were captured in order through the afternoon and the blanks drift the same way (e.g. up-right
  amplitude at 344 Hz falls 0.28 decades per hour in blanks and objects alike). y can't be learned
  from this batch.
- **The IMU x/y didn't work in this batch.** `imu_x_mm` is uncorrelated with the typed x (r = 0.03),
  is off by a median 53 mm (y: 87 mm), and imu_x and imu_y track each other (r = -0.91). The yaw is fine
  (SD 1.8°). The `imu_axes` were also left at `x,y` with the head turned. Use typed positions.

## Locator (spine left/right of the head)

Ridge regression on share log amplitude of 3 discs, 50-190 Hz (39 features). Held out one y row at
a time (7 folds, ridge strength picked inside each fold).

| | |
|---|---|
| mean abs error per stamp | **3.7 mm** (guessing the mean: 5.9 mm) |
| RMS error | 4.5 mm |
| side correct when the spine is 5-10 mm off | 88 % |
| mean of 5 reseats at one spot | 3.2 mm |
| error shared by a whole row | SD 1.3 mm |
| stamp-to-stamp error inside a row | SD 4.4 mm |

The predictions are shrunk toward 0 (slope 0.63), typical for ridge with this much noise. The 50-190 Hz
band was chosen on this same data from four candidates (full band: 3.7 vs 4.1 mm), so expect a
little worse on new data. Up-down position relative to a process isn't modelled yet (see above).

![locator](locator_cv.png)

## Cobb angle: feasible, with limits

`cobb.py feasibility` bends a virtual spine to a known Cobb angle (single C-curve over the 7 rows,
120 mm), gives each row the real held-out errors from one row of this batch, fits the curve and
compares. This works because a stamp's error depends on the head-to-spine offset, not on where the
spine is, as long as each vertebra still sits square under the head.

| true Cobb | full raster (25 stamps/row) | 3 stamps/row over the spine |
|---|---|---|
| 0° | 5.0 ± 3.8°, 11 % read as ≥ 10° | 9.8 ± 7.2°, 42 % read as ≥ 10° |
| 10° | 10.3 ± 5.5° | 12.6 ± 9.3° |
| 20° | 19.9 ± 6.2°, 94 % read as ≥ 10° | 20.5 ± 10.9° |
| 30° | 29.8 ± 5.8° | 29.3 ± 11.5° |
| 45° | 44.8 ± 5.4° | 44.9 ± 10.4° |

The real healthy phantom, measured with all its stamps, reads **6.3°** (c-curve) and 7.1° (quadratic),
which matches the 0° row.

So today: a full raster tells a healthy spine from a 20-30°+ curve, with about ±6° per scan.
That's not enough for the 10° threshold or for the ±3-5° that X-ray Cobb readers achieve. The
precision comes from the per-row spine estimate: ±1.5 mm per row gives about ±4°, ±1 mm gives ±3°
(7 rows, 120 mm). The current raster gets about ±2 mm per row.

## Do you need to stamp scoliotic phantoms?

**Not to train.** The locator learns the head-to-spine offset, and a healthy phantom with the head
moved left/right gives exactly those offsets. Training on a scoliotic phantom only adds the same
offsets with less control.

**Yes, a little, to validate**, because healthy data can't test three things a real curve brings:
1. each vertebra is *tilted* under the head (up to ~10-20° in a 30-45° curve),
2. vertebrae are *rotated axially*, which moves the spinous process toward the concave side, so the
   process line under-reads the X-ray Cobb,
3. the lateral offset changes *within* one head footprint.

Cheapest order:

1. **Tilt test on the healthy phantom (about 1 hour, no new phantom).** Turn the phantom (or the
   head) 15° in the skin plane and stamp x = -10, 0, +10 at two y rows, 5 reseats each (30 stamps).
   If `locator.py predict` stays unbiased, item 1 is covered. If it's biased, add tilted stamps to
   training (still healthy phantom).
2. **One scoliotic phantom, validation only.** One curve near 30° with a known (designed) Cobb, and
   axial rotation in the print if you can. Stamp the same 7 rows as the healthy raster. The minimum
   is the 5 x-offsets × 3 reseats per row (105 stamps), because the raster average is what makes
   Cobb work. One pass gives one Cobb reading; a second pass on another day tells you the repeatability.
   No blanks needed for the model (one every 30-45 min is enough to watch disc health).

## Next for better accuracy (healthy phantom)

- Fix recorded dl (GPIO33) so all four discs count.
- Randomise the order of x and y (this batch ran row by row, so y is confounded with time).
- A second session on another day, so the CV can hold out a whole session.
- Fix the IMU position (or use a printed grid), so a raster doesn't need typed positions.
- More stamps per spot don't help much (5-stamp mean 3.2 vs 3.7 mm); more x offsets (±2.5, ±7.5, ±15)
  and more rows help the Cobb fit more.
