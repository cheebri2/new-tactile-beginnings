# Locator cross-validation (2026-11-23)

Data `pc/final_data`, head turn `cw90`, discs left out (recorded names): dl. Phantom discs used: dl, dr, ur.
176 stamps; each y_mm group held out in turn (nested ridge alpha; final alpha 10).

| | value |
|---|---|
| mean abs error per stamp | 4.52 mm (calibrated) |
| RMS error per stamp | 5.86 mm |
| guessing the mean x | 5.94 mm |
| mean of the 5 repeats per spot | 3.48 mm |
| side right/wrong for abs(x) >= 5 mm | 91 % correct |
| error shared by a whole row (SD of row bias) | 2.12 mm |
| error stamp-to-stamp inside a row (SD) | 5.47 mm |
| shrink before calibration (slope) | 0.63; predictions are divided back out |

Prediction by true x (mean, SD, n):

| true x | mean | SD | n |
|---|---|---|---|
| -10 | -9.0 | 4.7 | 35 |
| -5 | -4.8 | 3.9 | 34 |
| 0 | -1.8 | 8.0 | 37 |
| 5 | +4.2 | 5.4 | 35 |
| 10 | +11.5 | 6.1 | 35 |

Mean abs error by y row: y=0: 3.7, y=20: 5.2, y=40: 4.1, y=60: 3.2, y=80: 6.6, y=100: 5.2, y=120: 3.7

Mean abs error by how far the nearest process is above the sensor centre (dy): dy=-15: 5.2 (n=25), dy=-10: 4.1 (n=25), dy=+0: 3.5 (n=50), dy=+5: 5.1 (n=51), dy=+10: 5.2 (n=25)
