# Locator cross-validation (2026-11-23)

Data `pc/final_data`, head turn `cw90`, discs left out (recorded names): dl. Phantom discs used: dl, dr, ur.
176 stamps; each y_mm group held out in turn (nested ridge alpha; final alpha 10).

| | value |
|---|---|
| mean abs error per stamp | 3.70 mm |
| RMS error per stamp | 4.53 mm |
| guessing the mean x | 5.94 mm |
| mean of the 5 repeats per spot | 3.23 mm |
| side right/wrong for abs(x) >= 5 mm | 88 % correct |
| error shared by a whole row (SD of row bias) | 1.33 mm |
| error stamp-to-stamp inside a row (SD) | 4.42 mm |
| slope of prediction vs truth | 0.63 (1 = unbiased) |

Prediction by true x (mean, SD, n):

| true x | mean | SD | n |
|---|---|---|---|
| -10 | -6.0 | 3.0 | 35 |
| -5 | -3.3 | 2.4 | 34 |
| 0 | -1.5 | 5.0 | 37 |
| 5 | +2.3 | 3.4 | 35 |
| 10 | +6.9 | 3.8 | 35 |

Mean abs error by y row: y=0: 3.8, y=20: 4.4, y=40: 2.8, y=60: 3.5, y=80: 3.7, y=100: 4.6, y=120: 3.1
