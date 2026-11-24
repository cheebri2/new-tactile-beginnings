# Severe scoliosis phantom, 2026-11-23: validation

Data: `pc/final_data` (object `severe_scoliosis`, session 2026-11-23), 75 stamps: 5 vertebrae
(stamp y = 0, 30, 60, 85, 125 mm) × head at x = -10, -5, 0, +5, +10 mm × 3 reseats. Head turned cw90
as on 11-22. Design shifts of the vertebrae from the mold centreline (bri), bottom to top:
**-14, 0, +4, -3, -17 mm**. Typed x is taken as the head's offset from *that vertebra's* centre (the
data fits this, not "offset from the mold centreline": that reading gives a 14° spine).

Model: `pc/models/locator_2026-11-22.json`, trained on the healthy phantom only (never saw these stamps).
Reproduce from `pc/`: `python ../analysis/severe_2026-11-23/validate_severe.py`.

## Locator

| vertebra (y) | 0 | 30 | 60 | 85 | 125 | all |
|---|---|---|---|---|---|---|
| mean abs error (mm) | 8.5 | 6.1 | 3.4 | 3.1 | 3.1 | **4.9** |

Healthy cross-validation was 4.5 mm, so it carries over to the scoliotic phantom, except the bottom
vertebra: every stamp there reads 4-16 mm right, whatever the true x (slope 0.2). It's the mold's
edge vertebra (and the most shifted, -14 mm).

## Cobb angle

| | design shifts | measured (median of 15 stamps per vertebra) |
|---|---|---|
| spine x, bottom to top (mm) | -14, 0, +4, -3, -17 | -20.4, +3.2, +4.9, +0.5, -15.1 |
| **Cobb, segment tilts** | **44.3°** | **59.5° (± 3.5° bootstrap)** |
| Cobb, c-curve fit | 49.8° | 61.0° (± 3.3°) |

The segment method reproduces the design 45° from the design shifts, so it is the one to report.
Measured over-reads by 15°. Most of it is the bottom vertebra (6.4 mm too far left makes the bottom
segment tilt 38° instead of 25°). With the bottom vertebra right, the measured angle would be 51°.
The other four vertebrae are within 1-3.5 mm.

Equations (`pc/cobb.py`): spine position at vertebra i, X_i = h_i - d_i (head x from the centreline
minus the predicted head-right-of-spine offset, median of stamps); process height Y_i = stamp y + 47 mm.
Segment tilt θ_k = atan((X_k+1 - X_k) / (Y_k+1 - Y_k)); **Cobb = max θ - min θ**. The c-curve fit
X(y) = c + A sin(π (y - Y_1) / L) gives Cobb = 2 atan(|A| π / L).

## Caveats

- One scan; this is one Cobb reading, not a repeatability.
- The head positions h_i here come from the design shifts plus typed x. In real use h_i must be known
  in a fixed frame (head kept on the mold centreline, or a working IMU/grid). If the head is centred
  by eye on each vertebra, the sensor reads ~0 and the curve is in where the hand went.
- Using only the x = 0 stamps (3 per vertebra) gives 59.8°; one x = 0 stamp per vertebra gives 60 ± 6°.
