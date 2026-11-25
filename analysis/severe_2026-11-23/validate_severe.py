"""Validate the healthy-trained locator and the Cobb estimate on the severe scoliosis phantom.

Run from pc/:  python ../analysis/severe_2026-11-23/validate_severe.py --data final_data \
                   --model models/locator_2026-11-22.json

Vertebra positions come from bri's mold map (piezo/locate.py MOLDS["severe"]). Typed x is the head's
offset from the mold centreline (bri, 2026-11-24) for rows in --centreline-rows (default all), else from that row's
vertebra centre. Head position from the centreline = x (centreline rows) or vertebra x + x (others); the
measured spine is head - predicted offset, median per vertebra, placed at the map's process height.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from cobb import fit_c, fit_poly, fit_segments  # noqa: E402
from piezo import locate  # noqa: E402

LEVEL_Y = [0, 30, 60, 85, 125]          # stamp rows, bottom to top

p = argparse.ArgumentParser()
p.add_argument("--data", default="final_data")
p.add_argument("--model", default="models/locator_2026-11-22.json")
p.add_argument("--object", default="severe_scoliosis")
p.add_argument("--out", default="../analysis/severe_2026-11-23")
p.add_argument("--mold", default="severe")
p.add_argument("--centreline-rows", type=float, nargs="*", default=[0, 30, 60, 85, 125],
               help="stamp rows whose typed x is from the mold centreline")
a = p.parse_args()

m = locate.load_model(a.model)
trials, feats = locate.load(a.data, m["turn"], m["bad_discs"])
st = trials[trials.object == a.object].reset_index(drop=True)
X, *_ = locate.FEATURES[m.get("feature_set", "share-amp")](feats, st.trial_id, m["channels"], m["freqs"],
                                                          m["f_min"], m["f_max"])
cal = m["cv"]["calibration"]
st["pred_x_mm"] = (locate.ridge_predict(m, X) - cal["intercept"]) / cal["slope"]
xs, ys = locate.mold(a.mold)
o = np.argsort(ys)
SHIFT, Y = xs[o], ys[o]                  # bottom to top, matching LEVEL_Y
shift = dict(zip(LEVEL_Y, SHIFT))
st["true_spine_mm"] = st.y_mm.map(shift)
cent = st.y_mm.isin(a.centreline_rows)
st["head_x_mm"] = np.where(cent, st.x_mm, st.true_spine_mm + st.x_mm)
st["x_label_mm"] = st.head_x_mm - st.true_spine_mm     # true head offset from the vertebra
st["err_mm"] = st.pred_x_mm - st.x_label_mm
st["spine_est_mm"] = st.head_x_mm - st.pred_x_mm
st.to_csv(Path(a.out) / "severe_predictions.csv", index=False)

est = st.groupby("y_mm").spine_est_mm.median().reindex(LEVEL_Y).to_numpy()
rng = np.random.default_rng(0)
groups = {y: st[st.y_mm == y].spine_est_mm.to_numpy() for y in LEVEL_Y}
boot = np.array([[f(Y, np.array([np.median(rng.choice(groups[y], len(groups[y]))) for y in LEVEL_Y]))[0]
                  for f in (fit_segments, fit_c)] for _ in range(2000)])

print(f"{len(st)} stamps. Locator error: MAE {st.err_mm.abs().mean():.2f} mm, by vertebra (bottom->top): "
      + ", ".join(f"{st[st.y_mm == y].err_mm.abs().mean():.1f}" for y in LEVEL_Y))
print("vertebra   true   measured (median of stamps)")
for y, t, e in zip(LEVEL_Y, SHIFT, est):
    print(f"  y={y:>3}   {t:+5.1f}   {e:+5.1f}")
for name, f in [("segment tilts", fit_segments), ("c-curve", fit_c)]:
    print(f"Cobb {name:13s}: design {f(Y, SHIFT)[0]:5.1f}  measured {f(Y, est)[0]:5.1f}")
print(f"bootstrap over stamps: segment {boot[:, 0].mean():.1f} ± {boot[:, 0].std():.1f}, "
      f"c-curve {boot[:, 1].mean():.1f} ± {boot[:, 1].std():.1f}")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(SHIFT, Y, "o-", color="k", label=f"design ({fit_segments(Y, SHIFT)[0]:.0f}°)")
ax[0].plot(est, Y, "s--", color="#c0392b", label=f"measured ({fit_segments(Y, est)[0]:.0f}°)")
ax[0].scatter(st.spine_est_mm, st.y_mm.map(dict(zip(LEVEL_Y, Y))) + rng.uniform(-2, 2, len(st)), s=8, alpha=0.35,
              color="#c0392b", label="one stamp")
ax[0].set(xlabel="spine x from mold centreline (mm, + = right)", ylabel="height of process (mm)",
          title="Severe phantom: vertebra positions", xlim=(-40, 25))
ax[0].legend(fontsize=8)
jit = rng.uniform(-0.8, 0.8, len(st))
for y, c in zip(LEVEL_Y, plt.cm.viridis(np.linspace(0, 0.9, 5))):
    d = st.y_mm == y
    ax[1].scatter(st.x_label_mm[d] + jit[d], st.pred_x_mm[d], s=12, color=c, label=f"y={y}")
ax[1].plot([-14, 14], [-14, 14], "k--", lw=1)
ax[1].set(xlabel="true head offset from that vertebra (mm)", ylabel="predicted (mm)",
          title=f"Locator on severe (trained on healthy): MAE {st.err_mm.abs().mean():.1f} mm")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(Path(a.out) / "severe_validation.png", dpi=150)
