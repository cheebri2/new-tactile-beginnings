"""Centred stamps only (x = 0 from the mold centreline): what each vertebra reads, severe vs healthy.
Run from pc/ with final_data present:  python ../analysis/severe_2026-11-23/centred_only.py"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from piezo import locate  # noqa: E402
from locator import stamps  # noqa: E402
from cobb import fit_c, fit_segments  # noqa: E402

m = locate.load_model("models/locator_phase_2026-11-22.json")
tr, fe = locate.load("final_data", m["turn"], m["bad_discs"])
F = locate.FEATURES[m["feature_set"]]
cal = m["cv"]["calibration"]
h, s = stamps(tr, ["healthy_vertebra"]), stamps(tr, ["severe_scoliosis"])
# healthy: held-out-row predictions (the model saw the other rows) so healthy isn't scored in-sample
Xh, *_ = F(fe, h.trial_id, m["channels"], m["freqs"], m["f_min"], m["f_max"])
raw = locate.grouped_cv(Xh, h.x_mm.to_numpy(float), h.y_mm.to_numpy())
h["spine"] = h.x_mm - (raw - cal["intercept"]) / cal["slope"]
Xs, *_ = F(fe, s.trial_id, m["channels"], m["freqs"], m["f_min"], m["f_max"])
s["spine"] = s.x_mm - (locate.ridge_predict(m, Xs) - cal["intercept"]) / cal["slope"]
xs, ys = locate.mold("severe"); o = np.argsort(ys); xs, ys = xs[o], ys[o]
xa, ya = locate.mold("aligned"); oa = np.argsort(ya); xa, ya = xa[oa], ya[oa]
srows, hrows = [0, 30, 60, 85, 125], [0, 20, 60, 80, 100]
S0 = s[s.x_mm == 0].pivot_table(index="y_mm", columns="repeat", values="spine").loc[srows].dropna(axis=1)
H0 = h[h.x_mm == 0].pivot_table(index="y_mm", columns="repeat", values="spine").loc[hrows].dropna(axis=1)
lv = ["T12", "T11", "T10", "T9", "T8"]
print("SEVERE, x = 0 stamps, spine position read (mm from centreline)")
print(pd.DataFrame(S0.values.round(1), index=[f"{l} (row {r})" for l, r in zip(lv, srows)],
                   columns=[f"stamp {c}" for c in S0.columns]).assign(median=S0.median(axis=1).round(1).values, design=xs))
print("\nHEALTHY, x = 0 stamps")
print(pd.DataFrame(H0.values.round(1), index=[f"{l} (row {r})" for l, r in zip(lv, hrows)],
                   columns=[f"stamp {c}" for c in H0.columns]).assign(median=H0.median(axis=1).round(1).values, design=xa))
X = xs
for name, D, Y in [("severe", S0, ys), ("healthy", H0, ya)]:
    def stats(v):
        _, bow, f = fit_c(Y, v)
        return (f"spread SD {v.std():4.1f} mm, range {np.ptp(v):4.1f} mm, C-bow {bow:+5.1f} mm "
                f"(left over {np.std(v - f(Y)):.1f}), segment Cobb {fit_segments(Y, v)[0]:3.0f} deg")
    print(f"\n{name}  median of passes: {stats(D.median(axis=1).to_numpy())}")
    for c in D.columns:
        print(f"   single pass {c}: {stats(D[c].to_numpy())}")
    print(f"   design:        {stats(X if name == 'severe' else xa)}")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(1, 2, figsize=(9.5, 5.2), sharex=True)
for a, D, X, Y, title in [(ax[0], H0, xa, ya, "Healthy (aligned) mold"), (ax[1], S0, xs, ys, "Severe mold")]:
    a.plot(X, Y, "o-", color="k", lw=2, label="design")
    for c in D.columns:
        a.plot(D[c].to_numpy(), Y, "s:", ms=5, alpha=0.6, label=f"pass {c} (x = 0 stamps)")
    a.plot(D.median(axis=1).to_numpy(), Y, "D-", color="#c0392b", lw=2, label="median of passes")
    a.axvline(0, color="0.6", lw=0.8, ls="--")
    a.set(title=title, xlabel="spine x read by the sensor (mm, + = right)", xlim=(-25, 25))
ax[0].set_ylabel("process height above mold bottom (mm)")
ax[0].legend(fontsize=7, loc="upper right")
ax[1].legend(fontsize=7.5, loc="lower right")
fig.suptitle("Centred stamps only (head at x = 0), phase-between-discs model")
fig.tight_layout()
fig.savefig("../analysis/severe_2026-11-23/centred_only.png", dpi=150)
