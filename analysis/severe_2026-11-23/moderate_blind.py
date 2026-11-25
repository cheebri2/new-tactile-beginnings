"""Blind test on the moderate mold: train the phase-between-discs model on the healthy (11-22) and
severe (11-23) molds, then read the moderate mold's spine from its centred stamps (x = 0) only.
Run from pc/ with final_data present: python ../analysis/severe_2026-11-23/moderate_blind.py"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from piezo import locate  # noqa: E402
from locator import stamps  # noqa: E402
from cobb import fit_c, fit_segments  # noqa: E402

BAD = sys.argv[1].split(",") if len(sys.argv) > 1 else ["dl"]
tr, fe = locate.load("final_data", "cw90", BAD)
print("left out (recorded names):", BAD)
h = stamps(tr, ["healthy_vertebra"])
s = stamps(tr, ["severe_scoliosis"]); s = s[s.session == "2026-11-23"]
mo = stamps(tr, ["mod_scoliosis", "moderate_scoliosis"])
new = stamps(tr, ["healthy", "severe_scoliosis"]); new = new[new.session == "2026-11-25"]

Xh, _, ch, fr = locate.relphase_matrix(fe, h.trial_id, f_max=200)
F = lambda d: locate.relphase_matrix(fe, d.trial_id, ch, fr, f_max=200)[0]  # noqa: E731
Xs, Xm, Xn = F(s), F(mo), F(new)
xs, ys = locate.mold("severe"); o = np.argsort(ys)
S = dict(zip([0, 30, 60, 85, 125], xs[o]))
yh = h.x_mm.to_numpy(float); ysv = s.x_mm.to_numpy(float) - s.y_mm.map(S).to_numpy()
X = np.vstack([Xh, Xs]); y = np.r_[yh, ysv]
g = np.r_[[f"h{v:g}" for v in h.y_mm], [f"s{v:g}" for v in s.y_mm]]
raw = locate.grouped_cv(X, y, g); sl, ic = np.polyfit(y, raw, 1)
m = locate.ridge_fit(X, y, locate.pick_alpha(X, y, g))
pred = lambda Z: (locate.ridge_predict(m, Z) - ic) / sl  # noqa: E731
print(f"trained on {len(yh)} healthy + {len(ysv)} severe stamps; CV MAE {np.mean(np.abs((raw - ic) / sl - y)):.2f} mm\n")

# moderate: rows are the 5 vertebrae bottom to top (T12..T8)
mx, my = locate.mold("moderate"); o = np.argsort(my); mx, my = mx[o], my[o]
rows = sorted(mo.y_mm.unique())
mo = mo.assign(spine=mo.x_mm - pred(Xm))
lv = ["T12", "T11", "T10", "T9", "T8"]
tab = mo.pivot_table(index="y_mm", columns="repeat", values="spine").loc[rows]
print("MODERATE, centred stamps, spine x read (mm from centreline):")
print(pd.DataFrame(tab.values.round(1), index=[f"{l} (row {r:g}, process {p:.0f} mm, sensor centre {r + 47:.0f})"
                                               for l, r, p in zip(lv, rows, my)],
                   columns=[f"pass {c}" for c in tab.columns]).assign(median=tab.median(axis=1).round(1).values, design=mx))


def show(name, v):
    c, bow, _ = fit_c(my, v)
    return f"{name:14s} spine {np.round(v, 1)}  segment Cobb {fit_segments(my, v)[0]:5.1f}  c-curve Cobb {c:5.1f}  C-bow {bow:+5.1f} mm"


print("\n" + show("design", mx))
print(show("median of 3", tab.median(axis=1).to_numpy()))
for c in tab.columns:
    print(show(f"pass {c}", tab[c].to_numpy()))

# same-day controls: centred stamps on healthy and severe taken on 11-25
xa, ya = locate.mold("aligned")
new = new.assign(spine=new.x_mm - pred(Xn), centre=new.y_mm + 47)
new["design"] = [np.interp(c, np.sort(locate.mold("aligned" if ob == "healthy" else "severe")[1]),
                           locate.mold("aligned" if ob == "healthy" else "severe")[0][np.argsort(locate.mold("aligned" if ob == "healthy" else "severe")[1])])
                 for c, ob in zip(new.centre, new.object)]
print("\n2026-11-25 controls (x = 0):")
print(new.groupby(["object", "y_mm"])[["spine", "design"]].agg(["mean"]).round(1).to_string())
mo.to_csv(f"../analysis/severe_2026-11-23/moderate_blind_predictions_{'-'.join(BAD)}.csv", index=False)
