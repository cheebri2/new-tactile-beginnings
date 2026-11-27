"""Model trained on all 5 vertebrae, read out on the middle three only (T11, T10, T9).
Uses middle_three_pred_all5.csv from middle_three.py (severe: each pass held out; moderate: blind).
Run from pc/: python ../analysis/severe_2026-11-23/middle_three_readout.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from piezo import locate  # noqa: E402
from cobb import fit_segments  # noqa: E402

OUT = Path("../analysis/severe_2026-11-23")
MID = ["T11", "T10", "T9"]
d = pd.read_csv(OUT / "middle_three_pred_all5.csv")
d = d[d.level.isin(MID)]

# middle-3 Cobb is about half the full Cobb on a single C-curve; scale from the two curved mold maps
ratio = {}
for m in ("moderate", "severe"):
    x, y = locate.mold(m); o = np.argsort(y); x, y = x[o], y[o]
    ratio[m] = fit_segments(y[1:4], x[1:4])[0] / fit_segments(y, x)[0]
SCALE = 1 / np.mean(list(ratio.values()))
print("mid3 / full Cobb on the mold maps:", {k: round(v, 3) for k, v in ratio.items()}, f"-> scale x{SCALE:.2f}\n")

rows = []
for case in ("severe", "moderate"):
    mx, my = locate.mold(case); D = dict(zip(locate.LEVELS, mx)); Y = dict(zip(locate.LEVELS, my))
    yy = np.array([Y[l] for l in MID]); des = np.array([D[l] for l in MID])
    full = fit_segments(my, mx)[0]; dmid = fit_segments(yy, des)[0]
    rows.append({"case": case, "pass": "design", **dict(zip(MID, des.round(1))), "mid3 Cobb": round(dmid, 1),
                 "scaled Cobb": round(dmid * SCALE, 1), "full design Cobb": round(full, 1)})
    c = d[(d.case == case) & (d.x_mm == 0)]
    tab = c.pivot_table(index="level", columns="repeat", values="spine").reindex(MID)
    for p in list(tab.columns) + ["median"]:
        v = (tab.median(axis=1) if p == "median" else tab[p]).to_numpy()
        sel = d[d.case == case] if p == "median" else d[(d.case == case) & (d.repeat == p)]
        cm = fit_segments(yy, v)[0]
        rows.append({"case": case, "pass": p, **dict(zip(MID, v.round(1))),
                     "centred err (mm)": round(float(np.mean(np.abs(v - des))), 1),
                     "all-stamp MAE (mm)": round(float(sel.err.abs().mean()), 1),
                     "mid3 Cobb": round(cm, 1), "scaled Cobb": round(cm * SCALE, 1), "full design Cobb": round(full, 1)})
res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(OUT / "middle_three_readout.csv", index=False)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axs = plt.subplots(1, 3, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
cols = {1: "#7fa7d6", 2: "#3b6ea5", 3: "#1f3f66"}
for ax, case in zip(axs[:2], ("severe", "moderate")):
    mx, my = locate.mold(case); o = np.argsort(my)
    ax.plot(mx[o], my[o], "-", color="0.75", lw=1.5)
    r = res[res.case == case]
    yy = [dict(zip(locate.LEVELS, my))[l] for l in MID]
    ax.plot(r[r["pass"] == "design"][MID].iloc[0], yy, "k-o", lw=2.2, label="mold design")
    for p in (1, 2, 3):
        ax.plot(r[r["pass"] == p][MID].iloc[0], yy, "o--", color=cols[p], lw=1, ms=4, label=f"pass {p}")
    ax.plot(r[r["pass"] == "median"][MID].iloc[0], yy, "s-", color="#c0392b", lw=2, label="median of passes")
    for l, y_ in zip(MID, yy):
        ax.text(-24, y_, l, va="center", fontsize=8, color="0.4")
    ax.set(xlim=(-25, 20), ylim=(65, 145), xlabel="spine x from mold centreline (mm)",
           ylabel="height above mold bottom (mm)", title=f"{case}: middle three, centred stamps")
    ax.title.set_fontsize(10)
ax = axs[2]
for i, case in enumerate(("severe", "moderate")):
    r = res[res.case == case]
    full = r["full design Cobb"].iloc[0]
    ax.plot([i - 0.3, i + 0.3], [full, full], "k-", lw=2.2)
    for j, p in enumerate((1, 2, 3)):
        ax.plot(i - 0.15 + 0.1 * j, r[r["pass"] == p]["scaled Cobb"].iloc[0], "o", color=cols[p])
    ax.plot(i + 0.2, r[r["pass"] == "median"]["scaled Cobb"].iloc[0], "s", color="#c0392b", ms=8)
ax.axhspan(0, 10, color="0.93", zorder=0)
ax.set(xticks=[0, 1], xticklabels=["severe", "moderate"], xlim=(-0.6, 1.6), ylim=(0, 110),
       ylabel="Cobb (deg)", title=f"Middle-3 Cobb x{SCALE:.2f}\nline = design full Cobb")
ax.title.set_fontsize(10)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "middle_three_readout.png", dpi=150)
print(f"-> {OUT}/middle_three_readout.csv, middle_three_readout.png")
