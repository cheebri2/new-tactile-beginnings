"""Train on all 5 vertebrae vs only the middle three (T11, T10, T9), and compare on the severe mold
(each pass held out in turn) and the moderate mold (blind, centred stamps).
Phase between recorded ul and dr only (the only two discs healthy on 11-22, 11-23 and 11-25).
Run from pc/ with final_data present: python ../analysis/severe_2026-11-23/middle_three.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from piezo import locate  # noqa: E402
from locator import stamps, levels  # noqa: E402
from cobb import fit_segments  # noqa: E402

OUT = Path("../analysis/severe_2026-11-23")
MID = ["T11", "T10", "T9"]
LV = ["T12", "T11", "T10", "T9", "T8"]

tr, fe = locate.load("final_data", "cw90", ["dl", "ur"])
h = stamps(tr, ["healthy_vertebra"])
s = stamps(tr, ["severe_scoliosis"]); s = s[s.session == "2026-11-23"].reset_index(drop=True)
mo = stamps(tr, ["mod_scoliosis"]).reset_index(drop=True)
for d in (h, s, mo):
    d["level"] = levels(d)[0]

Xh, _, ch, fr = locate.relphase_matrix(fe, h.trial_id)
F = lambda d: locate.relphase_matrix(fe, d.trial_id, ch, fr)[0]  # noqa: E731
Xs, Xm = F(s), F(mo)
sx, sy = locate.mold("severe")
S = dict(zip(locate.LEVELS, sx))
h["label"] = h.x_mm.astype(float)                       # aligned spine ~ centreline
s["label"] = s.x_mm - s.level.map(S)                    # head relative to that vertebra
s["spine_true"] = s.level.map(S)
mx, my = locate.mold("moderate"); M = dict(zip(locate.LEVELS, mx)); MY = dict(zip(locate.LEVELS, my))
SY = dict(zip(locate.LEVELS, sy))


def fit(hm, sm):
    """Ridge on the healthy rows hm and severe rows sm, with shrinkage calibration from grouped CV."""
    X = np.vstack([Xh[hm], Xs[sm]]); y = np.r_[h.label[hm], s.label[sm]]
    g = np.r_[[f"h{v:g}" for v in h.y_mm[hm]], [f"s{v:g}" for v in s.y_mm[sm]]]
    raw = locate.grouped_cv(X, y, g); sl, ic = np.polyfit(y, raw, 1)
    m = locate.ridge_fit(X, y, locate.pick_alpha(X, y, g))
    return lambda Z: (locate.ridge_predict(m, Z) - ic) / sl, float(np.mean(np.abs((raw - ic) / sl - y)))


def cobb(levels_, x, ymap, use):
    k = [LV.index(l) for l in use]
    yy = np.array([ymap[LV[i]] for i in k]); return fit_segments(yy, np.asarray(x)[k])[0]


rows = []
for train_name, keep in [("all 5", LV), ("middle 3", MID)]:
    hm = h.level.isin(keep).to_numpy()
    # severe: hold out each pass (repeat) in turn
    sev = []
    for r in sorted(s.repeat.unique()):
        sm = (s.level.isin(keep) & (s.repeat != r)).to_numpy()
        pred, _ = fit(hm, sm)
        te = (s.repeat == r).to_numpy()
        sev.append(s[te].assign(pred=pred(Xs[te])))
    sev = pd.concat(sev); sev["spine"] = sev.x_mm - sev.pred; sev["err"] = sev.pred - sev.label
    pred, cv = fit(hm, s.level.isin(keep).to_numpy())
    mod = mo.assign(spine=mo.x_mm - pred(Xm))
    mod["err"] = mod.spine - mod.level.map(M)
    pd.concat([sev.assign(case="severe"), mod.assign(case="moderate")]).assign(trained=train_name)[
        ["case", "trained", "trial_id", "x_mm", "y_mm", "repeat", "level", "spine", "err"]].to_csv(
        OUT / f"middle_three_pred_{train_name.replace(' ', '')}.csv", index=False)
    for case, d, ymap, design in [("severe", sev, SY, S), ("moderate", mod, MY, M)]:
        cen = d[d.x_mm == 0]
        tab = cen.pivot_table(index="level", columns="repeat", values="spine").reindex(LV)
        for p in list(tab.columns) + ["median"]:
            v = tab.median(axis=1) if p == "median" else tab[p]
            rows.append({"trained on": train_name, "case": case, "pass": p,
                         **{l: round(float(v[l]), 1) for l in LV},
                         "Cobb 5": round(cobb(LV, v, ymap, LV), 1), "Cobb mid3": round(cobb(LV, v, ymap, MID), 1)})
        for scope, sel in [("all 5", LV), ("middle 3", MID)]:
            dd = d[d.level.isin(sel)]
            rows.append({"trained on": train_name, "case": case, "pass": f"MAE {scope}",
                         "Cobb 5": round(float(dd.err.abs().mean()), 1),
                         "Cobb mid3": round(float(dd[dd.x_mm == 0].err.abs().mean()), 1)})
    print(f"trained on {train_name}: {hm.sum()} healthy + {s.level.isin(keep).sum()} severe stamps, CV MAE {cv:.2f} mm")

res = pd.DataFrame(rows)
for case, ymap, design in [("severe", SY, S), ("moderate", MY, M)]:
    v = pd.Series(design)
    print(f"\n{case.upper()} design spine {[design[l] for l in LV]}  Cobb 5 {cobb(LV, v, ymap, LV):.1f}  "
          f"Cobb mid3 {cobb(LV, v, ymap, MID):.1f}")
    print(res[res.case == case].drop(columns="case").to_string(index=False))
print("\n'MAE' rows: Cobb 5 column = mean abs spine error of all stamps in scope (mm), Cobb mid3 column = "
      "same for centred stamps only")
res.to_csv(OUT / "middle_three.csv", index=False)

# figure: centred-stamp medians vs design for both training sets
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axs = plt.subplots(1, 2, figsize=(9, 6), sharey=False)
for ax, (case, ymap, design) in zip(axs, [("severe", SY, S), ("moderate", MY, M)]):
    yy = [ymap[l] for l in LV]
    ax.plot([design[l] for l in LV], yy, "k-o", lw=2, label="mold design")
    for (tn, c) in [("all 5", "#3b6ea5"), ("middle 3", "#c0392b")]:
        r = res[(res.case == case) & (res["trained on"] == tn) & (res["pass"] == "median")].iloc[0]
        ax.plot([r[l] for l in LV], yy, "o--", color=c,
                label=f"trained on {tn}: Cobb {r['Cobb 5']:.0f}° (mid3 {r['Cobb mid3']:.0f}°)")
    ax.axhspan(min(ymap[l] for l in MID) - 5, max(ymap[l] for l in MID) + 5, color="0.93", zorder=0)
    for l, y_ in zip(LV, yy):
        ax.text(ax.get_xlim()[0] if False else -38, y_, l, va="center", fontsize=8, color="0.4")
    ax.set(xlim=(-40, 25), xlabel="spine x from mold centreline (mm)", ylabel="height above mold bottom (mm)",
           title=f"{case}: design Cobb {cobb(LV, pd.Series(design), ymap, LV):.1f}°, centred stamps, median of passes")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13)); ax.title.set_fontsize(9)
fig.tight_layout(); fig.savefig(OUT / "middle_three.png", dpi=150)
print(f"-> {OUT}/middle_three.csv, middle_three.png")
