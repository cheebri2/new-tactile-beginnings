"""Phase-between-discs model trained on healthy + severe stamps, scored on held-out severe stamps.
Run from pc/ with final_data present: python ../analysis/severe_2026-11-23/mixed_phase.py"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from piezo import locate  # noqa: E402
from locator import stamps  # noqa: E402
from cobb import fit_c, fit_segments  # noqa: E402

tr, fe = locate.load("final_data", "cw90", ["dl"])
h, s = stamps(tr, ["healthy_vertebra"]), stamps(tr, ["severe_scoliosis"])
Xh, _, ch, fr = locate.relphase_matrix(fe, h.trial_id, f_max=200)
Xs, *_ = locate.relphase_matrix(fe, s.trial_id, ch, fr, f_max=200)
xs, ys = locate.mold("severe"); o = np.argsort(ys); rows = [0, 30, 60, 85, 125]
S = dict(zip(rows, xs[o])); Y = ys[o]
yh = h.x_mm.to_numpy(float); gh = np.array([f"h{v:g}" for v in h.y_mm])
x = s.x_mm.to_numpy(float); Sv = s.y_mm.map(S).to_numpy(); yl = x - Sv       # head offset from the spine
gs = s.y_mm.to_numpy(); rep = s.repeat.to_numpy()


def fit_predict(train, test):
    Xtr = np.vstack([Xh, Xs[train]]); ytr = np.r_[yh, yl[train]]
    gtr = np.r_[gh, [f"s{v:g}" for v in gs[train]]]
    raw = locate.grouped_cv(Xtr, ytr, gtr); sl, ic = np.polyfit(ytr, raw, 1)
    m = locate.ridge_fit(Xtr, ytr, locate.pick_alpha(Xtr, ytr, gtr))
    return (locate.ridge_predict(m, Xs[test]) - ic) / sl


def score(name, pred, test):
    e = pred - yl[test]; sp = x[test] - pred
    med = pd.Series(sp).groupby(gs[test]).median().reindex(rows).to_numpy()
    co, *_ = np.linalg.lstsq(np.c_[x[test], Sv[test], np.ones(test.sum())], pred, rcond=None)
    c0 = test & (x == 0)
    bows = []
    if c0.any():
        d = pd.DataFrame({"y": gs[test], "r": rep[test], "v": sp})[x[test] == 0].pivot_table(index="y", columns="r", values="v")
        bows = [fit_c(Y, d[c].reindex(rows).to_numpy())[1] for c in d if d[c].notna().all() and len(d) == 5]
    print(f"{name:44s} MAE {np.mean(np.abs(e)):.2f} mm | pred {co[0]:.2f}x {co[1]:+.2f}S | spine {np.round(med, 1)} "
          f"| Cobb {fit_segments(Y, med)[0]:.1f} (c-curve {fit_c(Y, med)[0]:.1f}) | centred bows {np.round(bows, 1)}")
    return med


print(f"design spine {xs[o]}, Cobb {fit_segments(Y, xs[o])[0]:.1f} (c-curve {fit_c(Y, xs[o])[0]:.1f}), bow {fit_c(Y, xs[o])[1]:.1f}\n")
none = np.zeros(len(x), bool)
score("healthy only", fit_predict(none, ~none), ~none)

# hold out one severe row at a time
pred = np.zeros(len(x))
for r in rows:
    t = gs == r
    pred[t] = fit_predict(~t, t)
score("healthy + other 4 severe rows -> held-out row", pred, ~none)

# hold out one reseat pass at a time (each pass = one stamp per x per row)
pred = np.zeros(len(x))
for k in np.unique(rep):
    t = rep == k
    pred[t] = fit_predict(~t, t)
score("healthy + other 2 severe passes -> held-out pass", pred, ~none)
for k in np.unique(rep):
    t = rep == k
    score(f"   held-out pass {k}", pred[t], t)

# half and half: train on passes 1, test on 2+3 and the other way
for tr_k in ([1], [2], [3]):
    t = ~np.isin(rep, tr_k)
    score(f"healthy + severe pass {tr_k} -> passes {sorted(set(rep[t]))}", fit_predict(~t, t), t)

# does adding severe hurt the healthy mold? hold out each healthy row, train on the other healthy rows + all severe
ph = np.zeros(len(yh))
for r in np.unique(gh):
    t = gh == r
    Xtr = np.vstack([Xh[~t], Xs]); ytr = np.r_[yh[~t], yl]; gtr = np.r_[gh[~t], [f"s{v:g}" for v in gs]]
    raw = locate.grouped_cv(Xtr, ytr, gtr); sl, ic = np.polyfit(ytr, raw, 1)
    m = locate.ridge_fit(Xtr, ytr, locate.pick_alpha(Xtr, ytr, gtr))
    ph[t] = (locate.ridge_predict(m, Xh[t]) - ic) / sl
xa, ya = locate.mold("aligned"); ya = np.sort(ya)
d = pd.DataFrame({"y": h.y_mm, "r": h.repeat, "v": yh - ph})[yh == 0].pivot_table(index="y", columns="r", values="v").loc[[0, 20, 60, 80, 100]].dropna(axis=1)
print(f"\nhealthy held-out rows with severe in training: MAE {np.mean(np.abs(ph - yh)):.2f} mm, "
      f"centred bows {np.round([fit_c(ya, d[c].to_numpy())[1] for c in d], 1)}, "
      f"Cobb of medians {fit_segments(ya, d.median(axis=1).to_numpy())[0]:.1f}")
