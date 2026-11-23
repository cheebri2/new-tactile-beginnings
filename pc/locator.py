"""Train and use the spinous-process locator (how far the head is left/right of the spine).

  python locator.py train --data final_data --turn cw90 --bad-discs dl
  python locator.py predict --model models/locator.json --data some_new_data

train   fits ridge regression on "share" log amplitudes (piezo/locate.py), scores it with
        grouped cross-validation (default: hold out one y row at a time) and saves
          models/locator.json                       the model
          <results>/predictions.csv                 every stamp: true x, held-out prediction
          <results>/locator_cv.png                  prediction vs truth, error by row
          <results>/report.md                       the numbers
predict adds the model's x estimate to every non-blank stamp in --data (predictions.csv in
        the results folder). spine_x_mm = head x - predicted offset, using the typed x/y or,
        with --position imu, the IMU track.

--turn  how the head was turned on the phantom when the data was taken ('cw90' = head's up side
        to the phantom's right, bri's 2026-11-22 batch). Discs are renamed to phantom positions
        before training, so a model trained on turned data can be used on data taken square.
--bad-discs  recorded names of discs to leave out (the physical disc; it stays bad when turned).
"""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from piezo import locate


def stamps(trials):
    t = trials[(trials.test_type != "blank") & (trials.object.fillna("none") != "none")]
    ok = (t.timing == "events") & (t.stream_lost_blocks.fillna(0) == 0) & (t.stream_bad_blocks.fillna(0) == 0)
    if (~ok).any():
        print(f"dropping {int((~ok).sum())} stamps with missing timing or lost USB data")
    return t[ok].reset_index(drop=True)


def train(a):
    trials, feats = locate.load(a.data, a.turn, a.bad_discs)
    st = stamps(trials)
    X, names, chans, freqs = locate.share_matrix(feats, st.trial_id, f_min=a.f_min, f_max=a.f_max)
    y = st.x_mm.to_numpy(float)
    groups = st[a.group].astype(str).to_numpy()
    print(f"{len(st)} stamps, {X.shape[1]} features (phantom discs {', '.join(chans)}, "
          f"{len(freqs)} freqs {freqs[0]:g}-{freqs[-1]:g} Hz), {len(set(groups))} CV groups by {a.group}")

    pred = locate.grouped_cv(X, y, groups)              # nested: alpha picked inside each fold
    err = pred - y
    alpha = locate.pick_alpha(X, y, groups)
    model = locate.ridge_fit(X, y, alpha)

    res = st[["trial_id", "session", "repeat", "x_mm", "y_mm", "imu_x_mm", "imu_y_mm"]].copy()
    res["pred_x_mm"] = pred
    res["error_mm"] = err
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "predictions.csv", index=False)

    base = float(np.mean(np.abs(y - y.mean())))
    by_x = res.groupby("x_mm").pred_x_mm.agg(["mean", "std", "size"])
    by_row = res.assign(ae=np.abs(err)).groupby("y_mm").ae.mean()
    row_bias = res.groupby("y_mm").error_mm.mean()
    cond = res.groupby(["x_mm", "y_mm"]).pred_x_mm.mean().reset_index()
    side = np.abs(y) >= 5
    metrics = {
        "n_stamps": int(len(y)), "cv_group": a.group, "alpha": alpha,
        "mae_mm": float(np.mean(np.abs(err))), "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "baseline_mae_mm": base,
        "mae_5stamp_mean_mm": float(np.mean(np.abs(cond.pred_x_mm - cond.x_mm))),
        "side_correct_pct": float(100 * np.mean(np.sign(pred[side]) == np.sign(y[side]))),
        "row_bias_sd_mm": float(row_bias.std()),
        "within_row_sd_mm": float(res.groupby("y_mm").error_mm.std().mean()),
        "slope": float(np.polyfit(y, pred, 1)[0]),
    }
    meta = {"trained": dt.datetime.now().isoformat(timespec="seconds"), "data": str(a.data),
            "turn": a.turn, "bad_discs": list(a.bad_discs), "features": "share log10 amp",
            "f_min": a.f_min, "f_max": a.f_max, "target": "x_mm (head right of spine, mm)", "cv": metrics}
    locate.save_model(a.model, model, chans, freqs, meta)

    lines = [f"# Locator cross-validation ({dt.date.today()})", "",
             f"Data `{a.data}`, head turn `{a.turn}`, discs left out (recorded names): "
             f"{', '.join(a.bad_discs) or 'none'}. Phantom discs used: {', '.join(chans)}.",
             f"{len(y)} stamps; each {a.group} group held out in turn (nested ridge alpha; final alpha {alpha:g}).", "",
             "| | value |", "|---|---|",
             f"| mean abs error per stamp | {metrics['mae_mm']:.2f} mm |",
             f"| RMS error per stamp | {metrics['rmse_mm']:.2f} mm |",
             f"| guessing the mean x | {base:.2f} mm |",
             f"| mean of the 5 repeats per spot | {metrics['mae_5stamp_mean_mm']:.2f} mm |",
             f"| side right/wrong for abs(x) >= 5 mm | {metrics['side_correct_pct']:.0f} % correct |",
             f"| error shared by a whole row (SD of row bias) | {metrics['row_bias_sd_mm']:.2f} mm |",
             f"| error stamp-to-stamp inside a row (SD) | {metrics['within_row_sd_mm']:.2f} mm |",
             f"| slope of prediction vs truth | {metrics['slope']:.2f} (1 = unbiased) |", "",
             "Prediction by true x (mean, SD, n):", "", "| true x | mean | SD | n |", "|---|---|---|---|"]
    lines += [f"| {x:g} | {r['mean']:+.1f} | {r['std']:.1f} | {int(r['size'])} |" for x, r in by_x.iterrows()]
    lines += ["", "Mean abs error by y row: " + ", ".join(f"y={k:g}: {v:.1f}" for k, v in by_row.items())]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    plot(res, out / "locator_cv.png", metrics)
    print("\n".join(lines))
    print(f"\nmodel -> {a.model}\nresults -> {out}/")


def plot(res, path, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.8, 0.8, len(res))
    ax[0].scatter(res.x_mm + jit, res.pred_x_mm, s=12, alpha=0.5, color="#3b6ea5", label="one stamp (held out)")
    g = res.groupby("x_mm").pred_x_mm
    ax[0].errorbar(g.mean().index, g.mean(), yerr=g.std(), fmt="o", color="#c0392b", capsize=4, label="mean ± SD")
    lim = [res.x_mm.min() - 4, res.x_mm.max() + 4]
    ax[0].plot(lim, lim, "k--", lw=1, label="perfect")
    ax[0].set(xlabel="true head offset x (mm, + = head right of spine)", ylabel="predicted x (mm)",
              title=f"Held-out by row: MAE {m['mae_mm']:.1f} mm (guess-the-mean {m['baseline_mae_mm']:.1f})")
    ax[0].legend(fontsize=8)
    e = res.groupby("y_mm").error_mm
    ax[1].bar(e.mean().index, e.mean(), width=8, color="#7f8c8d", label="row bias (mean error)")
    ax[1].errorbar(e.mean().index, e.mean(), yerr=e.std(), fmt="none", color="k", capsize=3, label="± SD in row")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="y row (mm up the phantom)", ylabel="prediction error (mm)", title="Error by raster row")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def predict(a):
    m = locate.load_model(a.model)
    turn = a.turn if a.turn is not None else m["turn"]
    bad = a.bad_discs if a.bad_discs is not None else m["bad_discs"]
    trials, feats = locate.load(a.data, turn, bad)
    st = stamps(trials)
    X, *_ = locate.share_matrix(feats, st.trial_id, m["channels"], m["freqs"], m["f_min"], m["f_max"])
    st["pred_x_mm"] = locate.ridge_predict(m, X)
    hx, hy = ("imu_x_mm", "imu_y_mm") if a.position == "imu" else ("x_mm", "y_mm")
    st["head_x_mm"], st["head_y_mm"] = st[hx], st[hy]
    st["spine_x_mm"] = st.head_x_mm - st.pred_x_mm
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["trial_id", "x_mm", "y_mm", "head_x_mm", "head_y_mm", "pred_x_mm", "spine_x_mm"]
    st[cols].to_csv(out / "predictions.csv", index=False)
    print(st[cols].round(1).to_string(index=False))
    print(f"\n-> {out}/predictions.csv   (Cobb: python cobb.py estimate {out}/predictions.csv)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="final_data")
    t.add_argument("--turn", default="none", help="none | cw90 | ccw90 | 180")
    t.add_argument("--bad-discs", nargs="*", default=[], help="recorded names, e.g. dl")
    t.add_argument("--group", default="y_mm", help="trials.csv column to hold out together (y_mm, session, ...)")
    t.add_argument("--f-min", type=float, default=50)
    t.add_argument("--f-max", type=float, default=200,
                   help="top of the band (default 200 Hz: 50-190 Hz did best on the 2026-11-22 batch)")
    t.add_argument("--model", default="models/locator.json")
    t.add_argument("--results", default="training/results/locator")
    q = sub.add_parser("predict")
    q.add_argument("--model", default="models/locator.json")
    q.add_argument("--data", required=True)
    q.add_argument("--turn", default=None, help="default: same as the training data")
    q.add_argument("--bad-discs", nargs="*", default=None)
    q.add_argument("--position", choices=["typed", "imu"], default="typed")
    q.add_argument("--results", default="training/results/locator_predict")
    a = p.parse_args()
    train(a) if a.cmd == "train" else predict(a)


if __name__ == "__main__":
    main()
