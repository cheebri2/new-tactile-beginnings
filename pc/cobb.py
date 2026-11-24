"""Cobb angle from located spine positions, and how well it can work with the current locator.

  python cobb.py estimate training/results/locator_predict/predictions.csv
  python cobb.py feasibility training/results/locator/predictions.csv

estimate     takes the locator's per-stamp spine position (spine_x_mm = head x - predicted offset,
             from `locator.py predict`), takes the median per y row, fits the spine line and
             reports the Cobb angle two ways:
               c-curve  x = c + A sin(pi (y - y0) / span) through the rows (one curve, ends straight
                        up, same model as figures/mock_spine_heatmap); Cobb = 2 atan(A pi / span)
               poly     quadratic (or --deg) through the rows; Cobb = largest minus smallest tangent
                        tilt over the scanned span (handles S-curves with --deg 3)
feasibility  Monte Carlo with the locator's REAL held-out errors (`locator.py train` writes them):
             bend a virtual spine to a known Cobb angle, give every row the errors of real stamps
             from one held-out row, fit, and compare. The head is stamped at fixed spots
             from the straight midline (where each vertebra would be if centred), as on a
             phantom whose vertebrae are flat and only shifted sideways. A stamp's error does not depend on where the
             spine is, only on the head-to-spine offset, so healthy-phantom errors carry over to a
             bent spine as long as each vertebra still sits square under the head (see report.md:
             that is exactly the part healthy data cannot test).

Signs: x + = right, y + = up the phantom (toward the head). A tilt angle is atan(dx/dy).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fit_c(y, x):
    """(Cobb deg, signed amplitude mm, fitted x(y)) for a single half-sine curve over the rows."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    y0, span = y.min(), np.ptp(y)
    s = np.sin(np.pi * (y - y0) / span)
    (c, a), *_ = np.linalg.lstsq(np.c_[np.ones_like(s), s], x, rcond=None)
    cobb = 2 * np.degrees(np.arctan(abs(a) * np.pi / span))
    return cobb, a, (lambda yy: c + a * np.sin(np.pi * (np.asarray(yy) - y0) / span))


def fit_poly(y, x, deg=2):
    """(Cobb deg, tilt angles deg along the span, polynomial) = max - min tangent tilt."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    p = np.polyfit(y, x, deg)
    yy = np.linspace(y.min(), y.max(), 200)
    tilt = np.degrees(np.arctan(np.polyval(np.polyder(p), yy)))
    return float(tilt.max() - tilt.min()), tilt, np.poly1d(p)


def fit_segments(y, x):
    """(Cobb deg, tilt of each segment deg): tilt between neighbouring vertebrae
    theta_k = atan((x_k+1 - x_k) / (y_k+1 - y_k)); Cobb = most tilted one way minus the other.
    Closest to the clinical definition (end vertebrae = the most tilted ones). Needs no curve shape,
    but each point's error goes straight into two tilts, so it is the noisiest."""
    o = np.argsort(y)
    y, x = np.asarray(y, float)[o], np.asarray(x, float)[o]
    t = np.degrees(np.arctan(np.diff(x) / np.diff(y)))
    return float(t.max() - t.min()), t


def c_curve(cobb_deg, y, y0, span):
    a = np.tan(np.radians(cobb_deg / 2)) * span / np.pi
    return a * np.sin(np.pi * (np.asarray(y) - y0) / span)


def rows_from_stamps(df, x_col="spine_x_mm", y_col="head_y_mm", max_dy=15.0):
    """(y, spine x) per vertebra: median of the stamps whose nearest process is within max_dy of the
    sensor centre, placed at that vertebra's stamp y. Without a level column: one point per y row."""
    if "level" in df and "dy_mm" in df:
        d = df[df.dy_mm.abs() <= max_dy].assign(vy=lambda t: t[y_col] + 47.0 + t.dy_mm)  # process height
        r = d.groupby("level").agg(vy=("vy", "median"), x=(x_col, "median"))
        return r.vy.to_numpy(float), r.x.to_numpy(float)
    r = df.groupby(df[y_col].round(0))[x_col].median()
    return r.index.to_numpy(float), r.to_numpy(float)


def estimate(a):
    df = pd.read_csv(a.predictions)
    y_col = "head_y_mm" if "head_y_mm" in df else "y_mm"
    if "spine_x_mm" not in df:
        df["spine_x_mm"] = df["x_mm"] - df["pred_x_mm"]      # training output: head x typed
    y, x = rows_from_stamps(df, "spine_x_mm", y_col, a.max_dy)
    if len(y) < 3:
        raise SystemExit("need stamps in at least 3 y rows")
    c_c, amp, _ = fit_c(y, x)
    c_p, _, _ = fit_poly(y, x, a.deg)
    c_s, tilts = fit_segments(y, x)
    print("level y(mm)  spine x (mm, median of stamps)")
    for yy, xx in zip(y, x):
        print(f"     {yy:5.0f}  {xx:+6.1f}")
    print(f"\nCobb, c-curve fit: {c_c:.1f} deg (bow {amp:+.1f} mm)\nCobb, poly deg {a.deg}: {c_p:.1f} deg"
          f"\nCobb, segment tilts: {c_s:.1f} deg (tilts {', '.join(f'{v:+.1f}' for v in tilts)})")


def feasibility(a):
    df = pd.read_csv(a.predictions)
    rng = np.random.default_rng(a.seed)
    df["err_spine"] = -(df["pred_x_mm"] - df["x_mm"])        # spine estimate - truth
    if "dy_mm" in df and a.max_dy is not None:
        df = df[df.dy_mm.abs() <= a.max_dy]                  # only rows with a process near the centre
    rows = sorted(df.y_mm.unique())
    pools = [df[df.y_mm == r] for r in rows]
    y_rows = (np.asarray(a.level_y, float) if a.level_y else
              np.asarray(rows, float) if a.levels is None else np.linspace(0, a.span, a.levels))
    y0, span = y_rows.min(), np.ptp(y_rows)

    grid = np.array(sorted(df.x_mm.unique()))

    def one_draw(cobb, heads, n_per_head):
        """Head stamped at `heads` mm from the straight midline, n_per_head reseats each. A stamp's
        label is head - shift; its error is drawn from real stamps at the nearest trained offset."""
        shift = c_curve(cobb, y_rows, y0, span)
        xs = []
        for s in shift:
            pool = pools[rng.integers(len(pools))]                    # a whole real row: keeps row bias
            e = []
            for h in heads:
                lab = grid[np.argmin(np.abs(grid - (h - s)))]
                ee = pool[pool.x_mm == lab].err_spine.to_numpy()
                e += list(rng.choice(ee, size=n_per_head, replace=len(ee) < n_per_head))
            xs.append(s + np.mean(e))
        x = np.array(xs)
        return fit_c(y_rows, x)[0], fit_poly(y_rows, x, 2)[0]

    out = []
    schemes = [("1 stamp per vertebra, head on midline", [0], 1),
               ("3 reseats per vertebra, head on midline", [0], 3),
               ("head at -10, 0, +10 mm, 3 reseats each (9)", [-10, 0, 10], 3),
               ("head at -10..+10 mm, 2 reseats each (10)", [-10, -5, 0, 5, 10], 2),
               ("raster -10..+10 mm, 5 reseats each (25)", [-10, -5, 0, 5, 10], 5)]
    for scheme, heads, n in schemes:
        for cobb in a.cobbs:
            est = np.array([one_draw(cobb, heads, n) for _ in range(a.draws)])
            for k, method in enumerate(["c-curve", "poly2"]):
                e = est[:, k]
                out.append({"scheme": scheme, "stamps_per_row": n * len(heads), "true_cobb": cobb,
                            "method": method, "mean_est": e.mean(), "bias": e.mean() - cobb, "sd": e.std(),
                            "p95_abs_err": np.percentile(np.abs(e - cobb), 95),
                            "pct_read_ge_10": 100 * np.mean(e >= 10)})
    res = pd.DataFrame(out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(a.out) / "cobb_feasibility.csv", index=False)
    show = res[res.method == "c-curve"]
    print(f"{len(y_rows)} levels over {span:.0f} mm, errors from real rows y = {', '.join(f'{r:g}' for r in rows)}, "
          f"{a.draws} draws each, c-curve fit\n")
    print(show.pivot_table(index="scheme", columns="true_cobb", values="mean_est", sort=False).round(1)
          .to_string(), "\n\nSD of the estimate (deg):")
    print(show.pivot_table(index="scheme", columns="true_cobb", values="sd", sort=False).round(1).to_string())
    print("\n% of scans read as >= 10 deg (scoliosis threshold):")
    print(show.pivot_table(index="scheme", columns="true_cobb", values="pct_read_ge_10", sort=False).round(0).to_string())
    plot(res, Path(a.out) / "cobb_feasibility.png", len(y_rows), span)
    print(f"\n-> {a.out}/cobb_feasibility.csv, cobb_feasibility.png")


def plot(res, path, n_rows, span):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = res[res.method == "c-curve"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cols = ["#c0392b", "#e67e22", "#27ae60", "#3b6ea5"]
    for (scheme, g), c in zip(r.groupby("scheme", sort=False), cols):
        ax.errorbar(g.true_cobb, g.mean_est, yerr=g.sd, fmt="o-", color=c, capsize=3, label=scheme)
    m = r.true_cobb.max() + 5
    ax.plot([0, m], [0, m], "k--", lw=1, label="perfect")
    ax.axhspan(0, 10, color="0.92", zorder=0)
    ax.text(m - 1, 4, "below 10° = not scoliosis", ha="right", fontsize=8, color="0.4")
    ax.set(xlabel="true Cobb angle (deg)", ylabel="estimated Cobb (deg, mean ± SD)",
           title=f"Cobb from the locator's real held-out errors\n{n_rows} vertebrae over {span:.0f} mm, c-curve fit")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("predictions")
    e.add_argument("--deg", type=int, default=2)
    e.add_argument("--max-dy", type=float, default=15.0,
                   help="use stamps whose nearest process is within this of the sensor centre (mm)")
    f = sub.add_parser("feasibility")
    f.add_argument("predictions", help="predictions.csv written by locator.py train")
    f.add_argument("--cobbs", type=float, nargs="+", default=[0, 10, 20, 30],
                   help="true angles to simulate (above ~30 deg the shift leaves the trained +-10 mm)")
    f.add_argument("--level-y", type=float, nargs="*", default=[48.1, 81.3, 109.3, 133.4, 163.3],
                   help="process heights to simulate (default: the severe/moderate mold map, T12..T8). "
                        "Pass nothing to use the real rows")
    f.add_argument("--max-dy", type=float, default=None,
                   help="draw errors only from real rows whose nearest process is within this of the "
                        "sensor centre (needs level/dy_mm in predictions.csv)")
    f.add_argument("--levels", type=int, default=None, help="evenly spaced rows to simulate instead")
    f.add_argument("--span", type=float, default=120, help="mm covered by --levels")
    f.add_argument("--draws", type=int, default=2000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", default="training/results/cobb")
    a = p.parse_args()
    estimate(a) if a.cmd == "estimate" else feasibility(a)


if __name__ == "__main__":
    main()
