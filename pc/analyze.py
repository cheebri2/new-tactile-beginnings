"""Compare test conditions: averages across repeats, and how repeatable they are.

  python analyze.py                                   # every trial in ./data
  python analyze.py --session 2026-11-14 --norm share
  python analyze.py --where "depth_mm == 15" --test blank object offset
  python analyze.py --conditions "blank d=15" "object vertebra d=15"   # pick and order conditions

Writes to results/<name>/:
  amplitude.png       mean +/- SD per condition for each disc (thin lines = individual repeats)
  phase.png           phase vs the up-right disc, circular mean +/- SD
  repeatability.png   between-condition / within-condition spread per disc and frequency
  pairs.png           effect size for each pair of conditions, disc x frequency
  trials_pca.png      every trial on the first two principal components, by condition
  summary.csv         per condition x disc x frequency: n, mean, sd, cv, phase
  separation.csv      per disc x frequency: within SD, between SD, ratio, ICC
  pairs.csv           per pair of conditions x disc x frequency: ratio and effect size d
  report.md           the headline numbers
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from piezo import CHANNEL_NAMES, CHANNELS, REF_CHANNEL, stats

# Fixed categorical order (colourblind-checked); a condition keeps its colour in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Physical layout of the discs, so each panel sits where its disc is.
LAYOUT = {"ul": (0, 0), "ur": (0, 1), "dl": (1, 0), "dr": (1, 1)}
YLABEL = {"none": "amplitude (ADC counts, peak)", "share": "share of 4-disc total",
          "blank": "amplitude / session blank"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--out", help="output folder (default results/<norm>_<sessions>)")
    p.add_argument("--session", nargs="*", help="only these sessions")
    p.add_argument("--test", nargs="*", help="only these test types")
    p.add_argument("--where", help='pandas query on trial metadata, e.g. "depth_mm == 15 and load_g == 200"')
    p.add_argument("--conditions", nargs="*", help="only these condition labels, in this order")
    p.add_argument("--norm", choices=["none", "share", "blank"], default="none",
                   help="none: raw counts; share: fraction of the 4 discs; blank: / same-session blank")
    p.add_argument("--fmin", type=float, default=0)
    p.add_argument("--fmax", type=float, default=1e9)
    return p.parse_args()


def setup_axes(title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout="constrained")
    fig.suptitle(title, fontsize=12)
    for ax in axes.flat:
        ax.grid(color="#e4e3df", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    return fig, axes


def plot_amplitude(f, conds, colors, ylabel, path):
    fig, axes = setup_axes("Amplitude by condition: line = mean, band = ±1 SD, thin lines = each repeat")
    for ch in CHANNELS:
        ax = axes[LAYOUT[ch]]
        for c in conds:
            d = f[(f.channel == ch) & (f.condition == c)]
            for _, r in d.groupby("trial_id"):
                ax.plot(r.freq_hz, r.value, color=colors[c], lw=0.7, alpha=0.35)
            s = d.groupby("freq_hz")["value"].agg(["mean", "std", "count"]).reset_index()
            ax.fill_between(s.freq_hz, s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0),
                            color=colors[c], alpha=0.18, lw=0)
            ax.plot(s.freq_hz, s["mean"], color=colors[c], lw=2, label=f"{c} (n={d.trial_id.nunique()})")
        ax.set_title(CHANNEL_NAMES[ch], fontsize=10, loc="left")
        ax.set_ylabel(ylabel)
    for ax in axes[1]:
        ax.set_xlabel("frequency (Hz)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_phase(f, conds, colors, path):
    fig, axes = setup_axes(f"Phase relative to {CHANNEL_NAMES[REF_CHANNEL]} (positive = lags): "
                           "circular mean ± SD")
    for ch in CHANNELS:
        ax = axes[LAYOUT[ch]]
        ax.set_title(CHANNEL_NAMES[ch], fontsize=10, loc="left")
        if ch == REF_CHANNEL:
            ax.text(0.5, 0.5, "reference disc (phase 0)", transform=ax.transAxes, ha="center", color="#6b6a64")
            continue
        for c in conds:
            d = f[(f.channel == ch) & (f.condition == c)]
            s = d.groupby("freq_hz")["phase_deg"].apply(lambda v: pd.Series(stats.circ_mean_sd(v),
                                                                             index=["m", "sd"])).unstack()
            ax.errorbar(s.index, s["m"], yerr=s["sd"], color=colors[c], lw=1.6, capsize=2,
                        marker="o", ms=3, label=c)
        ax.set_ylim(-180, 180)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_ylabel("phase (deg)")
    for ax in axes[1]:
        ax.set_xlabel("frequency (Hz)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_repeatability(sep, path):
    fig, axes = setup_axes("Can conditions be told apart? between-condition SD / within-condition (repeat) SD")
    for ch in CHANNELS:
        ax = axes[LAYOUT[ch]]
        d = sep[sep.channel == ch]
        ax.axhspan(0, 1, color="#f3d9d3", alpha=0.6, lw=0)
        ax.axhline(1, color="#8a4b3c", lw=1)
        ax.plot(d.freq_hz, d.ratio, color=PALETTE[0], lw=2, marker="o", ms=4)
        ax.text(0.01, 0.02, "below 1: repeats scatter more than conditions differ",
                transform=ax.transAxes, fontsize=8, color="#8a4b3c")
        ax.set_title(f"{CHANNEL_NAMES[ch]}   (median repeat spread {d.within_cv_pct.median():.0f}%)",
                     fontsize=10, loc="left")
        ax.set_ylabel("between / within")
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("frequency (Hz)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pairs(pairs, path):
    keys = list(pairs.groupby(["cond_a", "cond_b"], sort=False).groups)
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 1.6 + 1.5 * len(keys)), squeeze=False,
                             layout="constrained")
    lim = max(3.0, float(np.nanpercentile(pairs.d.abs(), 98)))
    for ax, (a, b) in zip(axes[:, 0], keys):
        d = pairs[(pairs.cond_a == a) & (pairs.cond_b == b)]
        m = d.pivot_table(index="channel", columns="freq_hz", values="d").reindex(CHANNELS)
        im = ax.imshow(m.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_yticks(range(len(CHANNELS)), [CHANNEL_NAMES[c] for c in CHANNELS], fontsize=8)
        step = max(1, len(m.columns) // 15)
        ax.set_xticks(range(0, len(m.columns), step), [f"{v:g}" for v in m.columns[::step]], fontsize=8)
        ax.set_title(f"{b}  vs  {a}   (red = higher in {b}; |d| > 2 is a clear difference)",
                     fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("frequency (Hz)")
    fig.colorbar(im, ax=axes[:, 0], label="effect size d (log amplitude)", shrink=0.8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pca(m, cond, conds, colors, acc, path):
    if len(m) < 3:
        return
    scores, var = stats.pca(m)
    fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
    for c in conds:
        sel = (cond == c).to_numpy()
        ax.scatter(scores[sel, 0], scores[sel, 1], s=60, color=colors[c], edgecolor="white",
                   linewidth=1.5, label=c, zorder=3)
    ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}% of variance)" if len(var) > 1 else "PC2")
    acc_txt = f"leave-one-out nearest-centroid accuracy {acc[0]:.0%} (chance {acc[1]:.0%})" if acc[2] else ""
    ax.set_title("Each dot is one trial. Tight, separate clusters = repeatable and distinguishable\n"
                 + acc_txt, fontsize=10, loc="left")
    ax.grid(color="#e4e3df", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(path, a, trials, summary, sep, pairs, acc, conds):
    lines = [f"# Condition comparison ({a.norm} amplitude)", "",
             f"{len(trials)} trials, {len(conds)} conditions, sessions: "
             f"{', '.join(sorted(trials.session.astype(str).unique()))}", "",
             "| condition | trials | median lock quality | median SNR (dB) |", "|---|---|---|---|"]
    for c in conds:
        t = trials[trials.condition == c]
        snr = summary[summary.condition == c].snr_db_median.median()
        lines.append(f"| {c} | {len(t)} | {t.align_quality.median():.1f} | {snr:.0f} |")
    if len(sep):
        lines += ["", "## Repeatability (log amplitude)", "",
                  "Within = spread between repeats of the same condition (reseating). "
                  "Ratio = between-condition SD / within SD; below 1 means repeats scatter more "
                  "than the conditions differ.", "",
                  "| disc | median repeat spread | median ratio | best ratio (freq) | median ICC |",
                  "|---|---|---|---|---|"]
        for ch in CHANNELS:
            d = sep[sep.channel == ch]
            if d.empty:
                continue
            b = d.loc[d.ratio.idxmax()]
            lines.append(f"| {CHANNEL_NAMES[ch]} | {d.within_cv_pct.median():.0f}% | {d.ratio.median():.2f} | "
                         f"{b.ratio:.2f} ({b.freq_hz:g} Hz) | {d.icc.median():.2f} |")
    if len(pairs):
        lines += ["", "## Pairs of conditions", "",
                  "| pair | features with abs(d) > 2 | features with no overlap | strongest (disc, freq, ratio, d) |",
                  "|---|---|---|---|"]
        for (ca, cb), d in pairs.groupby(["cond_a", "cond_b"], sort=False):
            best = d.loc[d.d.abs().idxmax()]
            lines.append(f"| {cb} vs {ca} | {(d.d.abs() > 2).sum()} / {len(d)} | {(~d.overlap).sum()} | "
                         f"{CHANNEL_NAMES[best.channel]}, {best.freq_hz:g} Hz, x{best.ratio_b_over_a:.2f}, "
                         f"d={best.d:+.1f} |")
    if acc[2]:
        lines += ["", f"Leave-one-out nearest-centroid accuracy over the whole sweep: **{acc[0]:.0%}** "
                      f"(chance {acc[1]:.0%}, {acc[2]} trials). With only a few repeats per condition "
                      "treat this as a sanity check, not a result."]
    path.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    a = parse_args()
    data = Path(a.data)
    trials = pd.read_csv(data / "trials.csv")
    feats = pd.read_csv(data / "features.csv")

    if a.session:
        trials = trials[trials.session.astype(str).isin(a.session)]
    if a.test:
        trials = trials[trials.test_type.isin(a.test)]
    if a.where:
        trials = trials.query(a.where)
    if a.conditions:
        trials = trials[trials.condition.isin(a.conditions)]
    if trials.empty:
        raise SystemExit("No trials match those filters. Conditions available:\n  "
                         + "\n  ".join(pd.read_csv(data / "trials.csv").condition.unique()))
    feats = feats[feats.trial_id.isin(trials.trial_id) & feats.freq_hz.between(a.fmin, a.fmax)]
    f = stats.normalise(feats, trials, a.norm)
    f = f.merge(trials[["trial_id", "condition"]], on="trial_id")
    conds = a.conditions or list(dict.fromkeys(trials.sort_values("time").condition))
    conds = [c for c in conds if c in set(f.condition)]
    if len(conds) > len(PALETTE):
        raise SystemExit(f"{len(conds)} conditions; pick up to {len(PALETTE)} with --conditions or filters")
    colors = {c: PALETTE[i] for i, c in enumerate(conds)}

    sessions = "_".join(sorted(trials.session.astype(str).unique()))[:60]
    out = Path(a.out or f"results/{a.norm}_{sessions}")
    out.mkdir(parents=True, exist_ok=True)

    summary = stats.condition_summary(f)
    sep = stats.separation(f)
    pairs = stats.pairwise(f, conds)
    m, cond = stats.trial_matrix(f)
    acc = stats.leave_one_out_accuracy(m, cond) if len(conds) > 1 else (np.nan, np.nan, 0)
    summary.to_csv(out / "summary.csv", index=False)
    sep.to_csv(out / "separation.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)

    plot_amplitude(f, conds, colors, YLABEL[a.norm], out / "amplitude.png")
    plot_phase(f, conds, colors, out / "phase.png")
    if len(sep):
        plot_repeatability(sep, out / "repeatability.png")
    plot_pairs(pairs, out / "pairs.png")
    plot_pca(m, cond, conds, colors, acc, out / "trials_pca.png")
    print(write_report(out / "report.md", a, trials, summary, sep, pairs, acc, conds))
    print(f"\nFigures and tables in {out}/")
    if any(trials.groupby("condition").size() < 3):
        print("Note: some conditions have fewer than 3 repeats; their spread is not meaningful yet.")


if __name__ == "__main__":
    main()
