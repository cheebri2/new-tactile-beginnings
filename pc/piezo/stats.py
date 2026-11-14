"""Group trials by condition and measure how repeatable each feature is.

All repeatability numbers use log amplitude, because reseating changes coupling
multiplicatively (a disc that seats 20% better reads 20% higher at every frequency).
"""

import itertools

import numpy as np
import pandas as pd

from . import CHANNELS


def normalise(feats: pd.DataFrame, trials: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Adds column `value` (the amplitude analysed) according to mode:
    none   ADC counts as measured
    share  each disc's fraction of the four discs at that frequency (cancels overall level/preload)
    blank  divided by the mean blank (test_type == blank) of the same session, same disc and frequency
    """
    f = feats.copy()
    if mode == "none":
        f["value"] = f["amp"]
    elif mode == "share":
        f["value"] = f["amp"] / f.groupby(["trial_id", "freq_hz"])["amp"].transform("sum")
    elif mode == "blank":
        f = f.merge(trials[["trial_id", "session", "test_type"]], on="trial_id", how="left")
        ref = (f[f.test_type == "blank"].groupby(["session", "channel", "freq_hz"])["amp"]
               .mean().rename("blank_amp").reset_index())
        f = f.merge(ref, on=["session", "channel", "freq_hz"], how="left")
        missing = f["blank_amp"].isna() & (f.test_type != "blank")
        if missing.any():
            sess = sorted(f.loc[missing, "session"].astype(str).unique())
            print(f"note: no blank trials for session(s) {sess}; those trials are dropped")
        f["value"] = f["amp"] / f["blank_amp"]
        f = f.dropna(subset=["value"]).drop(columns=["session", "test_type", "blank_amp"])
    else:
        raise ValueError(mode)
    return f


def circ_mean_sd(deg):
    z = np.exp(1j * np.radians(np.asarray(deg, float)))
    m = z.mean()
    r = min(abs(m), 1.0)
    return float(np.degrees(np.angle(m))), float(np.degrees(np.sqrt(-2 * np.log(max(r, 1e-12)))))


def condition_summary(f: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and CV of `value` and circular mean/SD of phase per condition x channel x frequency."""
    def agg(g):
        pm, ps = circ_mean_sd(g["phase"])
        v = g["value"]
        return pd.Series({"n": len(g), "mean": v.mean(), "sd": v.std(ddof=1),
                          "cv_pct": 100 * v.std(ddof=1) / v.mean() if v.mean() else np.nan,
                          "min": v.min(), "max": v.max(),
                          "phase_mean_deg": pm, "phase_sd_deg": ps,
                          "snr_db_median": g["snr_db"].median()})
    return f.groupby(["condition", "channel", "freq_hz"], sort=False).apply(agg, include_groups=False).reset_index()


def separation(f: pd.DataFrame) -> pd.DataFrame:
    """Per channel x frequency: within-condition spread vs between-condition spread (log amplitude).

    within_sd   pooled SD across repeats of the same condition (the reseat/placement spread)
    between_sd  SD of the condition means
    ratio       between_sd / within_sd. Below ~1 the conditions are not distinguishable
                from repeat-to-repeat scatter at that feature.
    icc         one-way intraclass correlation: share of total variance due to condition.
                Near 1 = repeats agree and conditions differ; near 0 or below = repeats dominate.
    """
    rows = []
    g = f.assign(lv=np.log(f["value"].clip(lower=1e-9)))
    for (ch, fr), d in g.groupby(["channel", "freq_hz"], sort=False):
        groups = [x["lv"].to_numpy() for _, x in d.groupby("condition") if len(x) >= 1]
        k = len(groups)
        n = np.array([len(x) for x in groups])
        dfw = (n - 1).sum()
        if k < 2 or dfw < 1:
            continue
        ssw = sum(((x - x.mean()) ** 2).sum() for x in groups)
        msw = ssw / dfw
        means = np.array([x.mean() for x in groups])
        grand = np.concatenate(groups).mean()
        msb = (n * (means - grand) ** 2).sum() / (k - 1)
        n0 = (n.sum() - (n ** 2).sum() / n.sum()) / (k - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else np.nan
        within = np.sqrt(msw)
        between = means.std(ddof=1)
        rows.append({"channel": ch, "freq_hz": fr, "within_sd": within, "between_sd": between,
                     "ratio": between / within if within > 0 else np.nan, "icc": icc,
                     "within_cv_pct": 100 * (np.exp(within) - 1)})
    return pd.DataFrame(rows)


def pairwise(f: pd.DataFrame, conditions) -> pd.DataFrame:
    """Effect size d = (mean_b - mean_a) / pooled SD on log amplitude, for every pair of conditions."""
    rows = []
    g = f.assign(lv=np.log(f["value"].clip(lower=1e-9)))
    for a, b in itertools.combinations(conditions, 2):
        for (ch, fr), d in g.groupby(["channel", "freq_hz"], sort=False):
            xa = d.loc[d.condition == a, "lv"].to_numpy()
            xb = d.loc[d.condition == b, "lv"].to_numpy()
            if len(xa) < 2 or len(xb) < 2:
                continue
            sp = np.sqrt(((len(xa) - 1) * xa.var(ddof=1) + (len(xb) - 1) * xb.var(ddof=1))
                         / (len(xa) + len(xb) - 2))
            rows.append({"cond_a": a, "cond_b": b, "channel": ch, "freq_hz": fr,
                         "ratio_b_over_a": float(np.exp(xb.mean() - xa.mean())),
                         "d": (xb.mean() - xa.mean()) / sp if sp > 0 else np.nan,
                         "overlap": bool(min(xa.max(), xb.max()) >= max(xa.min(), xb.min()))})
    return pd.DataFrame(rows)


def trial_matrix(f: pd.DataFrame):
    """Trials x (channel, freq) matrix of log amplitude, plus each trial's condition."""
    m = f.assign(lv=np.log(f["value"].clip(lower=1e-9))).pivot_table(
        index="trial_id", columns=["channel", "freq_hz"], values="lv")
    m = m.reindex(columns=sorted(m.columns, key=lambda c: (CHANNELS.index(c[0]), c[1])))
    m = m.dropna(axis=1)
    cond = f.drop_duplicates("trial_id").set_index("trial_id").loc[m.index, "condition"]
    return m, cond


def leave_one_out_accuracy(m: pd.DataFrame, cond: pd.Series):
    """Nearest-centroid classifier, each trial held out in turn. A rough 'can repeats be told
    apart from other conditions' number, not a model. Returns (accuracy, chance, n_scored)."""
    x = m.to_numpy()
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    labels = cond.to_numpy()
    hits = scored = 0
    for i in range(len(x)):
        mask = np.arange(len(x)) != i
        cands = [c for c in np.unique(labels[mask]) if (labels[mask] == c).sum() >= 1]
        if labels[i] not in cands or (labels[mask] == labels[i]).sum() < 1:
            continue
        cents = np.array([x[mask & (labels == c)].mean(0) for c in cands])
        pred = cands[int(np.argmin(((cents - x[i]) ** 2).sum(1)))]
        hits += pred == labels[i]
        scored += 1
    k = len(np.unique(labels))
    return (hits / scored if scored else np.nan), 1 / k if k else np.nan, scored


def pca(m: pd.DataFrame, n=2):
    x = m.to_numpy()
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    return u[:, :n] * s[:n], var[:n]
