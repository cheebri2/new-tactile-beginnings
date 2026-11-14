"""Where trials live on disk.

data/
  raw/<trial_id>.csv.gz     raw samples: t_us, ur, ul, dl, dr (ADC counts)
  raw/<trial_id>.json       everything about the trial: metadata, sweep, firmware info
  trials.csv                one row per trial (metadata + capture quality)
  features.csv              tidy: one row per trial x channel x frequency
  dataset_wide.csv          one row per trial, every feature as a column (for ML)

The raw files are the source of truth. `python process.py` rebuilds the three CSVs from
them, so the feature method can change without recapturing.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import CHANNELS

META_FIELDS = ["test_type", "object", "x_mm", "y_mm", "depth_mm", "rotation_deg",
               "load_g", "session", "repeat", "notes"]
CONDITION_FIELDS = ["test_type", "object", "x_mm", "y_mm", "depth_mm", "rotation_deg", "load_g"]


def condition_label(meta: dict) -> str:
    """Readable label for a test condition, e.g. 'object vertebra x=5 y=0 d=15'."""
    parts = [str(meta.get("test_type", ""))]
    obj = meta.get("object")
    if obj not in (None, "", "none") and obj != meta.get("test_type"):
        parts.append(str(obj))
    has_object = obj not in (None, "", "none")
    for key, short in [("x_mm", "x"), ("y_mm", "y"), ("depth_mm", "d"),
                       ("rotation_deg", "rot"), ("load_g", "load")]:
        v = meta.get(key)
        if v is None or v != v or v == "":
            continue
        if key in ("x_mm", "y_mm") and not has_object:
            continue  # position means nothing without an object
        if key == "rotation_deg" and float(v) == 0:
            continue
        parts.append(f"{short}={float(v):g}")
    return " ".join(parts)


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")


def save_trial(data_dir: Path, trial_id: str, t_us, adc, meta: dict):
    raw = Path(data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(np.asarray(adc), columns=CHANNELS)
    df.insert(0, "t_us", np.asarray(t_us, np.int64))
    df.to_csv(raw / f"{trial_id}.csv.gz", index=False)
    (raw / f"{trial_id}.json").write_text(json.dumps(meta, indent=2, default=float))


def load_trial(data_dir: Path, trial_id: str):
    raw = Path(data_dir) / "raw"
    meta = json.loads((raw / f"{trial_id}.json").read_text())
    df = pd.read_csv(raw / f"{trial_id}.csv.gz")
    return df["t_us"].to_numpy(), df[CHANNELS].to_numpy(), meta


def list_trials(data_dir: Path):
    return sorted(p.name[:-5] for p in (Path(data_dir) / "raw").glob("*.json"))


def trial_row(meta: dict, info: dict) -> dict:
    row = {"trial_id": meta["trial_id"], "time": meta.get("time"), "condition": condition_label(meta)}
    row.update({k: meta.get(k) for k in META_FIELDS})
    row.update(info)
    row["calibration"] = meta.get("calibration", "")
    return row


def append_csv(path: Path, rows):
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        df = df.drop_duplicates(subset=[c for c in ("trial_id", "channel", "freq_hz") if c in df],
                                keep="last")
    df.to_csv(path, index=False)


def next_repeat(data_dir: Path, meta: dict) -> int:
    """1 + the number of trials already saved for this condition in this session."""
    path = Path(data_dir) / "trials.csv"
    if not path.exists():
        return 1
    df = pd.read_csv(path)
    same = (df["condition"] == condition_label(meta)) & (df["session"].astype(str) == str(meta["session"]))
    return int(same.sum()) + 1


def wide_table(trials: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """One row per trial: metadata, then amp_<ch>_<f>, phase_<ch>_<f>, snr_<ch>_<f>."""
    f = feats.copy()
    f["key"] = f["channel"] + "_" + f["freq_hz"].map(lambda v: f"{v:g}")
    parts = []
    for col, prefix in [("amp", "amp"), ("phase_deg", "phase"), ("snr_db", "snr")]:
        p = f.pivot_table(index="trial_id", columns="key", values=col, sort=False)
        p.columns = [f"{prefix}_{c}" for c in p.columns]
        parts.append(p)
    return trials.set_index("trial_id").join(pd.concat(parts, axis=1)).reset_index()
