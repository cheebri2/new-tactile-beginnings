"""Rebuild trials.csv, features.csv and dataset_wide.csv from the raw files.

Run this after changing the feature code, or after editing a trial's .json
(e.g. fixing a wrong offset in the metadata).

  python process.py            # uses ./data
  python process.py --data D:/piezo/data
"""

import argparse
from pathlib import Path

import pandas as pd

from piezo import store
from piezo.features import extract
from piezo.schedule import Sweep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    data = Path(p.parse_args().data)

    trial_rows, feat_rows = [], []
    ids = store.list_trials(data)
    for i, tid in enumerate(ids, 1):
        t_us, adc, meta = store.load_trial(data, tid)
        sweep = Sweep.from_dict(meta["sweep"])
        adc_read_us = float(meta.get("sense_info", {}).get("adc_read_us", 0))
        try:
            rows, info = extract(t_us, adc, sweep, adc_read_us)
        except ValueError as e:
            print(f"skip {tid}: {e}")
            continue
        info.update({f"stream_{k}": v for k, v in meta.get("stream", {}).items()})
        trial_rows.append(store.trial_row(meta, info))
        feat_rows += [dict(trial_id=tid, **r) for r in rows]
        print(f"[{i}/{len(ids)}] {tid}  lock quality {info['align_quality']:.1f}")

    trials, feats = pd.DataFrame(trial_rows), pd.DataFrame(feat_rows)
    trials.to_csv(data / "trials.csv", index=False)
    feats.to_csv(data / "features.csv", index=False)
    store.wide_table(trials, feats).to_csv(data / "dataset_wide.csv", index=False)
    print(f"wrote {len(trials)} trials to {data}/trials.csv, features.csv, dataset_wide.csv")


if __name__ == "__main__":
    main()
