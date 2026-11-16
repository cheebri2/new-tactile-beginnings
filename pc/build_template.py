"""Build the centred-vertebra template that align_test.py compares against.

Needs a data folder recorded as blank blocks, each followed by an object block at one x
offset (the 2026-11-15 layout). Every object trial is compared with the blank block
just before it.

  python build_template.py --data data --out models/centred_template.json
  python build_template.py --data data --session 2026-11-15 --where "depth_mm == 15"
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from piezo import align


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="models/centred_template.json")
    p.add_argument("--session", nargs="*", help="only these sessions")
    p.add_argument("--where", help="pandas query on trials.csv, e.g. \"depth_mm == 15\"")
    p.add_argument("--channels", nargs="+", default=list(align.DEFAULT_CHANNELS))
    p.add_argument("--bound", type=float, default=align.DEFAULT_BOUND)
    a = p.parse_args()

    data = Path(a.data)
    trials = pd.read_csv(data / "trials.csv")
    feats = pd.read_csv(data / "features.csv")
    if a.session:
        trials = trials[trials.session.astype(str).isin(a.session)]
    if a.where:
        trials = trials.query(a.where)
    feats = feats[feats.trial_id.isin(trials.trial_id)]

    tpl = align.build_template(trials, feats, a.channels, a.bound,
                               source=f"{data.resolve().name}: {len(trials)} trials, "
                                      f"sessions {sorted(trials.session.astype(str).unique())}")
    meta = json.loads((data / "raw" / f"{trials.trial_id.iloc[0]}.json").read_text())
    tpl["sweep"] = meta["sweep"]
    tpl["depth_mm"] = meta.get("depth_mm")
    align.save_template(tpl, a.out)

    print(f"template -> {a.out}")
    print(f"  channels {tpl['channels']}, {len(tpl['freqs'])} freqs "
          f"{tpl['freqs'][0]:g}-{tpl['freqs'][-1]:g} Hz, {tpl['n_blank_blocks']} blank blocks, bound {tpl['bound']}")
    print("  training trials vs centred template (in-sample, so optimistic):")
    for x, d in tpl["training_distance"].items():
        flag = ["aligned" if v <= tpl["bound"] else "not" for v in d]
        print(f"    x={float(x):+5.1f} mm  n={tpl['positions'][x]['n']}  distance "
              + "  ".join(f"{v:.2f}({f[0]})" for v, f in zip(d, flag)))


if __name__ == "__main__":
    main()
