"""Live alignment test: calibrate on a blank, then call each placement aligned or not.

  python align_test.py --port COM3                         # live, uses the shipped template
  python align_test.py --port COM3 --plan -10 -5 0 5 10 --reps 2   # tells you where to place it, random order
  python align_test.py --simulate                          # no hardware (fake data, checks the flow only)
  python align_test.py --replay data                       # re-run the method on saved trials (calibration runs)

Steps:
1. Baseline: the sensor head on blank silicone (no vertebra), same pressure you will use for
   the test. --baseline-sweeps sweeps (default 3) are averaged. Type b at any prompt to redo it.
2. Test: place the head over the vertebra, type the true x offset (mm, + = right) so the trial
   is labelled, press Enter. After the sweep (~37 s) it prints:
     ALIGNED / NOT ALIGNED, the distance from the centred template vs the bound, and the
     closest trained offset as a rough x estimate.
3. Every sweep (baselines too) is saved like capture.py does (to training/data by default), with the true x and the
   prediction, so this run becomes new training data. predictions.csv gets one line per test.

The template (models/centred_template_2026-11-15.json) was learned from one session of
25 vertebra trials, so treat the answers as a first check, not a validated classifier.
"""

import argparse
import datetime as dt
import random
import sys
import time
from pathlib import Path

import pandas as pd

from piezo import ACTIVE_CHANNELS, DATA_DIR, align, simulate, store, stream
from piezo.features import extract
from piezo.schedule import Sweep

HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HERE / "models" / "centred_template_2026-11-15.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", help="ESP32 serial port, e.g. COM3")
    p.add_argument("--simulate", action="store_true", help="fake data, no hardware")
    p.add_argument("--replay", metavar="DATA", help="re-run the method on a saved data folder")
    p.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    p.add_argument("--bound", type=float, help="override the template's error bound")
    p.add_argument("--baseline-sweeps", type=int, default=3)
    p.add_argument("--plan", type=float, nargs="+", help="x offsets to test, shuffled for you")
    p.add_argument("--reps", type=int, default=1, help="repeats of each --plan offset")
    p.add_argument("--depth", type=float, help="skin thickness, mm (default: the template's)")
    p.add_argument("--session", default=dt.date.today().isoformat())
    p.add_argument("--notes", default="")
    p.add_argument("--data", default=DATA_DIR, help=f"where trials are saved (default ./{DATA_DIR})")
    p.add_argument("--rate", type=int, default=2000)
    return p.parse_args()


def show(pred, true_x=None):
    verdict = "ALIGNED" if pred["aligned"] else "NOT ALIGNED"
    print(f"\n  >>> {verdict}   distance {pred['distance']:.2f} (bound {pred['bound']:.2f})")
    print(f"      rough position: {pred['side']}, about x = {pred['est_x']:+.1f} mm "
          f"(closest trained offset {pred['nearest_x']:+g} mm)")
    print("      distance to each trained offset: "
          + "  ".join(f"{x:+g}:{d:.2f}" for x, d in sorted(pred["dists"].items())))
    if true_x is not None and true_x == true_x:
        truth = abs(true_x) < 2.5
        ok = "correct" if truth == pred["aligned"] else "WRONG"
        print(f"      truth x = {true_x:+g} mm -> {ok}")


def summary(rows):
    df = pd.DataFrame(rows)
    df = df[df.true_x.notna()]
    if df.empty:
        return
    df["truth_aligned"] = df.true_x.abs() < 2.5
    print("\nSummary (aligned = |x| < 2.5 mm):")
    for x, g in df.groupby("true_x"):
        print(f"  x={x:+5g} mm  n={len(g)}  called aligned {int(g.aligned.sum())}/{len(g)}  "
              f"distance {g.distance.min():.2f}-{g.distance.max():.2f}  est x {g.est_x.mean():+.1f}")
    acc = (df.truth_aligned == df.aligned).mean()
    print(f"  overall {acc * 100:.0f}% correct over {len(df)} tests")


# ---------------------------------------------------------------- replay

def replay(a, tpl):
    data = Path(a.replay)
    trials = pd.read_csv(data / "trials.csv").sort_values("time")
    base, last_blank, rows = None, False, []
    for _, r in trials.iterrows():
        t_us, adc, meta = store.load_trial(data, r.trial_id)
        sweep = Sweep.from_dict(meta["sweep"])
        feats, _ = extract(t_us, adc, sweep, float(meta.get("board_info", {}).get("adc_read_us", 0)),
                           meta.get("events"))
        feats = pd.DataFrame(feats)
        if r.test_type == "blank":
            if base is None or not last_blank:
                base = align.Baseline(tpl["channels"], tpl["freqs"])
            base.add(feats, r.trial_id)
            last_blank = True
            continue
        last_blank = False
        if base is None:
            continue
        pred = align.predict(*base.change(feats), tpl, a.bound)
        print(f"{r.trial_id}  true x {r.x_mm:+g}  -> {'ALIGNED' if pred['aligned'] else 'not'}  "
              f"d={pred['distance']:.2f}  est x {pred['est_x']:+.1f}")
        rows.append(dict(trial_id=r.trial_id, true_x=r.x_mm, **{k: v for k, v in pred.items() if k != "dists"}))
    summary(rows)
    print("\nNote: replaying the trials the template was built from is in-sample, so this is optimistic.")


# ---------------------------------------------------------------- live

class Rig:
    def __init__(self, a, sweep):
        self.a, self.sweep, self.ser = a, sweep, None
        self.board_info = {"rate_hz": a.rate, "adc_read_us": 0.0, "fw": "simulated"}
        if a.simulate:
            return
        if not a.port:
            sys.exit("--port is required (or use --simulate / --replay)")
        self.ser = stream.open_port(a.port, stream.BAUD)
        self.ser.write(b"off\nx\n")
        time.sleep(0.3)
        stream.ask(self.ser, f"r{a.rate}")
        reply = stream.ask(self.ser, sweep.cfg_command())
        if reply.startswith("ERR"):
            sys.exit(f"Board rejected the template's sweep settings: {reply}")
        self.board_info = stream.parse_info(reply)
        print(f"Board: {self.board_info}")

    def sweep_once(self, meta):
        if self.a.simulate:
            t_us, adc, events = simulate.capture(meta, self.sweep, fs=self.a.rate)
            stats = {"blocks": 0, "bad_blocks": 0, "lost_blocks": 0, "overruns": 0}
        else:
            print(f"  sweeping ({self.sweep.duration_s:.0f} s), keep still...")
            t_us, adc, events, stats = stream.record_sweep(self.ser, self.sweep.duration_s)
            if len(t_us) < self.a.rate:
                sys.exit("No samples from the board. Check the port and that firmware/esp32 is flashed.")
        rows, info = extract(t_us, adc, self.sweep, float(self.board_info.get("adc_read_us", 0)), events)
        info.update({f"stream_{k}": v for k, v in stats.items()})
        feats = pd.DataFrame(rows)
        snr = feats.groupby("channel")["snr_db"].median()
        print("  median SNR dB: " + "  ".join(f"{c}={snr[c]:.0f}" for c in ACTIVE_CHANNELS))
        if snr[list(align.DEFAULT_CHANNELS)].min() < 10:
            print("  WARNING: up-left or down-left under 10 dB SNR. Check contact and that the amp is on.")
        if info["timing"] != "events" or stats["lost_blocks"] or stats["bad_blocks"]:
            print("  WARNING: missing sweep events or lost data (USB cable/port?).")
        return t_us, adc, events, stats, info, feats

    def save(self, meta, t_us, adc, events, stats, info, feats):
        data = Path(self.a.data)
        now = dt.datetime.now()
        repeat = store.next_repeat(data, meta)
        trial_id = store.safe(f"{now:%Y%m%d-%H%M%S}_{meta['test_type']}_r{repeat:02d}")
        meta = dict(meta, trial_id=trial_id, time=now.isoformat(timespec="seconds"), repeat=repeat,
                    stream=stats, board_info=self.board_info, events=events)
        store.save_trial(data, trial_id, t_us, adc, meta)
        f = feats.copy()
        f.insert(0, "trial_id", trial_id)
        store.append_csv(data / "trials.csv", [store.trial_row(meta, info)])
        store.append_csv(data / "features.csv", f.to_dict("records"))
        return trial_id


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"


def take_baseline(rig, tpl, meta_base):
    base = align.Baseline(tpl["channels"], tpl["freqs"])
    n = rig.a.baseline_sweeps
    print(f"\n=== Baseline: {n} sweep(s) on BLANK silicone (no vertebra), normal pressure ===")
    while len(base) < n:
        cmd = ask(f"\nBaseline {len(base) + 1}/{n}: seat the head on the blank, then Enter (q to quit) ")
        if cmd == "q":
            return None
        meta = dict(meta_base, test_type="blank", object="none", x_mm=0.0,
                    notes=(meta_base["notes"] + " align_test baseline").strip())
        t_us, adc, events, stats, info, feats = rig.sweep_once(meta)
        tid = rig.save(meta, t_us, adc, events, stats, info, feats)
        base.add(feats, tid)
        print(f"  saved {tid}")
    if len(base) > 1:
        print(f"\n  baseline spread {base.spread():.3f} log10 (typical in training: ~0.03; "
              f"above ~0.1 means the seating moved, consider b to redo it)")
    return base


def live(a, tpl):
    sweep = Sweep.from_dict(tpl["sweep"])
    rig = Rig(a, sweep)
    meta_base = {"object": "vertebra", "y_mm": 0.0, "depth_mm": a.depth or tpl.get("depth_mm"),
                 "rotation_deg": 0.0, "session": a.session, "notes": a.notes,
                 "calibration": Path(a.template).name, "sweep": sweep.to_dict()}
    plan = [x for x in (a.plan or []) for _ in range(a.reps)]
    random.shuffle(plan)
    print(f"Template: {Path(a.template).name} ({tpl['source']})")
    print(f"Sweep {sweep.freqs[0]:g}-{sweep.freqs[-1]:g} Hz step {sweep.f_step:g}, "
          f"{sweep.duration_s:.0f} s each. Bound {a.bound or tpl['bound']}.")

    base = take_baseline(rig, tpl, meta_base)
    rows = []
    while base is not None:
        nxt = plan[0] if plan else None
        hint = f"place it at x = {nxt:+g} mm, Enter (or type the actual x)" if nxt is not None \
            else "type the true x in mm (e.g. 0, -5, 7.5), or ? if unknown"
        cmd = ask(f"\n=== Test {len(rows) + 1}{f' ({len(plan)} left in plan)' if plan else ''}: "
                  f"vertebra under the head; {hint}. b = redo baseline, q = quit\n> ")
        if cmd == "q":
            break
        if cmd == "b":
            base = take_baseline(rig, tpl, meta_base)
            continue
        if cmd == "" and nxt is not None:
            true_x = nxt
        elif cmd in ("?", ""):
            true_x = float("nan")
        else:
            try:
                true_x = float(cmd)
            except ValueError:
                print("  not a number; try again")
                continue
        if plan and true_x == nxt:
            plan.pop(0)
        known = true_x == true_x
        meta = dict(meta_base, x_mm=true_x if known else None,
                    test_type=("object" if known and true_x == 0 else "offset" if known else "other"),
                    notes=(meta_base["notes"] + " align_test").strip())
        t_us, adc, events, stats, info, feats = rig.sweep_once(meta)
        pred = align.predict(*base.change(feats), tpl, a.bound)
        show(pred, true_x)
        meta["prediction"] = {k: v for k, v in pred.items() if k != "dists"}
        meta["prediction"]["dists"] = {f"{k:g}": v for k, v in pred["dists"].items()}
        meta["baseline_trials"] = base.trial_ids
        if ask("  keep this trial? [Y/s=skip] ") == "s":
            continue
        tid = rig.save(meta, t_us, adc, events, stats, info, feats)
        row = dict(trial_id=tid, time=dt.datetime.now().isoformat(timespec="seconds"), true_x=true_x,
                   baseline=";".join(base.trial_ids),
                   **{k: v for k, v in pred.items() if k != "dists"})
        rows.append(row)
        store.append_csv(Path(a.data) / "predictions.csv", [row])
        print(f"  saved {tid}")
        if a.plan and not plan:
            print("\nPlan finished.")
            break

    if rig.ser is not None:
        rig.ser.close()
    summary(rows)
    data = Path(a.data)
    if (data / "trials.csv").exists():
        store.wide_table(pd.read_csv(data / "trials.csv"), pd.read_csv(data / "features.csv")) \
            .to_csv(data / "dataset_wide.csv", index=False)
    print(f"\nTrials saved in {data}/, predictions in {data}/predictions.csv")


def main():
    a = parse_args()
    tpl = align.load_template(a.template)
    if a.replay:
        replay(a, tpl)
    else:
        live(a, tpl)


if __name__ == "__main__":
    main()
