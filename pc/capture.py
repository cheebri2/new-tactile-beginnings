"""Run repeated sweep trials for one test condition and save raw data + features.

Examples
  python capture.py --port COM3 --drive-port COM4 --test blank --depth 15 --repeats 5
  python capture.py --port COM3 --drive-port COM4 --test object --object vertebra --depth 15 --repeats 5
  python capture.py --port COM3 --drive-port COM4 --test offset --object vertebra --x 5 --depth 15
  python capture.py --simulate --test object --object vertebra --repeats 3     # no hardware

Between repeats it waits for Enter so you can lift and reseat the sensor head
(that reseat is what makes the repeat spread honest). Type s to skip saving the
last trial or q to stop.
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from piezo import CHANNELS, stream, store
from piezo.features import extract
from piezo.schedule import Sweep
from piezo import simulate

TEST_TYPES = ["blank", "object", "offset", "depth", "rotation", "load", "noise", "other"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group("hardware")
    io.add_argument("--port", help="sensing ESP32 serial port, e.g. COM3")
    io.add_argument("--drive-port", help="drive ESP32 port. Without it, press BOOT on the drive board when asked")
    io.add_argument("--rate", type=int, default=2000, help="sample rate in Hz (default 2000)")
    io.add_argument("--simulate", action="store_true", help="fake data, no hardware")

    m = p.add_argument_group("test metadata (saved with every trial)")
    m.add_argument("--test", required=True, choices=TEST_TYPES, help="what kind of test this is")
    m.add_argument("--object", default="none", help="object under the skin, e.g. vertebra, disc, none")
    m.add_argument("--x", type=float, default=0.0, help="object offset left/right from centre, mm (+ = right)")
    m.add_argument("--y", type=float, default=0.0, help="object offset up/down from centre, mm (+ = up)")
    m.add_argument("--depth", type=float, help="skin thickness over the object, mm")
    m.add_argument("--rotation", type=float, default=0.0, help="sensor head rotation, degrees")
    m.add_argument("--load", type=float, help="weight/preload on the sensor head, g")
    m.add_argument("--session", default=dt.date.today().isoformat(), help="session name (default today)")
    m.add_argument("--notes", default="", help="free text saved with every trial")
    m.add_argument("--calibration", default="", help="calibration file used, if any")

    s = p.add_argument_group("sweep")
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--f-start", type=float, default=Sweep.f_start)
    s.add_argument("--f-stop", type=float, default=Sweep.f_stop)
    s.add_argument("--f-step", type=float, default=Sweep.f_step)
    s.add_argument("--tone-ms", type=int, default=Sweep.tone_ms)
    s.add_argument("--gap-ms", type=int, default=Sweep.gap_ms)
    s.add_argument("--amp", type=int, default=Sweep.amp, help="drive amplitude, DAC counts 0..127")
    p.add_argument("--data", default="data", help="data folder (default ./data)")
    return p.parse_args()


def record_hardware(sense, drive, sweep, rate):
    """One sweep. Returns (t_us, adc, stream stats)."""
    s = stream.SenseStream(sense)
    s.start()
    time.sleep(0.5)  # some samples before the sweep
    if drive is not None:
        drive.reset_input_buffer()
        drive.write(b"sweep\n")
        end = time.time() + sweep.duration_s + 5
        while time.time() < end:
            line = drive.readline().decode(errors="ignore").strip()
            if line.startswith("SWEEP_END") or line.startswith("SWEEP_ABORTED"):
                break
        else:
            print("  warning: drive board never said SWEEP_END")
        time.sleep(0.5)
    else:
        print(f"  >>> press BOOT on the drive board now (recording {sweep.duration_s + 4:.0f} s)")
        time.sleep(sweep.duration_s + 4)
    raw = s.stop()
    t_us, adc, stats = stream.parse_blocks(raw)
    if len(t_us) < rate:
        sys.exit("No samples from the sensing board. Check the port and that firmware/sense is flashed.")
    return t_us, adc, stats


def main():
    a = parse_args()
    data = Path(a.data)
    sweep = Sweep(f_start=a.f_start, f_stop=a.f_stop, f_step=a.f_step, tone_ms=a.tone_ms,
                  gap_ms=a.gap_ms, amp=a.amp)
    meta_base = {"test_type": a.test, "object": a.object, "x_mm": a.x, "y_mm": a.y,
                 "depth_mm": a.depth, "rotation_deg": a.rotation, "load_g": a.load,
                 "session": a.session, "notes": a.notes, "calibration": a.calibration,
                 "sweep": sweep.to_dict()}

    sense = drive = None
    sense_info, drive_info = {"rate_hz": a.rate, "adc_read_us": 0.0, "fw": "simulated"}, {}
    if not a.simulate:
        if not a.port:
            sys.exit("--port is required (or use --simulate)")
        sense = stream.open_port(a.port, stream.SENSE_BAUD)
        sense.write(b"x\n")
        time.sleep(0.2)
        sense_info = stream.parse_info(stream.ask(sense, f"r{a.rate}"))
        if a.drive_port:
            drive = stream.open_port(a.drive_port, stream.DRIVE_BAUD)
            drive_info = stream.parse_info(stream.ask(drive, sweep.cfg_command()))
        else:
            print("No --drive-port: the drive board must hold the same sweep settings as its defaults.")
        print(f"Sensing board: {sense_info}")
        if drive_info:
            print(f"Drive board:   {drive_info}")

    label = store.condition_label(meta_base)
    print(f"\nCondition: {label}   ({sweep.duration_s:.1f} s per sweep, {len(sweep.freqs)} frequencies "
          f"{sweep.freqs[0]:g}-{sweep.freqs[-1]:g} Hz)")
    done = 0
    while done < a.repeats:
        repeat = store.next_repeat(data, meta_base)
        cmd = input(f"\nRepeat {repeat} ({done + 1}/{a.repeats}): seat the sensor, then Enter "
                    f"(q to quit) ").strip().lower()
        if cmd == "q":
            break
        if a.simulate:
            t_us, adc = simulate.capture(meta_base, sweep, fs=a.rate)
            stats = {"blocks": 0, "bad_blocks": 0, "lost_blocks": 0, "overruns": 0}
        else:
            t_us, adc, stats = record_hardware(sense, drive, sweep, a.rate)

        now = dt.datetime.now()
        trial_id = store.safe(f"{now:%Y%m%d-%H%M%S}_{a.test}_r{repeat:02d}")
        meta = dict(meta_base, trial_id=trial_id, time=now.isoformat(timespec="seconds"),
                    repeat=repeat, stream=stats, sense_info=sense_info, drive_info=drive_info)
        try:
            rows, info = extract(t_us, adc, sweep, float(sense_info.get("adc_read_us", 0)))
        except ValueError as e:
            print(f"  could not process this trial: {e}. Not saved.")
            continue
        info.update({f"stream_{k}": v for k, v in stats.items()})
        feats = pd.DataFrame(rows)
        snr = feats.groupby("channel")["snr_db"].median()
        print(f"  {len(t_us)} samples at {info['fs_hz']:.0f} Hz, sweep found at {info['sweep_start_s']:.2f} s "
              f"(lock quality {info['align_quality']:.1f})")
        print("  median SNR dB: " + "  ".join(f"{c}={snr[c]:.0f}" for c in CHANNELS))
        print("  peak amp:      " + "  ".join(
            f"{c}={feats[feats.channel == c].amp.max():.1f}@{feats.loc[feats[feats.channel == c].amp.idxmax(), 'freq_hz']:g}Hz"
            for c in CHANNELS))
        if info["align_quality"] < 3 or stats["lost_blocks"] or stats["bad_blocks"]:
            print("  WARNING: weak sweep lock or lost data. Check the drive is on and the skin is touching.")
        if input("  keep this trial? [Y/s=skip] ").strip().lower() == "s":
            continue

        store.save_trial(data, trial_id, t_us, adc, meta)
        feats.insert(0, "trial_id", trial_id)
        store.append_csv(data / "trials.csv", [store.trial_row(meta, info)])
        store.append_csv(data / "features.csv", feats.to_dict("records"))
        print(f"  saved {trial_id}")
        done += 1

    for ser in (sense, drive):
        if ser is not None:
            ser.close()
    if done:
        trials = pd.read_csv(data / "trials.csv")
        feats = pd.read_csv(data / "features.csv")
        store.wide_table(trials, feats).to_csv(data / "dataset_wide.csv", index=False)
    print(f"\n{done} trial(s) saved in {data}/. Compare conditions with: python analyze.py")


if __name__ == "__main__":
    main()
