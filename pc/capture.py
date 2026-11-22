"""Run repeated sweep trials for one test condition and save raw data + features.

Examples
  python capture.py --port COM3 --test blank --depth 15 --repeats 5
  python capture.py --port COM3 --test object --object vertebra --depth 15 --repeats 5
  python capture.py --port COM3 --test offset --object vertebra --x 5 --depth 15
  python capture.py --simulate --test object --object vertebra --repeats 3     # no hardware

Between repeats it waits for Enter so you can lift and reseat the sensor head
(that reseat is what makes the repeat spread honest). Type s to skip saving the
last trial or q to stop.

IMU (phantom tests): for every test except blank, it first asks you to put the head at
the phantom's lower-left corner and zeroes the IMU there (0,0, zero angle). The board
then tracks the head while you move it, and every trial saves the head's angle and
estimated x/y (imu_* columns in trials.csv, raw/<trial>_imu.csv.gz). Slide the head in
one quick move and hold it still before pressing Enter. --no-zero keeps the previous
corner; --imu off skips the IMU.
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from piezo import ACTIVE_CHANNELS, DATA_DIR, imu, stream, store
from piezo.features import extract
from piezo.schedule import Sweep
from piezo import simulate

TEST_TYPES = ["blank", "object", "offset", "depth", "rotation", "noise", "other"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group("hardware")
    io.add_argument("--port", help="ESP32 serial port, e.g. COM3")
    io.add_argument("--rate", type=int, default=2000, help="sample rate in Hz (default 2000)")
    io.add_argument("--simulate", action="store_true", help="fake data, no hardware")

    m = p.add_argument_group("test metadata (saved with every trial)")
    m.add_argument("--test", required=True, choices=TEST_TYPES, help="what kind of test this is")
    m.add_argument("--object", default="none", help="object under the skin, e.g. vertebra, disc, none")
    m.add_argument("--x", type=float, default=0.0, help="object offset left/right from centre, mm (+ = right)")
    m.add_argument("--y", type=float, default=0.0, help="object offset up/down from centre, mm (+ = up)")
    m.add_argument("--depth", type=float, help="skin thickness over the object, mm")
    m.add_argument("--rotation", type=float, default=0.0, help="sensor head rotation, degrees")
    m.add_argument("--session", default=dt.date.today().isoformat(), help="session name (default today)")
    m.add_argument("--notes", default="", help="free text saved with every trial")
    m.add_argument("--calibration", default="", help="calibration file used, if any")

    g = p.add_argument_group("IMU (phantom tests)")
    g.add_argument("--imu", choices=["auto", "on", "off"], default="auto",
                   help="track the head with the BNO055 (auto = every test except blank)")
    g.add_argument("--no-zero", action="store_true",
                   help="keep the lower-left corner from the last run instead of zeroing again")
    g.add_argument("--imu-axes", default="x,y",
                   help="board axes that point to phantom right,up, e.g. 'x,y' or '-y,x' (default x,y)")

    s = p.add_argument_group("sweep")
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--f-start", type=float, default=Sweep.f_start)
    s.add_argument("--f-stop", type=float, default=Sweep.f_stop)
    s.add_argument("--f-step", type=float, default=Sweep.f_step)
    s.add_argument("--tone-ms", type=int, default=Sweep.tone_ms)
    s.add_argument("--gap-ms", type=int, default=Sweep.gap_ms)
    s.add_argument("--amp", type=int, default=Sweep.amp, help="drive amplitude, DAC counts 0..127")
    p.add_argument("--data", default=DATA_DIR, help=f"data folder (default ./{DATA_DIR})")
    return p.parse_args()


def setup_imu(ser, a) -> bool:
    """Check the BNO055 and zero it at the phantom's lower-left corner. False = no IMU."""
    try:
        st = imu.status(ser)
    except TimeoutError:
        print("IMU: the board did not answer 'imu' (old firmware?). Continuing without IMU.")
        return False
    if not st.get("ok"):
        print("IMU: no BNO055 found (check UART2 wiring). Continuing without IMU.")
        return False
    if a.no_zero:
        if not st.get("zeroed"):
            sys.exit("IMU was never zeroed since the board powered up. Run without --no-zero.")
        print(f"IMU: keeping the previous corner, head now at x={st['x_mm']:.0f} y={st['y_mm']:.0f} mm")
        return True
    input("\nIMU zero: put the sensor head on the LOWER-LEFT corner of the phantom, square to its\n"
          "edges (board x axis pointing right along the phantom), hold it still, then Enter ")
    print("  hold still...")
    try:
        reply = imu.zero(ser)
    except TimeoutError:
        reply = "ERR,no reply"
    if reply.startswith("ERR"):
        sys.exit(f"IMU zero failed: {reply}")
    z = stream.parse_info(reply)
    print("  zeroed: this spot is (0, 0).")
    if z.get("gravity_odd"):
        print("  WARNING: the IMU's gravity check looks wrong at zero; x/y may be rotated or mirrored.")
    return True


def show_pose(ser, axes):
    try:
        st = imu.status(ser)
    except TimeoutError:
        return
    x, y = imu.remap(st.get("x_mm", float("nan")), st.get("y_mm", float("nan")), axes)
    note = "  (still moving?)" if st.get("moving") else ""
    if st.get("long_move"):
        note += "  (last move took > 5 s: estimate less reliable)"
    print(f"  IMU: head at x={float(x):.0f} mm, y={float(y):.0f} mm from the corner, "
          f"yaw {st.get('yaw_deg', 0):+.1f} deg{note}")


def report_imu(s):
    if not s.get("imu_samples"):
        print("  IMU: no samples received during the sweep.")
        return
    print(f"  IMU: x={s['imu_x_mm']:.0f} mm y={s['imu_y_mm']:.0f} mm  yaw {s['imu_yaw_deg']:+.1f} "
          f"pitch {s['imu_pitch_deg']:+.1f} roll {s['imu_roll_deg']:+.1f} deg  "
          f"(calib gyr {s['imu_calib_gyr']}/3 acc {s['imu_calib_acc']}/3)")
    if s["imu_turn_during_sweep_deg"] > 2:
        print(f"  WARNING: the head turned {s['imu_turn_during_sweep_deg']:.1f} deg during the sweep.")
    if s["imu_stale"]:
        print("  WARNING: IMU data stopped during the sweep.")


def main():
    a = parse_args()
    data = Path(a.data)
    sweep = Sweep(f_start=a.f_start, f_stop=a.f_stop, f_step=a.f_step, tone_ms=a.tone_ms,
                  gap_ms=a.gap_ms, amp=a.amp)
    meta_base = {"test_type": a.test, "object": a.object, "x_mm": a.x, "y_mm": a.y,
                 "depth_mm": a.depth, "rotation_deg": a.rotation,
                 "session": a.session, "notes": a.notes, "calibration": a.calibration,
                 "sweep": sweep.to_dict()}

    ser = None
    board_info = {"rate_hz": a.rate, "adc_read_us": 0.0, "fw": "simulated"}
    if not a.simulate:
        if not a.port:
            sys.exit("--port is required (or use --simulate)")
        ser = stream.open_port(a.port, stream.BAUD)
        ser.write(b"off\nx\n")
        time.sleep(0.3)
        stream.ask(ser, f"r{a.rate}")
        reply = stream.ask(ser, sweep.cfg_command())
        if reply.startswith("ERR"):
            sys.exit(f"Board rejected the sweep settings: {reply}")
        board_info = stream.parse_info(reply)
        print(f"Board: {board_info}")

    use_imu = not a.simulate and (a.imu == "on" or (a.imu == "auto" and a.test != "blank"))
    if use_imu:
        use_imu = setup_imu(ser, a)
        meta_base["imu_axes"] = a.imu_axes

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
        imu_samples = None
        if a.simulate:
            t_us, adc, events = simulate.capture(meta_base, sweep, fs=a.rate)
            stats = {"blocks": 0, "bad_blocks": 0, "lost_blocks": 0, "overruns": 0}
        else:
            if use_imu:
                show_pose(ser, a.imu_axes)
                imu.mark(ser, repeat)
            print(f"  sweeping ({sweep.duration_s:.0f} s)...")
            if use_imu:
                t_us, adc, events, stats, imu_samples = stream.record_sweep(ser, sweep.duration_s, with_imu=True)
            else:
                t_us, adc, events, stats = stream.record_sweep(ser, sweep.duration_s)
            if len(t_us) < a.rate:
                sys.exit("No samples from the board. Check the port and that firmware/esp32 is flashed.")

        now = dt.datetime.now()
        trial_id = store.safe(f"{now:%Y%m%d-%H%M%S}_{a.test}_r{repeat:02d}")
        meta = dict(meta_base, trial_id=trial_id, time=now.isoformat(timespec="seconds"),
                    repeat=repeat, stream=stats, board_info=board_info, events=events)
        try:
            rows, info = extract(t_us, adc, sweep, float(board_info.get("adc_read_us", 0)), events)
        except ValueError as e:
            print(f"  could not process this trial: {e}. Not saved.")
            continue
        info.update({f"stream_{k}": v for k, v in stats.items()})
        if use_imu:
            meta["imu_summary"] = imu.summarize(imu_samples, events, a.imu_axes)
            report_imu(meta["imu_summary"])
        feats = pd.DataFrame(rows)
        snr = feats.groupby("channel")["snr_db"].median()
        print(f"  {len(t_us)} samples at {info['fs_hz']:.0f} Hz, sweep started at {info['sweep_start_s']:.2f} s")
        print("  median SNR dB: " + "  ".join(f"{c}={snr[c]:.0f}" for c in ACTIVE_CHANNELS))
        print("  peak amp:      " + "  ".join(
            f"{c}={feats[feats.channel == c].amp.max():.1f}@{feats.loc[feats[feats.channel == c].amp.idxmax(), 'freq_hz']:g}Hz"
            for c in ACTIVE_CHANNELS))
        if info["timing"] != "events" or stats["lost_blocks"] or stats["bad_blocks"]:
            print("  WARNING: missing sweep events or lost data (USB cable/port?).")
        if snr.min() < 10:
            print("  WARNING: a disc is under 10 dB SNR. Check the amp is on and every disc touches the skin.")
        if input("  keep this trial? [Y/s=skip] ").strip().lower() == "s":
            continue

        store.save_trial(data, trial_id, t_us, adc, meta)
        if imu_samples is not None and len(imu_samples):
            store.save_imu(data, trial_id, imu.to_frame(imu_samples, a.imu_axes))
        feats.insert(0, "trial_id", trial_id)
        store.append_csv(data / "trials.csv", [store.trial_row(meta, info)])
        store.append_csv(data / "features.csv", feats.to_dict("records"))
        print(f"  saved {trial_id}")
        done += 1

    if ser is not None:
        ser.close()
    if done:
        trials = pd.read_csv(data / "trials.csv")
        feats = pd.read_csv(data / "features.csv")
        store.wide_table(trials, feats).to_csv(data / "dataset_wide.csv", index=False)
    print(f"\n{done} trial(s) saved in {data}/. Compare conditions with: python analyze.py")


if __name__ == "__main__":
    main()
