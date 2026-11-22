"""IMU stream format, flash log format, and the drift-corrected tracker on synthetic moves."""

import math

import numpy as np

from piezo import imu, stream


def test_imu_blocks_round_trip_alongside_piezo_data():
    t = np.arange(400, dtype=np.int64) * 500 + (2**32 - 50_000)  # crosses the micros() wrap
    adc = np.random.default_rng(2).integers(0, 4096, (400, 4)).astype(np.uint16)
    data = stream.encode_blocks(t, adc)
    imu_bytes = b"".join(stream.encode_imu(t[k], yaw=0.5 * k, x=k * 1.5, y=-k, flags=1, calib=0b11110000)
                         for k in range(0, 400, 20))
    half = len(data) // 2
    cut = data.find(b"\xA5\x5A", half)
    buf = data[:cut] + imu_bytes + data[cut:]

    t2, adc2, _, st = stream.parse_blocks(buf)  # old parser still sees all the piezo data
    assert np.array_equal(adc2, adc) and st["bad_blocks"] == 0
    rec = stream.parse_imu(buf, t2[0])
    assert len(rec) == 20
    assert np.array_equal(rec["t_us"], t2[::20])
    assert np.allclose(rec["x_mm"], np.arange(0, 400, 20) * 1.5)
    assert np.allclose(rec["yaw_deg"], np.arange(0, 400, 20) * 0.5)
    assert rec["calib"][0] == 0b11110000


def test_summary_uses_the_sweep_window_and_axes():
    rec = np.zeros(10, stream.IMU_DTYPE)
    rec["t_us"] = np.arange(10) * 10_000
    rec["x_mm"], rec["y_mm"] = 50.0, 20.0
    rec["x_mm"][0] = 999  # before the sweep: ignored
    rec["flags"] = imu.FLAG_ZEROED
    events = [{"event": "sweep_start", "t_us": 5_000}, {"event": "sweep_end", "t_us": 95_000}]
    s = imu.summarize(rec, events, axes="-y,x")
    assert s["imu_x_mm"] == -20.0 and s["imu_y_mm"] == 50.0
    assert s["imu_move_during_sweep_mm"] == 0 and s["imu_zeroed"]


def test_flash_log_round_trip():
    recs = [{"type": "boot", "t_ms": 0}, {"type": "zero", "t_ms": 1000},
            {"type": "imu", "t_ms": 1040, "x_mm": 12.5, "y_mm": -3.0, "yaw_deg": 1.25, "flags": 3},
            {"type": "mark", "t_ms": 1100, "value": 7.0}, {"type": "boot", "t_ms": 0}]
    buf = imu.encode_log(recs) + b"\xFF" * 64
    df = imu.decode_log(buf)
    assert list(df["type"]) == ["boot", "zero", "imu", "mark", "boot"]
    assert list(df["session"]) == [1, 1, 1, 1, 2]
    assert df.loc[2, "x_mm"] == 12.5 and df.loc[2, "moving"] and df.loc[3, "value"] == 7.0
    bad = bytearray(buf)
    bad[2 * 32 + 5] ^= 0xFF
    assert len(imu.decode_log(bytes(bad))) == 4 and imu.decode_log(bytes(bad)).attrs["bad_records"] == 1


# ---------------------------------------------------------------- tracker on synthetic moves

def min_jerk(t, T):
    s = np.clip(t / T, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def simulate_session(seed=0, world_yaw_deg=25.0, turn_deg=15.0, bias=(0.04, -0.03, 0.02),
                     noise=0.03, move_s=1.0, vib=0.0):
    """Head starts at the corner (board squared to the phantom), then: slide +60 mm x, sweep
    (held still, drive on), slide +40 mm y while turning the head by turn_deg, sweep,
    slide -30 mm x / +10 mm y, sweep. Returns truth (phantom frame, mm) and tracker output.

    BNO055-like errors: constant accelerometer bias (sensor frame), white noise, gyro noise.
    world_yaw_deg is the arbitrary heading the IMU-mode fusion starts with."""
    rng = np.random.default_rng(seed)
    fs = 100.0
    plan = [("still", 2.0, None, 0), ("move", move_s, (60, 0), 0), ("sweep", 19.0, None, 0),
            ("move", move_s * 0.8, (0, 40), turn_deg), ("sweep", 19.0, None, 0),
            ("move", move_s * 1.2, (-30, 10), 0), ("sweep", 19.0, None, 0), ("still", 1.0, None, 0)]
    psi_w = math.radians(world_yaw_deg)
    tr = imu.Tracker()
    tr.request_zero()
    pos, theta = np.zeros(2), 0.0
    truth, est = [], []
    t_us = 0
    for kind, dur, delta, turn in plan:
        n = int(dur * fs)
        tt = np.arange(n) / fs
        if kind == "move":
            d = np.asarray(delta, float) / 1000.0
            s = min_jerk(tt, dur)
            sdd = np.gradient(np.gradient(s, 1 / fs), 1 / fs)
            sd = np.gradient(s, 1 / fs)
        for k in range(n):
            if kind == "move":
                p, acc_p = pos + d * s[k], d * sdd[k]
                th = theta + math.radians(turn) * s[k]
                yaw_rate = math.degrees(math.radians(turn) * sd[k])
            else:
                p, acc_p, th, yaw_rate = pos, np.zeros(2), theta, 0.0
            yaw = psi_w + th
            q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            c, sn = math.cos(-th), math.sin(-th)  # phantom frame -> sensor frame
            a_s = np.array([c * acc_p[0] - sn * acc_p[1], sn * acc_p[0] + c * acc_p[1], 0.0])
            a_s += np.asarray(bias) + rng.normal(0, noise, 3)
            if kind == "sweep" and vib:
                a_s += vib * np.sin(2 * np.pi * 37.0 * k / fs) * np.array([1, 0.5, 0.2])
            gyro = rng.normal(0, 0.1, 3)
            gyro[2] += yaw_rate + (8.0 if kind == "move" else 0.0)  # hands wobble while sliding
            out = tr.update(t_us, q, a_s, [0, 0, 9.81], gyro, drive_on=(kind == "sweep"))
            truth.append(p * 1000)
            est.append((out["x_mm"], out["y_mm"], kind, out["yaw_deg"]))
            t_us += 10_000
        if kind == "move":
            pos, theta = pos + d, theta + math.radians(turn)
    return np.array(truth), est


def sweep_errors(truth, est):
    """Error (mm) at the end of each sweep, where capture.py reads the position."""
    errs = []
    for i in range(1, len(est)):
        if est[i - 1][2] == "sweep" and est[i][2] != "sweep":
            errs.append(math.hypot(est[i - 1][0] - truth[i - 1][0], est[i - 1][1] - truth[i - 1][1]))
    return errs


def test_tracker_recovers_slides_within_a_few_mm():
    truth, est = simulate_session()
    errs = sweep_errors(truth, est)
    assert len(errs) == 3
    assert max(errs) < 5.0, errs


def test_tracker_holds_position_through_vibration_during_sweeps():
    truth, est = simulate_session(vib=0.8)  # strong 37 Hz shake while the amp plays
    assert max(sweep_errors(truth, est)) < 5.0


def test_slow_moves_are_worse_and_naive_integration_is_useless():
    _, est_fast = simulate_session(move_s=1.0)
    truth, est_slow = simulate_session(move_s=4.0)
    fast, slow = max(sweep_errors(truth, est_fast)), max(sweep_errors(truth, est_slow))
    assert slow > fast
    # Plain double integration of the same biased accelerometer over the ~65 s session:
    t = 65.0
    assert 0.5 * 0.04 * t**2 * 1000 > 50_000  # tens of metres


def test_angles_are_relative_to_the_zero_pose():
    tr = imu.Tracker()
    tr.request_zero()
    q0 = np.array([math.cos(0.2), 0, 0, math.sin(0.2)])
    for k in range(30):
        out = tr.update(k * 10_000, q0, [0, 0, 0], [0, 0, 9.81], [0, 0, 0])
    assert out["zeroed"] and abs(out["yaw_deg"]) < 1e-6
    turn = math.radians(10) / 2
    q1 = imu._mul(q0, np.array([math.cos(turn), 0, 0, math.sin(turn)]))
    out = tr.update(31 * 10_000, q1, [0, 0, 0], [0, 0, 9.81], [0, 0, 0])
    assert abs(out["yaw_deg"] - 10) < 1e-3


def test_zero_works_with_large_uncalibrated_offsets():
    """Regression: an uncalibrated BNO055 at rest reads e.g. 0.4 m/s^2 and 2.5 deg/s. That is
    not motion; the board used to wait for |acc| < 0.25 forever and report 'never held still'."""
    rng = np.random.default_rng(3)
    tr = imu.Tracker()
    tr.request_zero()
    q = np.array([1.0, 0, 0, 0])
    for k in range(300):  # 3 s at rest
        lia = np.array([0.35, -0.2, 0.1]) + rng.normal(0, 0.03, 3)
        gyro = np.array([2.5, -1.0, 0.8]) + rng.normal(0, 0.1, 3)
        out = tr.update(k * 10_000, q, lia, [0, 0, 9.81], gyro)
        if out["zeroed"]:
            break
    assert out["zeroed"], (tr.acc_dev, tr.gyro_dev)
    assert k < 150  # within about 1.5 s


def test_tracker_with_offsets_still_tracks_slides():
    truth, est = simulate_session(bias=(0.35, -0.2, 0.1))
    assert max(sweep_errors(truth, est)) < 5.0

