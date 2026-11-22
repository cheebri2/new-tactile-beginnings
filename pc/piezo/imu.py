"""BNO055 IMU on the sensor head: angle and estimated x/y on the phantom.

The board does the tracking itself (firmware/esp32/src/tracker.h) because the head moves
between trials, when the laptop is not streaming. `Tracker` below is the same algorithm in
Python, so it can be tested on synthetic moves (tests/test_imu.py) and tuned offline.

Position from an accelerometer drifts fast (a 0.05 m/s^2 bias is 2.5 cm after 1 s, 2.5 m
after 10 s). The tracker only integrates while the head slides, forces velocity to zero
whenever it is still or a sweep is playing, removes leftover velocity as a linear drift
over each slide, and re-estimates the bias while still. Treat the result as
centimetre-level; the x/y you type stays the truth label.
"""

import math
import struct

import numpy as np
import pandas as pd

from . import stream

FLAG_ZEROED, FLAG_MOVING, FLAG_DRIVE, FLAG_LONG, FLAG_GRAVITY_ODD, FLAG_STALE = 1, 2, 4, 8, 16, 32


# ---------------------------------------------------------------- board commands

def status(ser) -> dict:
    """One 'IMU,...' line from the board as a dict (board must not be streaming)."""
    return stream.parse_info(stream.ask(ser, "imu", expect="IMU"))


def zero(ser, timeout=12.0) -> str:
    """Set the lower-left corner as (0,0) and the current pose as zero angle.
    The board waits for 0.25 s of stillness, so hold the head still."""
    return stream.ask(ser, "imu zero", expect="OK,zero", timeout=timeout)


def mark(ser, n):
    ser.write(f"mark {n}\n".encode())


# ---------------------------------------------------------------- axes

def remap(x, y, axes="x,y"):
    """Map the board's x/y to the phantom's (right, up). axes names, for phantom right and
    phantom up, which board axis to use: e.g. 'x,y' (default), '-y,x' (board turned 90 deg)."""
    parts = [p.strip() for p in axes.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--imu-axes needs two entries like 'x,y' or '-y,x', got {axes!r}")
    src = {"x": np.asarray(x, float), "y": np.asarray(y, float)}
    out = []
    for p in parts:
        sign = -1.0 if p.startswith("-") else 1.0
        key = p.lstrip("+-")
        if key not in src:
            raise ValueError(f"--imu-axes entries must be x or y (with optional -), got {p!r}")
        out.append(sign * src[key])
    return out[0], out[1]


# ---------------------------------------------------------------- per-trial summary

def summarize(imu: np.ndarray, events, axes="x,y") -> dict:
    """Pose during the sweep: median position/angles and how much the head moved."""
    if imu is None or len(imu) == 0:
        return {"imu_samples": 0}
    t = imu["t_us"]
    start = next((e["t_us"] for e in events or [] if e["event"] == "sweep_start"), t[0])
    end = next((e["t_us"] for e in events or [] if e["event"] in ("sweep_end", "sweep_aborted")), t[-1])
    sel = (t >= start) & (t <= end)
    if not sel.any():
        sel = np.ones(len(t), bool)
    d = imu[sel]
    x, y = remap(d["x_mm"], d["y_mm"], axes)
    flags = np.bitwise_or.reduce(d["flags"].astype(int))
    calib = int(d["calib"][-1])
    finite = np.isfinite(x) & np.isfinite(y)
    out = {
        "imu_samples": int(sel.sum()),
        "imu_x_mm": float(np.median(x[finite])) if finite.any() else float("nan"),
        "imu_y_mm": float(np.median(y[finite])) if finite.any() else float("nan"),
        "imu_yaw_deg": float(np.median(d["yaw_deg"])),
        "imu_pitch_deg": float(np.median(d["pitch_deg"])),
        "imu_roll_deg": float(np.median(d["roll_deg"])),
        "imu_move_during_sweep_mm": float(np.hypot(np.ptp(x[finite]), np.ptp(y[finite]))) if finite.any() else float("nan"),
        "imu_turn_during_sweep_deg": float(np.ptp(d["yaw_deg"])),
        "imu_zeroed": bool(flags & FLAG_ZEROED),
        "imu_long_move": bool(flags & FLAG_LONG),
        "imu_stale": bool(flags & FLAG_STALE),
        "imu_calib_sys": calib >> 6, "imu_calib_gyr": (calib >> 4) & 3,
        "imu_calib_acc": (calib >> 2) & 3,
        "imu_axes": axes,
    }
    return out


def to_frame(imu: np.ndarray, axes="x,y") -> pd.DataFrame:
    df = pd.DataFrame({k: imu[k] for k in imu.dtype.names})
    df["x_mm"], df["y_mm"] = remap(imu["x_mm"], imu["y_mm"], axes)
    return df


# ---------------------------------------------------------------- flash log

LOG_REC = struct.Struct("<BBBBIhhhhfffHH")  # 32 bytes, see firmware/esp32/src/flash_log.h
LOG_TYPES = {1: "imu", 2: "zero", 3: "sweep_start", 4: "sweep_end", 5: "mark", 6: "boot"}
assert LOG_REC.size == 32


def encode_log(records) -> bytes:
    """Inverse of decode_log, for tests. records: dicts with type, t_ms, yaw, ... ."""
    out = bytearray()
    code = {v: k for k, v in LOG_TYPES.items()}
    for r in records:
        body = struct.pack("<BBBBIhhhhfffH", code[r["type"]], r.get("flags", 0), r.get("calib", 0), 0,
                           r["t_ms"], round(r.get("yaw_deg", 0) * 100), round(r.get("pitch_deg", 0) * 100),
                           round(r.get("roll_deg", 0) * 100), 0, r.get("x_mm", 0.0), r.get("y_mm", 0.0),
                           r.get("value", 0.0), 0)
        out += body + struct.pack("<H", sum(body) & 0xFFFF)
    return bytes(out)


def decode_log(buf: bytes) -> pd.DataFrame:
    """Flash log bytes -> one row per record. A new `boot` record starts a new session."""
    rows, session, bad = [], 0, 0
    for off in range(0, len(buf) - 31, 32):
        rec = buf[off:off + 32]
        if rec[0] == 0xFF:
            break
        if sum(rec[:30]) & 0xFFFF != struct.unpack_from("<H", rec, 30)[0]:
            bad += 1
            continue
        (typ, flags, calib, _, t_ms, yaw, pitch, roll, _, x, y, value, _, _) = LOG_REC.unpack(rec)
        name = LOG_TYPES.get(typ, str(typ))
        if name == "boot":
            session += 1
        rows.append({"session": session, "type": name, "t_s": t_ms / 1000.0,
                     "yaw_deg": yaw / 100, "pitch_deg": pitch / 100, "roll_deg": roll / 100,
                     "x_mm": x, "y_mm": y, "value": value, "flags": flags,
                     "moving": bool(flags & FLAG_MOVING), "calib": calib})
    df = pd.DataFrame(rows)
    df.attrs["bad_records"] = bad
    return df


# ---------------------------------------------------------------- tracker (mirror of tracker.h)

def _rotate(q, v):
    w, x, y, z = q
    return np.array([
        (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - w * z) * v[1] + 2 * (x * z + w * y) * v[2],
        2 * (x * y + w * z) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - w * x) * v[2],
        2 * (x * z - w * y) * v[0] + 2 * (y * z + w * x) * v[1] + (1 - 2 * (x * x + y * y)) * v[2],
    ])


def _mul(a, b):
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])


def _euler(q):
    w, x, y, z = q
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return yaw, pitch, roll


class Tracker:
    still_gyro_dps = 3.0
    still_acc_ms2 = 0.25
    still_samples = 25
    bias_alpha = 0.02
    long_move_s = 5.0

    def __init__(self):
        self.zero_pending = self.zeroed = self.moving = self.long_move = self.gravity_odd = False
        self.last_t = None
        self.still_count = 0
        self.q0, self.psi0 = np.array([1.0, 0, 0, 0]), 0.0
        self.bias = np.zeros(3)
        self.pos, self.vel, self.seg, self.seg_t = np.zeros(2), np.zeros(2), np.zeros(2), 0.0

    def request_zero(self):
        self.zero_pending = True

    def _integrate(self, a, dt):
        self.vel += a * dt
        self.seg += self.vel * dt
        self.seg_t += dt

    def update(self, t_us, q, lia, grv, gyro, drive_on=False) -> dict:
        q, lia, grv, gyro = (np.asarray(v, float) for v in (q, lia, grv, gyro))
        dt = 0.0 if self.last_t is None else (t_us - self.last_t) * 1e-6
        if dt < 0 or dt > 0.05:
            dt = 0.0
        self.last_t = t_us

        a = lia - self.bias
        quiet = np.linalg.norm(gyro) < self.still_gyro_dps and np.linalg.norm(a) < self.still_acc_ms2
        self.still_count = self.still_count + 1 if quiet else 0
        still = drive_on or self.still_count >= self.still_samples
        if still and not drive_on:
            self.bias += self.bias_alpha * (lia - self.bias)

        if self.zero_pending and self.still_count >= self.still_samples:
            self.q0 = q.copy()
            xw = _rotate(q, [1.0, 0, 0])
            self.psi0 = math.atan2(xw[1], xw[0])
            gw = _rotate(q, grv)
            g = np.linalg.norm(grv)
            self.gravity_odd = not (g > 5.0 and gw[2] > 0.9 * g)
            self.pos[:] = 0
            self.vel[:] = 0
            self.seg[:] = 0
            self.seg_t = 0.0
            self.moving = self.long_move = False
            self.zeroed, self.zero_pending = True, False

        aw = _rotate(q, a)
        c, s = math.cos(-self.psi0), math.sin(-self.psi0)
        ap = np.array([c * aw[0] - s * aw[1], s * aw[0] + c * aw[1]])

        if self.zeroed:
            if not still:
                if not self.moving:
                    self.moving = True
                    self.vel[:] = 0
                    self.seg[:] = 0
                    self.seg_t = 0.0
                self._integrate(ap, dt)
            elif self.moving:
                self._integrate(ap, dt)
                self.pos += self.seg - self.vel * self.seg_t / 2
                self.long_move = self.seg_t > self.long_move_s
                self.moving = False
                self.vel[:] = 0
                self.seg[:] = 0
                self.seg_t = 0.0

        qr = _mul(np.array([self.q0[0], *(-self.q0[1:])]), q) if self.zeroed else q
        yaw, pitch, roll = _euler(qr)
        p = self.pos + (self.seg if self.moving else 0)
        return {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
                "x_mm": 1000 * p[0] if self.zeroed else float("nan"),
                "y_mm": 1000 * p[1] if self.zeroed else float("nan"),
                "moving": self.moving, "zeroed": self.zeroed}
