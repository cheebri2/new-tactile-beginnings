"""Serial link to the ESP32 and the binary block format of firmware/esp32."""

import struct
import threading
import time

import numpy as np

MAGIC = b"\xA5\x5A"
EVENT_MAGIC = b"\xA5\x5B"
IMU_MAGIC = b"\xA5\x5C"
HEADER = struct.Struct("<HIH")  # n, seq, overruns (after the 2 magic bytes)
EVENT = struct.Struct("<BIf")   # type, t_us, freq_hz
IMU = struct.Struct("<IhhhffhhBB")  # t_us, yaw/pitch/roll (0.01 deg), x_mm, y_mm, ax/ay (mm/s^2), flags, calib
IMU_DTYPE = np.dtype([("t_us", "<i8"), ("yaw_deg", "<f4"), ("pitch_deg", "<f4"), ("roll_deg", "<f4"),
                      ("x_mm", "<f4"), ("y_mm", "<f4"), ("ax_ms2", "<f4"), ("ay_ms2", "<f4"),
                      ("flags", "u1"), ("calib", "u1")])
FRAME = np.dtype([("t_us", "<u4"), ("adc", "<u2", (4,))])
BAUD = 921600
EVENT_NAMES = {1: "sweep_start", 2: "tone_start", 3: "tone_end", 4: "sweep_end", 5: "sweep_aborted"}


def parse_blocks(buf: bytes):
    """Decode every valid block in `buf`.

    Returns (t_us int64 unwrapped, adc uint16 [N,4], events, stats dict). events is a list of
    {"event", "t_us", "freq_hz"} on the same clock as t_us. Corrupt blocks are skipped and
    counted; a gap in the block counter means blocks were lost.
    """
    times, adcs, raw_events = [], [], []
    stats = {"blocks": 0, "bad_blocks": 0, "lost_blocks": 0, "overruns": 0}
    last_seq = None
    i = 0
    while True:
        i = buf.find(b"\xA5", i)
        if i < 0 or i + 2 + HEADER.size > len(buf):
            break
        if buf[i + 1:i + 2] == EVENT_MAGIC[1:]:
            end = i + 2 + EVENT.size
            if end + 2 <= len(buf) and sum(buf[i + 2:end]) & 0xFFFF == struct.unpack_from("<H", buf, end)[0]:
                raw_events.append(EVENT.unpack_from(buf, i + 2))
                i = end + 2
            else:
                i += 1
            continue
        if buf[i + 1:i + 2] != MAGIC[1:]:
            i += 1
            continue
        n, seq, over = HEADER.unpack_from(buf, i + 2)
        body_start = i + 2
        body_end = i + 2 + HEADER.size + n * FRAME.itemsize
        if n == 0 or n > 1000 or body_end + 2 > len(buf):
            i += 1  # false magic, or an incomplete block at the end
            continue
        checksum = struct.unpack_from("<H", buf, body_end)[0]
        if sum(buf[body_start:body_end]) & 0xFFFF != checksum:
            stats["bad_blocks"] += 1
            i += 1
            continue
        frames = np.frombuffer(buf, FRAME, count=n, offset=body_start + HEADER.size)
        times.append(frames["t_us"].astype(np.uint64))
        adcs.append(frames["adc"].copy())
        stats["blocks"] += 1
        stats["overruns"] += over
        if last_seq is not None and seq != last_seq + 1:
            stats["lost_blocks"] += max(0, seq - last_seq - 1)
        last_seq = seq
        i = body_end + 2
    if not times:
        return np.zeros(0, np.int64), np.zeros((0, 4), np.uint16), [], stats
    t = np.concatenate(times).astype(np.int64)
    # micros() wraps every ~71.6 minutes
    wraps = np.concatenate([[0], np.cumsum(np.diff(t) < -(1 << 31))])
    t = t + wraps * (1 << 32)
    events = []
    for kind, tu, f in raw_events:
        te = int(tu)
        while te < t[0] - (1 << 31):
            te += 1 << 32
        events.append({"event": EVENT_NAMES.get(kind, str(kind)), "t_us": te, "freq_hz": float(f)})
    return t, np.concatenate(adcs), events, stats


def parse_imu(buf: bytes, ref_t_us=None) -> np.ndarray:
    """Decode every IMU block (A5 5C) in `buf` into a structured array (IMU_DTYPE).
    t_us is unwrapped against ref_t_us (e.g. the first piezo sample) so it shares the
    piezo clock. Piezo/event blocks are skipped."""
    rows = []
    i = 0
    while True:
        i = buf.find(IMU_MAGIC, i)
        if i < 0:
            break
        end = i + 2 + IMU.size
        if end + 2 > len(buf):
            break
        if sum(buf[i + 2:end]) & 0xFFFF == struct.unpack_from("<H", buf, end)[0]:
            rows.append(IMU.unpack_from(buf, i + 2))
            i = end + 2
        else:
            i += 1
    out = np.zeros(len(rows), IMU_DTYPE)
    if not rows:
        return out
    r = list(zip(*rows))
    t = np.array(r[0], np.int64)
    if ref_t_us is not None:
        ref = int(ref_t_us)
        t = t + ((ref - t + (1 << 31)) >> 32 << 32)  # nearest wrap to the reference
    else:
        t = t + np.concatenate([[0], np.cumsum(np.diff(t) < -(1 << 31))]) * (1 << 32)
    out["t_us"] = t
    for k, name in enumerate(["yaw_deg", "pitch_deg", "roll_deg"], 1):
        out[name] = np.array(r[k]) / 100.0
    out["x_mm"], out["y_mm"] = r[4], r[5]
    out["ax_ms2"], out["ay_ms2"] = np.array(r[6]) / 1000.0, np.array(r[7]) / 1000.0
    out["flags"], out["calib"] = r[8], r[9]
    return out


def encode_imu(t_us, yaw=0.0, pitch=0.0, roll=0.0, x=0.0, y=0.0, ax=0.0, ay=0.0, flags=1, calib=0) -> bytes:
    """Inverse of parse_imu for one sample (tests)."""
    body = IMU.pack(int(t_us) & 0xFFFFFFFF, round(yaw * 100), round(pitch * 100), round(roll * 100),
                    x, y, round(ax * 1000), round(ay * 1000), flags, calib)
    return IMU_MAGIC + body + struct.pack("<H", sum(body) & 0xFFFF)


def encode_blocks(t_us, adc, frames_per_block=50) -> bytes:
    """Inverse of parse_blocks (used by tests and the simulator)."""
    out = bytearray()
    for seq, s in enumerate(range(0, len(t_us), frames_per_block)):
        tt = np.asarray(t_us[s:s + frames_per_block], np.uint64) & 0xFFFFFFFF
        frames = np.zeros(len(tt), FRAME)
        frames["t_us"] = tt
        frames["adc"] = adc[s:s + frames_per_block]
        body = HEADER.pack(len(tt), seq, 0) + frames.tobytes()
        out += MAGIC + body + struct.pack("<H", sum(body) & 0xFFFF)
    return bytes(out)


def encode_event(name, t_us, freq=0.0) -> bytes:
    code = {v: k for k, v in EVENT_NAMES.items()}[name]
    body = EVENT.pack(code, int(t_us) & 0xFFFFFFFF, freq)
    return EVENT_MAGIC + body + struct.pack("<H", sum(body) & 0xFFFF)


def parse_info(line: str) -> dict:
    """'INFO,fw=piezo-sweep-2.0,rate_hz=2000,...' -> dict."""
    out = {}
    for part in line.strip().split(",")[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def open_port(port, baud):
    import serial  # pyserial

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = port, baud, 0.2
    ser.dtr = False  # keep DTR/RTS low so opening the port does not reset the board
    ser.rts = False
    ser.open()
    time.sleep(0.3)
    ser.reset_input_buffer()
    return ser


def ask(ser, cmd, expect="INFO", timeout=3.0) -> str:
    """Send a text command and return the first reply line starting with `expect`."""
    ser.reset_input_buffer()
    ser.write((cmd + "\n").encode())
    end = time.time() + timeout
    while time.time() < end:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith(expect) or line.startswith("ERR"):
            return line
    raise TimeoutError(f"no {expect} reply to '{cmd}' on {ser.port}. Is the right firmware flashed?")


class Stream:
    """Reads binary blocks from the ESP32 in a background thread."""

    def __init__(self, ser):
        self.ser = ser
        self.buf = bytearray()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.ser.write(b"x\n")
        time.sleep(0.1)
        self.ser.reset_input_buffer()
        self.buf.clear()
        self._stop.clear()
        self.ser.write(b"b\n")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                self.buf += chunk

    def stop(self):
        self.ser.write(b"x\n")
        time.sleep(0.15)
        self._stop.set()
        self._thread.join(timeout=1)
        self.buf += self.ser.read(self.ser.in_waiting)
        return bytes(self.buf)


def record_sweep(ser, duration_s, pre_s=0.5, post_s=0.5, with_imu=False):
    """Stream, play one sweep, stop when the board reports the sweep finished.
    Returns (t_us, adc, events, stats), plus the IMU samples (parse_imu) if with_imu."""
    s = Stream(ser)
    s.start()
    time.sleep(pre_s)
    ser.write(b"sweep\n")
    deadline = time.time() + duration_s + 5
    time.sleep(duration_s * 0.9)
    while time.time() < deadline:
        _, _, events, _ = parse_blocks(bytes(s.buf[-6000:]))
        if any(e["event"] in ("sweep_end", "sweep_aborted") for e in events):
            break
        time.sleep(0.1)
    else:
        print("  warning: the board never reported the end of the sweep")
    time.sleep(post_s)
    buf = s.stop()
    t_us, adc, events, stats = parse_blocks(buf)
    if not with_imu:
        return t_us, adc, events, stats
    return t_us, adc, events, stats, parse_imu(buf, t_us[0] if len(t_us) else None)
