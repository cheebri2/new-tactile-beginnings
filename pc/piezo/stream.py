"""Serial link to the two ESP32s and the binary block format of firmware/sense."""

import struct
import threading
import time

import numpy as np

MAGIC = b"\xA5\x5A"
HEADER = struct.Struct("<HIH")  # n, seq, overruns (after the 2 magic bytes)
FRAME = np.dtype([("t_us", "<u4"), ("adc", "<u2", (4,))])
SENSE_BAUD = 921600
DRIVE_BAUD = 115200


def parse_blocks(buf: bytes):
    """Decode every valid block in `buf`.

    Returns (t_us uint64 unwrapped, adc uint16 [N,4], stats dict). Corrupt blocks are
    skipped and counted; a gap in the block counter means blocks were lost.
    """
    times, adcs = [], []
    stats = {"blocks": 0, "bad_blocks": 0, "lost_blocks": 0, "overruns": 0}
    last_seq = None
    i = 0
    while True:
        i = buf.find(MAGIC, i)
        if i < 0 or i + 2 + HEADER.size > len(buf):
            break
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
        return np.zeros(0, np.uint64), np.zeros((0, 4), np.uint16), stats
    t = np.concatenate(times).astype(np.int64)
    # micros() wraps every ~71.6 minutes
    wraps = np.concatenate([[0], np.cumsum(np.diff(t) < -(1 << 31))])
    t = t + wraps * (1 << 32)
    return t, np.concatenate(adcs), stats


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


def parse_info(line: str) -> dict:
    """'INFO,fw=sense-1.0,rate_hz=2000,...' -> dict."""
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


class SenseStream:
    """Reads binary blocks from the sensing board in a background thread."""

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
