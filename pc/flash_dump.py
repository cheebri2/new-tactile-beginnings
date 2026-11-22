"""Read the board's flash log (IMU track + sweep events + trial marks) into a CSV.

The board keeps this log on its W25Q32 flash as a backup of the IMU track, 25 records a
second while the IMU is zeroed, plus a record at every sweep start/end, zero and trial mark.

  python flash_dump.py --port COM3                      # -> training/data/flash_log_<time>.csv
  python flash_dump.py --port COM3 --out log.csv
  python flash_dump.py --port COM3 --erase              # wipe it (10-25 s) before a new session
  python flash_dump.py --file dump.bin                  # decode a saved raw dump

Columns: session (counts power-ups), type (imu, zero, sweep_start, sweep_end, mark, boot),
t_s (seconds since that power-up), yaw/pitch/roll_deg, x_mm/y_mm (from the lower-left
corner, board axes), value (trial number for marks), moving, flags, calib.
"""

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

from piezo import DATA_DIR, imu, stream


def read_dump(ser, timeout=120.0) -> bytes:
    ser.write(b"x\n")
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b"flash dump\n")
    end = time.time() + 5
    header = ""
    while time.time() < end and not header.startswith("FLASH"):
        header = ser.readline().decode(errors="ignore").strip()
    if not header.startswith("FLASH"):
        sys.exit("No reply to 'flash dump'. Is firmware 3.0+ flashed?")
    n = int(stream.parse_info(header)["bytes"])
    print(f"reading {n} bytes ({n // 32} records)...")
    buf = bytearray()
    end = time.time() + timeout
    while len(buf) < n and time.time() < end:
        buf += ser.read(min(65536, n - len(buf)))
    if len(buf) < n:
        print(f"warning: only got {len(buf)} of {n} bytes")
    return bytes(buf)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", help="ESP32 serial port, e.g. COM3")
    p.add_argument("--file", help="decode a raw dump saved earlier instead of reading the board")
    p.add_argument("--out", help="output CSV (default training/data/flash_log_<time>.csv)")
    p.add_argument("--erase", action="store_true", help="erase the whole log (asks first)")
    a = p.parse_args()

    if a.file:
        buf = Path(a.file).read_bytes()
    else:
        if not a.port:
            sys.exit("--port or --file is required")
        ser = stream.open_port(a.port, stream.BAUD)
        if a.erase:
            if input("Erase the whole flash log? Type yes: ").strip().lower() != "yes":
                return
            print(stream.ask(ser, "flash erase", expect="OK,erased", timeout=90))
            return
        buf = read_dump(ser)
        ser.close()
        raw = Path(DATA_DIR) / f"flash_log_{dt.datetime.now():%Y%m%d-%H%M%S}.bin"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(buf)

    df = imu.decode_log(buf)
    out = Path(a.out) if a.out else Path(DATA_DIR) / f"flash_log_{dt.datetime.now():%Y%m%d-%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    bad = df.attrs.get("bad_records", 0)
    sessions = df["session"].nunique() if len(df) else 0
    print(f"{len(df)} records, {sessions} power-up(s), {bad} corrupt -> {out}")


if __name__ == "__main__":
    main()
