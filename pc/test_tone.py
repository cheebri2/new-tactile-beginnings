"""Play a continuous 150 Hz tone on the amp, for checking the hardware.

  python test_tone.py --port COM3            # default amplitude
  python test_tone.py --port COM3 --amp 60   # quieter (0..127 DAC counts around mid-scale)

Press Enter (or Ctrl+C) to stop.
"""

import argparse

from piezo import stream


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="ESP32 serial port, e.g. COM3")
    p.add_argument("--amp", type=int, choices=range(128), metavar="0..127")
    a = p.parse_args()

    ser = stream.open_port(a.port, stream.BAUD)
    try:
        ser.write(b"x\n")  # make sure the board is not streaming, so it replies in text
        reply = stream.ask(ser, "test" if a.amp is None else f"test {a.amp}", expect="OK")
        print(reply)
        if reply.startswith("ERR"):
            return
        try:
            input("Playing 150 Hz. Press Enter to stop... ")
        except KeyboardInterrupt:
            pass
    finally:
        ser.write(b"off\n")
        ser.close()
    print("Stopped.")


if __name__ == "__main__":
    main()
