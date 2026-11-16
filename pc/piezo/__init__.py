"""Laptop side of the piezo sweep pipeline: capture, feature extraction, analysis."""

CHANNELS = ["ur", "ul", "dl", "dr"]  # up_right GPIO35, up_left 32, down_left 33, down_right 34
CHANNEL_NAMES = {"ur": "up right", "ul": "up left", "dl": "down left", "dr": "down right"}
# Still recorded in the raw files, but left out of features and analysis. Empty the list to restore.
DISABLED_CHANNELS = ["dr"]  # GPIO34 board fault, 2026-11-15
ACTIVE_CHANNELS = [c for c in CHANNELS if c not in DISABLED_CHANNELS]
REF_CHANNEL = "ur"  # phases are reported relative to this channel
DATA_DIR = "training/data"  # default for capture/process/analyze; calibration runs are in "data"
