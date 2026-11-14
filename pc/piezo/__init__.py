"""Laptop side of the piezo sweep pipeline: capture, feature extraction, analysis."""

CHANNELS = ["ur", "ul", "dl", "dr"]  # up_right GPIO34, up_left 35, down_left 32, down_right 33
CHANNEL_NAMES = {"ur": "up right", "ul": "up left", "dl": "down left", "dr": "down right"}
REF_CHANNEL = "ur"  # phases are reported relative to this channel
