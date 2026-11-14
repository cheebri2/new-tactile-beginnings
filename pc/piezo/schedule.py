"""The stepped sweep the drive ESP32 plays. Must match firmware/drive (the capture
script sends `cfg` to the drive board so they always agree)."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Sweep:
    f_start: float = 20.0
    f_stop: float = 300.0
    f_step: float = 10.0
    tone_ms: int = 600
    gap_ms: int = 100
    lead_ms: int = 1000   # silence before the first tone, also used as the noise reference
    amp: int = 100        # DAC counts, 0..127
    ramp_ms: int = 10
    settle_ms: int = 150  # skipped at the start of each tone before measuring
    tail_ms: int = 50     # skipped at the end of each tone

    @property
    def freqs(self) -> np.ndarray:
        n = int(np.floor((self.f_stop - self.f_start) / self.f_step + 1e-6)) + 1
        return self.f_start + self.f_step * np.arange(n)

    @property
    def tone_starts_s(self) -> np.ndarray:
        """Tone start times relative to the sweep start (start of the lead silence)."""
        k = np.arange(len(self.freqs))
        return (self.lead_ms + k * (self.tone_ms + self.gap_ms)) / 1000.0

    @property
    def windows_s(self):
        """(start, end) of the analysis window inside each tone, relative to sweep start."""
        a = self.tone_starts_s + self.settle_ms / 1000.0
        return a, self.tone_starts_s + (self.tone_ms - self.tail_ms) / 1000.0

    @property
    def noise_window_s(self):
        """Window in the lead silence, same length as a tone window, for the noise floor."""
        length = (self.tone_ms - self.settle_ms - self.tail_ms) / 1000.0
        end = self.lead_ms / 1000.0 - 0.05
        return max(0.05, end - length), end

    @property
    def duration_s(self) -> float:
        return (self.lead_ms + len(self.freqs) * (self.tone_ms + self.gap_ms)) / 1000.0

    def cfg_command(self) -> str:
        return (f"cfg {self.f_start:g} {self.f_stop:g} {self.f_step:g} {self.tone_ms} "
                f"{self.gap_ms} {self.lead_ms} {self.amp} {self.ramp_ms}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Sweep":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
