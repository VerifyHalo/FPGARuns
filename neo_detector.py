"""
Software replica of the NEO seizure detector.
"""

import numpy as np

# Defaults from run_tests.py
_THRESHOLD        = 170_000   # NEO magnitude threshold (ADC² units)
_WINDOW_TIMEOUT   = 300       # consecutive non-detections → seizure end  (@ 1 kHz)
_TRANSITION_COUNT = 50        # consecutive detections → seizure start     (@ 1 kHz)
_BASE_RATE        = 1_000.0   # Hz


class NeoDetector:
    """NEO-based seizure detector matching the FPGAPipelines neo-branch pipeline."""

    def __init__(
        self,
        sample_rate: float,
        threshold: int = _THRESHOLD,
        window_timeout: int = _WINDOW_TIMEOUT,
        transition_count: int = _TRANSITION_COUNT,
    ):
        scale = sample_rate / _BASE_RATE
        self.threshold        = threshold
        self.window_timeout   = int(window_timeout   * scale)
        self.transition_count = int(transition_count * scale)

    # ------------------------------------------------------------------
    def detect(self, raw_uint16: np.ndarray) -> list[tuple[int, int]]:
        x = raw_uint16.astype(np.int64) - 32_768

        # NEO: ψ[n] = x[n]² - x[n-1]·x[n+1]
        neo = np.zeros(len(x), dtype=np.int64)
        neo[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
        triggered = np.abs(neo) > self.threshold

        return self._state_machine(triggered)

    # ------------------------------------------------------------------
    def _state_machine(self, triggered: np.ndarray) -> list[tuple[int, int]]:
        """Sequential state machine"""
        seizures: list[tuple[int, int]] = []
        NORMAL, SEIZURE = 0, 1
        state = NORMAL
        det_count = timeout_count = 0
        seizure_start = 0

        tc = self.transition_count
        wt = self.window_timeout

        for i, det in enumerate(triggered):
            if state == NORMAL:
                if det:
                    det_count += 1
                    timeout_count = 0
                    if det_count >= tc:
                        state = SEIZURE
                        seizure_start = max(0, i - (tc - 1))
                        det_count = 0
                else:
                    det_count = 0
            else:   # SEIZURE
                if det:
                    timeout_count = 0
                else:
                    timeout_count += 1
                    if timeout_count >= wt:
                        end = i - wt          # last confirmed detection
                        seizures.append((seizure_start, max(seizure_start, end)))
                        state = NORMAL
                        det_count = timeout_count = 0

        if state == SEIZURE:
            seizures.append((seizure_start, len(triggered) - 1))

        return seizures
