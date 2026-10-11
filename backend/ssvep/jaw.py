"""
Double jaw-clench detector: a hands-free "start the flash" trigger.

Why a jaw clench is easy to detect
----------------------------------
Clenching fires the masseter/temporalis muscles. Their electrical activity
(EMG) is broadband and high-frequency, and so strong that it swamps every EEG
channel -- even the occipital ones -- for as long as the clench lasts. EEG
itself has very little power above ~20 Hz, so the 20-45 Hz band at rest is
quiet and a clench shows up as a burst that is many times larger.

Algorithm (runs on the live stream, sample by sample, causal)
-------------------------------------------------------------
1. Band-pass 20-45 Hz (stays below the 50/60 Hz mains notches).
2. Envelope = mean absolute value across channels, smoothed over ~0.05 s
   (short, so a fast double clench ~0.3 s apart stays two separate bursts).
3. Baseline = running median of the envelope over the last ~10 s of
   non-burst data, so the threshold adapts to each person/session.
4. Burst = envelope above `factor` x baseline (hysteresis to end it).
   Bursts shorter than `min_burst` are spikes; longer than `max_burst` are
   chewing/talking/movement -- both are ignored.
5. Double clench = two valid bursts whose onsets are `min_gap`..`max_gap`
   seconds apart. Then a `refractory` pause so one double fires once.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from . import config as cfg


class JawClenchDetector:
    def __init__(self, sfreq: float = cfg.SAMPLING_RATE, n_channels: int = cfg.NUM_CHANNELS,
                 factor: float = cfg.JAW_THRESHOLD_FACTOR,
                 band=(20.0, 45.0), smooth: float = 0.05,
                 min_burst: float = 0.03, max_burst: float = 1.0,
                 min_gap: float = 0.15, max_gap: float = 1.5,
                 refractory: float = 2.0, baseline_seconds: float = 10.0,
                 warmup: float = 3.0):
        self.sfreq = sfreq
        self.factor = factor
        self.min_burst, self.max_burst = min_burst, max_burst
        self.min_gap, self.max_gap = min_gap, max_gap
        self.refractory = refractory
        self.warmup = warmup

        self._sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
        self._zi = np.repeat(sosfilt_zi(self._sos)[:, np.newaxis, :], n_channels, axis=1)
        self._alpha = 1.0 / max(1.0, smooth * sfreq)      # EMA smoothing
        self._env = 0.0
        # Baseline: envelope history, decimated to ~10 values/s to stay cheap.
        self._hist = deque(maxlen=int(baseline_seconds * 10))
        self._decim = max(1, int(sfreq / 10))
        self._n = 0                                        # samples seen

        self._in_burst = False
        self._burst_start = 0.0
        self._last_onset = None    # onset time of the previous valid burst
        self._quiet_until = 0.0    # refractory end time

        self.baseline = None       # exposed for debugging / calibration
        self.envelope = 0.0
        # False while the board sends a dead-flat signal (electrode lost
        # contact / amplifier railed). Nothing is detected or learned then.
        self.signal_ok = True
        self._holdoff_until = 0.0  # ignore the filter transient after a dropout

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Forget pending bursts (e.g. after a flash); keep the baseline."""
        self._in_burst = False
        self._last_onset = None

    def threshold(self) -> float | None:
        return None if self.baseline is None else self.factor * self.baseline

    def feed(self, samples: np.ndarray) -> list[float]:
        """
        Process new raw samples, shape (channels, n). Returns the times (in
        seconds since the detector started) of any double clenches detected.
        """
        if samples.size == 0:
            return []
        filtered, self._zi = sosfilt(self._sos, samples, axis=1, zi=self._zi)
        rect = np.abs(filtered).mean(axis=0)

        # A dead-flat chunk on every channel is not EEG: the electrodes lost
        # contact. Learning a "resting level" from it drove the old baseline to
        # ~0, after which every sample looked like a clench and it locked up.
        flat = samples.shape[1] > 1 and float(np.ptp(samples, axis=1).max()) == 0.0
        if flat:
            self.signal_ok = False
            self.reset()
        elif not self.signal_ok:
            self.signal_ok = True
            # The jump back to a real signal rings through the filter: wait 1 s.
            self._holdoff_until = (self._n + samples.shape[1]) / self.sfreq + 1.0

        events = []
        for value in rect:
            self._n += 1
            t = self._n / self.sfreq
            self._env += self._alpha * (value - self._env)
            self.envelope = self._env
            if flat or t < self._holdoff_until:
                continue

            # Resting level = median over the last ~10 s. Always updated: short
            # clenches barely move a median, and updating during long bursts too
            # means it re-adapts within seconds if the signal level changes.
            if self._n % self._decim == 0:
                self._hist.append(self._env)
                if len(self._hist) >= 10:
                    self.baseline = float(np.median(self._hist))
            if not self.baseline or t < self.warmup:
                continue

            thr = self.factor * self.baseline
            if not self._in_burst and self._env > thr:
                self._in_burst, self._burst_start = True, t
            elif self._in_burst and self._env < 0.7 * thr:     # hysteresis
                self._in_burst = False
                if t >= self._quiet_until and self._on_burst(self._burst_start, t - self._burst_start):
                    events.append(t)
            elif self._in_burst and t - self._burst_start > 3 * self.max_burst:
                self.reset()   # way too long for a clench: artefact, start over
        return events

    def _on_burst(self, onset: float, duration: float) -> bool:
        """Register one finished burst; True if it completes a double clench."""
        if not (self.min_burst <= duration <= self.max_burst):
            self._last_onset = None        # spike or sustained artefact
            return False
        if self._last_onset is not None and \
                self.min_gap <= onset - self._last_onset <= self.max_gap:
            self._last_onset = None
            self._quiet_until = onset + duration + self.refractory
            return True
        self._last_onset = onset           # first clench of a possible pair
        return False
