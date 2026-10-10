"""
Central configuration for the NeuroPawn SSVEP-TRCA pipeline.

Everything that you might want to tweak (serial port, electrode count,
stimulus frequencies, timing, filter settings, key bindings, file paths)
lives here so that the rest of the code never hard-codes "magic numbers".

Import it everywhere as:

    from ssvep import config as cfg
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
# Serial port the Knight board enumerates on.
#   Windows -> "COM3", "COM4", ...
#   Linux   -> "/dev/ttyUSB0", "/dev/ttyACM0", ...
#   macOS   -> "/dev/cu.usbserial-XXXX"
SERIAL_PORT = "COM11"

# Number of EEG channels you have wired up on the board (1-8).
NUM_CHANNELS = 8

# The board's fixed sampling rate. The Knight board streams at 125 Hz.
SAMPLING_RATE = 125  # Hz

# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
# Monitor index PsychoPy draws the flicker on (0 = primary, 1 = secondary).
STIMULUS_SCREEN = 0
FULLSCREEN = True
WINDOW_SIZE = [1920, 1200]      # only used when FULLSCREEN is False
# The flicker frequencies assume this refresh rate. Use a 60 Hz monitor.
EXPECTED_REFRESH_HZ = 60

# Per-channel gain used in the `chon_{ch}_{gain}` command (see board.py).
CHANNEL_GAIN = 12

# --------------------------------------------------------------------------- #
# Electrode montage (10-20 system)
# --------------------------------------------------------------------------- #
# SSVEP is generated in the primary visual cortex, so every electrode sits
# over the occipital / parieto-occipital scalp. The order below maps
# board channel 1..8 -> physical electrode. Reference/ground go on the
# earlobes or mastoids (A1/A2).
ELECTRODE_LABELS = ["Oz", "O1", "O2", "PO7", "PO8", "PO3", "PO4", "POz"]

# --------------------------------------------------------------------------- #
# Stimulus frequencies
# --------------------------------------------------------------------------- #
# One flicker frequency per on-screen target. These values were chosen so
# that a 60 Hz monitor can render them with an (almost) integer number of
# frames per cycle, which keeps the real flicker stable:
#   6.67 Hz ~ 9 frames, 8.57 Hz ~ 7 frames, 12 Hz = 5 frames, 15 Hz = 4 frames.
# 10 Hz is deliberately avoided: it sits in the alpha band (the brain's own
# 8-12 Hz idle rhythm) and its 2nd harmonic (20 Hz) collides with 6.67 Hz's
# 3rd. In the first recordings it was the most-confused target (mostly with
# 12 Hz). No frequency here is a harmonic of another.
# Changing these invalidates existing training data -- re-record afterwards.
STIMULUS_FREQUENCIES = [6.67, 8.57, 12.00, 15.00]  # Hz
N_TARGETS = len(STIMULUS_FREQUENCIES)

# --------------------------------------------------------------------------- #
# Trial timing (seconds)
# --------------------------------------------------------------------------- #
# A single trial runs: [cue] -> [flicker] -> [capture].
CUE_DURATION = 1.0        # how long the "look here" cue is shown
FLICKER_DURATION = 1.5    # how long the targets actually flash
INTER_TRIAL_INTERVAL = 0.5  # blank rest between trials (gaze-shift time)

# TRCA analysis window (must match how the data was recorded):
GAZE_DURATION = 1.0       # length of EEG used for classification [s]
VISUAL_LATENCY = 0.15     # delay before the cortex "locks on" [s]

# Number of samples grabbed from the ring buffer after each flicker period.
# 1.5 s * 125 Hz = 187.5 -> 188 samples covers the whole flicker window.
CAPTURE_SAMPLES = int(round(FLICKER_DURATION * SAMPLING_RATE)) + 1  # 188

# --------------------------------------------------------------------------- #
# Pre-processing filters (BrainFlow)
# --------------------------------------------------------------------------- #
BANDPASS_LOW = 3.0        # Hz  - remove drift / DC
BANDPASS_HIGH = 48.0      # Hz  - keep SSVEP fundamentals + first harmonics
BANDPASS_ORDER = 2

# Mains-noise notches. Keep both so the same code works in 50 Hz and 60 Hz
# countries; the filter that does not match your grid is simply harmless.
NOTCH_BANDS = [(49.0, 51.0), (59.0, 61.0)]
NOTCH_ORDER = 4

# --------------------------------------------------------------------------- #
# TRCA filter bank
# --------------------------------------------------------------------------- #
# meegkit expects [[(passband), (stopband)], ...]. Each sub-band isolates a
# harmonic range so the ensemble classifier can exploit SSVEP harmonics.
FILTERBANK = [
    [(6, 48), (4, 60)],
    [(14, 48), (10, 60)],
    [(22, 48), (16, 60)],
    [(30, 48), (24, 60)],
    [(38, 48), (32, 60)],
    [(46, 48), (40, 60)],
]
USE_ENSEMBLE_TRCA = True   # ensemble TRCA is more accurate than plain TRCA

# --------------------------------------------------------------------------- #
# Key bindings for real-time control
# --------------------------------------------------------------------------- #
# Maps the predicted target index (0..N_TARGETS-1) to a keyboard key that is
# "pressed" via pydirectinput. Reorder / change to taste.
#   index 0 (6.67 Hz) -> 'w'   (forward)
#   index 1 (8.57 Hz) -> 's'   (back)
#   index 2 (12.0 Hz) -> 'a'   (left)
#   index 3 (15.0 Hz) -> 'd'   (right)
KEY_MAP = {0: "w", 1: "s", 2: "a", 3: "d"}

# --------------------------------------------------------------------------- #
# File paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where training blocks are saved / loaded from (block_{block}_{trial}.csv).
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "training_data")
