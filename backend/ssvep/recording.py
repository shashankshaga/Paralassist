"""
Background EEG acquisition process.

Why a separate process?
------------------------
PsychoPy must flip the screen on every monitor refresh (every ~16.6 ms on a
60 Hz display) to keep the flicker frequencies exact. If we asked it to also
talk to the board, run filters and TRCA, those calls could block a flip and
smear the stimulus timing -- which corrupts the very SSVEP we are trying to
measure. So the board lives in its own `multiprocessing.Process`:

  * The PsychoPy (parent) process only ever sets a few shared flags.
  * This (child) process owns the board, and on the *rising edge* of the
    recording flag it grabs the latest window from the ring buffer and either
    saves it (collect mode) or classifies it and presses a key (predict mode).

Shared state (all `multiprocessing.Value`):
  recording_flag : bool  - parent sets True to request one capture.
  block_index    : int   - which training block we're on (collect mode).
  label_index    : int   - the target the user is cued to look at (collect mode).
  last_prediction: int   - most recent predicted target (predict mode, -1 = none).
"""

from __future__ import annotations

import os
from multiprocessing import Process, Value, Event

from . import config as cfg


class RecordingProcess(Process):
    """Owns the board in a child process and reacts to the recording flag."""

    def __init__(self, mode: str, data_dir: str = cfg.TRAINING_DATA_DIR,
                 n_blocks: int | None = None, key_map: dict | None = None,
                 labels: dict | None = None, log_dir: str | None = None):
        super().__init__()
        assert mode in ("collect", "predict")
        self.mode = mode
        self.data_dir = data_dir
        self.n_blocks = n_blocks
        self.key_map = key_map if key_map is not None else cfg.KEY_MAP
        # If given ({target: name}), predict mode only prints the name and
        # never presses a key -- handy for testing the classifier.
        self.labels = labels
        # If given, every prediction (and the EEG window behind it) is logged here.
        self.log_dir = log_dir

        # Shared, process-safe state read/written by the parent.
        self.recording_flag = Value("b", False)
        self.block_index = Value("i", 1)
        self.label_index = Value("i", 0)
        self.last_prediction = Value("i", -1)
        self.prediction_count = Value("i", 0)  # bumped after each prediction

        self._running = Event()
        self._running.set()

    # --------------------------------------------------------------------- #
    # Child-process entry point
    # --------------------------------------------------------------------- #
    def run(self) -> None:
        # Import inside run() so the heavy libraries load in the CHILD process.
        import numpy as np
        from .board import KnightBoard
        from .preprocessing import filter_eeg, extract_channel_matrix, crop_indices

        board = KnightBoard(cfg.SERIAL_PORT, cfg.NUM_CHANNELS, cfg.CHANNEL_GAIN)
        board.start_stream()

        crop = crop_indices()

        # Predict mode needs a trained model up front.
        model = None
        if self.mode == "predict":
            import pydirectinput as pdi
            pdi.FAILSAFE = False
            from .trca_model import fit_model
            model = fit_model(self.data_dir, self.n_blocks)

        if self.mode == "collect":
            os.makedirs(self.data_dir, exist_ok=True)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        prev_flag = False
        while self._running.is_set():
            flag = self.recording_flag.value

            # Act only on the rising edge (False -> True) so each request
            # triggers exactly one capture.
            if flag and not prev_flag:
                data = board.get_latest(cfg.CAPTURE_SAMPLES)
                if data.shape[1] >= cfg.CAPTURE_SAMPLES:
                    filter_eeg(data, board.eeg_channels, board.sr)
                    window = extract_channel_matrix(data, board.eeg_channels)

                    if self.mode == "collect":
                        self._save(window, np)
                    else:
                        self._predict(window, model, crop, np, pdi)
                else:
                    print(f"[{self.mode}] WARNING: only {data.shape[1]} of "
                          f"{cfg.CAPTURE_SAMPLES} samples in buffer - trial "
                          f"skipped. Is SERIAL_PORT ({cfg.SERIAL_PORT}) the board?")

            prev_flag = flag

        board.stop_stream()

    # --------------------------------------------------------------------- #
    # Mode-specific handlers
    # --------------------------------------------------------------------- #
    def _save(self, window, np) -> None:
        """Write one captured trial to block_{block}_{label+1}.csv."""
        block = self.block_index.value
        trial = self.label_index.value + 1  # file trial index is 1-based
        path = os.path.join(self.data_dir, f"block_{block}_{trial}.csv")
        header = ",".join(cfg.ELECTRODE_LABELS[:window.shape[1]])
        np.savetxt(path, window, delimiter=",", header=header,
                   comments="", fmt="%.7f")
        print(f"[collect] saved {os.path.basename(path)}")

    def _predict(self, window, model, crop, np, pdi) -> None:
        """Classify one captured trial and press the mapped key."""
        # meegkit expects (samples, channels, trials); add a trial axis + crop.
        trial = np.expand_dims(window, axis=2)[crop]
        target = int(model.predict(trial)[0])
        self.last_prediction.value = target

        freq = cfg.STIMULUS_FREQUENCIES[target]
        n = self.prediction_count.value + 1
        if self.log_dir:
            self._log(n, target, freq, window, np)
        # Bump the counter last, so the parent only sees it once logging is done.
        self.prediction_count.value = n

        if self.labels is not None:
            print(f"\n>>> flash {n}: {self.labels[target].upper()} <<<   "
                  f"(target {target}, {freq} Hz)\n", flush=True)
            return

        key = self.key_map.get(target)
        print(f"[predict] target {target} ({freq} Hz) -> key '{key}'")
        if key is not None:
            pdi.press(key)

    def _log(self, n, target, freq, window, np) -> None:
        """Append one row to predictions.csv and save the EEG window used."""
        import datetime
        name = f"flash_{n:03d}.csv"
        header = ",".join(cfg.ELECTRODE_LABELS[:window.shape[1]])
        np.savetxt(os.path.join(self.log_dir, name), window, delimiter=",",
                   header=header, comments="", fmt="%.7f")

        summary = os.path.join(self.log_dir, "predictions.csv")
        new_file = not os.path.exists(summary)
        label = self.labels[target] if self.labels else self.key_map.get(target)
        with open(summary, "a", encoding="utf-8") as f:
            if new_file:
                f.write("flash,time,target,label,freq_hz,eeg_file\n")
            f.write(f"{n},{datetime.datetime.now().isoformat(timespec='seconds')},"
                    f"{target},{label},{freq},{name}\n")

    def stop(self) -> None:
        self._running.clear()
