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
  last_prediction: int   - most recent final decision (predict mode):
                           target index, -1 = none yet, REJECTED = models disagreed.
  last_trca / last_eegnet: int - each model's own prediction (-1 = not used).
"""

from __future__ import annotations

import os
import time
from multiprocessing import Process, Value, Event

from . import config as cfg

# last_prediction value meaning "TRCA and EEGNet disagreed: look again".
REJECTED = -2
# last_prediction value meaning "the EEG of this flash was flat or wildly
# abnormal (electrode contact lost): not classified".
BAD_SIGNAL = -3


class RecordingProcess(Process):
    """Owns the board in a child process and reacts to the recording flag."""

    def __init__(self, mode: str, data_dir: str = cfg.TRAINING_DATA_DIR,
                 n_blocks: int | None = None, key_map: dict | None = None,
                 labels: dict | None = None, log_dir: str | None = None,
                 eegnet_path: str | None = cfg.EEGNET_MODEL_PATH,
                 jaw_trigger: bool = cfg.JAW_TRIGGER):
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
        # Trained EEGNet used to cross-check TRCA (None = TRCA only).
        self.eegnet_path = eegnet_path
        # Watch for double jaw clenches (predict mode only) -- see jaw.py.
        self.jaw_trigger = jaw_trigger and mode == "predict"

        # Shared, process-safe state read/written by the parent.
        self.recording_flag = Value("b", False)
        self.block_index = Value("i", 1)
        self.label_index = Value("i", 0)
        self.last_prediction = Value("i", -1)
        self.prediction_count = Value("i", 0)  # bumped after each prediction
        self.last_trca = Value("i", -1)
        self.last_eegnet = Value("i", -1)
        self.eegnet_active = Value("b", False)  # set once the child has loaded it
        # Jaw trigger: the parent sets jaw_enabled only while the screen is idle;
        # the child bumps jaw_count on every double clench seen while enabled.
        self.jaw_enabled = Value("b", False)
        self.jaw_count = Value("i", 0)
        # Live muscle level / resting level (shown as a meter; 0 = not ready).
        self.jaw_ratio = Value("d", 0.0)
        # Set by the child once the board, models and detector are all set up.
        self.ready = Value("b", False)

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
            self._eegnet = self._load_eegnet()

        if self.mode == "collect":
            os.makedirs(self.data_dir, exist_ok=True)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        jaw = self._start_jaw_detector(board) if self.jaw_trigger else None
        self.ready.value = True

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
            if jaw is not None:
                self._poll_jaw(board, jaw)
            time.sleep(0.002)  # don't spin a CPU core at 100% while waiting

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

    def _start_jaw_detector(self, board):
        """Create the jaw-clench detector and the state used to feed it."""
        from .jaw import JawClenchDetector

        self._jaw_ts_row = board.board_shim.get_timestamp_channel(board.board_id)
        self._jaw_last_ts = 0.0
        self._jaw_next_poll = 0.0
        self._jaw_was_enabled = False
        # Trace of what the detector sees, for diagnosing missed/false triggers.
        self._jaw_trace = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self._jaw_trace = open(os.path.join(self.log_dir, "jaw_trace.csv"),
                                   "w", encoding="utf-8")
            self._jaw_trace.write("time_s,level,resting,threshold,ratio,enabled,trigger\n")
        self._jaw_t0 = time.monotonic()
        print(f"[jaw] double-clench trigger on "
              f"(threshold {cfg.JAW_THRESHOLD_FACTOR}x resting level)")
        return JawClenchDetector(sfreq=board.sr, n_channels=len(board.eeg_channels))

    def _poll_jaw(self, board, jaw) -> None:
        """Feed samples that arrived since the last poll (~every 40 ms)."""
        now = time.monotonic()
        if now < self._jaw_next_poll:
            return
        self._jaw_next_poll = now + 0.04

        # Non-destructive peek at the last 2 s; keep only samples not seen yet
        # (by timestamp), so the trial capture's buffer is never disturbed.
        data = board.get_latest(2 * board.sr)
        if data.shape[1] == 0:
            return
        new = data[:, data[self._jaw_ts_row] > self._jaw_last_ts]
        if new.shape[1] == 0:
            return
        self._jaw_last_ts = new[self._jaw_ts_row, -1]

        # Always feed (keeps the filter and resting level continuous), but only
        # act on clenches while the parent says the screen is idle.
        events = jaw.feed(new[board.eeg_channels])
        enabled = bool(self.jaw_enabled.value)
        if enabled and not self._jaw_was_enabled:
            jaw.reset()        # drop half a pair left over from before the flash
            events = []
        self._jaw_was_enabled = enabled
        fired = enabled and bool(events)
        if fired:
            self.jaw_count.value += 1
            print("[jaw] double clench detected -> starting flash", flush=True)

        if not jaw.signal_ok:
            ratio = -1.0           # meter shows "NO SIGNAL"
        else:
            ratio = 0.0 if not jaw.baseline else jaw.envelope / jaw.baseline
        self.jaw_ratio.value = ratio
        if self._jaw_trace is not None:
            thr = jaw.threshold()
            self._jaw_trace.write(
                f"{now - self._jaw_t0:.3f},{jaw.envelope:.1f},{jaw.baseline or 0:.1f},"
                f"{thr or 0:.1f},{ratio:.2f},{int(enabled)},{int(fired)}\n")
            self._jaw_trace.flush()

    def _load_eegnet(self):
        """
        Load the EEGNet checkpoint if it may be used, else return None.

        It is used only if it exists, passed the accuracy gate when trained, and
        was trained on exactly the data TRCA was just fitted on.
        """
        if not self.eegnet_path or not os.path.exists(self.eegnet_path):
            print("[predict] no EEGNet model found -> TRCA only")
            return None
        from . import eegnet

        model, meta = eegnet.load_checkpoint(self.eegnet_path)
        accs = (f"TRCA {meta['trca_cv_accuracy']:.1f}%, "
                f"EEGNet {meta['eegnet_cv_accuracy']:.1f}%")
        if meta["data_fingerprint"] != eegnet.data_fingerprint(self.data_dir):
            print("[predict] EEGNet was trained on different data -> TRCA only "
                  "(re-run train_eegnet.py)")
            return None
        if not meta["use_eegnet"]:
            print(f"[predict] EEGNet failed the gate ({accs}) -> TRCA only")
            return None
        print(f"[predict] EEGNet loaded ({accs}) -> TRCA + EEGNet must agree")
        self.eegnet_active.value = True
        return model

    def _predict(self, window, model, crop, np, pdi) -> None:
        """Classify one captured trial and, if accepted, press the mapped key."""
        from .hybrid import decide

        # Refuse to classify garbage: a flat or huge window means the electrodes
        # (usually reference/ground) lost contact, and any guess would be random.
        level = float(np.median(window.std(axis=0)))
        low, high = cfg.SIGNAL_STD_RANGE
        if not (low <= level <= high):
            n = self.prediction_count.value + 1
            self.last_trca.value = self.last_eegnet.value = -1
            self.last_prediction.value = BAD_SIGNAL
            if self.log_dir:
                self._log(n, None, None, None, window, np, label="BAD_SIGNAL")
            self.prediction_count.value = n
            print(f"\n>>> flash {n}: BAD SIGNAL ({level:.0f} uV, expected "
                  f"{low:.0f}-{high:.0f}) - check electrodes <<<\n", flush=True)
            return

        # meegkit expects (samples, channels, trials); add a trial axis + crop.
        trial = np.expand_dims(window, axis=2)[crop]
        trca_target = int(model.predict(trial)[0])

        eegnet_target = None
        if self._eegnet is not None:
            from .eegnet import live_window, predict
            eegnet_target = int(predict(self._eegnet, live_window(window))[0])

        # Both must agree (if EEGNet is in use); None = rejected, look again.
        target = decide(trca_target, eegnet_target)
        self.last_trca.value = trca_target
        self.last_eegnet.value = -1 if eegnet_target is None else eegnet_target
        self.last_prediction.value = REJECTED if target is None else target

        n = self.prediction_count.value + 1
        if self.log_dir:
            self._log(n, trca_target, eegnet_target, target, window, np)
        # Bump the counter last, so the parent only sees it once logging is done.
        self.prediction_count.value = n

        def name(t):
            return (self.labels or {}).get(t, str(t)).upper()

        both = ("" if eegnet_target is None else
                f"   [TRCA {name(trca_target)} / EEGNet {name(eegnet_target)}]")
        if target is None:
            print(f"\n>>> flash {n}: NO MATCH - look again <<<{both}\n", flush=True)
            return

        freq = cfg.STIMULUS_FREQUENCIES[target]
        if self.labels is not None:
            print(f"\n>>> flash {n}: {name(target)} <<<   "
                  f"(target {target}, {freq} Hz){both}\n", flush=True)
            return

        key = self.key_map.get(target)
        print(f"[predict] target {target} ({freq} Hz) -> key '{key}'{both}")
        if key is not None:
            pdi.press(key)

    def _log(self, n, trca_target, eegnet_target, target, window, np,
             label: str | None = None) -> None:
        """Append one row to predictions.csv and save the EEG window used."""
        import datetime
        name = f"flash_{n:03d}.csv"
        header = ",".join(cfg.ELECTRODE_LABELS[:window.shape[1]])
        np.savetxt(os.path.join(self.log_dir, name), window, delimiter=",",
                   header=header, comments="", fmt="%.7f")

        summary = os.path.join(self.log_dir, "predictions.csv")
        new_file = not os.path.exists(summary)
        names = self.labels if self.labels else self.key_map
        if target is None:
            label, freq = label or "REJECTED", ""
        else:
            label, freq = names.get(target), cfg.STIMULUS_FREQUENCIES[target]
        trca = "" if trca_target is None else trca_target
        eeg = "" if eegnet_target is None else eegnet_target
        final = "" if target is None else target
        with open(summary, "a", encoding="utf-8") as f:
            if new_file:
                f.write("flash,time,trca,eegnet,final,label,freq_hz,eeg_file\n")
            f.write(f"{n},{datetime.datetime.now().isoformat(timespec='seconds')},"
                    f"{trca},{eeg},{final},{label},{freq},{name}\n")

    def wait_until_ready(self, timeout: float = 180.0) -> bool:
        """
        Block (in the parent) until the child has finished setting up the board
        and models. Returns False if the child died or `timeout` passed.

        Replaces a fixed sleep: setup time varies (board config ~20 s, plus
        TRCA training and loading PyTorch/EEGNet), and anything started before
        the child is ready is silently lost.
        """
        start = time.monotonic()
        while not self.ready.value:
            if not self.is_alive():
                return False
            if time.monotonic() - start > timeout:
                return False
            time.sleep(0.1)
        return True

    def stop(self) -> None:
        self._running.clear()
