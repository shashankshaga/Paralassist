"""
realtime_control.py
===================

Use your trained SSVEP-TRCA classifier to control the keyboard in real time.

How it works
------------
1. A background process connects to the Knight board and fits a TRCA model on
   the data you recorded with `collect_training_data.py`.
2. The four squares flicker continuously. You look at whichever target you
   want to trigger.
3. After each 1.5 s flicker window the classifier decides which frequency you
   were looking at and "presses" the mapped key (see cfg.KEY_MAP). Point the
   window you want to control (a game, a robot teleop, etc.) so it has focus.

Default key map (edit ssvep/config.py -> KEY_MAP to change):
    6.67 Hz (top-left)     -> w
    8.57 Hz (top-right)    -> s
    12.0 Hz (bottom-left)  -> a
    15.0 Hz (bottom-right) -> d

Run it with:  python realtime_control.py
Press Escape to stop.
"""

from __future__ import annotations

from psychopy import core

from ssvep import config as cfg
from ssvep.recording import RecordingProcess
from ssvep.stimulus import build_window, build_stimuli, show_message, run_trial


def main() -> None:
    win = build_window()
    show_message(win, "Setting up board and training TRCA...\nPlease wait (~30 s).")

    # Background board reader in PREDICT mode: fits TRCA on your training data,
    # then classifies + presses a key after every flicker window.
    recorder = RecordingProcess(mode="predict", data_dir=cfg.TRAINING_DATA_DIR)
    recorder.start()
    core.wait(30)

    squares, cues = build_stimuli(win)

    show_message(win, "Ready! Look at a target to send its key.\nEscape to quit.")
    core.wait(2)

    try:
        # Free-running control loop. There is no "correct" target here, so we
        # do not draw a cue; the user simply looks wherever they want.
        while True:
            # target_idx only affects the (unused) cue and label in predict
            # mode; pass -1 so no cue frame is highlighted.
            keep_going = run_trial(win, squares, cues, target_idx=-1,
                                   recording_process=recorder)
            if not keep_going:
                break
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        recorder.join(timeout=5)
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
