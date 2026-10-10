"""
test_directions.py
==================

Test the trained classifier one flash at a time, without pressing any keys.

The four squares sit still (black) until you press SPACE. Then they flash once
for FLICKER_DURATION seconds; look at the square you want. When the flash ends,
the classifier's guess is shown in the middle of the screen and printed in the
terminal, e.g.  >>> flash 3: LEFT <<<

Every flash is also logged to test_logs/<session>/: one row per flash in
predictions.csv, plus flash_NNN.csv holding the filtered EEG window it used.

Square -> direction:
    top-left     (6.67 Hz) -> LEFT
    top-right    (8.57 Hz) -> RIGHT
    bottom-left  (12.0 Hz) -> UP
    bottom-right (15.0 Hz) -> DOWN

Run it with:  python test_directions.py
Press Escape to quit.
"""

from __future__ import annotations

import datetime
import os

from psychopy import core, event, visual

from ssvep import config as cfg
from ssvep.recording import RecordingProcess
from ssvep.stimulus import build_window, build_stimuli, show_message, run_trial

# Train on the latest recording (whatever collect_training_data.py last
# wrote). It must have been recorded with the current STIMULUS_FREQUENCIES.
DATA_DIR = cfg.TRAINING_DATA_DIR

# One folder per session, e.g. test_logs/2026-10-10_11-05-12/
LOG_DIR = os.path.join(cfg.PROJECT_ROOT, "test_logs",
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Target index (same order as the squares in stimulus.py) -> direction name.
DIRECTIONS = {0: "left", 1: "right", 2: "up", 3: "down"}

# Where each direction label is drawn: just above the top squares and just
# below the bottom squares, so the text never overlaps the flicker.
_LABEL_POS = [(-0.5, 0.85), (0.5, 0.85), (-0.5, -0.85), (0.5, -0.85)]

# How many past guesses to list on screen.
HISTORY_LEN = 8

# How long to wait for the classifier after a flash before giving up [s].
PREDICTION_TIMEOUT = 2.0


def draw_idle(win, squares, texts) -> None:
    """Draw the still (black) squares plus all the on-screen text."""
    for sq in squares:
        sq.fillColor = [-1, -1, -1]
        sq.draw()
    for txt in texts:
        txt.draw()
    win.flip()


def wait_for_prediction(recorder, count_before: int) -> bool:
    """Block until the recorder reports a new prediction (or time out)."""
    clock = core.Clock()
    while clock.getTime() < PREDICTION_TIMEOUT:
        if recorder.prediction_count.value > count_before:
            return True
        core.wait(0.02)
    return False


def main() -> None:
    win = build_window()
    show_message(win, "Setting up board and training TRCA...\nPlease wait (~30 s).")

    recorder = RecordingProcess(mode="predict", data_dir=DATA_DIR,
                                labels=DIRECTIONS, log_dir=LOG_DIR)
    recorder.start()
    core.wait(30)

    squares, cues = build_stimuli(win)

    # Everything text-based lives in the empty column between the squares.
    labels = [visual.TextStim(win, text=DIRECTIONS[i].upper(), pos=pos,
                              color="white", height=0.08)
              for i, pos in enumerate(_LABEL_POS)]
    result = visual.TextStim(win, text="", pos=(0, 0.1), color="yellow",
                             height=0.12, bold=True)
    detail = visual.TextStim(win, text="", pos=(0, -0.02), color="white",
                             height=0.045, wrapWidth=0.48)
    prompt = visual.TextStim(win, text="SPACE = flash\nESC = quit",
                             pos=(0, -0.15), color="grey", height=0.04,
                             wrapWidth=0.48)
    history = visual.TextStim(win, text="", pos=(0, -0.3), color="white",
                              height=0.04, wrapWidth=0.48, anchorVert="top")
    texts = [*labels, result, detail, prompt, history]
    past = []

    # The terminal usually keeps keyboard focus after launch (especially with
    # a second monitor), which swallows SPACE. Pull focus to the stimulus window.
    try:
        win.winHandle.activate()
    except AttributeError:
        pass
    event.clearEvents()

    print(f"\nReady. Press SPACE in the stimulus window to flash.\n"
          f"Logging to {LOG_DIR}\n", flush=True)
    flash = 0
    try:
        while True:
            draw_idle(win, squares, texts)
            keys = event.getKeys(keyList=["space", "escape"])
            if "escape" in keys:
                break
            if "space" not in keys:
                continue

            flash += 1
            print(f"flash {flash}: flashing...", flush=True)
            count_before = recorder.prediction_count.value
            if not run_trial(win, squares, cues, target_idx=-1,
                             recording_process=recorder):
                break
            event.clearEvents()  # ignore SPACE presses made mid-flash

            if wait_for_prediction(recorder, count_before):
                target = recorder.last_prediction.value
                name = DIRECTIONS[target].upper()
                freq = cfg.STIMULUS_FREQUENCIES[target]
                result.text = name
                detail.text = f"flash {flash}  |  {freq} Hz"
                past.insert(0, f"{flash}: {name}")
            else:
                result.text = "no data"
                detail.text = f"flash {flash}  |  board sent too few samples"
                past.insert(0, f"{flash}: (no data)")
                print(f"flash {flash}: no prediction received", flush=True)
            history.text = "\n".join(past[:HISTORY_LEN])
    finally:
        recorder.stop()
        recorder.join(timeout=5)
        print(f"\n{flash} flash(es) logged to {LOG_DIR}", flush=True)
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
