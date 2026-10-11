"""
test_directions.py
==================

Test the trained classifier one flash at a time, without pressing any keys.

The four squares sit still (black) until you press SPACE or clench your jaw
twice (~0.3-1.5 s apart; see ssvep/jaw.py and JAW_* in config.py). Then they flash once
for FLICKER_DURATION seconds; look at the square you want. When the flash ends,
the classifier's guess is shown in the middle of the screen and printed in the
terminal, e.g.  >>> flash 3: LEFT <<<

If a trained EEGNet (models/eegnet.pt, from train_eegnet.py) passed its
accuracy gate, TRCA and EEGNet must agree: if they disagree the screen shows
NO MATCH and you simply flash again.

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
from ssvep.recording import BAD_SIGNAL, REJECTED, RecordingProcess
from ssvep.stimulus import build_window, build_stimuli, show_message, run_trial

# Train on the latest recording (whatever collect_training_data.py last
# wrote). It must have been recorded with the current STIMULUS_FREQUENCIES.
DATA_DIR = cfg.TRAINING_DATA_DIR

# One folder per session, e.g. test_logs/2026-10-10_11-05-12/
LOG_DIR = os.path.join(cfg.PROJECT_ROOT, "test_logs",
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Target index (same order as the squares in stimulus.py) -> direction name.
DIRECTIONS = dict(enumerate(cfg.TARGET_LABELS))

# Where each direction label is drawn: just above the top squares and just
# below the bottom squares, so the text never overlaps the flicker.
_LABEL_POS = [(-0.5, 0.85), (0.5, 0.85), (-0.5, -0.85), (0.5, -0.85)]

# Idle instructions shown between the squares.
PROMPT = ("SPACE or 2x jaw clench = flash\nESC = quit" if cfg.JAW_TRIGGER
          else "SPACE = flash\nESC = quit")

# How many past guesses to list on screen.
HISTORY_LEN = 8

# How long to wait for the classifier after a flash before giving up [s].
PREDICTION_TIMEOUT = 2.0


class JawMeter:
    """Live bar: current jaw-muscle level vs. the trigger threshold."""

    WIDTH, HEIGHT, Y = 0.4, 0.035, 0.38

    def __init__(self, win):
        self.factor = cfg.JAW_THRESHOLD_FACTOR
        self.full = 2 * self.factor            # bar is full at 2x threshold
        left = -self.WIDTH / 2
        self.back = visual.Rect(win, width=self.WIDTH, height=self.HEIGHT,
                                pos=(0, self.Y), fillColor=[-0.6] * 3, lineColor=None)
        self.fill = visual.Rect(win, width=0.001, height=self.HEIGHT, pos=(left, self.Y),
                                anchor="left", fillColor="grey", lineColor=None)
        x_thr = left + self.WIDTH * self.factor / self.full
        self.mark = visual.Line(win, start=(x_thr, self.Y - self.HEIGHT),
                                end=(x_thr, self.Y + self.HEIGHT), lineColor="white")
        self.text = visual.TextStim(win, text="", pos=(0, self.Y + 0.06),
                                    color="grey", height=0.035)

    def update(self, ratio: float) -> None:
        if ratio < 0:
            self.text.text = "NO SIGNAL - check electrodes"
            self.text.color = "red"
            self.fill.width = 0.001
            return
        self.text.color = "grey"
        if ratio == 0:
            self.text.text = "jaw: warming up..."
            self.fill.width = 0.001
            return
        self.fill.width = max(0.001, self.WIDTH * min(ratio / self.full, 1.0))
        self.fill.fillColor = "limegreen" if ratio >= self.factor else "grey"
        self.text.text = f"jaw {ratio:.1f}x  (trigger at {self.factor}x)"

    def draw(self) -> None:
        for item in (self.back, self.fill, self.mark, self.text):
            item.draw()


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
    if not recorder.wait_until_ready():
        print("Board/model setup failed - see the error above.", flush=True)
        show_message(win, "Setup failed - see the terminal.")
        core.wait(3)
        recorder.stop()
        recorder.join(timeout=5)
        win.close()
        core.quit()

    squares, cues = build_stimuli(win)

    # Everything text-based lives in the empty column between the squares.
    labels = [visual.TextStim(win, text=DIRECTIONS[i].upper(), pos=pos,
                              color="white", height=0.08)
              for i, pos in enumerate(_LABEL_POS)]
    result = visual.TextStim(win, text="", pos=(0, 0.1), color="yellow",
                             height=0.12, bold=True)
    detail = visual.TextStim(win, text="", pos=(0, -0.02), color="white",
                             height=0.045, wrapWidth=0.48)
    prompt = visual.TextStim(win, text=PROMPT,
                             pos=(0, -0.15), color="grey", height=0.04,
                             wrapWidth=0.48)
    history = visual.TextStim(win, text="", pos=(0, -0.3), color="white",
                              height=0.04, wrapWidth=0.48, anchorVert="top")
    texts = [*labels, result, detail, prompt, history]
    meter = JawMeter(win) if cfg.JAW_TRIGGER else None
    if meter is not None:
        texts.append(meter)
    past = []

    # The terminal usually keeps keyboard focus after launch (especially with
    # a second monitor), which swallows SPACE. Pull focus to the stimulus window.
    try:
        win.winHandle.activate()
    except AttributeError:
        pass
    event.clearEvents()

    how = "SPACE or clench your jaw twice" if cfg.JAW_TRIGGER else "SPACE"
    print(f"\nReady. Press {how} to flash.\n"
          f"Logging to {LOG_DIR}\n", flush=True)
    flash = 0
    jaw_seen = recorder.jaw_count.value
    try:
        while True:
            # Jaw clenches only count while this idle screen is showing.
            recorder.jaw_enabled.value = True
            if meter is not None:
                meter.update(recorder.jaw_ratio.value)
            draw_idle(win, squares, texts)
            keys = event.getKeys(keyList=["space", "escape"])
            if "escape" in keys:
                break
            jaw_fired = recorder.jaw_count.value != jaw_seen
            if "space" not in keys and not jaw_fired:
                continue
            recorder.jaw_enabled.value = False
            jaw_seen = recorder.jaw_count.value

            flash += 1
            trigger = "jaw clench" if jaw_fired and "space" not in keys else "SPACE"
            print(f"flash {flash}: flashing ({trigger})...", flush=True)
            if trigger == "jaw clench":
                # Let the clench's muscle activity fade (it would swamp the EEG
                # window) and give the eyes time to settle on a square.
                prompt.text = "clench detected\nlook at a square..."
                draw_idle(win, squares, texts)
                core.wait(cfg.JAW_START_DELAY)
                prompt.text = PROMPT
            count_before = recorder.prediction_count.value
            if not run_trial(win, squares, cues, target_idx=-1,
                             recording_process=recorder):
                break
            event.clearEvents()  # ignore SPACE presses made mid-flash

            if wait_for_prediction(recorder, count_before):
                target = recorder.last_prediction.value
                votes = ""
                if recorder.eegnet_active.value and target != BAD_SIGNAL:
                    votes = (f"\nTRCA {DIRECTIONS[recorder.last_trca.value].upper()}"
                             f" / EEGNet {DIRECTIONS[recorder.last_eegnet.value].upper()}")
                if target == BAD_SIGNAL:
                    result.color = "red"
                    result.text = "BAD SIGNAL"
                    detail.text = f"flash {flash}  |  check electrodes\n(reference / ground)"
                    past.insert(0, f"{flash}: (bad signal)")
                elif target == REJECTED:
                    result.color = "red"
                    result.text = "NO MATCH"
                    detail.text = f"flash {flash}  |  look again{votes}"
                    past.insert(0, f"{flash}: (no match)")
                else:
                    name = DIRECTIONS[target].upper()
                    freq = cfg.STIMULUS_FREQUENCIES[target]
                    result.color = "yellow"
                    result.text = name
                    detail.text = f"flash {flash}  |  {freq} Hz{votes}"
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
