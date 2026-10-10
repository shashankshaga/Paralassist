"""
collect_training_data.py
========================

Record your own SSVEP training data for the NeuroPawn Knight board.

How it works
------------
The screen shows four flickering squares (one per frequency). For each
*block* you are cued to look at every target once, in a randomised order.
Each 1.5 s flicker window is captured from the board and saved to
`training_data/block_{block}_{trial}.csv`, where `trial` (1..4) encodes which
target you looked at. Collect several blocks so the classifier has enough
repeats to learn a stable spatial filter.

Tips for good data
------------------
* Look ONLY at the cued square and keep your gaze steady for the whole flash.
* Blink between trials (during the rest), not during the flicker.
* Collect data the same way you will use the BCI later -- same posture,
  lighting and electrode placement -- so the training set reflects reality.

Run it with:  python collect_training_data.py
Press Escape at any time to stop early (data already saved is kept).
"""

from __future__ import annotations

import random

from psychopy import core

from ssvep import config as cfg
from ssvep.recording import RecordingProcess
from ssvep.stimulus import build_window, build_stimuli, show_message, run_trial

# Number of times to cue every target. More blocks = better classifier,
# but a longer, more tiring session. 6-16 is a sensible range.
N_BLOCKS = 6


def main() -> None:
    win = build_window()
    show_message(win, "Setting up the Knight board...\nThis takes ~30 seconds, please wait.")

    # Start the background board reader in COLLECT mode and wait for the
    # channel-configuration sequence to finish before the first trial.
    recorder = RecordingProcess(mode="collect", data_dir=cfg.TRAINING_DATA_DIR)
    recorder.start()
    core.wait(30)

    squares, cues = build_stimuli(win)

    try:
        for block in range(1, N_BLOCKS + 1):
            recorder.block_index.value = block

            show_message(win, f"Block {block} of {N_BLOCKS}\n\nGet ready...")
            core.wait(2)

            # Randomise the order targets are cued in this block. The label is
            # saved by target identity (not presentation order), so the data
            # stays correctly labelled while avoiding an order confound.
            order = list(range(cfg.N_TARGETS))
            random.shuffle(order)

            for target_idx in order:
                if not run_trial(win, squares, cues, target_idx, recorder):
                    raise KeyboardInterrupt

        show_message(win, "Done! Training data saved.\nYou can close this window.")
        core.wait(2)
    except KeyboardInterrupt:
        print("Stopped early by user.")
    finally:
        recorder.stop()
        recorder.join(timeout=5)
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
