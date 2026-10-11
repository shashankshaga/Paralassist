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

Run it with:  python collect_training_data.py            (6 blocks)
              python collect_training_data.py --blocks 12
Any data already in training_data/ is first moved to training_data_<date_time>/
so two sessions never get mixed.
Press Escape at any time to stop early (data already saved is kept).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import random
import shutil

from psychopy import core

from ssvep import config as cfg
from ssvep.recording import RecordingProcess
from ssvep.stimulus import build_window, build_stimuli, show_message, run_trial

# Default number of times to cue every target. More blocks = better classifier
# (EEGNet especially), but a longer, more tiring session. 6-16 is sensible.
N_BLOCKS = 6


def archive_existing_data(data_dir: str) -> None:
    """Move any previous session out of the way so sessions never mix."""
    if not glob.glob(os.path.join(data_dir, "block_*_*.csv")):
        return
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup = f"{data_dir}_{stamp}"
    shutil.move(data_dir, backup)
    print(f"Previous training data moved to {backup}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record SSVEP training data.")
    parser.add_argument("--blocks", type=int, default=N_BLOCKS,
                        help=f"number of blocks to record (default {N_BLOCKS})")
    n_blocks = parser.parse_args().blocks

    archive_existing_data(cfg.TRAINING_DATA_DIR)

    win = build_window()
    show_message(win, "Setting up the Knight board...\nThis takes ~30 seconds, please wait.")

    # Start the background board reader in COLLECT mode and wait for the
    # channel-configuration sequence to finish before the first trial.
    recorder = RecordingProcess(mode="collect", data_dir=cfg.TRAINING_DATA_DIR)
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

    try:
        for block in range(1, n_blocks + 1):
            recorder.block_index.value = block

            show_message(win, f"Block {block} of {n_blocks}\n\nGet ready...")
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
