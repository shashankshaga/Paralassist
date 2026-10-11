"""
train_eegnet.py
===============

Train the EEGNet classifier locally on the recorded training data and decide
whether it is good enough to cross-check TRCA during live use.

Steps
-----
1. Leave-one-block-out validation of TRCA and EEGNet on training_data/.
2. Gate: EEGNet is used live only if its accuracy is at most
   hybrid.EEGNET_MARGIN (10) points below TRCA's.
3. Train the final EEGNet on all blocks and save models/eegnet.pt together
   with both accuracies, the gate decision and a fingerprint of the data.

Live scripts load the .pt file and only use it if the gate passed AND the
fingerprint matches the training data TRCA is fitted on.

Run it with:  python train_eegnet.py
"""

from __future__ import annotations

import time

import torch

from ssvep import config as cfg, eegnet, hybrid
from ssvep.trca_model import cross_validate_accuracy


def main() -> None:
    X, y, blocks = eegnet.load_trials(cfg.TRAINING_DATA_DIR)
    n_blocks = len(set(blocks.tolist()))
    print(f"Loaded {len(y)} trials ({n_blocks} blocks) from {cfg.TRAINING_DATA_DIR}")
    if n_blocks < 2:
        raise SystemExit("Need at least 2 blocks for leave-one-block-out validation.")

    t0 = time.time()
    trca_acc = cross_validate_accuracy(X, y, blocks)
    print(f"\nTRCA   leave-one-block-out accuracy: {trca_acc:.1f}%")

    print(f"EEGNet leave-one-block-out validation ({n_blocks} models, in parallel)...")
    eegnet_acc, per_block = eegnet.cross_validate(X, y, blocks, **eegnet.TRAIN_PARAMS)
    print("  per block:", {b: f"{a:.0f}%" for b, a in per_block.items()})
    print(f"EEGNet leave-one-block-out accuracy: {eegnet_acc:.1f}%")

    use = hybrid.should_use_eegnet(trca_acc, eegnet_acc)
    need = trca_acc - hybrid.EEGNET_MARGIN
    print(f"\nGate: EEGNet needs >= {need:.1f}% -> "
          + ("PASSED: live mode = TRCA + EEGNet must agree"
             if use else "FAILED: live mode = TRCA only"))

    print("\nTraining final EEGNet on all blocks...")
    torch.set_num_threads(1)  # tiny model: one thread is ~2x faster than many
    model = eegnet.train_model(X, y, blocks, **eegnet.TRAIN_PARAMS)
    metadata = dict(
        trca_cv_accuracy=trca_acc,
        eegnet_cv_accuracy=eegnet_acc,
        eegnet_cv_per_block=per_block,
        use_eegnet=use,
        margin=hybrid.EEGNET_MARGIN,
        n_trials=int(len(y)),
        n_blocks=n_blocks,
        data_fingerprint=eegnet.data_fingerprint(cfg.TRAINING_DATA_DIR),
        labels=list(cfg.TARGET_LABELS),
        stimulus_frequencies=list(cfg.STIMULUS_FREQUENCIES),
        train_params=eegnet.TRAIN_PARAMS,
        trained_on="local",
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    eegnet.save_checkpoint(cfg.EEGNET_MODEL_PATH, model, metadata)
    print(f"Saved {cfg.EEGNET_MODEL_PATH}  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
