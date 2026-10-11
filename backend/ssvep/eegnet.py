"""
EEGNet data preparation, shared by cloud training and local inference.

Training (Databricks) and live prediction (this PC) must turn EEG into model
input in *exactly* the same way, or the network sees different data than it
learned from. Both sides therefore call the functions in this module.

This module only needs numpy/pandas (no BrainFlow, no PsychoPy), so it can be
imported on a Databricks cluster straight from the Git folder.

From trial to model input
-------------------------
Each training CSV is one recorded flicker trial, already detrended, band-pass
(3-48 Hz) and notch filtered by `preprocessing.filter_eeg` at record time, so
no further filtering is done here. Shape (samples, channels) = (189, 8).

1. Windowing: skip the visual latency (first 0.15 s = 19 samples) and cut
   1.0 s windows (125 samples). For training, several overlapping windows are
   cut from each trial (`stride` samples apart) -- a standard augmentation that
   multiplies the tiny dataset and makes the network robust to small timing
   shifts. Live inference uses only the first window, which is exactly the
   window TRCA uses.
2. Normalisation: each window is z-scored per channel (zero mean, unit
   variance). Raw amplitudes here are ~1000 uV and vary between sessions with
   electrode contact; z-scoring removes that scale so the network only has to
   learn the *shape* of the response.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from . import config as cfg

# Analysis window, in samples (same as preprocessing.crop_indices()).
WINDOW_START = int(np.floor(cfg.VISUAL_LATENCY * cfg.SAMPLING_RATE + 0.5))   # 19
WINDOW_LENGTH = int(np.floor(cfg.GAZE_DURATION * cfg.SAMPLING_RATE + 0.5))   # 125

_FILE_RE = re.compile(r"block_(\d+)_(\d+)\.csv$")


def load_trials(data_dir: str):
    """
    Load every block_{block}_{trial}.csv in `data_dir`.

    Works with local folders and Databricks Volume paths (/Volumes/...).

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_channels, n_samples)
        Channels-first, the layout PyTorch/braindecode expect.
    y : np.ndarray, shape (n_trials,)
        Target index 0..N_TARGETS-1 (file trial number - 1).
    blocks : np.ndarray, shape (n_trials,)
        Block number of each trial, used for leave-one-block-out validation.
    """
    trials, labels, blocks = [], [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "block_*_*.csv"))):
        match = _FILE_RE.search(os.path.basename(path))
        if not match:
            continue
        block, trial = int(match.group(1)), int(match.group(2))
        trials.append(pd.read_csv(path).values.T)   # -> (channels, samples)
        labels.append(trial - 1)
        blocks.append(block)

    if not trials:
        raise FileNotFoundError(f"No block_*_*.csv trials found in {data_dir!r}.")

    lengths = {t.shape[1] for t in trials}
    if len(lengths) > 1:
        raise ValueError(f"Trials have different lengths {sorted(lengths)} in {data_dir!r}.")

    return (np.stack(trials).astype(np.float32),
            np.array(labels, dtype=np.int64),
            np.array(blocks, dtype=np.int64))


def normalize(windows: np.ndarray) -> np.ndarray:
    """Z-score each channel of each window: (..., channels, samples)."""
    mean = windows.mean(axis=-1, keepdims=True)
    std = windows.std(axis=-1, keepdims=True)
    return ((windows - mean) / (std + 1e-6)).astype(np.float32)


def make_windows(X: np.ndarray, y: np.ndarray, blocks: np.ndarray,
                 stride: int | None = 5):
    """
    Cut normalised 1 s windows out of each trial.

    Parameters
    ----------
    X, y, blocks : arrays from `load_trials`.
    stride : int or None
        Samples between consecutive window starts. None -> one window per trial
        (the live-inference window only), e.g. for evaluation.

    Returns
    -------
    windows : (n_windows, channels, WINDOW_LENGTH) float32
    y_win, blocks_win : labels and block numbers, repeated per window.
    """
    last_start = X.shape[-1] - WINDOW_LENGTH
    if last_start < WINDOW_START:
        raise ValueError(f"Trials of {X.shape[-1]} samples are too short for a "
                         f"{WINDOW_LENGTH}-sample window starting at {WINDOW_START}.")
    starts = [WINDOW_START] if stride is None else range(WINDOW_START, last_start + 1, stride)

    windows = np.concatenate([X[..., s:s + WINDOW_LENGTH] for s in starts])
    reps = len(starts)
    return normalize(windows), np.tile(y, reps), np.tile(blocks, reps)


def live_window(window: np.ndarray) -> np.ndarray:
    """
    Turn one live capture into a model input batch of size 1.

    `window` is the (samples, channels) matrix produced by
    `preprocessing.extract_channel_matrix` right after `filter_eeg`, i.e. the
    same thing that gets saved as a training CSV.
    """
    x = window.T[:, WINDOW_START:WINDOW_START + WINDOW_LENGTH]   # (channels, samples)
    return normalize(x[np.newaxis])                               # (1, channels, samples)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
# braindecode EEGNet (https://huggingface.co/braindecode/EEGNet), sized for our
# data. Architecture-only: there are no pretrained weights, it is trained from
# scratch on our recordings. The shape arguments come from config so the cloud
# and local copies can never disagree; the rest are the card's defaults.
EEGNET_PARAMS = dict(
    n_chans=cfg.NUM_CHANNELS,          # 8 occipital electrodes
    n_outputs=cfg.N_TARGETS,           # 4 targets
    n_times=WINDOW_LENGTH,             # 125 samples = 1.0 s at 125 Hz
    sfreq=cfg.SAMPLING_RATE,
    F1=8,                              # temporal filters
    D=2,                               # spatial filters per temporal filter
    kernel_length=64,                  # ~0.5 s temporal kernel (~2 Hz resolution)
    drop_prob=0.25,
    # braindecode's default (0.01) makes BatchNorm's running statistics adapt
    # so slowly that, on a dataset this small, they are still near their initial
    # values after training -- the model then scores chance (25%) in eval mode
    # even when training accuracy is ~100%. PyTorch's default 0.1 fixes it.
    batch_norm_momentum=0.1,
)


def build_model(**overrides):
    """Create an untrained braindecode EEGNet. Keyword args override EEGNET_PARAMS."""
    from braindecode.models import EEGNet  # torch is only needed from here on

    return EEGNet(**{**EEGNET_PARAMS, **overrides})


# --------------------------------------------------------------------------- #
# Training, validation, export
# --------------------------------------------------------------------------- #
TRAIN_PARAMS = dict(epochs=150, batch_size=32, lr=1e-3, weight_decay=1e-2,
                    stride=5, seed=0)


def train_model(X, y, blocks, epochs=150, batch_size=32, lr=1e-3,
                weight_decay=1e-2, stride=5, seed=0, on_epoch=None):
    """
    Train a fresh EEGNet on trials X (n_trials, channels, samples).

    `on_epoch(epoch, loss)` is called after every epoch (e.g. for MLflow).
    Returns the model in eval mode.
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    windows, labels, _ = make_windows(X, y, blocks, stride=stride)
    xb, yb = torch.from_numpy(windows), torch.from_numpy(labels)

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xb))
        total = 0.0
        for i in range(0, len(xb), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xb[idx]), yb[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if on_epoch is not None:
            on_epoch(epoch, total / len(xb))
    return model.eval()


def predict(model, windows: np.ndarray) -> np.ndarray:
    """Predicted target per window; `windows` is (n, channels, samples), normalised."""
    import torch

    with torch.no_grad():
        return model(torch.from_numpy(windows)).argmax(1).numpy()


def _fold_accuracy(X, y, blocks, block, train_kw):
    """Train without `block`, return accuracy (%) on it. Runs in a worker process."""
    import torch

    # EEGNet is tiny, so multi-threading costs more than it gains: one thread
    # per model is ~2x faster, and the folds run in parallel instead.
    torch.set_num_threads(1)
    test = blocks == block
    model = train_model(X[~test], y[~test], blocks[~test], **train_kw)
    windows, labels, _ = make_windows(X[test], y[test], blocks[test], stride=None)
    return float((predict(model, windows) == labels).mean() * 100)


def cross_validate(X, y, blocks, n_jobs: int | None = None, **train_kw):
    """
    Leave-one-block-out accuracy, tested on the live-inference window only.

    The folds are independent, so they are trained in parallel across
    `n_jobs` processes (default: CPU cores - 2). Results are identical to a
    sequential run because every fold is seeded.

    Returns (accuracy_percent, {block: accuracy_percent}).
    """
    from concurrent.futures import ProcessPoolExecutor

    unique = [int(b) for b in np.unique(blocks)]
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 2)
    n_jobs = min(n_jobs, len(unique))

    if n_jobs == 1:
        accs = [_fold_accuracy(X, y, blocks, b, train_kw) for b in unique]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            accs = list(pool.map(_fold_accuracy, *zip(*[(X, y, blocks, b, train_kw)
                                                        for b in unique])))
    per_block = dict(zip(unique, accs))
    return float(np.mean(accs)), per_block


def data_fingerprint(data_dir: str) -> str:
    """SHA-1 over every trial CSV's name and contents: identifies a dataset."""
    import hashlib

    h = hashlib.sha1()
    for path in sorted(glob.glob(os.path.join(data_dir, "block_*_*.csv"))):
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def save_checkpoint(path: str, model, metadata: dict) -> None:
    """Save weights + architecture + metadata (accuracies, gate, labels...) to .pt."""
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "eegnet_params": EEGNET_PARAMS,
                "metadata": metadata}, path)


def load_checkpoint(path: str):
    """Rebuild the trained EEGNet from a .pt file. Returns (model, metadata)."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(**ckpt["eegnet_params"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["metadata"]
