"""
calibrate_jaw.py
================

Check (and tune) the double-jaw-clench trigger on YOUR signal. ~45 s.

The terminal tells you what to do:
  * RELAX for 12 s   -> measures your resting muscle level
  * 4x "CLENCH TWICE" -> two short, firm clenches, about half a second apart
  * 1x "CLENCH ONCE"  -> must NOT trigger

Afterwards it reports how far each clench rose above your resting level, which
cues the detector caught, and a suggested JAW_THRESHOLD_FACTOR for config.py.
The raw recording is saved to jaw_calibration/ so it can be re-analysed.

Run it with:  python calibrate_jaw.py

Free mode:    python calibrate_jaw.py --free 60
    Records 60 s with no cues (a seconds counter is printed so you can pace
    yourself, e.g. one clench every 5 s), then lists every muscle burst found.
Re-analyse:   python calibrate_jaw.py --file jaw_calibration/<name>.npz
"""

from __future__ import annotations

import argparse
import datetime
import os
import time

import numpy as np

from ssvep import config as cfg
from ssvep.board import KnightBoard
from ssvep.jaw import JawClenchDetector

REST = 12.0          # s of rest at the start
CUE_EVERY = 6.0      # s between cues
CUES = ["CLENCH TWICE"] * 4 + ["CLENCH ONCE"]
TAIL = 4.0           # s after the last cue


def countdown_print(text: str) -> None:
    print(f"\n   >>> {text} <<<", flush=True)


def start_board():
    board = KnightBoard(cfg.SERIAL_PORT, cfg.NUM_CHANNELS, cfg.CHANNEL_GAIN)
    board.start_stream()
    board.board_shim.get_board_data()      # discard set-up data
    return board


def save(eeg, sfreq, prefix, **extra) -> str:
    out_dir = os.path.join(cfg.PROJECT_ROOT, "jaw_calibration")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, prefix + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".npz")
    np.savez(path, eeg=eeg, sfreq=sfreq, **extra)
    print(f"\nSaved recording to {path}")
    return path


def record_free(seconds: float) -> None:
    """Record with no cues; the user clenches at their own pace."""
    board = start_board()
    print(f"\nRECORDING {seconds:.0f} s. Relax for the first 5 s, then clench "
          f"whenever you like (e.g. every 5 s).", flush=True)
    start = time.monotonic()
    tick = 0
    while (elapsed := time.monotonic() - start) < seconds:
        if elapsed >= tick:
            print(f"  {tick:3d} s", flush=True)
            tick += 1
        time.sleep(0.01)
    data = board.board_shim.get_board_data()
    board.stop_stream()
    eeg = data[board.eeg_channels]
    save(eeg, board.sr, "free_")
    list_bursts(eeg, board.sr)


def detector_trace(eeg, sfreq, factor=cfg.JAW_THRESHOLD_FACTOR, step=5):
    """Replay `eeg` through the detector; return (t, envelope, baseline, doubles)."""
    det = JawClenchDetector(sfreq=sfreq, n_channels=eeg.shape[0], factor=factor)
    env, base, events = [], [], []
    for i in range(0, eeg.shape[1], step):
        events += det.feed(eeg[:, i:i + step])
        env.append(det.envelope)
        base.append(det.baseline or np.nan)
    t = (np.arange(len(env)) + 1) * step / sfreq
    return t, np.array(env), np.array(base), events


def list_bursts(eeg, sfreq, factor=cfg.JAW_THRESHOLD_FACTOR) -> None:
    """Print every stretch where muscle activity rose above the threshold."""
    t, env, base, doubles = detector_trace(eeg, sfreq, factor)
    ratio = env / base
    above = np.nan_to_num(ratio) > factor
    edges = np.flatnonzero(np.diff(above.astype(int)))
    starts = [i + 1 for i in edges if not above[i]]
    ends = [i + 1 for i in edges if above[i]]
    if above[0]:
        starts = [0] + starts
    if len(ends) < len(starts):
        ends.append(len(above) - 1)

    print(f"\nRecording: {t[-1]:.1f} s, resting level {np.nanmedian(base):.1f}, "
          f"threshold {factor}x")
    print(f"Muscle bursts above {factor}x resting:")
    if not starts:
        print("  none found")
    prev = None
    for a, b in zip(starts, ends):
        gap = "" if prev is None else f"   (+{t[a] - prev:.1f} s since previous)"
        print(f"  {t[a]:6.1f} s  lasting {t[b] - t[a]:.2f} s  peak {np.nanmax(ratio[a:b + 1]):5.1f}x{gap}")
        prev = t[a]
    print(f"Double clenches the trigger would fire on: {[round(e, 1) for e in doubles] or 'none'}")
    quiet = ratio[np.nan_to_num(ratio) <= factor]
    if quiet.size:
        print(f"Between bursts the signal stays at {np.nanmedian(quiet):.1f}x "
              f"(95% of the time below {np.nanpercentile(quiet, 95):.1f}x).")


def main() -> None:
    board = KnightBoard(cfg.SERIAL_PORT, cfg.NUM_CHANNELS, cfg.CHANNEL_GAIN)
    board.start_stream()
    board.board_shim.get_board_data()      # discard set-up data
    start = time.monotonic()

    print(f"\nRELAX: keep your jaw loose and still for {REST:.0f} s...", flush=True)
    cue_times = []
    for i, cue in enumerate(CUES):
        t_cue = REST + i * CUE_EVERY
        while time.monotonic() - start < t_cue:
            time.sleep(0.01)
        cue_times.append(time.monotonic() - start)
        countdown_print(cue)
        time.sleep(1.5)
        print("   relax...", flush=True)
    while time.monotonic() - start < REST + len(CUES) * CUE_EVERY - CUE_EVERY + TAIL:
        time.sleep(0.05)

    data = board.board_shim.get_board_data()
    board.stop_stream()
    eeg = data[board.eeg_channels]

    out_dir = os.path.join(cfg.PROJECT_ROOT, "jaw_calibration")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".npz")
    np.savez(path, eeg=eeg, sfreq=board.sr, cue_times=np.array(cue_times), cues=np.array(CUES))
    print(f"\nSaved recording to {path}")
    analyse(eeg, board.sr, cue_times)


def analyse(eeg: np.ndarray, sfreq: float, cue_times, factor: float = cfg.JAW_THRESHOLD_FACTOR) -> None:
    """Replay the recording through the detector and summarise."""
    det = JawClenchDetector(sfreq=sfreq, n_channels=eeg.shape[0], factor=factor)
    env, base, events = [], [], []
    step = 5
    for i in range(0, eeg.shape[1], step):
        events += det.feed(eeg[:, i:i + step])
        env.append(det.envelope)
        base.append(det.baseline or np.nan)
    env, base = np.array(env), np.array(base)
    t = (np.arange(len(env)) + 1) * step / sfreq
    rest_level = np.nanmedian(base)

    print(f"\nResting level: {rest_level:.1f}   (current threshold = {factor}x = {factor * rest_level:.1f})")
    peaks = []
    for cue, t_cue in zip(CUES, cue_times):
        win = (t >= t_cue) & (t < t_cue + 3.0)
        ratio = float(np.nanmax(env[win] / base[win])) if win.any() else float("nan")
        hit = [e for e in events if t_cue <= e < t_cue + 4.0]
        ok = bool(hit) == (cue == "CLENCH TWICE")
        peaks.append(ratio)
        print(f"  {cue:13s} at {t_cue:5.1f}s: peak {ratio:5.1f}x resting  -> "
              f"{'TRIGGERED' if hit else 'no trigger'}  {'OK' if ok else '<-- WRONG'}")
    stray = [e for e in events if not any(tc <= e < tc + 4.0 for tc in cue_times)]
    print(f"  false triggers outside the cues: {len(stray)}")

    weakest = np.nanmin(peaks[:-1]) if len(peaks) > 1 else np.nan
    if np.isfinite(weakest):
        suggestion = max(2.5, round(0.5 * weakest, 1))
        print(f"\nWeakest double-clench peak: {weakest:.1f}x resting.")
        print(f"Suggested JAW_THRESHOLD_FACTOR = {suggestion}  (half the weakest clench, min 2.5)")
        if weakest < 3:
            print("Clenches barely stand out: clench harder, or check electrode contact.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the jaw-clench trigger.")
    parser.add_argument("--free", type=float, metavar="SECONDS",
                        help="record with no cues for SECONDS, then list bursts")
    parser.add_argument("--file", help="re-analyse a saved .npz recording")
    args = parser.parse_args()
    if args.file:
        rec = np.load(args.file)
        list_bursts(rec["eeg"], float(rec["sfreq"]))
    elif args.free:
        record_free(args.free)
    else:
        main()
