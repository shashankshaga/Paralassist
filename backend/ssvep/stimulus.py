"""
PsychoPy flicker stimuli and the single-trial routine.

The four targets are white squares in the corners of the screen. Each one is
turned white/black every frame according to the sign of a sine wave at its
assigned frequency:

    colour = white if sin(2*pi*f*t) >= 0 else black

Because the update happens once per `win.flip()` (i.e. locked to the monitor
refresh), the on-screen flicker is frame-accurate. On a 60 Hz display the
chosen frequencies (6.67 / 8.57 / 12 / 15 Hz) land on near-integer frame
counts, so the flicker stays rock-steady -- this is exactly what we verified
with an Arduino + photoresistor rig.

A trial has three phases:
    1. CUE      : a coloured frame highlights the target the user must look at.
    2. FLICKER  : all four targets flash for FLICKER_DURATION seconds.
    3. CAPTURE  : the recording flag is raised so the child process grabs the
                  window that just elapsed.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
from psychopy import visual, core, event, logging

from . import config as cfg

# Corner positions in normalised units: top-left, top-right, bottom-left, bottom-right.
_POSITIONS = [(-0.5, 0.5), (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]
_SIZE = (0.5, 0.5)


def build_window() -> visual.Window:
    """Create the full-screen (or windowed) PsychoPy window."""
    # Windows display scaling (e.g. 150%) makes PsychoPy see a shrunken screen
    # (1280x800 instead of 1920x1200). Opting into per-monitor DPI awareness
    # makes it render at the panel's real resolution.
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass
    # Hide PsychoPy's harmless console warnings (e.g. "Monitor specification
    # not found" -- we draw in normalised units, so no calibration is needed).
    logging.console.setLevel(logging.ERROR)

    return visual.Window(
        size=cfg.WINDOW_SIZE,
        fullscr=cfg.FULLSCREEN,
        screen=cfg.STIMULUS_SCREEN,
        winType="pyglet",
        color=[0, 0, 0],
        colorSpace="rgb",
        units="norm",
        allowGUI=False,
    )


def build_stimuli(win: visual.Window):
    """Create the four flicker squares and their cue frames."""
    squares, cues = [], []
    for pos in _POSITIONS:
        squares.append(visual.Rect(
            win=win, units="norm", width=_SIZE[0], height=_SIZE[1],
            pos=pos, anchor="center", fillColor="white", lineColor="black",
        ))
        # A thick coloured outline used to mark the current target.
        cues.append(visual.Rect(
            win=win, units="norm", width=_SIZE[0] * 1.15, height=_SIZE[1] * 1.15,
            pos=pos, anchor="center", fillColor=None, lineColor="red",
            lineWidth=8,
        ))
    return squares, cues


def show_message(win: visual.Window, text: str) -> None:
    """Draw a centred one-off message (e.g. 'Setting up board...')."""
    visual.TextStim(win, text=text, pos=(0, 0), color="white", height=0.08).draw()
    win.flip()


def run_trial(win, squares, cues, target_idx, recording_process,
              freqs=cfg.STIMULUS_FREQUENCIES) -> bool:
    """
    Run one cue -> flicker -> capture trial.

    Parameters
    ----------
    target_idx : int
        Which square the user should look at this trial (0..3). Used to draw
        the cue and, in collect mode, to label the saved data. Pass -1 for
        free-running (predict) mode, where no cue is shown.
    recording_process : RecordingProcess
        Shared background acquisition process; we toggle its recording flag.

    Returns
    -------
    bool
        True to continue, False if the user pressed Escape.
    """
    clock = core.Clock()
    cued = 0 <= target_idx < len(cues)

    # --- Phase 1: CUE ---------------------------------------------------- #
    # Highlight the target so the user knows where to look. In collect mode we
    # publish the label so the child saves under the right filename. In predict
    # mode (target_idx == -1) we skip the cue entirely.
    recording_process.recording_flag.value = False
    if cued:
        recording_process.label_index.value = target_idx

        clock.reset()
        while clock.getTime() < cfg.CUE_DURATION:
            for sq in squares:
                sq.fillColor = [-1, -1, -1]
                sq.draw()
            cues[target_idx].draw()
            win.flip()
            if _escape_pressed():
                return False

    # --- Phase 2: FLICKER ------------------------------------------------ #
    clock.reset()
    while clock.getTime() < cfg.FLICKER_DURATION:
        t = clock.getTime()
        _flicker_frame(squares, freqs, t)
        win.flip()
        if _escape_pressed():
            return False

    # --- Phase 3: CAPTURE ------------------------------------------------ #
    # Blank the squares and raise the flag. The rising edge makes the child
    # grab the last CAPTURE_SAMPLES (the flicker window that just ended).
    for sq in squares:
        sq.fillColor = [-1, -1, -1]
        sq.draw()
    win.flip()
    recording_process.recording_flag.value = True

    # Short rest / gaze-shift; also gives the child time to save or predict.
    core.wait(cfg.INTER_TRIAL_INTERVAL)
    return True


def _flicker_frame(squares, freqs, t: float) -> None:
    """Set each square white/black based on the sign of its sine wave at time t."""
    # Frames land at t = k / refresh. For 12 and 15 Hz some of those frames
    # fall exactly on a zero-crossing of the sine, so sub-millisecond timing
    # jitter flips them randomly (15 Hz came out as 010001000... instead of
    # 11001100...). Sampling a quarter-frame later keeps every frequency used
    # here at least 1/4 frame away from a zero-crossing.
    t = t + 0.25 / cfg.EXPECTED_REFRESH_HZ
    for sq, f in zip(squares, freqs):
        white = np.sin(2 * np.pi * f * t) >= 0
        sq.fillColor = [1, 1, 1] if white else [-1, -1, -1]
        sq.draw()


def _escape_pressed() -> bool:
    return "escape" in event.getKeys(keyList=["escape"])
