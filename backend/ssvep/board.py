"""
Thin, well-documented wrapper around the NeuroPawn Knight board.

Why the board needs "setting up" before it streams anything useful
------------------------------------------------------------------
When BrainFlow first connects, every EEG channel on the Knight board is
*powered down*. If you start streaming immediately you just get flat lines.
Each channel has to be switched on and (optionally) tied into the board's
right-leg-drive (RLD / bias) circuit, which actively cancels common-mode
noise (mains hum, movement). That is what `configure_channels()` does with
two commands per channel:

    chon_{ch}_{gain}   -> turn channel `ch` ON with the given PGA gain
    rldadd_{ch}        -> add channel `ch` to the bias/RLD feedback loop

These commands are sent one-by-one with short sleeps, because the board's
firmware needs a moment to apply each register write. Configuring all 8
channels therefore takes ~20-30 s -- this is normal and only happens once
at start-up.
"""

from __future__ import annotations

import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels


class KnightBoard:
    """Connect to, configure, and stream from a NeuroPawn Knight board."""

    def __init__(self, serial_port: str, num_channels: int, gain: int = 12):
        self.num_channels = num_channels
        self.gain = gain

        # BrainFlow needs the serial port; everything else is fixed for this board.
        # Only show BrainFlow errors. Its info/warning output is just noise here:
        # it dumps the connection JSON and warns about every config command we
        # send while streaming (which the Knight board requires).
        BoardShim.set_log_level(LogLevels.LEVEL_ERROR.value)

        params = BrainFlowInputParams()
        params.serial_port = serial_port

        self.board_shim = BoardShim(BoardIds.NEUROPAWN_KNIGHT_BOARD.value, params)
        self.board_id = self.board_shim.get_board_id()

        # Ask BrainFlow which rows of the data matrix hold EEG and the rate.
        self.eeg_channels = self.board_shim.get_exg_channels(self.board_id)
        self.sr = self.board_shim.get_sampling_rate(self.board_id)

    # --------------------------------------------------------------------- #
    # Streaming lifecycle
    # --------------------------------------------------------------------- #
    def start_stream(self, buffer_size: int = 450_000) -> None:
        """Open the session, start streaming, and power up every channel."""
        self.board_shim.prepare_session()
        self.board_shim.start_stream(buffer_size)
        print("Stream started - configuring channels (this takes a moment)...")
        time.sleep(2)  # let the stream settle before sending config commands
        self.configure_channels()
        print("Board ready.")

    def configure_channels(self) -> None:
        """Switch each channel on and add it to the bias/RLD loop."""
        for ch in range(1, self.num_channels + 1):
            time.sleep(0.5)
            on_cmd = f"chon_{ch}_{self.gain}"      # power channel on at `gain`
            self.board_shim.config_board(on_cmd)
            print(f"  sent {on_cmd}")

            time.sleep(1.0)
            rld_cmd = f"rldadd_{ch}"               # tie channel into bias drive
            self.board_shim.config_board(rld_cmd)
            print(f"  sent {rld_cmd}")
            time.sleep(0.5)

    def stop_stream(self) -> None:
        """Stop streaming and release the hardware session."""
        self.board_shim.stop_stream()
        self.board_shim.release_session()
        print("Stream stopped and session released.")

    # --------------------------------------------------------------------- #
    # Data access
    # --------------------------------------------------------------------- #
    def get_latest(self, num_samples: int):
        """
        Return the most recent `num_samples` columns WITHOUT draining the buffer.

        BrainFlow keeps incoming samples in a fixed-size ring buffer. There are
        two ways to read it:

          * get_board_data()          -> returns everything AND empties the
                                         buffer (destructive read).
          * get_current_board_data(n) -> returns a *copy* of the latest `n`
                                         samples and leaves the buffer intact.

        We always use the second form. Because it is non-destructive, the
        stimulus loop and the classifier can both "peek" at the freshest slice
        of EEG on every trial without ever losing samples or fighting over the
        buffer. The board keeps filling the ring in the background regardless.
        """
        return self.board_shim.get_current_board_data(num_samples)
