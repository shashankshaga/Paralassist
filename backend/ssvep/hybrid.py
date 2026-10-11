"""
Use TRCA and EEGNet together as a cross-check.

Rule
----
1. Gate: EEGNet is only used if its cross-validated accuracy is no more than
   `EEGNET_MARGIN` percentage points below TRCA's (e.g. TRCA 87.5% -> EEGNet
   needs >= 77.5%). Both accuracies must come from the same leave-one-block-out
   test on the same training data. If EEGNet fails the gate, TRCA decides alone.
2. Agreement: when both are used, each predicts a target for the flash. If they
   agree, that target is accepted. If they disagree, the flash is rejected
   (`decide` returns None) and the user is asked to look again.
"""

from __future__ import annotations

# EEGNet may be at most this many percentage points worse than TRCA.
EEGNET_MARGIN = 10.0


def should_use_eegnet(trca_acc: float, eegnet_acc: float,
                      margin: float = EEGNET_MARGIN) -> bool:
    """Accuracies in percent. True if EEGNet is within `margin` points of TRCA."""
    return eegnet_acc >= trca_acc - margin


def decide(trca_pred: int, eegnet_pred: int | None) -> int | None:
    """
    Final target for one flash, or None if the flash should be repeated.

    Pass eegnet_pred=None when EEGNet is not in use (failed the gate): the
    TRCA prediction is then accepted as-is.
    """
    if eegnet_pred is None:
        return trca_pred
    return trca_pred if trca_pred == eegnet_pred else None
