"""
evaluate_trca.py
================

Offline sanity check for your recorded training data. Runs leave-one-block-out
cross-validation and prints per-block accuracy plus the mean accuracy and
information transfer rate (ITR). No board or screen required.

Run it with:  python evaluate_trca.py
"""

from ssvep import config as cfg
from ssvep.trca_model import cross_validate

if __name__ == "__main__":
    cross_validate(cfg.TRAINING_DATA_DIR)
