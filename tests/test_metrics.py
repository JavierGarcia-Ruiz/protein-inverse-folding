"""Unit tests for inverse folding evaluation metrics."""

import numpy as np
import torch
import pytest

from pif.data.constants import blosum62_score
from pif.evaluation.metrics import (
    compute_sequence_recovery, compute_perplexity, compute_inverse_folding_metrics
)


def test_blosum62_score():
    # Identity comparison of positive hydrophobic residue
    score_self = blosum62_score("AAAA", "AAAA")
    assert score_self == 4.0

    # Conservative substitution (I <-> L is +2)
    score_cons = blosum62_score("IIII", "LLLL")
    assert score_cons == 2.0

    # Radical mutation (W <-> D is -4)
    score_rad = blosum62_score("WWWW", "DDDD")
    assert score_rad == -4.0


def test_compute_sequence_recovery():
    preds = np.array([0, 1, 2, 3])
    targets = np.array([0, 1, 4, 5])
    mask = np.ones(4, dtype=bool)

    rec = compute_sequence_recovery(preds, targets, mask)
    assert rec == 50.0


def test_compute_perplexity():
    # Perfect logits (very confident on true class)
    logits = torch.zeros(1, 10, 20)
    targets = torch.zeros(1, 10, dtype=torch.long)
    mask = torch.ones(1, 10, dtype=torch.bool)
    logits[0, :, 0] = 100.0

    ppl = compute_perplexity(logits, targets, mask)
    assert ppl < 1.05


def test_compute_inverse_folding_metrics():
    preds = [np.array([0, 1, 2, 3])]
    targs = [np.array([0, 1, 2, 4])]
    ss = ["HHHE"]
    env = ["CCCS"]
    cath = [1]

    metrics = compute_inverse_folding_metrics(preds, targs, ss, env, cath, ppl_value=5.2)
    assert metrics["sequence_recovery"] == 75.0
    assert metrics["perplexity"] == 5.2
    assert "recovery_helix" in metrics
    assert "recovery_core" in metrics

