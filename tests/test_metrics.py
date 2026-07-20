"""Tests for the metric core (S5). numpy/pandas only — no sklearn imported anywhere here."""

from __future__ import annotations

import numpy as np

from mlfactory.compute.metrics import (
    average_precision,
    calibration_table,
    expected_calibration_error,
    gain_table,
    ks_table,
    log_loss,
    precision_recall_f1,
    psi,
    rank_order_breaks,
    roc_auc,
    top_decile_lift,
)


# --- KS -------------------------------------------------------------------- #
def test_ks_high_on_separable():
    rng = np.random.default_rng(0)
    n = 4000
    y = np.concatenate([np.zeros(n), np.ones(n)])
    s = np.concatenate([rng.uniform(0, 1, n), rng.uniform(1, 2, n)])
    assert ks_table(y, s).ks > 0.4


def test_ks_low_on_random():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 8000)
    s = rng.normal(size=8000)
    assert ks_table(y, s).ks < 0.15


def test_ks_single_class_is_zero():
    assert ks_table(np.ones(100), np.arange(100.0)).ks == 0.0


# --- PSI ------------------------------------------------------------------- #
def test_psi_zero_for_identical():
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    assert psi(x, x) < 1e-6


def test_psi_positive_for_shift():
    rng = np.random.default_rng(3)
    exp = rng.normal(0, 1, 5000)
    act = rng.normal(3, 1, 5000)
    assert psi(exp, act) > 0.25  # a 3-sigma shift is "major"


# --- rank-order breaks ----------------------------------------------------- #
def _monotone_case():
    scores, labels, per = [], [], 100
    for d in range(10):
        bad = 0.05 + 0.09 * d  # bad-rate rises with score
        k = round(bad * per)
        scores += list(range(d * per, d * per + per))
        labels += [1] * k + [0] * (per - k)
    return np.array(labels), np.array(scores, dtype=float)


def test_rob_zero_when_monotone():
    y, s = _monotone_case()
    assert rank_order_breaks(y, s) == 0


def test_rob_positive_when_inverted():
    y, s = _monotone_case()
    assert rank_order_breaks(y, -s) > 0


# --- lift / gain ----------------------------------------------------------- #
def test_top_decile_lift_and_gain():
    # Top 10% by score are all positives; overall base rate 10% → lift ≈ 10x.
    y = np.array([1] * 100 + [0] * 900)
    s = np.arange(1000, 0, -1, dtype=float)  # first 100 are highest-scored
    assert top_decile_lift(y, s) > 9.0
    table = gain_table(y, s)
    assert table[0]["cum_capture"] == 1.0  # top decile captures all positives


def test_lift_random_near_one():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 5000)
    s = rng.normal(size=5000)
    assert 0.7 < top_decile_lift(y, s) < 1.3


# --- AUCs ------------------------------------------------------------------ #
def test_roc_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    rng = np.random.default_rng(6)
    yy = rng.integers(0, 2, 10000)
    assert abs(roc_auc(yy, rng.normal(size=10000)) - 0.5) < 0.03


def test_average_precision():
    y = np.array([0, 0, 1, 1])
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    # All positives ranked last → AP is low.
    assert average_precision(y, np.array([0.9, 0.8, 0.2, 0.1])) < 0.6


# --- threshold metrics ----------------------------------------------------- #
def test_precision_recall_f1_known():
    y = np.array([1, 1, 0, 0])
    r = precision_recall_f1(y, np.array([0.9, 0.4, 0.8, 0.1]), threshold=0.5)
    # predicted positive: idx0 (tp), idx2 (fp). tp=1, fp=1, fn=1.
    assert r["precision"] == 0.5
    assert r["recall"] == 0.5
    assert r["f1"] == 0.5


def test_log_loss():
    assert log_loss(np.array([1, 0]), np.array([0.999, 0.001])) < 0.01
    # log_loss of a 0.5-everywhere predictor equals ln(2).
    assert abs(log_loss(np.array([1, 0, 1, 0]), np.full(4, 0.5)) - np.log(2)) < 1e-3


# --- calibration ----------------------------------------------------------- #
def test_calibration_well_calibrated():
    rng = np.random.default_rng(7)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)  # perfectly calibrated by construction
    assert expected_calibration_error(y, p) < 0.03
    assert len(calibration_table(y, p)) > 0


def test_calibration_miscalibrated():
    rng = np.random.default_rng(8)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < np.clip(p - 0.3, 0, 1)).astype(int)  # over-confident
    assert expected_calibration_error(y, p) > 0.1
