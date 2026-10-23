"""Tests for the fundamentals exercises (kNN, softmax, BatchNorm).

These test the REFERENCE implementations (reference/exercises/) — the
owner studies them, then writes his own; the same style of tests is what
he should write against his versions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from reference.exercises.knn import (compute_distances_no_loops,
                                     compute_distances_one_loop,
                                     compute_distances_two_loops,
                                     knn_accuracy, predict_labels)
from reference.exercises.softmax import (numerical_gradient,
                                         softmax_loss_naive,
                                         softmax_loss_vectorized)
from reference.exercises.batchnorm import batchnorm_forward


# ----------------------------------------------------------------- kNN

def test_knn_versions_agree():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(7, 4))     # 7 train points, 4 features
    X_test = rng.normal(size=(3, 4))      # 3 test points
    d2 = compute_distances_two_loops(X_test, X_train)   # (3, 7)
    d1 = compute_distances_one_loop(X_test, X_train)
    d0 = compute_distances_no_loops(X_test, X_train)
    assert d2.shape == (3, 7)
    assert np.allclose(d2, d1, atol=1e-10)
    assert np.allclose(d2, d0, atol=1e-8)   # expansion version: looser tol


def test_knn_hand_example():
    # 2x2 worked example from docs/worked/knn.md (hand-checkable).
    X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    X_test = np.array([[3.0, 4.0], [0.1, 0.0]])
    y_train = np.array([0, 1, 2])
    d = compute_distances_two_loops(X_test, X_train)
    # test0: sqrt(9+16)=5, sqrt(4+16)=sqrt20, sqrt(9+9)=sqrt18; the nearest
    # train point to (3,4) is index 2 (sqrt18) -> label 2.
    assert d[0].tolist() == pytest.approx([5.0, np.sqrt(20), np.sqrt(18)])
    assert predict_labels(d, y_train, k=1).tolist() == [2, 0]
    assert knn_accuracy(np.array([0, 0]), np.array([0, 1])) == 0.5


# -------------------------------------------------------------- softmax

def test_softmax_naive_equals_vectorized():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 4))           # (6, 4)
    W = rng.normal(size=(4, 3)) * 0.5     # (4, 3)
    y = rng.integers(0, 3, size=6)
    ln, dn = softmax_loss_naive(X, W, y)
    lv, dv = softmax_loss_vectorized(X, W, y)
    assert ln == pytest.approx(lv)
    assert np.allclose(dn, dv, atol=1e-10)


def test_softmax_gradient_check():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, 3))
    W = rng.normal(size=(3, 4)) * 0.1
    y = rng.integers(0, 4, size=5)
    _, analytic = softmax_loss_vectorized(X, W, y)
    numeric = numerical_gradient(lambda w: softmax_loss_vectorized(X, w, y)[0],
                                 W.copy())
    # Relative error, the standard CS231n check.
    rel = np.max(np.abs(analytic - numeric) / np.maximum(
        1e-8, np.abs(analytic) + np.abs(numeric)))
    assert rel < 1e-6, rel


def test_softmax_large_scores_do_not_overflow():
    X = np.array([[1000.0, 0.0]])         # would overflow without max-subtract
    W = np.eye(2)
    y = np.array([0])
    loss, _ = softmax_loss_vectorized(X, W, y)
    assert np.isfinite(loss)


# ------------------------------------------------------------ batchnorm

def test_batchnorm_matches_pytorch_train_and_eval():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8, 5, 6, 6)).astype(np.float32) * 2 + 1  # (N,C,H,W)
    C = x.shape[1]
    gamma = rng.normal(size=C).astype(np.float32)
    beta = rng.normal(size=C).astype(np.float32)

    bn = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1)
    with torch.no_grad():
        bn.weight.copy_(torch.from_numpy(gamma))
        bn.bias.copy_(torch.from_numpy(beta))

    rm = np.zeros(C, dtype=np.float32)
    rv = np.ones(C, dtype=np.float32)
    out, rm2, rv2 = batchnorm_forward(x, gamma, beta, rm, rv, train=True)

    bn.train()
    ref = bn(torch.from_numpy(x)).detach().numpy()
    assert np.allclose(out, ref, atol=1e-4)

    bn.eval()  # uses the running stats the train pass just updated
    ref_eval = bn(torch.from_numpy(x)).detach().numpy()
    out_eval, _, _ = batchnorm_forward(x, gamma, beta, rm2, rv2, train=False)
    assert np.allclose(out_eval, ref_eval, atol=1e-4)


def test_batchnorm_bs1_train_is_degenerate_eval_is_not():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 6)).astype(np.float32)   # (N=2, C=6) rows
    from reference.exercises.batchnorm import batchnorm_train_vs_eval_demo
    demo = batchnorm_train_vs_eval_demo(x)
    train_out = demo["out_train_bs1"]    # (1, C)
    eval_out = demo["out_eval_bs1"]      # (1, C)
    # Train mode at batch 1: mean == the sample itself -> xhat = 0 -> the
    # output is exactly beta (here 0): BN erased the input.
    assert np.allclose(train_out, 0.0, atol=1e-5)
    # Eval mode: running stats (0, 1) -> output = x (gamma=1, beta=0).
    assert np.allclose(eval_out, x[:1], atol=1e-4)
