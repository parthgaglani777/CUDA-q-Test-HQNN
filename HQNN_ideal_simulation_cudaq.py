"""
Ideal (noise-free, exact state-vector) evaluation of trained HQNN weights on NVIDIA CUDA-Q.

    python HQNN_ideal_simulation_cudaq.py 11
    python HQNN_ideal_simulation_cudaq.py 11 --configs whinge wmse --min-pr 0.55

Differences from the Qiskit version
-----------------------------------
* The circuit is rebuilt from the JSON file written at training time, so feature-map
  reps / entanglement / ansatz can never differ from what the weights were trained with
  (the original used freps=1 here and freps=2 in training).
* The learned bias and zoom are applied.
* The decision threshold is chosen on TRAIN scores; test labels are only used for reporting.
  Same for the logistic-regression baseline.
* No dependency on hqnn_data_reupload_1.
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report

from hqnn_cudaq import (
    CudaqQNN, HERE, best_threshold, init_target, load_data, load_model, predict_scores,
)

NAMES = ["Benign (-1)", "Suspicious (+1)"]


def main():
    ap = argparse.ArgumentParser(description="HQNN ideal-simulation evaluation (CUDA-Q)")
    ap.add_argument("pnd", type=int)
    ap.add_argument("--configs", nargs="+", default=["whinge", "mhinge", "wmse", "wbce"])
    ap.add_argument("--data-dir", default=HERE)
    ap.add_argument("--out-dir", default=HERE, help="folder that contains weights/")
    ap.add_argument("--shots", type=int, default=None, help="None = exact expectation values")
    ap.add_argument("--target", default="auto")
    ap.add_argument("--mqpu", action="store_true")
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--beta", type=float, default=None, help=">1 favours recall, <1 favours precision")
    ap.add_argument("--min-pr", type=float, default=0.55, help="precision floor")
    ap.add_argument("--min-re", type=float, default=None, help="recall floor")
    a = ap.parse_args()

    init_target(a.target, a.mqpu, a.fp64)
    x_tr, y_tr, x_te, y_te = load_data(a.pnd, a.data_dir)
    print(f"x_test range [{x_te.min():.3f}, {x_te.max():.3f}] vs x_train range "
          f"[{x_tr.min():.3f}, {x_tr.max():.3f}]  (test features outside the training range are not clipped)")

    for config in a.configs:
        print("\n" + "=" * 30, config, "=" * 30)
        w, cfg, meta = load_model(a.out_dir, config, a.pnd)
        cfg.num_shots = a.shots
        print(f"Loaded architecture from JSON: fm={cfg.fm}(reps={cfg.freps}) ansatz={cfg.ansatz}(reps={cfg.areps}) "
              f"| bias={cfg.bias:.4f} zoom={cfg.zoom_scale}")
        qnn = CudaqQNN(x_tr.shape[1], cfg)
        assert qnn.num_weights == w.size, f"weights file has {w.size} entries, circuit needs {qnn.num_weights}"

        s_tr = predict_scores(qnn, x_tr, w, cfg.zoom_scale, cfg.bias)
        s_te = predict_scores(qnn, x_te, w, cfg.zoom_scale, cfg.bias)

        print(f"Train AUC-PR: {average_precision_score(y_tr, s_tr):.4f}")
        print(f"Test  AUC-PR: {average_precision_score(y_te, s_te):.4f}")

        thr = best_threshold(s_tr, y_tr, a.beta, a.min_pr, a.min_re)     # train labels only
        print("Test report @ train-chosen threshold:")
        print(classification_report(y_te, np.where(s_te >= thr, 1, -1), target_names=NAMES, zero_division=0))

    # ------------------------------------------------------------------ classical baseline
    print("\n" + "-" * 50 + "\nLogistic Regression baseline\n" + "-" * 50)
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000).fit(x_tr, y_tr)
    lr_tr, lr_te = lr.decision_function(x_tr), lr.decision_function(x_te)
    print(f"Test AUC-PR (Suspicious, +1): {average_precision_score(y_te, lr_te):.4f}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        thr = best_threshold(lr_tr, y_tr, a.beta, a.min_pr, a.min_re)    # train labels only
        print("Test report @ train-chosen threshold:")
        print(classification_report(y_te, np.where(lr_te >= thr, 1, -1), target_names=NAMES, zero_division=0))


if __name__ == "__main__":
    main()
