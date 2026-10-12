#
# QCentroid entrypoint  --  HQNN (hybrid quantum neural network) classifier on NVIDIA CUDA-Q
#
# DO NOT change the name of this file.
# DO NOT edit the signature of the run() function.
#
import contextlib
import io
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report

from hqnn_cudaq import (
    LOSSES, CudaqQNN, HQNNConfig, best_threshold, init_target, predict_scores, train_qnn,
)

logger = logging.getLogger("qcentroid-user-log")

# Every key a user may pass in solver_params (anything else is rejected, to catch typos).
DEFAULTS = {
    "configs": ["whinge", "mhinge", "wmse", "wbce"],   # loss configurations to train (one model each)
    "iters": 400,                  # COBYLA max evaluations per model
    "fm": "zz", "freps": 2, "fent": "pairwise",
    "ansatz": "realamplitude", "areps": 3, "aent": "pairwise",
    "obs": "local",                # 'local' or 'global'
    "weight_seeds": [42],
    "zoom": None,                  # tanh(zoom * raw); None = off
    "dynamic_bias": None,          # None | mse | wmse | hinge | whinge | bce | wbce
    "shots": None,                 # None = exact expectation values
    "min_pr": 0.55, "min_re": None, "beta": None,   # threshold criterion (chosen on TRAIN data)
    "target": "auto",              # auto | nvidia | qpp-cpu
    "fp64": False, "mqpu": False,
    "cache_fm": True,              # simulate the feature map once per sample and reuse it
    "require_gpu": True,           # fail immediately instead of running on the (very slow) CPU simulator
    "baseline": True,              # also fit a logistic-regression baseline
}


class _LogWriter(io.TextIOBase):
    """Forward print() output (training progress etc.) to the QCentroid dashboard log."""
    def __init__(self):
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                logger.info(line.rstrip())
        return len(s)

    def flush(self):
        if self._buf.strip():
            logger.info(self._buf.strip())
        self._buf = ""


def _labels(y, name):
    y = np.asarray(y, dtype=float).ravel()
    vals = set(np.unique(y).tolist())
    if vals <= {0.0, 1.0}:
        logger.info(f"{name}: labels are {{0,1}} -> mapping 0 to -1")
        y = np.where(y == 0, -1.0, 1.0)
    elif not vals <= {-1.0, 1.0}:
        raise ValueError(f"{name} must contain labels in {{-1,+1}} (or {{0,1}}); found {sorted(vals)}")
    return y.astype(int)


def _parse_input(d: dict):
    for k in ("x_train", "y_train", "x_test"):
        if k not in d:
            raise ValueError(f"input_data is missing '{k}'")
    x_tr = np.asarray(d["x_train"], dtype=float)
    x_te = np.asarray(d["x_test"], dtype=float)
    y_tr = _labels(d["y_train"], "y_train")
    y_te = _labels(d["y_test"], "y_test") if d.get("y_test") is not None else None
    if x_tr.ndim != 2 or x_te.ndim != 2 or x_tr.shape[1] != x_te.shape[1]:
        raise ValueError(f"x_train {x_tr.shape} / x_test {x_te.shape} must be 2-D with equal column counts")
    if len(y_tr) != len(x_tr) or (y_te is not None and len(y_te) != len(x_te)):
        raise ValueError("label vectors must match the number of rows of x_train / x_test")
    if not (y_tr == 1).any() or not (y_tr == -1).any():
        raise ValueError("y_train must contain both classes")
    return x_tr, y_tr, x_te, y_te


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    t_start = time.time()
    logger.info("Start of HQNN (CUDA-Q) solver...")

    solver_params = solver_params or {}
    unknown = set(solver_params) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown solver_params: {sorted(unknown)}. Allowed: {sorted(DEFAULTS)}")
    p = {**DEFAULTS, **solver_params}

    configs = [p["configs"]] if isinstance(p["configs"], str) else list(p["configs"])
    bad = [c for c in configs if c not in LOSSES]
    if bad:
        raise ValueError(f"Unknown configs {bad}; choose from {list(LOSSES)}")

    x_tr, y_tr, x_te, y_te = _parse_input(input_data)
    logger.info(f"x_train {x_tr.shape}, x_test {x_te.shape}, +1 fraction (train) {np.mean(y_tr == 1):.3f}")

    out_dir = Path("additional_output")
    out_dir.mkdir(exist_ok=True)

    with contextlib.redirect_stdout(_LogWriter()) as log:
        target = init_target(p["target"], bool(p["mqpu"]), bool(p["fp64"]))
        log.flush()
        if p["require_gpu"] and not target.startswith("nvidia"):
            raise RuntimeError(
                f"CUDA-Q is running on '{target}', not on a GPU. A 15-qubit CPU simulation is far too slow "
                "to train this model. Run on a GPU node, or set solver_params {'require_gpu': false} "
                "(only sensible for tiny test inputs)."
            )

        cimb = float(np.mean(y_tr == 1))          # fraction of class +1 (the minority)
        W1, W2 = 1.0 / (2.0 * cimb), 1.0 / (2.0 * (1.0 - cimb))
        names = {"-1": "benign (-1)", "1": "suspicious (+1)"}

        results = []
        for config in configs:
            logger.info(f"=== config {config} ===")
            loss_fn, weighted = LOSSES[config]
            cfg = HQNNConfig(
                fm=p["fm"], freps=int(p["freps"]), fent=p["fent"],
                ansatz=p["ansatz"], areps=int(p["areps"]), aent=p["aent"], obs_type=p["obs"],
                iters=int(p["iters"]), weight_seeds=[int(s) for s in p["weight_seeds"]],
                zoom_scale=p["zoom"], dynamic_bias=p["dynamic_bias"], num_shots=p["shots"],
            )
            qnn = CudaqQNN(x_tr.shape[1], cfg, cache_fm=bool(p["cache_fm"]))

            t0 = time.time()
            w, hist, seed, bias, best_loss = train_qnn(
                qnn, x_tr, y_tr, loss_fn, *((W1, W2) if weighted else (None, None)))
            train_s = time.time() - t0
            cfg.bias = float(bias)
            qnn.clear_cache()                      # release GPU memory held by cached states
            log.flush()

            s_tr = predict_scores(qnn, x_tr, w, cfg.zoom_scale, bias)
            s_te = predict_scores(qnn, x_te, w, cfg.zoom_scale, bias)
            thr = best_threshold(s_tr, y_tr, p["beta"], p["min_pr"], p["min_re"])   # TRAIN labels only
            pred = np.where(s_te >= thr, 1, -1)

            res = {
                "config": config,
                "best_train_loss": float(best_loss),
                "n_evaluations": len(hist),
                "train_seconds": round(train_s, 1),
                "weight_seed": int(seed),
                "bias": float(bias),
                "threshold": float(thr),
                "train_auc_pr": float(average_precision_score(y_tr, s_tr)),
                "weights": [float(v) for v in w],
                "test_scores": [float(v) for v in s_te],
                "test_predictions": [int(v) for v in pred],
                "loss_history": [float(v) for v in hist],
            }
            if y_te is not None:
                res["test_auc_pr"] = float(average_precision_score(y_te, s_te))
                rep = classification_report(y_te, pred, labels=[-1, 1], output_dict=True, zero_division=0,
                                            target_names=[names["-1"], names["1"]])
                res["test_report"] = rep
                logger.info(f"{config}: train AUC-PR {res['train_auc_pr']:.4f} | "
                            f"test AUC-PR {res['test_auc_pr']:.4f} | threshold {thr:.4f}")

            # additional_output/: everything needed to reload this model later
            (out_dir / f"model_{config}.json").write_text(json.dumps(
                {"config": config, "hqnn": asdict(cfg), "weights": res["weights"],
                 "threshold": res["threshold"], "bias": res["bias"], "n_features": int(x_tr.shape[1])},
                indent=2), encoding="utf-8")
            (out_dir / f"loss_history_{config}.csv").write_text(
                "evaluation,loss\n" + "\n".join(f"{i + 1},{v}" for i, v in enumerate(hist)) + "\n",
                encoding="utf-8")
            results.append(res)

        baseline = None
        if p["baseline"]:
            lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000).fit(x_tr, y_tr)
            d_tr, d_te = lr.decision_function(x_tr), lr.decision_function(x_te)
            thr = best_threshold(d_tr, y_tr, p["beta"], p["min_pr"], p["min_re"], verbose=False)
            baseline = {"threshold": float(thr),
                        "test_predictions": [int(v) for v in np.where(d_te >= thr, 1, -1)]}
            if y_te is not None:
                baseline["test_auc_pr"] = float(average_precision_score(y_te, d_te))
                logger.info(f"logistic-regression baseline: test AUC-PR {baseline['test_auc_pr']:.4f}")

    summary_lines = ["config,best_train_loss,train_auc_pr,test_auc_pr"] + [
        f"{r['config']},{r['best_train_loss']:.5f},{r['train_auc_pr']:.4f},{r.get('test_auc_pr', '')}"
        for r in results]
    (out_dir / "summary.csv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    output = {
        "target": target,
        "n_features": int(x_tr.shape[1]),
        "results": results,
        "logistic_regression_baseline": baseline,
        "total_seconds": round(time.time() - t_start, 1),
    }
    logger.info("End of HQNN solver, returning output.")
    return output
