"""
Hybrid Quantum Neural Network (HQNN) on NVIDIA CUDA-Q.

Port of the Qiskit / qiskit-machine-learning implementation (hqnn.py).

What changed vs. the Qiskit version
-----------------------------------
* Circuits are CUDA-Q kernels (@cudaq.kernel) instead of QuantumCircuit objects.
  The circuit *structure* (feature map, ansatz, entanglement edge lists) is baked
  into the kernel by `make_kernel(...)`; the kernel takes ONE argument:
        v = [x_0 ... x_{n_in-1}, theta_0 ... theta_{n_w-1}]      (inputs first, then weights)
  which is the same parameter order as the Qiskit circuit.
* EstimatorQNN is replaced by `CudaqQNN.forward`, which evaluates the whole
  training batch in ONE call to `cudaq.observe(kernel, H, V)` where V is a
  (n_samples x n_params) array (CUDA-Q "broadcasting").
* Expectation values are exact (state-vector) unless NUM_SHOTS is set. The old
  code never used NUM_SHOTS and ran with Aer's default (noisy) precision.
* COBYLA is scipy's COBYLA (qiskit-machine-learning's COBYLA is a thin wrapper
  around the same routine).

Bug fixes folded in (from the review of the original code)
----------------------------------------------------------
* `pnd` is a command-line argument and is passed everywhere (no import-time
  hardcoded PND = 2 / FileNotFoundError).
* One config object is used for training AND evaluation and is saved next to the
  weights (JSON), so feature-map reps can no longer differ between the scripts.
* Weights are written to ./weights/ (created automatically).
* `best_threshold` drops the dummy last point of precision_recall_curve so the
  precision / recall floor can really be "relaxed".
* The decision threshold is chosen on TRAIN scores, never on the test labels.
* Best weights, best loss and best bias come from the same (best) objective
  evaluation, not from the last one.
* Weighted-median bias normalises the cumulative weights.
* Hinge losses clamp at 0, as their docstrings say.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

import cudaq
from cudaq import spin

warnings.filterwarnings("ignore", message=".*cudaq-logical.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="cudaq")

HERE = os.path.dirname(os.path.abspath(__file__))
PI = float(np.pi)


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class HQNNConfig:
    # --- QNN architecture (saved with the weights and reused at evaluation) ---
    fm: str = "zz"                    # 'zz', 'z' or 'da' (dense angle)
    freps: int = 2                    # feature-map repetitions
    fent: str = "pairwise"            # 'full','linear','reverse_linear','circular','pairwise'
    ansatz: str = "realamplitude"     # 'realamplitude', 'nlocal', 'efficientsu2'
    areps: int = 3
    aent: str = "pairwise"
    obs_type: str = "local"           # 'local' (sum of Z, / n_qubits) or 'global' (Z^n)

    # --- training ---
    iters: int = 400                  # COBYLA max iterations
    weight_seeds: list = field(default_factory=lambda: [42])
    zoom_scale: float | None = None   # tanh(zoom * raw) amplification; None = off
    dynamic_bias: str | None = None   # None, 'mse','wmse','hinge','whinge','bce','wbce'
    num_shots: int | None = None      # None = exact expectation values (recommended)

    # --- filled in after training (saved to the JSON sidecar) ---
    bias: float = 0.0


# =============================================================================
# Backend selection
# =============================================================================
def init_target(target: str = "auto", mqpu: bool = False, fp64: bool = False) -> str:
    """
    'auto'  -> 'nvidia' if a GPU is visible, otherwise CPU simulator 'qpp-cpu'.
    'nvidia' options: fp64 (double precision; default is fp32), mqpu (multi-GPU
    batch parallelism, one circuit per GPU).
    """
    n_gpu = cudaq.num_available_gpus()
    if target == "auto":
        target = "nvidia" if n_gpu > 0 else "qpp-cpu"
    if target == "nvidia":
        if n_gpu == 0:
            raise RuntimeError("target 'nvidia' requested but no GPU is visible.")
        opts = [o for o, on in (("mqpu", mqpu), ("fp64", fp64)) if on]
        if opts:
            cudaq.set_target("nvidia", option=",".join(opts))
        else:
            cudaq.set_target("nvidia")
    else:
        cudaq.set_target(target)
    name = cudaq.get_target().name
    print(f"[CUDA-Q] target={name} | visible GPUs={n_gpu} | qpus={cudaq.get_target().num_qpus()}")
    return name


# =============================================================================
# Entanglement maps (same conventions as qiskit.circuit.library)
# =============================================================================
def entangler_pairs(n: int, ent: str) -> list[tuple[int, int]]:
    ent = ent.lower()
    linear = [(i, i + 1) for i in range(n - 1)]
    if ent == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if ent == "linear":
        return linear
    if ent == "reverse_linear":
        return [(i, i + 1) for i in reversed(range(n - 1))]
    if ent == "circular":
        return ([(n - 1, 0)] if n > 2 else []) + linear
    if ent == "pairwise":
        return [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, i + 1) for i in range(1, n - 1, 2)]
    raise ValueError(f"Entanglement '{ent}' not supported (use full/linear/reverse_linear/circular/pairwise).")


# =============================================================================
# Kernel factory
# =============================================================================
FM_IDS = {"zz": 0, "z": 1, "da": 2}
ANSATZ_IDS = {  # name -> (has_rz)
    "realamplitude": 0, "ra": 0, "nlocal": 0,
    "efficientsu2": 1, "su2": 1,
}


def make_kernel(n_features: int, cfg: HQNNConfig):
    """
    Returns (kernel, fm_only_kernel, from_state_kernel, n_qubits, n_inputs, n_weights).

    kernel(v: list[float]) with v = [inputs..., weights...].

    Gate conventions (verified numerically against the Qiskit circuits):
      zz : reps x [ H ; P(2 x_i) ; for each pair (i,j): CX(i,j) P(2 x_i x_j / pi)_j CX(i,j) ]
      z  : reps x [ H ; P(2 x_i) ]
      da : RY(2 x_2i) RZ(2 x_2i+1) per qubit, then CZ chain (linear / circular)
      realamplitude / nlocal(ry+cx): areps x [RY layer ; CX edges] ; final RY layer
      efficientsu2                  : areps x [RY layer ; RZ layer ; CX edges] ; final RY ; RZ
    """
    fm = cfg.fm.lower()
    if fm not in FM_IDS:
        raise NotImplementedError(
            f"Feature map '{cfg.fm}' is not ported. Supported: zz, z, da. "
            "(The 'su2' feature map in the original code reused the ansatz's parameter names "
            "and was not usable as an input encoder.)"
        )
    if cfg.ansatz.lower() not in ANSATZ_IDS:
        raise ValueError(f"Ansatz '{cfg.ansatz}' not recognised.")
    if cfg.ansatz.lower() == "sca" or cfg.aent == "sca" or cfg.fent == "sca":
        raise NotImplementedError("'sca' entanglement is not ported.")

    # ---- host-side structure (becomes compile-time constants inside the kernel)
    fm_id = FM_IDS[fm]
    if fm_id == 2:
        n_q = n_features // 2
        n_in = 2 * n_q
        if n_in != n_features:
            print(f"[warn] 'da' uses {n_in} of {n_features} features (odd count -> last feature unused).")
        if cfg.fent == "linear":
            pairs = [(i, i + 1) for i in range(n_q - 1)]
        elif cfg.fent == "circular":
            pairs = [(i, i + 1) for i in range(n_q - 1)] + [(n_q - 1, 0)]
        else:
            pairs = []
        dsrc = [int(a) for a, _ in pairs]
        ddst = [int(b) for _, b in pairs]
    else:
        n_q = n_features
        n_in = n_features
        pairs = entangler_pairs(n_q, cfg.fent) if fm_id == 0 else []
        dsrc, ddst = [], []
    fsrc = [int(a) for a, _ in (pairs if fm_id == 0 else [])]
    fdst = [int(b) for _, b in (pairs if fm_id == 0 else [])]

    apairs = entangler_pairs(n_q, cfg.aent)
    asrc = [int(a) for a, _ in apairs]
    adst = [int(b) for _, b in apairs]

    # CUDA-Q cannot infer the element type of an EMPTY captured list, so loop counts are
    # stored separately and every edge list is padded with a dummy entry (never read).
    n_fe, n_de, n_ae = len(fsrc), len(dsrc), len(asrc)
    fsrc, fdst, dsrc, ddst, asrc, adst = (l or [0] for l in (fsrc, fdst, dsrc, ddst, asrc, adst))

    has_rz = ANSATZ_IDS[cfg.ansatz.lower()]
    layer = n_q * (1 + has_rz)            # parameters per rotation layer
    n_w = layer * (cfg.areps + 1)
    freps, areps = int(cfg.freps), int(cfg.areps)
    nq = int(n_q)
    ni = int(n_in)

    # NOTE: do not name any local variable `x` / `h` / `y` / `z` in this scope:
    # those are CUDA-Q gate names used inside the kernels.
    @cudaq.kernel
    def feature_map(q: cudaq.qview, v: list[float]):
        if fm_id == 0:  # ZZ feature map
            for rep in range(freps):
                for i in range(nq):
                    h(q[i])
                for i in range(nq):
                    r1(2.0 * v[i], q[i])
                for e in range(n_fe):
                    a = fsrc[e]
                    b = fdst[e]
                    x.ctrl(q[a], q[b])
                    r1(2.0 * v[a] * v[b] / PI, q[b])
                    x.ctrl(q[a], q[b])
        if fm_id == 1:  # Z feature map
            for rep in range(freps):
                for i in range(nq):
                    h(q[i])
                for i in range(nq):
                    r1(2.0 * v[i], q[i])
        if fm_id == 2:  # dense angle encoding
            for i in range(nq):
                ry(2.0 * v[2 * i], q[i])
                rz(2.0 * v[2 * i + 1], q[i])
            for e in range(n_de):
                z.ctrl(q[dsrc[e]], q[ddst[e]])

    @cudaq.kernel
    def ansatz(q: cudaq.qview, v: list[float], off: int):
        # weights live at v[off : off + n_w]
        for r in range(areps):
            base = off + r * layer
            for i in range(nq):
                ry(v[base + i], q[i])
            if has_rz == 1:
                for i in range(nq):
                    rz(v[base + nq + i], q[i])
            for e in range(n_ae):
                x.ctrl(q[asrc[e]], q[adst[e]])
        base = off + areps * layer
        for i in range(nq):
            ry(v[base + i], q[i])
        if has_rz == 1:
            for i in range(nq):
                rz(v[base + nq + i], q[i])

    @cudaq.kernel
    def hqnn(v: list[float]):                   # full circuit: v = [inputs, weights]
        q = cudaq.qvector(nq)
        feature_map(q, v)
        ansatz(q, v, ni)

    @cudaq.kernel
    def hqnn_fm_only(v: list[float]):           # feature map only (for state caching)
        q = cudaq.qvector(nq)
        feature_map(q, v)

    @cudaq.kernel
    def hqnn_from_state(s: cudaq.State, w: list[float]):   # ansatz applied to a cached feature-map state
        q = cudaq.qvector(s)
        ansatz(q, w, 0)

    return hqnn, hqnn_fm_only, hqnn_from_state, n_q, n_in, n_w


def make_observable(n_qubits: int, obs_type: str):
    if obs_type.lower() == "local":          # sum_k Z_k
        ham = spin.z(0)
        for k in range(1, n_qubits):
            ham += spin.z(k)
        return ham
    if obs_type.lower() == "global":         # Z^{(x) n}
        ham = spin.z(0)
        for k in range(1, n_qubits):
            ham *= spin.z(k)
        return ham
    raise ValueError(f"Observable type '{obs_type}' not recognised.")


# =============================================================================
# QNN wrapper (replaces EstimatorQNN)
# =============================================================================
class CudaqQNN:
    def __init__(self, n_features: int, cfg: HQNNConfig, cache_fm: bool = False):
        self.cfg = cfg
        (self.kernel, self.k_fm, self.k_from_state,
         self.n_qubits, self.n_inputs, self.num_weights) = make_kernel(n_features, cfg)
        self.cache_fm = cache_fm
        self._state_cache: dict = {}
        self.hamiltonian = make_observable(self.n_qubits, cfg.obs_type)
        # 'local' = sum of n Z's -> divide by n so outputs live in [-1, 1]
        self.norm = float(self.n_qubits) if cfg.obs_type.lower() == "local" else 1.0
        print(
            f"CudaqQNN | fm={cfg.fm}(reps={cfg.freps}, ent={cfg.fent}) | "
            f"ansatz={cfg.ansatz}(reps={cfg.areps}, ent={cfg.aent}) | obs={cfg.obs_type} | "
            f"qubits={self.n_qubits} | inputs={self.n_inputs} | weights={self.num_weights}"
        )

    def clear_cache(self):
        """Drop cached feature-map states (they live in GPU memory)."""
        self._state_cache.clear()
        import gc
        gc.collect()

    def forward(self, X: np.ndarray, w: np.ndarray, use_cache: bool | None = None) -> np.ndarray:
        """Normalised expectation values in [-1, 1], one per row of X.
        use_cache=None -> follow self.cache_fm. Only the training set should be cached:
        cached States stay resident on the GPU and caching more data sets exhausted GPU memory."""
        X = np.asarray(X, dtype=np.float64)[:, : self.n_inputs]
        w = np.asarray(w, dtype=np.float64)
        V = np.ascontiguousarray(
            np.concatenate([X, np.broadcast_to(w, (X.shape[0], w.size))], axis=1)
        )
        kw = {} if self.cfg.num_shots is None else {"shots_count": int(self.cfg.num_shots)}

        n_qpu = cudaq.get_target().num_qpus()
        if (self.cache_fm if use_cache is None else use_cache) and n_qpu == 1:
            # The feature-map state depends only on x, never on the weights: simulate it ONCE per
            # sample and, on every later call, simulate only the ansatz starting from that state.
            key = (X.shape, hash(X.tobytes()))
            if key not in self._state_cache:
                print(f"    [cache] simulating feature-map states for {X.shape[0]} samples (once)...", flush=True)
                self._state_cache[key] = [cudaq.get_state(self.k_fm, row.tolist()) for row in X]
            wl = w.tolist()
            ev = np.array([
                cudaq.observe(self.k_from_state, self.hamiltonian, st, wl, **kw).expectation()
                for st in self._state_cache[key]
            ])
        elif n_qpu > 1:
            # multi-GPU ('nvidia' + option 'mqpu'): one asynchronous observe per sample, round-robin over GPUs
            futs = [
                cudaq.observe_async(self.kernel, self.hamiltonian, V[i], qpu_id=i % n_qpu, **kw)
                for i in range(V.shape[0])
            ]
            ev = np.array([f.get().expectation() for f in futs])
        else:
            # one call, whole batch (CUDA-Q broadcasting over the rows of V)
            res = cudaq.observe(self.kernel, self.hamiltonian, V, **kw)
            ev = np.array([r.expectation() for r in res])
        return ev / self.norm


# =============================================================================
# Loss functions       (labels t in {-1, +1};  +1 = suspicious = ~23 % MINORITY)
# =============================================================================
def weighted_hinge_loss(t, p, weights) -> float:
    """sum_i w_i * max(0, 1 - t_i p_i) / sum_i w_i"""
    t, p = np.ravel(t), np.ravel(p)
    return float((weights * np.maximum(0.0, 1.0 - t * p)).sum() / weights.sum())


def modified_weighted_hinge_loss(t, p, weights) -> float:
    """
    sum_i w_i * max(0, 1 - t_i p_i + (1 - p_i)(1 + t_i)) / sum_i w_i
        t = +1 : 3 (1 - p)   <- class +1 (suspicious, minority) is penalised 3x
        t = -1 : 1 + p
    """
    t, p = np.ravel(t), np.ravel(p)
    return float(
        (weights * np.maximum(0.0, 1.0 - t * p + (1.0 - p) * (1.0 + t))).sum() / weights.sum()
    )


def weighted_mse_loss(t, p, weights) -> float:
    t, p = np.ravel(t), np.ravel(p)
    return float((weights * (t - p) ** 2).sum() / weights.sum())


def weighted_bce_loss(t, p, weights, eps: float = 1e-7) -> float:
    """p in [-1,1] is mapped to a probability (1+p)/2 of class +1."""
    t, p = np.ravel(t), np.ravel(p)
    pr = np.clip((1.0 + p) / 2.0, eps, 1.0 - eps)
    t01 = np.where(t == -1, 0, 1)
    el = -(t01 * np.log(pr) + (1.0 - t01) * np.log(1.0 - pr))
    return float(np.sum(weights * el) / weights.sum())


LOSSES = {
    "hinge":   (weighted_hinge_loss, False),
    "whinge":  (weighted_hinge_loss, True),
    "mhinge":  (modified_weighted_hinge_loss, False),
    "wmhinge": (modified_weighted_hinge_loss, True),
    "mse":     (weighted_mse_loss, False),
    "wmse":    (weighted_mse_loss, True),
    "bce":     (weighted_bce_loss, False),
    "wbce":    (weighted_bce_loss, True),
}


# =============================================================================
# Prediction
# =============================================================================
def predict_scores(qnn: CudaqQNN, X, w, zoom_scale=None, bias: float = 0.0) -> np.ndarray:
    """Continuous score = tanh(zoom * raw) + bias   (monotone in raw; sign = class when bias is learned)."""
    raw = qnn.forward(X, w, use_cache=False)      # one batched observe; no cached States
    if zoom_scale is not None:
        raw = np.tanh(zoom_scale * raw)
    return raw + bias


# =============================================================================
# Threshold selection
# =============================================================================
def best_threshold(scores, y, beta=None, min_pr=None, min_re=None, verbose=True) -> float:
    """
    Choose a decision threshold from a PR curve. Call it with TRAIN (or validation) data only.

    precision_recall_curve returns len(thr)+1 precision/recall points; the last one
    (precision=1, recall=0) has no threshold and is dropped, otherwise the precision
    floor can never fail and the 'relax' branch is dead code.
    """
    pr, rec, thr = precision_recall_curve(y, scores)
    pr, rec = pr[:-1], rec[:-1]

    if beta:
        f = (1 + beta**2) * pr * rec / (beta**2 * pr + rec + 1e-12)
        idx = int(np.argmax(f))
        if verbose:
            print(f"Maximising F-beta (beta={beta})")
    elif min_pr:
        valid = pr >= min_pr
        if valid.any():
            idx = int(np.where(valid, rec, -1).argmax())
            if verbose:
                print(f"Flooring precision at {min_pr}: maximising recall")
        else:
            idx = int(np.argmax(pr))
            print(f"WARNING: no threshold reaches precision {min_pr}; using max-precision threshold "
                  f"(precision={pr[idx]:.3f}, recall={rec[idx]:.3f}).")
    elif min_re:
        valid = rec >= min_re
        if valid.any():
            idx = int(np.where(valid, pr, -1).argmax())
            if verbose:
                print(f"Flooring recall at {min_re}: maximising precision")
        else:
            idx = int(np.argmax(rec))
            print(f"WARNING: no threshold reaches recall {min_re}; using max-recall threshold.")
    else:
        f = 2 * pr * rec / (pr + rec + 1e-12)
        idx = int(np.argmax(f))
        if verbose:
            print("No criterion given: maximising F1")

    if verbose:
        print(f"Best threshold: {thr[idx]:.4f} (train precision={pr[idx]:.3f}, recall={rec[idx]:.3f})")
    return float(thr[idx])


# =============================================================================
# Training
# =============================================================================
def _weighted_median(values: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(values)
    cw = np.cumsum(w[order])
    idx = int(np.searchsorted(cw, 0.5 * cw[-1]))   # normalised: works for any weight scale
    return float(values[order][min(idx, len(values) - 1)])


def train_qnn(qnn: CudaqQNN, X, Y, loss_fn, w1=None, w2=None):
    """
    COBYLA training. Returns (best_weights, loss_history, best_seed, best_bias, best_loss).
    The best (weights, bias, loss) triple is tracked over *all* objective evaluations,
    so they always belong to the same point.
    """
    cfg = qnn.cfg
    zoom, dyn = cfg.zoom_scale, cfg.dynamic_bias

    if w1 is None or w2 is None:
        wn1 = wn2 = 1.0
        wvec = np.ones(len(Y))
    else:
        wn1, wn2 = w1, w2
        wvec = np.where(Y == 1, wn1, wn2)

    mask_pos, mask_neg = (Y == 1), (Y == -1)

    best = {"loss": np.inf, "w": None, "bias": 0.0, "seed": cfg.weight_seeds[0], "hist": None}

    for wseed in cfg.weight_seeds:
        rng = np.random.default_rng(wseed)
        w0 = rng.uniform(-np.pi, np.pi, size=qnn.num_weights)
        hist: list[float] = []
        run = {"loss": np.inf, "w": w0.copy(), "bias": 0.0}
        t0 = time.time()

        def objective(w: np.ndarray) -> float:
            raw = qnn.forward(X, w)
            if zoom is not None:
                raw = np.tanh(zoom * raw)

            if dyn in ("wmse", "mse"):
                b = float(np.sum(wvec * (Y - raw)) / np.sum(wvec))
            elif dyn in ("whinge", "hinge"):
                b = _weighted_median(Y - raw, wvec)
            elif dyn in ("wbce", "bce"):
                b = float(np.sum(wvec * (Y - raw)) / np.sum(wvec))
                for _ in range(2):  # Newton-Raphson refinement
                    u_pos = np.clip(1 + raw[mask_pos] + b, 1e-7, None)
                    u_neg = np.clip(1 - raw[mask_neg] - b, 1e-7, None)
                    num = wn1 * np.sum(1.0 / u_pos) - wn2 * np.sum(1.0 / u_neg)
                    den = wn1 * np.sum(1.0 / u_pos**2) + wn2 * np.sum(1.0 / u_neg**2)
                    b += num / den
            else:
                b = 0.0

            cost = loss_fn(Y, raw + b, wvec)
            hist.append(cost)
            if cost < run["loss"]:
                run["loss"], run["w"], run["bias"] = cost, np.array(w, copy=True), b
            if len(hist) % 25 == 0:
                print(f"    eval {len(hist):4d} | loss {cost:.5f} | best {run['loss']:.5f} | "
                      f"{(time.time() - t0) / len(hist):.2f}s/eval", flush=True)
            return cost

        minimize(objective, w0, method="COBYLA", options={"maxiter": cfg.iters})

        if run["loss"] < best["loss"]:
            best = {"loss": run["loss"], "w": run["w"], "bias": run["bias"],
                    "seed": wseed, "hist": hist}

    return best["w"], best["hist"], best["seed"], best["bias"], best["loss"]


# =============================================================================
# Data + persistence
# =============================================================================
def load_data(pnd: int, data_dir: str):
    rd = lambda n: pd.read_csv(os.path.join(data_dir, f"{n}_cmplx_{pnd}.csv"), header=None).to_numpy()
    x_tr, x_te = rd("x_train"), rd("x_test")
    y_tr, y_te = rd("y_train").ravel(), rd("y_test").ravel()
    print(f"x_train {x_tr.shape}  x_test {x_te.shape} | "
          f"+1 fraction: train {np.mean(y_tr == 1):.3f}, test {np.mean(y_te == 1):.3f}")
    return x_tr, y_tr, x_te, y_te


def weights_paths(out_dir: str, config: str, pnd: int):
    stem = os.path.join(out_dir, "weights", f"weights_cmplx_{config}_{pnd}")
    return stem + ".csv", stem + ".json"


def save_model(out_dir, config, pnd, weights, cfg: HQNNConfig, extra: dict):
    csv_path, json_path = weights_paths(out_dir, config, pnd)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    np.savetxt(csv_path, weights, delimiter=",")                 # plain one-column CSV
    meta = {"config": config, "pnd": pnd, "hqnn": asdict(cfg), **extra}
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {csv_path}\nSaved {json_path}")


def load_model(out_dir, config, pnd):
    csv_path, json_path = weights_paths(out_dir, config, pnd)
    w = np.loadtxt(csv_path, delimiter=",")
    with open(json_path) as f:
        meta = json.load(f)
    return w, HQNNConfig(**meta["hqnn"]), meta


# =============================================================================
# One config end-to-end
# =============================================================================
def run_config(config: str, pnd: int, data, cfg: HQNNConfig, out_dir: str,
               beta=None, min_pr=0.55, min_re=None, cache_fm: bool = False):
    x_tr, y_tr, x_te, y_te = data
    cimb = float(np.mean(y_tr == 1))                 # fraction of class +1 (minority)
    W1 = 1.0 / (2.0 * cimb)                          # weight for +1 (minority)  -> large
    W2 = 1.0 / (2.0 * (1.0 - cimb))                  # weight for -1 (majority)  -> small

    if config not in LOSSES:
        raise ValueError(f"unknown config '{config}', choose from {list(LOSSES)}")
    loss_fn, weighted = LOSSES[config]
    w1, w2 = (W1, W2) if weighted else (None, None)

    print(f"\n{'=' * 78}\n  CONFIG {config.upper()} | pnd={pnd} | W1(+1)={W1:.3f} W2(-1)={W2:.3f}\n{'=' * 78}")
    qnn = CudaqQNN(x_tr.shape[1], cfg, cache_fm=cache_fm)

    t0 = time.time()
    weights, hist, seed, bias, best_loss = train_qnn(qnn, x_tr, y_tr, loss_fn, w1, w2)
    print(f"Training done in {time.time() - t0:.1f}s | evaluations={len(hist)} | best loss={best_loss:.5f} "
          f"| bias={bias:.4f} | weight seed={seed}")
    cfg.bias = float(bias)
    qnn.clear_cache()                                    # release GPU memory held by cached states
    # Save immediately: a crash during prediction must not lose the training run.
    save_model(out_dir, config, pnd, weights, cfg,
               {"best_loss": float(best_loss), "threshold": None, "weight_seed": seed,
                "n_evals": len(hist), "loss_history": [float(v) for v in hist]})

    s_tr = predict_scores(qnn, x_tr, weights, cfg.zoom_scale, bias)
    s_te = predict_scores(qnn, x_te, weights, cfg.zoom_scale, bias)

    # Threshold from TRAIN scores only (never from y_test).
    thr = best_threshold(s_tr, y_tr, beta, min_pr, min_re)

    print(f"\nTrain AUC-PR: {average_precision_score(y_tr, s_tr):.4f}")
    print(f"Test  AUC-PR: {average_precision_score(y_te, s_te):.4f}")
    names = ["benign (-1)", "suspicious (+1)"]
    print("\n-- Train @ threshold --")
    print(classification_report(y_tr, np.where(s_tr >= thr, 1, -1), target_names=names, zero_division=0))
    print("-- Test @ train-chosen threshold --")
    print(classification_report(y_te, np.where(s_te >= thr, 1, -1), target_names=names, zero_division=0))

    save_model(out_dir, config, pnd, weights, cfg,
               {"best_loss": float(best_loss), "threshold": thr, "weight_seed": seed,
                "n_evals": len(hist), "loss_history": [float(v) for v in hist]})
    return {"config": config, "best_loss": best_loss, "threshold": thr,
            "test_auc_pr": float(average_precision_score(y_te, s_te))}


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="HQNN training on NVIDIA CUDA-Q")
    ap.add_argument("pnd", type=int, help="dataset id, e.g. 11 -> x_train_cmplx_11.csv")
    ap.add_argument("--configs", nargs="+", default=["whinge", "mhinge", "wmse", "wbce"],
                    choices=list(LOSSES))
    ap.add_argument("--data-dir", default=HERE)
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--freps", type=int, default=2)
    ap.add_argument("--areps", type=int, default=3)
    ap.add_argument("--fm", default="zz")
    ap.add_argument("--fent", default="pairwise")
    ap.add_argument("--ansatz", default="realamplitude")
    ap.add_argument("--aent", default="pairwise")
    ap.add_argument("--obs", default="local", choices=["local", "global"])
    ap.add_argument("--weight-seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--zoom", type=float, default=None)
    ap.add_argument("--dynamic-bias", default=None,
                    choices=[None, "mse", "wmse", "hinge", "whinge", "bce", "wbce"])
    ap.add_argument("--shots", type=int, default=None, help="None = exact expectation values")
    ap.add_argument("--target", default="auto", help="auto | nvidia | qpp-cpu | ...")
    ap.add_argument("--mqpu", action="store_true", help="nvidia target: spread circuits over all GPUs")
    ap.add_argument("--fp64", action="store_true", help="nvidia target: double precision (default fp32)")
    ap.add_argument("--cache-fm", action="store_true",
                    help="simulate the (weight-independent) feature map once per sample and reuse the state")
    ap.add_argument("--min-pr", type=float, default=0.55)
    ap.add_argument("--min-re", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    a = ap.parse_args()

    init_target(a.target, a.mqpu, a.fp64)
    data = load_data(a.pnd, a.data_dir)

    summary = []
    for config in a.configs:
        cfg = HQNNConfig(fm=a.fm, freps=a.freps, fent=a.fent, ansatz=a.ansatz, areps=a.areps,
                         aent=a.aent, obs_type=a.obs, iters=a.iters, weight_seeds=a.weight_seeds,
                         zoom_scale=a.zoom, dynamic_bias=a.dynamic_bias, num_shots=a.shots)
        summary.append(run_config(config, a.pnd, data, cfg, a.out_dir, a.beta, a.min_pr, a.min_re, a.cache_fm))

    print(f"\n{'=' * 78}\n SUMMARY\n{'=' * 78}")
    for r in summary:
        print(f"{r['config']:8s} best train loss {r['best_loss']:.4f} | test AUC-PR {r['test_auc_pr']:.4f}")


if __name__ == "__main__":
    main()
