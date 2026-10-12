"""
Optional check: the CUDA-Q kernels give the same expectation values as the
original Qiskit circuits (needs `pip install qiskit`; Qiskit is NOT needed to
train or evaluate).

    python verify_vs_qiskit.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import efficient_su2, n_local, real_amplitudes
from qiskit.circuit.library.data_preparation import z_feature_map, zz_feature_map
from qiskit.quantum_info import SparsePauliOp, Statevector

import cudaq
from hqnn_cudaq import HQNNConfig, CudaqQNN, init_target


def my_data_map(x):                       # identical to the original hqnn.py
    return x[0] if len(x) == 1 else x[0] * x[1] / np.pi


def qiskit_circuit(n_feat, c: HQNNConfig):
    if c.fm == "zz":
        fmap = zz_feature_map(n_feat, data_map_func=my_data_map, reps=c.freps, entanglement=c.fent)
    elif c.fm == "z":
        fmap = z_feature_map(n_feat, reps=c.freps)
    else:  # dense angle (as in original build_feature_map)
        nq = n_feat // 2
        fmap = QuantumCircuit(nq)
        xs = ParameterVector("x", n_feat)
        for i in range(nq):
            fmap.ry(2 * xs[2 * i], i)
            fmap.rz(2 * xs[2 * i + 1], i)
        for i in range(nq - 1 if c.fent in ("linear", "circular") else 0):
            fmap.cz(i, i + 1)
        if c.fent == "circular":
            fmap.cz(nq - 1, 0)
    nq = fmap.num_qubits
    if c.ansatz == "efficientsu2":
        ans = efficient_su2(nq, reps=c.areps, entanglement=c.aent)
    elif c.ansatz == "nlocal":
        ans = n_local(nq, rotation_blocks="ry", entanglement_blocks="cx", reps=c.areps, entanglement=c.aent)
    else:
        ans = real_amplitudes(nq, reps=c.areps, entanglement=c.aent)
    return fmap.compose(ans), fmap, ans


cases = [
    ("PRODUCTION  zz/pairwise + realamplitude/pairwise, 15q", 15, HQNNConfig(fm="zz", freps=2, fent="pairwise", ansatz="realamplitude", areps=3, aent="pairwise")),
    ("zz/full + efficientsu2/circular, 6q",                    6,  HQNNConfig(fm="zz", freps=2, fent="full", ansatz="efficientsu2", areps=2, aent="circular")),
    ("zz/reverse_linear + nlocal/linear, 7q",                  7,  HQNNConfig(fm="zz", freps=3, fent="reverse_linear", ansatz="nlocal", areps=2, aent="linear")),
    ("z + realamplitude/full, 6q",                             6,  HQNNConfig(fm="z", freps=2, ansatz="realamplitude", areps=2, aent="full")),
    ("da/circular + efficientsu2/pairwise, 12 feats -> 6q",    12, HQNNConfig(fm="da", fent="circular", ansatz="efficientsu2", areps=2, aent="pairwise")),
    ("global observable, zz/pairwise, 6q",                     6,  HQNNConfig(fm="zz", freps=2, fent="pairwise", ansatz="realamplitude", areps=3, aent="pairwise", obs_type="global")),
]

init_target("qpp-cpu")
rng = np.random.default_rng(0)
worst = 0.0
for name, n_feat, cfg in cases:
    qc, fmap, ans = qiskit_circuit(n_feat, cfg)
    nq = qc.num_qubits
    # parameter order in Qiskit: inputs first, then weights (same as in the original code)
    order = list(fmap.parameters) + list(ans.parameters)
    assert len(order) == qc.num_parameters
    obs = (SparsePauliOp.from_list([("I" * k + "Z" + "I" * (nq - k - 1), 1.0) for k in range(nq)])
           if cfg.obs_type == "local" else SparsePauliOp.from_list([("Z" * nq, 1.0)]))

    qnn = CudaqQNN(n_feat, cfg)
    assert qnn.num_weights == len(ans.parameters), (qnn.num_weights, len(ans.parameters))
    X = rng.uniform(0, np.pi, (5, n_feat))
    w = rng.uniform(-np.pi, np.pi, qnn.num_weights)
    got = qnn.forward(X, w) * qnn.norm

    ref = []
    for xi in X:
        vals = np.concatenate([xi[: len(fmap.parameters)], w])
        bound = qc.assign_parameters(dict(zip(order, vals)))
        ref.append(Statevector(bound).expectation_value(obs).real)
    err = float(np.max(np.abs(got - np.array(ref))))
    worst = max(worst, err)
    print(f"{'OK ' if err < 1e-5 else 'BAD'} {name:58s} max|diff| = {err:.2e}")

print("\nALL MATCH" if worst < 1e-5 else "\nMISMATCH - do not trust the port")
