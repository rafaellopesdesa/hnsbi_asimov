"""Setup, classifier ratios, and reference sampling."""

import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd

FEATURES = ["x1", "x2", "x3", "x4", "x5"]


def setup_workspace(repo, workspace):
    backend = repo / ".dependencies" / "nsbi-lhc-toolkit"
    if not backend.exists():
        backend.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/rafaellopesdesa/nsbi-lhc-toolkit.git",
                str(backend),
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(backend), "sparse-checkout", "set", "src"], check=True)
        subprocess.run(
            ["git", "-C", str(backend), "checkout", "832f3086bc7cf830f2b5d907cb2a2d693f50a391"],
            check=True,
        )
    sys.path.insert(0, str(backend / "src"))
    workspace.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)


def as_inference_session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )


def load_ratio_pack(model_dir, member=0):
    from nsbi_common_utils.training.utils import load_trained_model

    suffix = "" if member is None else str(member)
    scaler, model = load_trained_model(
        str(model_dir / f"model{suffix}.onnx"), str(model_dir / f"model_scaler{suffix}.bin")
    )
    return {"scaler": scaler, "model": as_inference_session(model)}


def predict_with_model(data, scaler, model, batch_size=10_000):
    scaled = np.asarray(scaler.transform(data), dtype=np.float32)
    input_name = model.get_inputs()[0].name
    scores = np.concatenate(
        [
            model.run(None, {input_name: scaled[start : start + batch_size]})[0].reshape(-1)
            for start in range(0, len(scaled), batch_size)
        ]
    ).astype(np.float64)
    scores = np.clip(scores, 0, 1 - 1e-9)
    return scores / (1 - scores)


def evaluate_ratio_packs(packs, values, batch_size=100_000):
    values = np.asarray(values, dtype=np.float32)
    chunks = []
    for start in range(0, len(values), batch_size):
        batch = pd.DataFrame(values[start : start + batch_size], columns=FEATURES)
        chunks.append(np.mean([predict_with_model(batch, **pack) for pack in packs], axis=0))
    return np.maximum(np.concatenate(chunks), 1e-12)


def ratio_training_dataframe(target, reference, n_events, seed):
    frames = []
    for label, (frame, sample_seed) in enumerate([(reference, seed + 1), (target, seed)]):
        frame = frame.sample(n=n_events, random_state=sample_seed)[FEATURES + ["weight"]].copy()
        frame["weights_normed"] = frame["weight"] / frame["weight"].sum()
        frame["train_labels"] = float(label)
        frames.append(frame)
    return pd.concat(frames[::-1], ignore_index=True).sample(
        frac=1, random_state=seed + 2, ignore_index=True
    )


def density_ratio_trainer(**kwargs):
    from nsbi_common_utils.training import neural_ratio_estimation as nre

    nre.plot_loss = lambda *args, **kwargs: None
    return nre.density_ratio_trainer(**kwargs)


def sample_selected_flow(flow, n_events, select, batch_size=65_536, double_needed=False):
    from utils_nf import flow_sample_x

    chunks, kept, generated, passed = [], 0, 0, 0
    while kept < n_events:
        needed = (n_events - kept) * (2 if double_needed else 1)
        values = flow_sample_x(
            flow, max(batch_size, min(needed, 4 * batch_size)), batch_size=batch_size
        )
        mask = select(pd.DataFrame(values, columns=FEATURES))
        chunks.append(values[mask])
        kept += int(mask.sum())
        passed += int(mask.sum())
        generated += len(values)
    return np.concatenate(chunks)[:n_events], passed / generated
