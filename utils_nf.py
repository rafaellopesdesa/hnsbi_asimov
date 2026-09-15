"""Streaming, normalizing-flow, and checkpoint helpers for this analysis."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

_UINT64_MASK = (1 << 64) - 1


def _splitmix64(values):
    """Vectorized SplitMix64 hash for deterministic row-level decisions."""
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(11400714819323198485)
        values = (values ^ values >> np.uint64(30)) * np.uint64(13787848793156543929)
        values = (values ^ values >> np.uint64(27)) * np.uint64(10723151780598845931)
    return values ^ values >> np.uint64(31)


def _hashed_uniform(row_indices, seed):
    hashed = _hash_rows(row_indices, seed)
    return (hashed >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)


def _hash_rows(row_indices, seed):
    seed_uint64 = np.uint64(int(seed) & _UINT64_MASK)
    with np.errstate(over="ignore"):
        seeded_indices = np.asarray(row_indices, dtype=np.uint64) + seed_uint64
    return _splitmix64(seeded_indices)


def _stream_split_masks(row_indices, *, presel_fraction, flow_train_fraction, seed):
    """Assign rows reproducibly to PRESEL, flow-train, or evaluation."""
    uniform = _hashed_uniform(row_indices, seed)
    flow_train_boundary = presel_fraction + (1.0 - presel_fraction) * flow_train_fraction
    return {
        "presel": uniform < presel_fraction,
        "flow_train": (uniform >= presel_fraction) & (uniform < flow_train_boundary),
        "eval": uniform >= flow_train_boundary,
    }


def _iter_parquet_batches(parquet_path, *, columns, batch_size):
    """Yield ``(global_row_indices, dataframe)`` without loading the file."""
    import pyarrow.parquet as pq

    columns = list(dict.fromkeys(columns))
    parquet_file = pq.ParquetFile(parquet_path, pre_buffer=False)
    row_offset = 0
    for record_batch in parquet_file.iter_batches(
        batch_size=int(batch_size), columns=columns, use_threads=True
    ):
        dataframe = record_batch.to_pandas()
        n_rows = len(dataframe)
        row_indices = np.arange(row_offset, row_offset + n_rows, dtype=np.uint64)
        row_offset += n_rows
        yield (row_indices, dataframe)


def _prepare_stream_batch(dataframe, features):
    for feature in features:
        dataframe[feature] = dataframe[feature].astype(np.float32, copy=False)
    dataframe["weight"] = dataframe["weight"].astype(np.float64, copy=False)
    return dataframe


def _update_priority_reservoir(retained, candidates, priorities, max_events):
    """Keep the rows with the smallest deterministic random priorities."""
    candidates = candidates.copy()
    candidates["_stream_priority"] = np.asarray(priorities, dtype=np.uint64)
    if retained is None:
        combined = candidates
    else:
        combined = pd.concat([retained, candidates], ignore_index=True, copy=False)
    if len(combined) > max_events:
        priority = combined["_stream_priority"].to_numpy(dtype=np.uint64)
        keep = np.argpartition(priority, max_events - 1)[:max_events]
        combined = combined.iloc[keep].reset_index(drop=True)
    return combined


def _finish_priority_reservoir(retained, columns):
    retained = retained.sort_values("_stream_priority", kind="stable")
    return retained.drop(columns="_stream_priority").reset_index(drop=True)


def sample_parquet_partition(
    parquet_path,
    *,
    features,
    partition,
    max_events,
    batch_size,
    presel_fraction,
    flow_train_fraction,
    split_seed,
    reservoir_seed,
):
    """Stream one parquet partition and retain at most ``max_events`` rows."""
    max_events = int(max_events)
    columns = [*features, "weight"]
    retained = None
    stats: dict[str, float | int] = {"inclusive_weight": 0.0, "partition_weight": 0.0}
    for row_indices, batch in _iter_parquet_batches(
        parquet_path, columns=columns, batch_size=batch_size
    ):
        batch = _prepare_stream_batch(batch, features)
        masks = _stream_split_masks(
            row_indices,
            presel_fraction=presel_fraction,
            flow_train_fraction=flow_train_fraction,
            seed=split_seed,
        )
        mask = masks[partition]
        stats["inclusive_weight"] += float(batch["weight"].sum())
        stats["partition_weight"] += float(batch.loc[mask, "weight"].sum())
        selected_indices = row_indices[mask]
        priority = _hash_rows(selected_indices, reservoir_seed)
        retained = _update_priority_reservoir(
            retained, batch.loc[mask, columns], priority, max_events
        )
    sample = _finish_priority_reservoir(retained, columns)
    return (sample, stats)


def accumulate_preselection_histogram(
    parquet_path,
    *,
    features,
    ratio_predictor,
    log_ratio_edges,
    batch_size,
    presel_fraction,
    flow_train_fraction,
    split_seed,
):
    """Accumulate a weighted PRESEL-ratio histogram on flow-training rows."""
    edges = np.asarray(log_ratio_edges, dtype=np.float64)
    columns = [*features, "weight"]
    histogram = np.zeros(len(edges) - 1, dtype=np.float64)
    stats: dict[str, float | int] = {"inclusive_weight": 0.0, "partition_weight": 0.0}
    ratio_min = float(np.exp(edges[0]))
    ratio_max = float(np.exp(edges[-1]))
    for row_indices, batch in _iter_parquet_batches(
        parquet_path, columns=columns, batch_size=batch_size
    ):
        batch = _prepare_stream_batch(batch, features)
        stats["inclusive_weight"] += float(batch["weight"].sum())
        mask = _stream_split_masks(
            row_indices,
            presel_fraction=presel_fraction,
            flow_train_fraction=flow_train_fraction,
            seed=split_seed,
        )["flow_train"]
        if not mask.any():
            continue
        partition_batch = batch.loc[mask, columns]
        ratio = np.asarray(
            ratio_predictor(partition_batch[list(features)]), dtype=np.float64
        ).reshape(-1)
        valid_ratio = ratio > 0.0
        log_ratio = np.log(np.clip(ratio[valid_ratio], ratio_min, ratio_max))
        histogram += np.histogram(
            log_ratio,
            bins=edges,
            weights=partition_batch["weight"].to_numpy(dtype=np.float64)[valid_ratio],
        )[0]
        stats["partition_weight"] += float(partition_batch["weight"].sum())
    return (histogram, stats)


def choose_preselection_ratio_cut(
    signal_histogram,
    background_histogram,
    log_ratio_edges,
    *,
    signal_inclusive_yield,
    background_inclusive_yield,
    signal_partition_weight,
    background_partition_weight,
    target_background_to_signal,
):
    """Choose the loosest histogrammed ratio cut reaching the B/S target."""
    signal_histogram = np.asarray(signal_histogram, dtype=np.float64)
    background_histogram = np.asarray(background_histogram, dtype=np.float64)
    edges = np.asarray(log_ratio_edges, dtype=np.float64)
    signal_yield = (
        float(signal_inclusive_yield)
        * np.cumsum(signal_histogram[::-1])
        / float(signal_partition_weight)
    )
    background_yield = (
        float(background_inclusive_yield)
        * np.cumsum(background_histogram[::-1])
        / float(background_partition_weight)
    )
    background_to_signal = np.divide(
        background_yield,
        signal_yield,
        out=np.full_like(background_yield, np.inf),
        where=signal_yield > 0.0,
    )
    valid = np.flatnonzero(
        (signal_yield > 0.0) & (background_to_signal <= float(target_background_to_signal))
    )
    best = valid[np.argmax(signal_yield[valid])]
    lower_edges_descending = edges[:-1][::-1]
    ratio_cut = float(np.exp(lower_edges_descending[best]))
    diagnostics = {
        "histogram_signal_yield": float(signal_yield[best]),
        "histogram_background_yield": float(background_yield[best]),
        "histogram_background_to_signal": float(background_to_signal[best]),
    }
    return (ratio_cut, diagnostics)


def collect_preselected_parquet(
    parquet_path,
    *,
    features,
    ratio_predictor,
    ratio_cut,
    max_train_events,
    max_eval_events,
    batch_size,
    presel_fraction,
    flow_train_fraction,
    split_seed,
    reservoir_seed,
):
    """Stream, classify, and retain bounded post-selection train/eval samples."""
    max_events = {"flow_train": int(max_train_events), "eval": int(max_eval_events)}
    columns = [*features, "weight"]
    retained: dict[str, pd.DataFrame | None] = {"flow_train": None, "eval": None}
    stats: dict[str, dict[str, float | int]] = {
        split: {"partition_weight": 0.0, "selected_weight": 0.0} for split in retained
    }
    for row_indices, batch in _iter_parquet_batches(
        parquet_path, columns=columns, batch_size=batch_size
    ):
        batch = _prepare_stream_batch(batch, features)
        masks = _stream_split_masks(
            row_indices,
            presel_fraction=presel_fraction,
            flow_train_fraction=flow_train_fraction,
            seed=split_seed,
        )
        relevant = masks["flow_train"] | masks["eval"]
        if not relevant.any():
            continue
        relevant_positions = np.flatnonzero(relevant)
        ratio = np.asarray(
            ratio_predictor(batch.loc[relevant, list(features)]), dtype=np.float64
        ).reshape(-1)
        passes = np.zeros(len(batch), dtype=bool)
        passes[relevant_positions] = ratio >= float(ratio_cut)
        for split_index, split in enumerate(["flow_train", "eval"]):
            split_mask = masks[split]
            selected_mask = split_mask & passes
            stats[split]["partition_weight"] += float(batch.loc[split_mask, "weight"].sum())
            stats[split]["selected_weight"] += float(batch.loc[selected_mask, "weight"].sum())
            selected_indices = row_indices[selected_mask]
            priority = _hash_rows(selected_indices, int(reservoir_seed) + split_index)
            retained[split] = _update_priority_reservoir(
                retained[split], batch.loc[selected_mask, columns], priority, max_events[split]
            )
    samples = {split: _finish_priority_reservoir(retained[split], columns) for split in retained}
    return (samples, stats)


@dataclass
class Standardizer:
    """Per-feature affine standardization used before flow training."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x, sample_weights=None):
        x = np.asarray(x, dtype=np.float32)
        if sample_weights is None:
            mean = x.mean(axis=0).astype(np.float32)
            std = x.std(axis=0).astype(np.float32)
        else:
            weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
            weights = weights / weights.sum()
            mean64 = np.sum(weights[:, None] * x.astype(np.float64), axis=0)
            variance64 = np.sum(weights[:, None] * (x.astype(np.float64) - mean64) ** 2, axis=0)
            mean = mean64.astype(np.float32)
            std = np.sqrt(variance64).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, z):
        z = np.asarray(z, dtype=np.float32)
        return (z * self.std + self.mean).astype(np.float32)

    @property
    def log_det_x_to_z_standardization(self):
        return float(-np.log(self.std).sum())


def _build_quadratic_spline(config):
    """Build a rational-quadratic spline coupling flow using ``nflows``."""
    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.nn.nets import ResidualNet
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
    from nflows.transforms.permutations import ReversePermutation
    from nflows.utils.torchutils import create_alternating_binary_mask

    n_features = int(config["n_features"])
    hidden_features = int(config["hidden_features"])
    hidden_layers = int(config["hidden_layers"])
    dropout_probability = float(config.get("dropout_probability", 0.0))

    def make_net(in_features, out_features):
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            num_blocks=hidden_layers,
            activation=torch.nn.functional.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=False,
        )

    transforms = []
    for layer_index in range(int(config["n_coupling_layers"])):
        mask = create_alternating_binary_mask(features=n_features, even=layer_index % 2 == 0)
        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_net,
                num_bins=int(config.get("spline_num_bins", 8)),
                tails="linear",
                tail_bound=float(config.get("spline_tail_bound", 3.0)),
                apply_unconditional_transform=False,
            )
        )
        if layer_index + 1 < int(config["n_coupling_layers"]):
            transforms.append(ReversePermutation(features=n_features))
    return Flow(
        transform=CompositeTransform(transforms), distribution=StandardNormal(shape=[n_features])
    )


def build_flow(model_config, device):
    return _build_quadratic_spline(model_config).to(device)


def checkpoint_path(sample_name, model_dir, flow_type):
    """Return a model-specific checkpoint path without architecture collisions."""
    return Path(model_dir) / f"{flow_type}_{sample_name}.pt"


def _save_flow(path, flow, scaler, features, model_config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": flow.state_dict(),
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "features": list(features),
            "model_config": dict(model_config),
        },
        path,
    )
    return path


def load_flow(sample_name, *, model_dir, flow_type, device):
    """Load a trained flow and its standardization from a checkpoint."""
    path = checkpoint_path(sample_name, model_dir, flow_type)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_features = list(checkpoint["features"])
    model_config = checkpoint["model_config"]
    flow = build_flow(model_config, device)
    flow.load_state_dict(checkpoint["state_dict"])
    flow.eval()
    scaler = Standardizer(
        mean=np.asarray(checkpoint["scaler_mean"], dtype=np.float32),
        std=np.asarray(checkpoint["scaler_std"], dtype=np.float32),
    )
    return {
        "flow": flow,
        "scaler": scaler,
        "features": saved_features,
        "model_config": model_config,
        "path": path,
    }


def _make_loaders(x_scaled, *, sample_weights=None, validation_fraction, batch_size, seed):
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    weight_tensor = None
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        weights = weights * (len(weights) / weights.sum())
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
    n_total = len(x_tensor)
    n_val = min(n_total - 1, max(1, int(round(validation_fraction * n_total))))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_total, generator=generator)
    if weight_tensor is None:
        train_ds = TensorDataset(x_tensor[permutation[:n_train]])
        val_ds = TensorDataset(x_tensor[permutation[n_train:]])
    else:
        train_ds = TensorDataset(
            x_tensor[permutation[:n_train]], weight_tensor[permutation[:n_train]]
        )
        val_ds = TensorDataset(
            x_tensor[permutation[n_train:]], weight_tensor[permutation[n_train:]]
        )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return (train_loader, val_loader)


def train_flow(
    sample_name,
    df_train,
    *,
    features,
    model_dir,
    model_config,
    training_config,
    device,
    sample_weights=None,
    load_if_available=True,
    seed=0,
):
    """Train (or load) one flow using notebook-selected hyperparameters."""
    model_config = dict(model_config)
    model_config["flow_type"] = "quadratic_spline"
    path = checkpoint_path(sample_name, model_dir, model_config["flow_type"])
    if load_if_available and path.exists():
        return load_flow(
            sample_name, model_dir=model_dir, flow_type=model_config["flow_type"], device=device
        )
    training_config = dict(training_config)
    batch_size = int(training_config["batch_size"])
    n_epochs = int(training_config["n_epochs"])
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    validation_fraction = float(training_config.get("validation_fraction", 0.2))
    patience = int(training_config.get("patience", n_epochs))
    gradient_clip = training_config.get("gradient_clip", 5.0)
    lr_scheduler_factor = float(training_config.get("lr_scheduler_factor", 0.2))
    lr_scheduler_patience = int(training_config.get("lr_scheduler_patience", 2))
    min_learning_rate = float(training_config.get("min_learning_rate", learning_rate * 0.001))
    x = df_train[list(features)].to_numpy(dtype=np.float32)
    selected_weights = sample_weights
    scaler = Standardizer.fit(x, sample_weights=selected_weights)
    x_scaled = scaler.transform(x)
    train_loader, val_loader = _make_loaders(
        x_scaled,
        sample_weights=selected_weights,
        validation_fraction=validation_fraction,
        batch_size=batch_size,
        seed=seed,
    )
    flow = build_flow(model_config, device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        threshold=0.0001,
        threshold_mode="abs",
        min_lr=min_learning_rate,
    )
    best_val = np.inf
    best_state = None
    stale_epochs = 0
    for epoch in range(1, n_epochs + 1):
        flow.train()
        for packed_batch in train_loader:
            batch = packed_batch[0].to(device)
            event_nll = -flow.log_prob(batch)
            if len(packed_batch) == 1:
                loss = event_nll.mean()
            else:
                batch_weights = packed_batch[1].to(device)
                loss = torch.sum(batch_weights * event_nll) / len(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=gradient_clip)
            optimizer.step()
        flow.eval()
        val_nll_sum = 0.0
        val_weight_sum = 0.0
        with torch.no_grad():
            for packed_batch in val_loader:
                batch = packed_batch[0].to(device)
                event_nll = -flow.log_prob(batch)
                if len(packed_batch) == 1:
                    val_nll_sum += float(event_nll.sum().detach().cpu())
                    val_weight_sum += len(batch)
                else:
                    batch_weights = packed_batch[1].to(device)
                    val_nll_sum += float(torch.sum(batch_weights * event_nll).detach().cpu())
                    val_weight_sum += float(batch_weights.sum().detach().cpu())
        val_loss = val_nll_sum / val_weight_sum
        learning_rate_before_step = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        print(f"{sample_name}: epoch {epoch}, validation NLL {val_loss:.4f}")
        if val_loss < best_val - 0.0001:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in flow.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if current_learning_rate < learning_rate_before_step:
            stale_epochs = 0
        elif stale_epochs >= patience:
            break
    if best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval()
    saved = _save_flow(path, flow, scaler, features=features, model_config=model_config)
    return {
        "flow": flow,
        "scaler": scaler,
        "features": list(features),
        "model_config": model_config,
        "path": saved,
    }


@torch.no_grad()
def flow_log_prob_x(flow_pack, x, batch_size=65536):
    """Evaluate log densities in the original, unstandardized coordinates."""
    features = list(flow_pack["features"])
    if isinstance(x, pd.DataFrame):
        x_array = x[features].to_numpy(dtype=np.float32)
    else:
        x_array = np.asarray(x, dtype=np.float32)
    flow = flow_pack["flow"]
    scaler = flow_pack["scaler"]
    device = next(flow.parameters()).device
    flow.eval()
    chunks = []
    for start in range(0, len(x_array), batch_size):
        x_scaled = scaler.transform(x_array[start : start + batch_size])
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
        log_p_scaled = flow.log_prob(x_tensor).detach().cpu().numpy()
        chunks.append(log_p_scaled + scaler.log_det_x_to_z_standardization)
    return np.concatenate(chunks)


@torch.no_grad()
def flow_sample_x(flow_pack, n, batch_size=65536):
    """Draw flow samples and return them in the original coordinates."""
    flow = flow_pack["flow"]
    scaler = flow_pack["scaler"]
    flow.eval()
    chunks = []
    remaining = int(n)
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        sample = flow.sample(current_batch).detach().cpu().numpy()
        chunks.append(scaler.inverse(sample))
        remaining -= current_batch
    return np.concatenate(chunks, axis=0)
