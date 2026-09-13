"""Pseudo-experiments for the nominal hNDE likelihood."""

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def build_compressed_q_model(
    q_values, signal_weights, background_weights, lam_sig, lam_bkg, n_bins
):
    q_values = np.asarray(q_values, dtype=np.float64)
    log_q = np.log(np.maximum(q_values, np.finfo(np.float64).tiny))
    lower, upper = np.quantile(log_q, [1e-06, 1.0 - 1e-06])
    interior = np.linspace(lower, upper, int(n_bins) - 1)
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    signal_probability = np.histogram(log_q, bins=edges, weights=signal_weights)[0].astype(
        np.float64
    )
    background_probability = np.histogram(log_q, bins=edges, weights=background_weights)[0].astype(
        np.float64
    )
    probability_floor = 1e-15
    signal_probability = signal_probability + probability_floor
    background_probability = background_probability + probability_floor
    signal_probability /= signal_probability.sum()
    background_probability /= background_probability.sum()
    q_binned = lam_sig / lam_bkg * signal_probability / background_probability
    return {
        "q": q_binned,
        "signal_probability": signal_probability,
        "background_probability": background_probability,
        "log_q_edges": edges,
    }


def make_toy_fitter(q, lam_sig, lam_bkg):
    q = jnp.asarray(q, dtype=jnp.float64)

    def fit_batch(counts):
        counts = jnp.asarray(counts, dtype=jnp.float64)
        q_values = q
        initial_mu = jnp.clip((jnp.sum(counts, axis=1) - lam_bkg) / lam_sig, 0.0, 12.0)
        score_at_zero = lam_sig - jnp.sum(counts * q_values, axis=1)

        def newton_step(_, mu):
            response = q_values / (1.0 + mu[:, None] * q_values)
            score = lam_sig - jnp.sum(counts * response, axis=1)
            information = jnp.sum(counts * response**2, axis=1)
            step = score / jnp.maximum(information, 1e-12)
            step = jnp.clip(step, -2.0, 2.0)
            return jnp.clip(mu - step, 0.0, 12.0)

        mu_hat = jax.lax.fori_loop(0, 16, newton_step, initial_mu)
        mu_hat = jnp.where(score_at_zero >= 0.0, 0.0, mu_hat)

        def statistic(mu_test):
            value = 2.0 * (
                (mu_test - mu_hat) * lam_sig
                - jnp.sum(
                    counts
                    * (jnp.log1p(mu_test * q_values) - jnp.log1p(mu_hat[:, None] * q_values)),
                    axis=1,
                )
            )
            return jnp.maximum(value, 0.0)

        return mu_hat, statistic(0.0)

    return jax.jit(fit_batch)


def run_toys(
    mu_true,
    n_toys,
    seed,
    signal_probability,
    background_probability,
    lam_sig,
    lam_bkg,
    batch_size,
    fit_batch,
):
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, n_toys, batch_size):
        shape = (min(batch_size, n_toys - start), len(signal_probability))
        signal_counts = rng.poisson(mu_true * lam_sig * signal_probability, size=shape)
        background_counts = rng.poisson(lam_bkg * background_probability, size=shape)
        mu_hat, q_zero = fit_batch(signal_counts + background_counts)
        chunks.append(pd.DataFrame({"mu_hat": np.asarray(mu_hat), "q_zero": np.asarray(q_zero)}))
    return pd.concat(chunks, ignore_index=True)


def _simulate_reconstructed_mixture(
    components, n_events, rng, feature_names, response_scale, response_resolution
):
    """Run one batch of the latent mixture and detector-response simulator."""
    n_events = int(n_events)
    fractions = np.asarray([component[0] for component in components], dtype=float)
    fractions /= fractions.sum()
    component_index = rng.choice(len(components), size=n_events, p=fractions)
    latent = np.empty((n_events, len(feature_names)), dtype=np.float64)
    for index, (_, mean, covariance) in enumerate(components):
        mask = component_index == index
        if np.any(mask):
            latent[mask] = rng.multivariate_normal(
                np.asarray(mean, dtype=float),
                np.asarray(covariance, dtype=float),
                size=int(mask.sum()),
            )
    reconstructed = np.asarray(response_scale, dtype=np.float64)[None, :] * latent + rng.normal(
        loc=0.0, scale=np.asarray(response_resolution, dtype=np.float64)[None, :], size=latent.shape
    )
    return pd.DataFrame(reconstructed.astype(np.float32, copy=False), columns=feature_names)


def load_or_generate_simulator_q_bank(
    *,
    sample_name,
    components,
    n_selected,
    seed,
    generation_batch_size,
    bank_dir,
    feature_names,
    response_scale,
    response_resolution,
    presel_ratio_cut,
    evaluate_presel_ratio,
    evaluate_ratio,
    ratio_normalization,
    lam_sig,
    lam_bkg,
    ratio_evaluation_batch_size=100000,
):
    """Load or generate an independent selected simulator bank of model q values."""
    bank_dir = Path(bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    path = bank_dir / f"{sample_name}_selected_q_{n_selected}_seed{seed}.npz"
    if path.exists():
        return np.load(path)["q"]
    rng = np.random.default_rng(int(seed))
    q_chunks = []
    n_kept = 0
    while n_kept < int(n_selected):
        generated = _simulate_reconstructed_mixture(
            components,
            generation_batch_size,
            rng,
            feature_names,
            response_scale,
            response_resolution,
        )
        passes = evaluate_presel_ratio(generated) >= presel_ratio_cut
        selected = generated.loc[passes, feature_names].reset_index(drop=True)
        if len(selected):
            ratio_signal = (
                evaluate_ratio("signal", selected, batch_size=ratio_evaluation_batch_size)
                / ratio_normalization["signal"]
            )
            ratio_background = (
                evaluate_ratio("background", selected, batch_size=ratio_evaluation_batch_size)
                / ratio_normalization["background"]
            )
            log_q = np.log(lam_sig / lam_bkg) + np.log(ratio_signal) - np.log(ratio_background)
            q_batch = np.exp(np.clip(log_q, -80.0, 80.0))
            n_to_keep = min(len(q_batch), int(n_selected) - n_kept)
            q_chunks.append(q_batch[:n_to_keep].astype(np.float64, copy=False))
            n_kept += n_to_keep
        del generated, selected, passes
    q_values = np.concatenate(q_chunks)[: int(n_selected)]
    np.savez(
        path,
        q=q_values,
        presel_ratio_cut=np.asarray(presel_ratio_cut),
        ratio_normalization_signal=np.asarray(ratio_normalization["signal"]),
        ratio_normalization_background=np.asarray(ratio_normalization["background"]),
        lam_sig=np.asarray(lam_sig),
        lam_bkg=np.asarray(lam_bkg),
    )
    return q_values


def probability_in_log_q_bins(q_values, log_q_edges):
    """Return the empirical simulator probability in fixed log-q bins."""
    log_q = np.log(np.maximum(q_values, np.finfo(np.float64).tiny))
    counts = np.histogram(log_q, bins=log_q_edges)[0].astype(np.float64)
    return counts / counts.sum()
