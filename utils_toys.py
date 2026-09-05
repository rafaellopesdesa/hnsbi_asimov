"""Pseudo-experiment helpers used by the hNDE reproduction notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "expected_compressed_profile",
    "load_or_generate_simulator_q_bank",
    "probability_in_log_q_bins",
    "run_compressed_simulator_toys",
    "run_exact_simulator_toys",
    "summarize_toys",
]


def _simulate_reconstructed_mixture(
    components,
    n_events,
    rng,
    feature_names,
    response_scale,
    response_resolution,
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

    reconstructed = (
        np.asarray(response_scale, dtype=np.float64)[None, :] * latent
        + rng.normal(
            loc=0.0,
            scale=np.asarray(response_resolution, dtype=np.float64)[None, :],
            size=latent.shape,
        )
    )
    return pd.DataFrame(
        reconstructed.astype(np.float32, copy=False),
        columns=feature_names,
    )


def _simulator_bank_path(bank_dir, sample_name, n_selected, seed):
    return Path(bank_dir) / (
        f"{sample_name}_selected_q_{int(n_selected):d}_seed{int(seed):d}.npz"
    )


def _simulator_bank_is_compatible(
    payload,
    n_selected,
    presel_ratio_cut,
    ratio_normalization,
    lam_sig,
    lam_bkg,
):
    required = {
        "q",
        "presel_ratio_cut",
        "ratio_normalization_signal",
        "ratio_normalization_background",
        "lam_sig",
        "lam_bkg",
    }
    if not required.issubset(payload.files):
        return False
    checks = [
        len(payload["q"]) == int(n_selected),
        np.isclose(
            float(payload["presel_ratio_cut"]),
            presel_ratio_cut,
            rtol=1.0e-12,
            atol=1.0e-15,
        ),
        np.isclose(
            float(payload["ratio_normalization_signal"]),
            ratio_normalization["signal"],
            rtol=1.0e-12,
            atol=1.0e-15,
        ),
        np.isclose(
            float(payload["ratio_normalization_background"]),
            ratio_normalization["background"],
            rtol=1.0e-12,
            atol=1.0e-15,
        ),
        np.isclose(float(payload["lam_sig"]), lam_sig),
        np.isclose(float(payload["lam_bkg"]), lam_bkg),
    ]
    return bool(np.all(checks))


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
    ratio_evaluation_batch_size=100_000,
    force_rebuild=False,
    clear_device_cache=None,
):
    """Load or generate an independent selected simulator bank of model q values."""
    bank_dir = Path(bank_dir)
    bank_dir.mkdir(parents=True, exist_ok=True)
    path = _simulator_bank_path(bank_dir, sample_name, n_selected, seed)
    if path.exists() and not force_rebuild:
        with np.load(path, allow_pickle=False) as payload:
            if _simulator_bank_is_compatible(
                payload,
                n_selected,
                presel_ratio_cut,
                ratio_normalization,
                lam_sig,
                lam_bkg,
            ):
                q_values = np.asarray(payload["q"], dtype=np.float64)
                print(f"Loaded {len(q_values):,} {sample_name} q values from {path}")
                return q_values
        print(f"Ignoring incompatible cached bank: {path}")

    rng = np.random.default_rng(int(seed))
    q_chunks = []
    n_kept = 0
    n_generated = 0
    n_passed = 0
    next_report = max(1, int(n_selected) // 10)

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
        n_generated += len(generated)
        n_passed += len(selected)

        if len(selected):
            ratio_signal = (
                evaluate_ratio(
                    "signal",
                    selected,
                    batch_size=ratio_evaluation_batch_size,
                )
                / ratio_normalization["signal"]
            )
            ratio_background = (
                evaluate_ratio(
                    "background",
                    selected,
                    batch_size=ratio_evaluation_batch_size,
                )
                / ratio_normalization["background"]
            )
            log_q = (
                np.log(lam_sig / lam_bkg)
                + np.log(ratio_signal)
                - np.log(ratio_background)
            )
            q_batch = np.exp(np.clip(log_q, -80.0, 80.0))
            n_to_keep = min(len(q_batch), int(n_selected) - n_kept)
            q_chunks.append(q_batch[:n_to_keep].astype(np.float64, copy=False))
            n_kept += n_to_keep

        if n_kept >= next_report or n_kept == int(n_selected):
            print(
                f"{sample_name:10s}: retained {n_kept:,}/{int(n_selected):,} "
                f"selected events after generating {n_generated:,}"
            )
            while next_report <= n_kept:
                next_report += max(1, int(n_selected) // 10)

        del generated, selected, passes
        if clear_device_cache is not None:
            clear_device_cache()

    q_values = np.concatenate(q_chunks)[: int(n_selected)]
    print(
        f"{sample_name:10s}: simulator PRESEL acceptance = "
        f"{n_passed / n_generated:.3%}"
    )
    np.savez(
        path,
        q=q_values,
        presel_ratio_cut=np.asarray(presel_ratio_cut),
        ratio_normalization_signal=np.asarray(ratio_normalization["signal"]),
        ratio_normalization_background=np.asarray(
            ratio_normalization["background"]
        ),
        lam_sig=np.asarray(lam_sig),
        lam_bkg=np.asarray(lam_bkg),
    )
    print(f"Saved simulator bank to {path}")
    return q_values


def probability_in_log_q_bins(q_values, log_q_edges):
    """Return the empirical simulator probability in fixed log-q bins."""
    log_q = np.log(np.maximum(q_values, np.finfo(np.float64).tiny))
    counts = np.histogram(log_q, bins=log_q_edges)[0].astype(np.float64)
    if counts.sum() == 0.0:
        raise ValueError("No simulator q values fall in the requested bins.")
    return counts / counts.sum()


def run_exact_simulator_toys(
    *,
    mu_true,
    n_toys,
    seed,
    signal_q,
    background_q,
    lam_sig,
    lam_bkg,
    fit_toy,
):
    """Generate event-level toys from simulator banks and fit the learned model."""
    rng = np.random.default_rng(seed)
    rows = []
    progress_every = max(1, int(n_toys) // 10)
    for toy_index in range(int(n_toys)):
        n_signal = int(rng.poisson(float(mu_true) * lam_sig))
        n_background = int(rng.poisson(lam_bkg))
        q_values = np.concatenate(
            (
                background_q[
                    rng.integers(0, len(background_q), size=n_background)
                ],
                signal_q[rng.integers(0, len(signal_q), size=n_signal)],
            )
        )
        mu_hat, t_mu, q_zero, information = fit_toy(
            q_values,
            mu_true,
            lam_sig,
            lam_bkg,
        )
        rows.append(
            {
                "mu_true": float(mu_true),
                "toy": toy_index,
                "n_events": n_signal + n_background,
                "n_signal": n_signal,
                "n_background": n_background,
                "mu_hat": mu_hat,
                "t_mu": t_mu,
                "q_zero": q_zero,
                "information": information,
            }
        )
        if (toy_index + 1) % progress_every == 0:
            print(
                f"Exact simulator toys, mu={mu_true:g}: "
                f"completed {toy_index + 1}/{int(n_toys)}"
            )
    return pd.DataFrame(rows)


def run_compressed_simulator_toys(
    *,
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
    """Generate binned simulator toys and fit the frozen learned likelihood."""
    rng = np.random.default_rng(seed)
    chunks = []
    signal_probability = np.asarray(signal_probability, dtype=np.float64)
    background_probability = np.asarray(background_probability, dtype=np.float64)
    if signal_probability.shape != background_probability.shape:
        raise ValueError("Signal and background probabilities must have matching bins.")

    signal_means = float(mu_true) * lam_sig * signal_probability
    background_means = lam_bkg * background_probability
    n_batches = int(np.ceil(int(n_toys) / int(batch_size)))
    for batch_index, start in enumerate(range(0, int(n_toys), int(batch_size))):
        current_batch_size = min(int(batch_size), int(n_toys) - start)
        signal_counts = rng.poisson(
            signal_means,
            size=(current_batch_size, len(signal_probability)),
        )
        background_counts = rng.poisson(
            background_means,
            size=(current_batch_size, len(background_probability)),
        )
        total_counts = signal_counts + background_counts
        mu_hat, t_mu, q_zero, information, fitted_score = fit_batch(
            total_counts,
            float(mu_true),
        )
        chunks.append(
            pd.DataFrame(
                {
                    "mu_true": float(mu_true),
                    "toy": np.arange(start, start + current_batch_size),
                    "n_events": total_counts.sum(axis=1),
                    "n_signal": signal_counts.sum(axis=1),
                    "n_background": background_counts.sum(axis=1),
                    "mu_hat": np.asarray(mu_hat),
                    "t_mu": np.asarray(t_mu),
                    "q_zero": np.asarray(q_zero),
                    "information": np.asarray(information),
                    "fitted_score": np.asarray(fitted_score),
                }
            )
        )
        if (batch_index + 1) % max(1, n_batches // 10) == 0:
            print(
                f"Compressed simulator toys, mu={mu_true:g}: "
                f"completed {start + current_batch_size:,}/{int(n_toys):,}"
            )
    return pd.concat(chunks, ignore_index=True)


def summarize_toys(label, results, mu_true):
    """Summarize fitted estimators and discovery statistics at one hypothesis."""
    subset = results.loc[results["mu_true"] == float(mu_true)]
    mu_hat = subset["mu_hat"].to_numpy(dtype=np.float64)
    q_zero = subset["q_zero"].to_numpy(dtype=np.float64)
    return {
        "sample": label,
        "toys": len(subset),
        "mean(mu_hat)": np.mean(mu_hat),
        "std(mu_hat)": np.std(mu_hat, ddof=1),
        "RMS(mu_hat-mu_A)": np.sqrt(np.mean((mu_hat - float(mu_true)) ** 2)),
        "P(mu_hat=0)": np.mean(mu_hat <= 1.0e-10),
        "mean(q_0)": np.mean(q_zero),
        "median(q_0)": np.median(q_zero),
    }


def expected_compressed_profile(
    mu_values,
    expected_counts,
    mu_hat,
    q_values,
    lam_sig,
):
    """Evaluate an expected-data profile with the learned compressed likelihood."""
    mu_values = np.asarray(mu_values, dtype=np.float64)
    expected_counts = np.asarray(expected_counts, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)

    def relative_nll(mu):
        return 2.0 * (
            float(mu) * lam_sig
            - np.sum(expected_counts * np.log1p(float(mu) * q_values))
        )

    minimum = relative_nll(mu_hat)
    return np.asarray(
        [max(0.0, relative_nll(mu) - minimum) for mu in mu_values]
    )
