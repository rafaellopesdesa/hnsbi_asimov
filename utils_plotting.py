"""Figures for the hNDE demonstration."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import ncx2, norm


def _weighted_correlation(values, weights):
    """Return a feature correlation matrix with optional event weights."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weight_sum = float(weights.sum())
    normalized = weights / weight_sum
    mean = np.sum(normalized[:, None] * values, axis=0)
    centered = values - mean
    covariance = (centered * normalized[:, None]).T @ centered
    scale = np.sqrt(np.diag(covariance))
    denominator = np.outer(scale, scale)
    correlation = covariance / denominator
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _highest_density_contour_levels(histogram, probabilities=(0.95, 0.68)):
    """Return density thresholds enclosing the requested probability masses."""
    values = np.asarray(histogram, dtype=float).ravel()
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    levels = []
    for probability in probabilities:
        index = min(np.searchsorted(cumulative, probability), len(ordered) - 1)
        levels.append(ordered[index])
    levels = np.unique(np.sort(levels))
    return levels[(levels > 0.0) & (levels < ordered[0])]


def plot_flow_pair_closure(
    sample_name,
    mc,
    generated,
    feature_names,
    n_bins_1d=60,
    n_bins_2d=35,
    contour_smoothing=1.0,
    quantile_range=(0.005, 0.995),
    mc_weights=None,
    generated_weights=None,
    mc_label="held-out MC",
    generated_label="flow sample",
    generated_color="C1",
    correlation_names=("MC", "flow"),
):
    """Pair plot of two event arrays, optionally with event weights."""
    mc = np.asarray(mc)
    generated = np.asarray(generated)
    feature_names = list(feature_names)
    if mc_weights is not None:
        mc_weights = np.asarray(mc_weights, dtype=np.float64).reshape(-1)
    if generated_weights is not None:
        generated_weights = np.asarray(generated_weights, dtype=np.float64).reshape(-1)
    q_low, q_high = quantile_range
    n_features = len(feature_names)
    limits = []
    for feature_index in range(n_features):
        low = min(
            np.quantile(mc[:, feature_index], q_low),
            np.quantile(generated[:, feature_index], q_low),
        )
        high = max(
            np.quantile(mc[:, feature_index], q_high),
            np.quantile(generated[:, feature_index], q_high),
        )
        limits.append((low, high))
    corr_mc = _weighted_correlation(mc, mc_weights)
    corr_generated = _weighted_correlation(generated, generated_weights)
    fig, axes = plt.subplots(
        n_features, n_features, figsize=(2.55 * n_features, 2.55 * n_features), squeeze=False
    )
    for row in range(n_features):
        for column in range(n_features):
            ax = axes[row, column]
            x_low, x_high = limits[column]
            if row == column:
                bins = np.linspace(x_low, x_high, n_bins_1d + 1)
                ax.hist(
                    mc[:, column],
                    bins=bins,
                    weights=mc_weights,
                    density=True,
                    histtype="step",
                    color="black",
                    lw=1.8,
                    label=mc_label,
                )
                ax.hist(
                    generated[:, column],
                    bins=bins,
                    weights=generated_weights,
                    density=True,
                    histtype="step",
                    color=generated_color,
                    lw=1.8,
                    ls="--",
                    label=generated_label,
                )
                ax.set_xlim(x_low, x_high)
                if row == 0:
                    ax.set_ylabel("density")
            elif row > column:
                y_low, y_high = limits[row]
                x_edges = np.linspace(x_low, x_high, n_bins_2d + 1)
                y_edges = np.linspace(y_low, y_high, n_bins_2d + 1)
                x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
                hist_mc, _, _ = np.histogram2d(
                    mc[:, column], mc[:, row], bins=(x_edges, y_edges), weights=mc_weights
                )
                hist_generated, _, _ = np.histogram2d(
                    generated[:, column],
                    generated[:, row],
                    bins=(x_edges, y_edges),
                    weights=generated_weights,
                )
                if contour_smoothing > 0.0:
                    hist_mc = gaussian_filter(hist_mc, sigma=contour_smoothing)
                    hist_generated = gaussian_filter(hist_generated, sigma=contour_smoothing)
                levels_mc = _highest_density_contour_levels(hist_mc)
                levels_generated = _highest_density_contour_levels(hist_generated)
                if len(levels_mc):
                    ax.contour(
                        x_centers,
                        y_centers,
                        hist_mc.T,
                        levels=levels_mc,
                        colors="black",
                        linewidths=1.5,
                    )
                if len(levels_generated):
                    ax.contour(
                        x_centers,
                        y_centers,
                        hist_generated.T,
                        levels=levels_generated,
                        colors=generated_color,
                        linestyles="--",
                        linewidths=1.5,
                    )
                ax.set_xlim(x_low, x_high)
                ax.set_ylim(y_low, y_high)
            else:
                rho_mc = corr_mc[row, column]
                rho_generated = corr_generated[row, column]
                delta_rho = rho_generated - rho_mc
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.62,
                    f"$\\rho_{{\\rm {correlation_names[0]}}}={rho_mc:+.3f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.text(
                    0.5,
                    0.44,
                    f"$\\rho_{{\\rm {correlation_names[1]}}}={rho_generated:+.3f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color=generated_color,
                )
                ax.text(
                    0.5,
                    0.25,
                    f"$\\Delta\\rho={delta_rho:+.3f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                )
            if row >= column:
                ax.tick_params(axis="x", labelbottom=row == n_features - 1, labelsize=8)
                ax.tick_params(axis="y", labelleft=column == 0, labelsize=8)
                if row == n_features - 1:
                    ax.set_xlabel(feature_names[column])
                if column == 0 and row > 0:
                    ax.set_ylabel(feature_names[row])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False
    )
    fig.suptitle(f"{sample_name}: feature and correlation closure", y=0.999)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return fig


def plot_log_density_truth_binned(truth, model, n_bins=25):
    groups = np.array_split(np.argsort(truth), n_bins)
    x = np.array([truth[g].mean() for g in groups])
    y = np.array([model[g].mean() for g in groups])
    delta = model - truth
    residual = np.array([delta[g].mean() for g in groups])
    sem = np.array([model[g].std(ddof=1) / np.sqrt(len(g)) for g in groups])
    residual_sem = np.array([delta[g].std(ddof=1) / np.sqrt(len(g)) for g in groups])
    fig, (ax, lower) = plt.subplots(
        2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    ax.errorbar(x, y, yerr=sem, fmt="o", ms=3, label="hNDE")
    ax.plot([x[0], x[-1]], [x[0], x[-1]], "k--", label="Perfect calibration")
    ax.set_ylabel(r"$\langle\log\widehat p_S(\mathbf{x})\rangle$")
    ax.legend()
    lower.errorbar(x, residual, yerr=residual_sem, fmt="o", ms=3)
    lower.axhline(0, color="black", lw=1)
    lower.set(
        xlabel=r"$\log p_S(\mathbf{x})$", ylabel=r"$\langle\Delta\log p\rangle$", ylim=(-0.1, 0.1)
    )
    return fig


def plot_toy_comparison(
    hybrid_toys, mu_A, sigma_A, q0_A, simulator_toys=None, asymptotic=True, log_q=False, n_bins=35
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    mu_max = max(hybrid_toys["mu_hat"].max() * 1.001, mu_A + 4 * sigma_A)
    q_max = max(hybrid_toys["q_zero"].max() * 1.001, ncx2.ppf(0.999, df=1, nc=q0_A))
    if simulator_toys is not None:
        mu_max = max(mu_max, simulator_toys["mu_hat"].max() * 1.001)
        q_max = max(q_max, simulator_toys["q_zero"].max() * 1.001)
    for axis, key, upper, xlabel in zip(
        axes, ["mu_hat", "q_zero"], [mu_max, q_max], [r"$\widehat\mu$", r"$q_0$"]
    ):
        edges = np.linspace(0, upper, n_bins + 1)
        counts = np.histogram(hybrid_toys[key], bins=edges)[0]
        axis.stairs(
            counts / len(hybrid_toys), edges, fill=True, alpha=0.3, color="C0", label="hNDE toys"
        )
        if simulator_toys is not None:
            counts = np.histogram(simulator_toys[key], bins=edges)[0]
            axis.errorbar(
                (edges[1:] + edges[:-1]) / 2,
                counts / len(simulator_toys),
                yerr=np.sqrt(counts) / len(simulator_toys),
                fmt="k.",
                label="Simulator toys",
            )
        if asymptotic:
            cdf = (
                norm.cdf((edges - mu_A) / sigma_A)
                if key == "mu_hat"
                else norm.cdf(np.sqrt(edges) - np.sqrt(q0_A))
            )
            probability = np.diff(cdf)
            probability[0] += cdf[0]
            axis.stairs(probability, edges, color="black", label="Asimov prediction")
            if key == "q_zero":
                axis.stairs(
                    np.diff(ncx2.cdf(edges, df=1, nc=q0_A)),
                    edges,
                    color="C3",
                    ls="--",
                    label=r"$\chi^2_1(q_{0,A})$",
                )
        axis.set(xlabel=xlabel, ylabel="Probability per bin")
        axis.legend(loc="upper right")
    if log_q:
        axes[1].set_yscale("log")
    fig.tight_layout()
    return fig
