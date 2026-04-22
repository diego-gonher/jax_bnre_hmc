"""Matplotlib helpers for training and inference diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import binom

from jax_bnre_hmc.plot_style import apply_plot_style

__all__ = [
    "plot_sbc_rank_histograms",
    "plot_tarp_ecp_curve",
    "save_training_diagnostic_plots",
]


def save_training_diagnostic_plots(
    run_dir: Path | str,
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_bce_losses: Sequence[float],
    val_bce_losses: Sequence[float],
    pj: Any,
    pm: Any,
    *,
    sigmoid_filename: str = "sigmoid.pdf",
    n_plot: int | None = None,
    figsize: tuple[float, float] = (10.0, 5.0),
    dpi: int = 150,
) -> None:
    """Save standard training plots under ``run_dir`` (losses, BCE-style losses, sigmoid curves).

    Args:
        run_dir: Hydra run directory for output plots.
        train_losses, val_losses: Total loss per epoch.
        train_bce_losses, val_bce_losses: BCE-style NRE loss per epoch.
        pj, pm: Sigmoid of joint / marginal logits (1D sequences); converted with ``numpy.asarray``.
        sigmoid_filename: Output name for the sigmoid plot (e.g. ``sigmoid.pdf`` or ``sigmoid_subset.pdf``).
        n_plot: If set, plot only ``pj[:n_plot]`` and ``pm[:n_plot]`` (e.g. long sequences).
        figsize: Figure size for all three figures.
        dpi: Rasterization DPI for ``savefig``.
    """
    apply_plot_style()
    run_dir = Path(run_dir)
    pj_arr = np.asarray(pj)
    pm_arr = np.asarray(pm)
    if n_plot is not None:
        n_plot = int(n_plot)
        pj_arr = pj_arr[:n_plot]
        pm_arr = pm_arr[:n_plot]

    plt.figure(figsize=figsize)
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.legend()
    plt.savefig(run_dir / "losses.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=figsize)
    plt.plot(train_bce_losses, label="train_bce_style_loss")
    plt.plot(val_bce_losses, label="val_bce_style_loss")
    plt.legend()
    plt.savefig(run_dir / "bce_style_losses.pdf", dpi=dpi, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=figsize)
    plt.plot(pj_arr, label="joint")
    plt.plot(pm_arr, label="marginal")
    plt.legend()
    plt.savefig(run_dir / sigmoid_filename, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_tarp_ecp_curve(
    output_dir: Path | str,
    alpha_grid: Any,
    ecp: Any,
    *,
    filename: str = "tarp_ecp_curve.pdf",
    figsize: tuple[float, float] = (4.0, 4.0),
    dpi: int = 256,
    bbox_inches: str = "tight",
) -> None:
    """Save the TARP empirical coverage probability vs credibility level plot.

    Args:
        output_dir: Directory for ``filename`` (typically ``hmc_results/``).
        alpha_grid: Credibility levels (x-axis), shape ``(K,)`` (array-like).
        ecp: Empirical coverage values (y-axis), same length as ``alpha_grid``.
        filename: Output file name.
        figsize: Figure size in inches.
        dpi: Rasterization DPI.
        bbox_inches: Passed to ``savefig`git `.
    """
    apply_plot_style()
    output_dir = Path(output_dir)
    alpha_np = np.asarray(alpha_grid)
    ecp_np = np.asarray(ecp)
    plt.figure(figsize=figsize)
    plt.plot(alpha_np, ecp_np, color='#00B7B5', linewidth=1.5, label='Coverage')
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel(r"Credibility Level ($\alpha$)", fontsize=14)
    plt.ylabel("Empirical Coverage Probability (ECP)", fontsize=14)
    plt.tick_params(axis="both", labelsize=11.5)
    plt.axis("square")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.savefig(output_dir / filename, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()


def plot_sbc_rank_histograms(
    ranks: np.ndarray,
    num_posterior_samples: int,
    labels: Optional[List[str]] = None,
    ks_pvals: Optional[np.ndarray] = None,
    bins: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    *,
    show_uniform_region: bool = True,
    uniform_region_alpha: float = 0.25,
    uniform_region_color: str = "gray",
    uniform_region_ci_level: float = 0.95,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[Figure, np.ndarray]:
    """Histogram of SBC ranks per dimension (marginals), with uniform reference line.

    Args:
        ranks: Shape ``(N, D)`` (one column per marginal / reduction).
        num_posterior_samples: ``S`` (same as in ``run_sbc_from_samples``); x-axis is ``[0, S]``.
        labels: Optional length-``D`` names for subplot titles.
        ks_pvals: Optional length-``D`` p-values shown in titles.
        bins: Number of histogram bins. If ``None``, uses ``min(max(1, N // 10), S + 1)``
            (same spirit as ``run_tarp_jax(..., num_bins=None)``: ~one bin per 10 simulations,
            capped by the rank lattice width ``S + 1``).
        figsize: Figure size; default scales with ``D``.
        show_uniform_region: If True, shade a binomial interval for counts per bin under uniformity.
        uniform_region_alpha: Alpha for the shaded band.
        uniform_region_color: Face color for the shaded band.
        uniform_region_ci_level: Central coverage for ``binom.ppf`` tails (e.g. 0.95).
        output_path: If set, ``savefig`` to this path.

    Returns:
        ``(fig, axes)`` with ``axes`` shape ``(D,)`` (flattened row of subplots).
    """
    apply_plot_style()
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.ndim != 2:
        raise ValueError(f"ranks must have shape (N, D), got {ranks.shape}")
    n, d = ranks.shape
    s = int(num_posterior_samples)
    if s <= 0:
        raise ValueError("num_posterior_samples must be positive.")

    if bins is not None:
        n_bins = int(bins)
    else:
        n_bins = min(max(1, n // 10), s + 1)
    n_bins = max(1, n_bins)
    expected_count = n / n_bins

    p_bin = 1.0 / n_bins
    alpha_tail = (1.0 - uniform_region_ci_level) / 2.0
    low_count = float(binom.ppf(alpha_tail, n, p_bin))
    high_count = float(binom.ppf(1.0 - alpha_tail, n, p_bin))
    bin_edges = np.linspace(0.0, float(s), n_bins + 1)

    if figsize is None:
        figsize = (max(3.0 * d, 3.5), 3.2)

    fig, axes_2d = plt.subplots(1, d, figsize=figsize, squeeze=False, sharey=True)
    axes = axes_2d.ravel()

    for j in range(d):
        ax = axes[j]
        if show_uniform_region:
            for bi in range(n_bins):
                ax.fill_between(
                    [bin_edges[bi], bin_edges[bi + 1]],
                    low_count,
                    high_count,
                    facecolor=uniform_region_color,
                    alpha=uniform_region_alpha,
                    zorder=0,
                    linewidth=0,
                )
        ax.hist(
            ranks[:, j],
            bins=n_bins,
            range=(0.0, float(s)),
            color="#00B7B5",
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        ax.axhline(expected_count, color="k", linestyle="--", linewidth=1.0, alpha=0.7, zorder=3)
        ax.set_xlim(0.0, float(s))
        ax.set_xlabel("rank", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        if j == 0:
            ax.set_ylabel("count", fontsize=16)

        title_parts: List[str] = []
        if labels is not None and j < len(labels):
            title_parts.append(str(labels[j]))
        else:
            title_parts.append(f"dim {j}")
        ax.set_title("\n".join(title_parts), fontsize=16)

        if ks_pvals is not None and j < len(ks_pvals):
            ax.text(
                0.95,
                0.95,
                f"KS p={float(ks_pvals[j]):.3g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
                zorder=4,
            )

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=256, bbox_inches="tight")
    return fig, axes
