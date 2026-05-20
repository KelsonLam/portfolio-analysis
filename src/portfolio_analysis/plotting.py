"""Charts for the portfolio analysis.

Four views: how the portfolio grew, how deep its drawdowns went, how the
holdings move together, and where the weight (and the risk) actually sits.
Matplotlib only. Each function returns the Figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cumulative_returns(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame | None = None,
    title: str = "Cumulative return",
):
    """Portfolio growth, optionally with each holding faded in behind it."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if asset_returns is not None:
        for col in asset_returns.columns:
            equity = (1.0 + asset_returns[col].fillna(0.0)).cumprod()
            ax.plot(equity.index, equity.values, alpha=0.25, linewidth=1)
    port_equity = (1.0 + portfolio_returns).cumprod()
    ax.plot(
        port_equity.index, port_equity.values,
        color="black", linewidth=2, label="Portfolio",
    )
    ax.set_title(title)
    ax.set_ylabel("Growth of 1 unit")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(returns: pd.Series, title: str = "Portfolio drawdown"):
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(drawdown.index, drawdown.values, 0.0, color="tab:red", alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_correlation(corr: pd.DataFrame, title: str = "Return correlation"):
    """A simple correlation heatmap with annotated cells."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_weight_vs_risk(
    contributions: pd.DataFrame, title: str = "Weight vs risk share"
):
    """Side-by-side bars of capital weight and risk share per holding."""
    labels = list(contributions.index)
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, contributions["Weight"], width, label="Weight")
    ax.bar(x + width / 2, contributions["Risk share"], width, label="Risk share")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Share of portfolio")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return path
