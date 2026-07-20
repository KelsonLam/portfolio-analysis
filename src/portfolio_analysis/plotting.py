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

from . import viz_style


def plot_cumulative_returns(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame | None = None,
    title: str = "Cumulative return",
):
    """Portfolio growth, with each holding in its own color behind it."""
    viz_style.apply()
    fig, ax = plt.subplots(figsize=(10, 5))
    if asset_returns is not None:
        for i, col in enumerate(asset_returns.columns):
            equity = (1.0 + asset_returns[col].fillna(0.0)).cumprod()
            color = viz_style.SERIES[i % len(viz_style.SERIES)]
            ax.plot(equity.index, equity.values, color=color, alpha=0.55,
                    linewidth=1.3, label=col)
    port_equity = (1.0 + portfolio_returns).cumprod()
    ax.plot(
        port_equity.index, port_equity.values,
        color=viz_style.INK, linewidth=2.4, label="Portfolio",
    )
    ax.set_title(title)
    ax.set_ylabel("Growth of 1 unit")
    ax.set_xlabel("Date")
    ax.legend(ncols=2)
    fig.tight_layout()
    return fig


def plot_drawdown(returns: pd.Series, title: str = "Portfolio drawdown"):
    viz_style.apply()
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(drawdown.index, drawdown.values, 0.0,
                    color=viz_style.RED, alpha=0.16)
    ax.plot(drawdown.index, drawdown.values, color=viz_style.RED, linewidth=1.4)
    trough = drawdown.idxmin()
    ax.annotate(f"{drawdown.min():.1%}", xy=(trough, drawdown.min()),
                xytext=(0, -12), textcoords="offset points",
                ha="center", fontsize=9, color=viz_style.INK, fontweight="bold")
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_correlation(corr: pd.DataFrame, title: str = "Return correlation"):
    """A simple correlation heatmap with annotated cells."""
    viz_style.apply()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap=viz_style.diverging_cmap())
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.grid(visible=False)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            v = corr.iloc[i, j]
            text_color = "white" if abs(v) > 0.6 else viz_style.INK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                   fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_weight_vs_risk(
    contributions: pd.DataFrame, title: str = "Weight vs risk share"
):
    """Side-by-side bars of capital weight and risk share per holding."""
    viz_style.apply()
    labels = list(contributions.index)
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - width / 2, contributions["Weight"], width,
               label="Weight", color=viz_style.BLUE)
    b2 = ax.bar(x + width / 2, contributions["Risk share"], width,
               label="Risk share", color=viz_style.ORANGE)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8,
                     color=viz_style.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Share of portfolio")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    return path
