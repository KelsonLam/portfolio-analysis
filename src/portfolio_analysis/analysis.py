"""Looking inside the portfolio: correlations, per-asset stats, and risk.

The headline portfolio numbers hide where the risk actually comes from. A 10%
weight in a volatile, uncorrelated name can carry far more risk than its weight
suggests. The risk contribution view makes that visible, which is the whole
point of thinking about a portfolio as a bundle of risk exposures rather than a
list of positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

TRADING_DAYS = 252


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of daily returns."""
    return asset_returns.corr()


def per_asset_summary(
    asset_returns: pd.DataFrame, risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """Annualized return, volatility, and Sharpe for each holding."""
    rows = {}
    for ticker in asset_returns.columns:
        series = asset_returns[ticker].dropna()
        rows[ticker] = {
            "Annual return": metrics.annualized_return(series),
            "Annual volatility": metrics.annualized_volatility(series),
            "Sharpe": metrics.sharpe_ratio(series, risk_free_rate),
        }
    return pd.DataFrame(rows).T


def risk_contributions(
    weights: pd.Series, asset_returns: pd.DataFrame
) -> pd.DataFrame:
    """Decompose portfolio volatility into each holding's contribution.

    Uses the standard marginal risk decomposition. For weight vector w and
    annualized covariance matrix S, portfolio volatility is sqrt(w' S w). Each
    holding's contribution to risk is w_i times (S w)_i / vol, and those
    contributions sum exactly to the portfolio volatility.
    """
    cols = list(asset_returns.columns)
    w = weights.reindex(cols).fillna(0.0).to_numpy()

    cov = asset_returns.cov().to_numpy() * TRADING_DAYS
    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

    if port_vol == 0:
        contributions = np.zeros_like(w)
    else:
        marginal = cov @ w / port_vol     # marginal contribution to risk
        contributions = w * marginal      # contribution to risk

    pct = contributions / port_vol if port_vol > 0 else np.zeros_like(w)

    return pd.DataFrame(
        {
            "Weight": w,
            "Risk contribution": contributions,
            "Risk share": pct,
        },
        index=cols,
    )
