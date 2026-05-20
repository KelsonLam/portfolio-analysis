"""Weights and the portfolio return series.

The baseline models a constant-weight portfolio: the holding weights are reset
to their targets every day. That is the cleanest definition of "a portfolio
with these weights" and it avoids the bookkeeping of letting weights drift with
prices. A periodic-rebalance version (monthly, quarterly) is a natural next
step, but daily constant weights is the honest, simple starting point.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def build_weights(
    holdings: Mapping[str, float | None], tickers: Sequence[str]
) -> pd.Series:
    """Turn a holdings mapping into normalized weights over ``tickers``.

    Tickers given an explicit weight keep it. Tickers with no weight (missing
    or ``None``) split whatever is left over equally. The result is normalized
    to sum to 1, so the input weights do not have to be exact.
    """
    weights = pd.Series(0.0, index=list(tickers), dtype=float)

    known = {t: float(holdings[t]) for t in tickers
             if t in holdings and holdings[t] is not None}
    unknown = [t for t in tickers if t not in known]

    for ticker, value in known.items():
        weights[ticker] = value

    if unknown:
        leftover = max(0.0, 1.0 - sum(known.values()))
        share = leftover / len(unknown) if leftover > 0 else 0.0
        for ticker in unknown:
            weights[ticker] = share

    total = weights.sum()
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return weights / total


def portfolio_returns(
    asset_returns: pd.DataFrame, weights: pd.Series
) -> pd.Series:
    """Daily return of the constant-weight portfolio.

    Each day's return is the weighted average of the asset returns, which is
    equivalent to rebalancing back to the target weights every day.
    """
    aligned = weights.reindex(asset_returns.columns).fillna(0.0)
    return asset_returns.mul(aligned, axis=1).sum(axis=1)
