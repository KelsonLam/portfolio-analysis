"""Periodic rebalancing, the realistic cousin of constant weights.

The baseline portfolio in portfolio.py rebalances to target weights every single
day, which is clean but unrealistic. In practice you set weights, let them drift
with the market, and reset them back on a schedule (monthly, quarterly). This
module models that: weights drift between rebalance dates and snap back to target
at the start of each new period.

The gap between this and the constant-weight version is the cost of letting
winners run versus the discipline of trimming them, which is a real portfolio
decision rather than a modelling detail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rebalanced_returns(
    asset_returns: pd.DataFrame, weights: pd.Series, freq: str = "M"
) -> pd.Series:
    """Daily portfolio returns with weights reset to target each ``freq`` period.

    ``freq`` is a pandas period alias: "M" monthly, "Q" quarterly, "Y" yearly,
    "W" weekly. Weights drift with returns inside each period.
    """
    cols = list(asset_returns.columns)
    target = weights.reindex(cols).fillna(0.0).to_numpy(dtype=float)
    if target.sum() <= 0:
        raise ValueError("weights must sum to a positive number.")
    target = target / target.sum()

    periods = asset_returns.index.to_period(freq)
    current = target.copy()
    out = np.empty(len(asset_returns))
    prev_period = None

    for t, (_, row) in enumerate(asset_returns.iterrows()):
        if prev_period is not None and periods[t] != prev_period:
            current = target.copy()              # rebalance at the new period
        r = np.nan_to_num(row.to_numpy(dtype=float))
        out[t] = float(np.dot(current, r))
        grown = current * (1.0 + r)              # let the weights drift
        total = grown.sum()
        current = grown / total if total > 0 else target.copy()
        prev_period = periods[t]

    return pd.Series(out, index=asset_returns.index)
