"""Tests for weights, portfolio returns, metrics, and risk decomposition.

Synthetic returns keep these fast and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_analysis.portfolio import build_weights, portfolio_returns
from portfolio_analysis import analysis, metrics


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


def test_weights_normalize_to_one():
    w = build_weights({"A": 2.0, "B": 2.0}, ["A", "B"])
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] == pytest.approx(0.5)


def test_missing_weight_splits_leftover():
    # A takes 0.6, B and C share the remaining 0.4 equally.
    w = build_weights({"A": 0.6, "B": None}, ["A", "B", "C"])
    assert w["A"] == pytest.approx(0.6)
    assert w["B"] == pytest.approx(0.2)
    assert w["C"] == pytest.approx(0.2)
    assert w.sum() == pytest.approx(1.0)


def test_portfolio_return_is_weighted_average():
    idx = _days(3)
    rets = pd.DataFrame(
        {"A": [0.10, -0.05, 0.02], "B": [0.00, 0.05, -0.01]}, index=idx
    )
    w = build_weights({"A": 0.5, "B": 0.5}, ["A", "B"])
    port = portfolio_returns(rets, w)
    expected = 0.5 * rets["A"] + 0.5 * rets["B"]
    pd.testing.assert_series_equal(port, expected, check_names=False)


def test_risk_contributions_sum_to_portfolio_vol():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(
        rng.normal(0.0004, 0.012, size=(1000, 4)),
        index=_days(1000), columns=list("ABCD"),
    )
    w = build_weights({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}, list("ABCD"))
    contrib = analysis.risk_contributions(w, rets)
    port_vol = metrics.annualized_volatility(portfolio_returns(rets, w))
    # The contributions are an exact decomposition of portfolio volatility.
    assert contrib["Risk contribution"].sum() == pytest.approx(port_vol, rel=1e-6)
    assert contrib["Risk share"].sum() == pytest.approx(1.0, rel=1e-6)


def test_per_asset_summary_shape():
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(250, 3)),
        index=_days(250), columns=["A", "B", "C"],
    )
    summary = analysis.per_asset_summary(rets)
    assert list(summary.index) == ["A", "B", "C"]
    assert "Sharpe" in summary.columns


def test_max_drawdown_negative_or_zero():
    r = pd.Series([0.02, -0.05, 0.01, -0.03], index=_days(4))
    assert metrics.max_drawdown(r) <= 0.0
