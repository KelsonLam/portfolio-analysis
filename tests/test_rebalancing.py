"""Tests for periodic rebalancing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_analysis.rebalancing import rebalanced_returns


def _days(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_single_asset_matches_its_own_returns():
    r = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.0]}, index=_days(4))
    out = rebalanced_returns(r, pd.Series({"A": 1.0}), freq="M")
    pd.testing.assert_series_equal(out, r["A"], check_names=False)


def test_zero_returns_give_zero():
    r = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]}, index=_days(2))
    out = rebalanced_returns(r, pd.Series({"A": 0.5, "B": 0.5}))
    assert (out == 0.0).all()


def test_first_day_is_target_weighted_average():
    r = pd.DataFrame({"A": [0.10, 0.0], "B": [0.00, 0.0]}, index=_days(2))
    out = rebalanced_returns(r, pd.Series({"A": 0.5, "B": 0.5}))
    # Day one uses the target weights: 0.5*0.10 + 0.5*0.0 = 0.05.
    assert out.iloc[0] == pytest.approx(0.05)


def test_runs_on_multi_period_data():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0.0003, 0.01, size=(300, 3)),
                     index=_days(300), columns=list("ABC"))
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    out = rebalanced_returns(r, w, freq="M")
    assert len(out) == 300
    assert out.notna().all()


def test_bad_weights_raise():
    r = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=_days(1))
    with pytest.raises(ValueError):
        rebalanced_returns(r, pd.Series({"A": 0.0, "B": 0.0}))
