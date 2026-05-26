"""Edge-case and validation tests for the portfolio builder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_analysis.portfolio import build_weights


def test_empty_tickers_raises():
    with pytest.raises(ValueError):
        build_weights({}, [])


def test_unknown_holding_raises():
    # A holding asked for but not in the available tickers should fail loudly.
    with pytest.raises(ValueError):
        build_weights({"AAPL": 0.5, "ZZZZ": 0.5}, ["AAPL"])


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        build_weights({"A": -0.2, "B": 0.5}, ["A", "B"])


def test_all_missing_weights_split_equally():
    w = build_weights({}, ["A", "B", "C", "D"])
    assert all(abs(v - 0.25) < 1e-12 for v in w)


def test_partial_weights_distribute_remainder():
    w = build_weights({"A": 0.4}, ["A", "B", "C"])
    assert w["A"] == pytest.approx(0.4)
    assert w["B"] == pytest.approx(0.3)
    assert w["C"] == pytest.approx(0.3)
