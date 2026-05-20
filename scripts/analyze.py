"""Command line entry point for the portfolio analysis.

Examples
--------
Analyze the portfolio in config.yaml::

    python scripts/analyze.py

Save the charts as well::

    python scripts/analyze.py --save-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_analysis.data import YFinanceLoader, daily_returns
from portfolio_analysis.portfolio import build_weights, portfolio_returns
from portfolio_analysis.metrics import format_summary, summarize
from portfolio_analysis import analysis, plotting


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a stock portfolio.")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
    )
    p.add_argument("--start", help="Override the start date (YYYY-MM-DD).")
    p.add_argument("--end", help="Override the end date (YYYY-MM-DD).")
    p.add_argument("--no-cache", action="store_true", help="Force a fresh download.")
    p.add_argument("--save-plots", action="store_true", help="Write charts to results/.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    holdings = cfg["holdings"]
    tickers = list(holdings.keys())
    start = args.start or cfg["period"]["start"]
    end = args.end or cfg["period"]["end"]
    rf = cfg.get("risk_free_rate", 0.0)

    print(f"Loading {len(tickers)} holdings from {start} to {end} ...")
    loader = YFinanceLoader(use_cache=not args.no_cache)
    prices = loader.load(tickers, start, end)
    rets = daily_returns(prices)

    # Build weights over the tickers that actually came back with data.
    weights = build_weights(holdings, list(rets.columns))
    port = portfolio_returns(rets, weights)

    print("\nPortfolio performance")
    print("-" * 40)
    print(format_summary(summarize(port, risk_free_rate=rf)))

    print("\nPer-holding summary")
    print("-" * 40)
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(analysis.per_asset_summary(rets, risk_free_rate=rf))

    contributions = analysis.risk_contributions(weights, rets)
    print("\nWeight vs risk contribution")
    print("-" * 40)
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(contributions)

    if args.save_plots:
        f1 = plotting.plot_cumulative_returns(port, rets)
        f2 = plotting.plot_drawdown(port)
        f3 = plotting.plot_correlation(analysis.correlation_matrix(rets))
        f4 = plotting.plot_weight_vs_risk(contributions)
        for fig, name in [
            (f1, "cumulative_returns"), (f2, "drawdown"),
            (f3, "correlation"), (f4, "weight_vs_risk"),
        ]:
            out = plotting.save_figure(fig, f"results/{name}.png")
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
