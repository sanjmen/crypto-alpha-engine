"""
Institutional Vectorized & Event-Driven Crypto Backtesting Engine.
Simulates portfolio equity curve, realistic taker/maker trading fees,
exchange slippage, and 8-hour perpetual funding rate cashflows.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """
    Comprehensive performance and risk metrics report with execution fee decomposition.
    """
    total_return_pct: float
    cagr_pct: float
    annualized_vol_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    gross_return_pct: float
    total_fees_pct: float
    total_funding_pnl_pct: float
    total_turnover: float
    equity_curve: pd.Series
    drawdown_curve: pd.Series


class VectorizedBacktester:
    """
    Vectorized simulation engine for multi-asset crypto portfolios.
    Enforces realistic transaction friction (taker fee + slippage, or passive maker)
    and perpetual funding cashflows.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        taker_fee: float = 0.0004,            # 0.04% taker fee
        maker_fee: float = 0.0002,            # 0.02% maker fee
        slippage_bps: float = 1.0,            # 1 bp slippage for takers
        execution_mode: str = "taker",        # "taker", "maker", or "hybrid"
        turnover_deadband: float = 0.0,       # Deadband buffer (e.g. 0.03 for 3%)
        periods_per_year: int = 8760,         # 8,760 hours in a year (1h bars)
    ):
        self.initial_capital = initial_capital
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage = slippage_bps / 10_000.0
        self.execution_mode = execution_mode.lower()
        self.turnover_deadband = turnover_deadband
        self.periods_per_year = periods_per_year

        # Determine effective cost per turnover unit
        if self.execution_mode == "maker":
            # Passive maker: 0.02% fee, 0 slippage (limit order posted at touch)
            self.total_cost_per_turnover = self.maker_fee
        elif self.execution_mode == "hybrid":
            # 75% passive maker, 25% taker fills
            self.total_cost_per_turnover = 0.75 * self.maker_fee + 0.25 * (self.taker_fee + self.slippage)
        else:
            # Aggressive taker: 0.04% fee + 1 bp slippage
            self.total_cost_per_turnover = self.taker_fee + self.slippage

    def run(
        self,
        returns_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        funding_rates_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Executes portfolio simulation with optional turnover deadband and passive execution.
        """
        # Align index and symbols
        common_idx = returns_df.index.intersection(weights_df.index)
        common_symbols = [col for col in returns_df.columns if col in weights_df.columns]

        rets = returns_df.loc[common_idx, common_symbols].fillna(0.0)
        target_weights = weights_df.loc[common_idx, common_symbols].fillna(0.0)

        # Apply turnover deadband filter (hysteresis buffer)
        if self.turnover_deadband > 0:
            effective_weights = target_weights.copy()
            prev_w = np.zeros(len(common_symbols))
            weights_arr = target_weights.values
            effective_arr = np.zeros_like(weights_arr)

            for t in range(len(weights_arr)):
                desired = weights_arr[t]
                delta = np.abs(desired - prev_w)
                # Keep previous weight if change is smaller than deadband
                rebalance_mask = delta > self.turnover_deadband
                new_w = np.where(rebalance_mask, desired, prev_w)
                effective_arr[t] = new_w
                prev_w = new_w

            weights = pd.DataFrame(effective_arr, index=common_idx, columns=common_symbols)
        else:
            weights = target_weights

        # 1. Gross portfolio return: sum(w_{t-1} * r_t)
        shifted_weights = weights.shift(1).fillna(0.0)
        gross_returns = (shifted_weights * rets).sum(axis=1)

        # 2. Transaction costs from weight turnover: sum(|w_t - w_{t-1}|) * cost
        weight_turnover = (weights - shifted_weights).abs().sum(axis=1)
        transaction_costs = weight_turnover * self.total_cost_per_turnover

        # 3. Funding rate cashflows: sum(w_{t-1} * funding_rate_t)
        funding_pnl = pd.Series(0.0, index=common_idx)
        if funding_rates_df is not None:
            f_common = [col for col in common_symbols if col in funding_rates_df.columns]
            if f_common:
                f_rates = funding_rates_df.reindex(index=common_idx, columns=f_common).fillna(0.0)
                # Longs pay when funding > 0, shorts receive when funding > 0
                funding_pnl = -(shifted_weights[f_common] * f_rates).sum(axis=1)

        # 4. Net portfolio returns
        net_returns = gross_returns - transaction_costs + funding_pnl

        # 5. Equity curve & Drawdowns
        equity = self.initial_capital * (1.0 + net_returns).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        # Metrics computation
        n_periods = len(net_returns)
        if n_periods < 2:
            raise ValueError("Insufficient periods to evaluate backtest performance.")

        total_return_pct = float((equity.iloc[-1] / self.initial_capital - 1.0) * 100.0)
        cagr_pct = float(((equity.iloc[-1] / self.initial_capital) ** (self.periods_per_year / n_periods) - 1.0) * 100.0)

        mean_ret = float(net_returns.mean())
        std_ret = float(net_returns.std()) or 1e-8
        annualized_vol_pct = float(std_ret * np.sqrt(self.periods_per_year) * 100.0)

        sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(self.periods_per_year))

        downside_std = float(net_returns[net_returns < 0].std()) or 1e-8
        sortino_ratio = float((mean_ret / downside_std) * np.sqrt(self.periods_per_year))

        max_dd_pct = float(abs(drawdown.min()) * 100.0)
        calmar_ratio = float(cagr_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        winning_periods = (net_returns > 0).sum()
        total_active_periods = (net_returns != 0).sum()
        win_rate_pct = float((winning_periods / total_active_periods * 100.0) if total_active_periods > 0 else 0.0)

        gross_gains = net_returns[net_returns > 0].sum()
        gross_losses = abs(net_returns[net_returns < 0].sum()) or 1e-8
        profit_factor = float(gross_gains / gross_losses)

        # Fee and component breakdown
        total_gross_return_pct = float(gross_returns.sum() * 100.0)
        total_fees_pct = float(transaction_costs.sum() * 100.0)
        total_funding_pct = float(funding_pnl.sum() * 100.0)
        total_turnover = float(weight_turnover.sum())

        return BacktestResult(
            total_return_pct=total_return_pct,
            cagr_pct=cagr_pct,
            annualized_vol_pct=annualized_vol_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_dd_pct,
            calmar_ratio=calmar_ratio,
            total_trades=int((weight_turnover > 0.01).sum()),
            win_rate_pct=win_rate_pct,
            profit_factor=profit_factor,
            gross_return_pct=total_gross_return_pct,
            total_fees_pct=total_fees_pct,
            total_funding_pnl_pct=total_funding_pct,
            total_turnover=total_turnover,
            equity_curve=equity,
            drawdown_curve=drawdown,
        )
