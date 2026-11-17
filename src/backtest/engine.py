"""
Institutional Vectorized & Event-Driven Crypto Backtesting Engine.
Simulates portfolio equity curve, realistic taker/maker trading fees,
exchange slippage, and 8-hour perpetual funding rate cashflows.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.market_impact import NonLinearMarketImpactModel


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

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) > 0 else 0.0


class VectorizedBacktester:
    """
    Vectorized simulation engine for multi-asset crypto portfolios.
    Enforces realistic transaction friction (taker fee + slippage, or passive maker)
    and perpetual funding cashflows.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        taker_fee: float = 0.0005,  # 0.05% Binance USDS-M VIP0 taker fee
        maker_fee: float = 0.0002,  # 0.02% maker fee
        slippage_bps: float = 1.0,  # 1 bp slippage for takers
        execution_mode: str = "taker",  # "taker", "maker", or "hybrid"
        turnover_deadband: float = 0.0,  # Deadband buffer (e.g. 0.03 for 3%)
        periods_per_year: int = 8760,  # 8,760 hours in a year (1h bars)
        impact_model: NonLinearMarketImpactModel | None = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage = slippage_bps / 10_000.0
        self.execution_mode = execution_mode.lower()
        if self.execution_mode not in {"taker", "maker", "hybrid"}:
            raise ValueError(f"Unsupported execution_mode: '{execution_mode}'. Expected 'taker', 'maker', or 'hybrid'.")
        self.turnover_deadband = turnover_deadband
        self.periods_per_year = periods_per_year
        self.impact_model = impact_model

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
        funding_rates_df: pd.DataFrame | None = None,
        volumes_df: pd.DataFrame | None = None,
        volatilities_df: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """
        Executes portfolio simulation with optional turnover deadband, passive execution,
        and dynamic market impact slippage modeling.
        """
        common_idx = returns_df.index.intersection(weights_df.index)
        common_symbols = [col for col in returns_df.columns if col in weights_df.columns]

        rets = returns_df.loc[common_idx, common_symbols].fillna(0.0)
        target_weights = weights_df.loc[common_idx, common_symbols].fillna(0.0)

        # Apply turnover deadband filter (hysteresis buffer)
        if self.turnover_deadband > 0:
            prev_weights = np.zeros(len(common_symbols))
            weights_arr = target_weights.values
            effective_arr = np.zeros_like(weights_arr)

            for bar_idx in range(len(weights_arr)):
                desired_weights = weights_arr[bar_idx]
                delta_weights = np.abs(desired_weights - prev_weights)
                rebalance_mask = delta_weights > self.turnover_deadband
                new_weights = np.where(rebalance_mask, desired_weights, prev_weights)
                effective_arr[bar_idx] = new_weights
                prev_weights = new_weights

            weights = pd.DataFrame(effective_arr, index=common_idx, columns=common_symbols)
        else:
            weights = target_weights

        # 1. Gross portfolio return: sum(w_{t-1} * r_t)
        shifted_weights = weights.shift(1).fillna(0.0)
        gross_returns = (shifted_weights * rets).sum(axis=1)

        # 2. Transaction costs from weight turnover: sum(|w_t - w_{t-1}|) * cost
        weight_turnover = (weights - shifted_weights).abs().sum(axis=1)
        if self.impact_model is not None and volumes_df is not None and volatilities_df is not None:
            impact_symbols = [c for c in common_symbols if c in volumes_df.columns and c in volatilities_df.columns]
            if impact_symbols:
                vols = volumes_df.reindex(index=common_idx, columns=impact_symbols).fillna(50_000_000.0)
                sigmas = volatilities_df.reindex(index=common_idx, columns=impact_symbols).fillna(0.005)
                turnover_mat = (weights[impact_symbols] - shifted_weights[impact_symbols]).abs()
                order_notional_mat = turnover_mat * self.initial_capital
                participation_rate = (order_notional_mat / np.maximum(vols, 1000.0)).clip(lower=1e-7, upper=1.0)
                y_coef = self.impact_model.params.Y_coefficient
                alpha_exp = self.impact_model.params.alpha_exponent
                impact_mat = y_coef * sigmas * (participation_rate**alpha_exp)
                half_spread = self.impact_model.params.min_slippage_bps / 10_000.0
                dynamic_slippage = np.maximum(half_spread, half_spread + impact_mat)
                effective_cost_mat = self.taker_fee + dynamic_slippage
                transaction_costs = (turnover_mat * effective_cost_mat).sum(axis=1)
                remaining_symbols = [c for c in common_symbols if c not in impact_symbols]
                if remaining_symbols:
                    remaining_turnover = (
                        (weights[remaining_symbols] - shifted_weights[remaining_symbols]).abs().sum(axis=1)
                    )
                    transaction_costs = transaction_costs + remaining_turnover * self.total_cost_per_turnover
            else:
                transaction_costs = weight_turnover * self.total_cost_per_turnover
        else:
            transaction_costs = weight_turnover * self.total_cost_per_turnover

        # 3. Funding rate cashflows: sum(w_{t-1} * funding_rate_t)
        funding_pnl = pd.Series(0.0, index=common_idx)
        if funding_rates_df is not None:
            funding_symbols = [col for col in common_symbols if col in funding_rates_df.columns]
            if funding_symbols:
                f_rates = funding_rates_df.reindex(index=common_idx, columns=funding_symbols).fillna(0.0)
                # Longs pay when funding > 0, shorts receive when funding > 0
                funding_pnl = -(shifted_weights[funding_symbols] * f_rates).sum(axis=1)

        # 4. Net portfolio returns
        net_returns = gross_returns - transaction_costs + funding_pnl

        # 5. Equity curve & Drawdowns (guard against negative equity bankruptcy)
        growth_factors = np.maximum(0.0, 1.0 + net_returns)
        equity = self.initial_capital * growth_factors.cumprod()
        running_max = equity.cummax().replace(0.0, np.nan)
        drawdown = ((equity - running_max) / running_max).fillna(0.0)

        # Metrics computation
        n_periods = len(net_returns)
        if n_periods < 2:
            raise ValueError("Insufficient periods to evaluate backtest performance.")

        final_equity = float(equity.iloc[-1])
        total_return_pct = float((final_equity / self.initial_capital - 1.0) * 100.0)
        if final_equity <= 0.0:
            cagr_pct = -100.0
        else:
            cagr_pct = float(
                ((final_equity / self.initial_capital) ** (self.periods_per_year / n_periods) - 1.0) * 100.0
            )

        mean_ret = float(net_returns.mean())
        raw_std = float(net_returns.std())
        std_ret = raw_std if (np.isfinite(raw_std) and raw_std >= 1e-8) else 1e-8
        annualized_vol_pct = float(std_ret * np.sqrt(self.periods_per_year) * 100.0)

        sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(self.periods_per_year))

        negative_rets = net_returns[net_returns < 0]
        raw_downside_std = float(negative_rets.std()) if len(negative_rets) > 1 else 0.0
        downside_std = raw_downside_std if (np.isfinite(raw_downside_std) and raw_downside_std >= 1e-8) else 1e-8
        sortino_ratio = float((mean_ret / downside_std) * np.sqrt(self.periods_per_year))

        max_dd_pct = float(abs(drawdown.min()) * 100.0)
        calmar_ratio = float(cagr_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        winning_periods = int((net_returns > 0).sum())
        total_active_periods = int((net_returns != 0).sum())
        win_rate_pct = float((winning_periods / total_active_periods * 100.0) if total_active_periods > 0 else 0.0)

        gross_gains = float(net_returns[net_returns > 0].sum())
        raw_losses = float(abs(net_returns[net_returns < 0].sum()))
        gross_losses = raw_losses if (np.isfinite(raw_losses) and raw_losses >= 1e-8) else 1e-8
        profit_factor = float(gross_gains / gross_losses)

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
