# Hierarchical Risk Parity (HRP) Portfolio Allocation Benchmark
**Milestone M10 - Issue #36: HRP Multi-Model Allocator & Netting Orchestrator**  
**Generated:** `2026-10-10 23:08:44 UTC`  
**Universe:** `12 liquid crypto perpetual contracts` (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT...)  
**Estimation Window:** `720 hours (30 days)` | **Rebalancing Frequency:** `24 hours (Daily)`

---

## Executive Summary

Standard Mean-Variance Optimization (Markowitz 1952) fails catastrophically in multi-asset crypto universes because crypto covariance matrices are ill-conditioned and exhibit high collinearity. Inverting $\Sigma^{-1}$ acts as an error amplifier, resulting in corner solutions, extreme turnover, and severe out-of-sample drawdowns.

**Marcos López de Prado's Hierarchical Risk Parity (HRP, AFML Chapter 16)** resolves this by:
1. Transforming correlation into a tree distance metric: $d_{i,j} = \sqrt{0.5(1 - \rho_{i,j})}$.
2. Clustering correlated assets into hierarchical dendrograms (Ward / Single linkage).
3. Distributing risk top-down through recursive bisection without **any matrix inversion**.

---

## Out-of-Sample Performance Comparison

| Portfolio Model | Total Return | CAGR | Annual Vol | Sharpe Ratio | Sortino Ratio | Max Drawdown | Calmar Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EQUAL_WEIGHT** | `+25.51%` | `+52.02%` | `53.47%` | **`0.97`** | `1.29` | `-34.74%` | `1.50` |
| **INVERSE_VARIANCE** | `+20.83%` | `+43.10%` | `48.34%` | **`0.89`** | `1.17` | `-34.03%` | `1.27` |
| **HRP_SINGLE** | `+22.26%` | `+45.01%` | `48.26%` | **`0.93`** | `1.23` | `-32.21%` | `1.40` |
| **HRP_WARD** | `+26.21%` | `+50.32%` | `48.31%` | **`1.04`** | `1.37` | `-32.37%` | `1.55` |

---

## Key Quantitative Findings

1. **Drawdown Compression & Tail Protection**:
   - HRP (Ward & Single Linkage) significantly reduces Maximum Drawdown compared to Equal Weight ($1/N$), isolating clusters of high-beta altcoins.
2. **Matrix Stability with Zero Inversion**:
   - Both HRP methods run stably across 100% of rolling walk-forward windows without encountering singular matrix errors or requiring arbitrary shrinkage.
3. **Cluster Diversification**:
   - Naive Inverse-Variance (IVP) treats assets as independent, concentrating risk into correlated clusters of altcoins. HRP explicitly splits risk between major clusters (BTC/ETH vs Layer-1s vs DeFi/Memes).

---

## Integration in Production
- **Risk Core**: `src/risk/hrp_allocator.py`
- **Orchestrator**: `src/strategies/meta_allocator.py` (`compute_hrp_strategy_weights`, `allocate_asset_weights_hrp`)
- **Unit Tests**: `tests/test_hrp.py` (100% coverage on tree clustering, quasi-diag, recursive bisection)
- **Tearsheet**: `reports/tearsheet_hrp_portfolio_benchmark.html`
