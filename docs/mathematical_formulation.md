# Mathematical Formulation & Quantitative Framework

## 1. Cross-Sectional Alpha & Neutralization (DataCrunch Engine)

In cross-sectional equity and crypto momentum, the objective is to predict the relative rank of asset $i \in \{1, \dots, N\}$ over a forward holding period $\tau$:
$$y_{i, t+\tau} = \frac{P_{i, t+\tau} - P_{i, t}}{P_{i, t}}$$

### 1.1 Cross-Sectional Gaussian Rank Transformation
Raw feature values $x_{i, t}$ exhibit heavy outliers and regime shifts across time. We map raw cross-sectional features to a standard normal distribution via rank-quantile transformation:
$$u_{i, t} = \frac{\text{Rank}(x_{i, t}) - 0.5}{N}$$
$$\tilde{x}_{i, t} = \Phi^{-1}(u_{i, t})$$
where $\Phi^{-1}$ is the inverse cumulative distribution function of $\mathcal{N}(0, 1)$.

### 1.2 Market Beta Neutralization
To eliminate systemic crypto market beta (e.g., BTC directional swings), the portfolio weights $w_t \in \mathbb{R}^N$ are orthogonalized against the market return vector $\beta \in \mathbb{R}^N$:
$$\tilde{w}_t = w_t - \frac{w_t^T \beta}{\|\beta\|_2^2} \beta$$
$$\sum_{i=1}^N \tilde{w}_{i, t} = 0 \quad (\text{Dollar Neutral}), \quad \tilde{w}_t^T \beta = 0 \quad (\text{Beta Neutral})$$

---

## 2. Intraday Volatility Estimators (Synth Engine)

Standard close-to-close sample volatility discards substantial intraday price path information. We leverage high-frequency OHLC estimators:

### 2.1 Parkinson Volatility (Extreme Value Estimator)
Under geometric Brownian motion with zero drift:
$$\sigma_{\text{Parkinson}}^2 = \frac{1}{4 \ln 2} \frac{1}{N} \sum_{k=1}^N \left( \ln \frac{H_k}{L_k} \right)^2 \approx \frac{0.361}{N} \sum_{k=1}^N \left( \ln \frac{H_k}{L_k} \right)^2$$
Parkinson volatility is approximately $5\times$ more statistically efficient than close-to-close variance.

### 2.2 Garman-Klass Volatility (OHLC Minimum-Variance Estimator)
Accounts for opening jumps and drift between consecutive periods:
$$\sigma_{\text{GK}}^2 = \frac{1}{N} \sum_{k=1}^N \left[ 0.5 \left( \ln \frac{H_k}{L_k} \right)^2 - (2\ln 2 - 1) \left( \ln \frac{C_k}{O_k} \right)^2 \right]$$
Garman-Klass achieves approximately $8\times$ the statistical efficiency of close-to-close variance.

### 2.3 Rogers-Satchell Volatility (Non-Zero Drift Estimator)
Robust to arbitrary trend drift $\mu \neq 0$:
$$\sigma_{\text{RS}}^2 = \frac{1}{N} \sum_{k=1}^N \left[ \ln \frac{H_k}{C_k} \ln \frac{H_k}{O_k} + \ln \frac{L_k}{C_k} \ln \frac{L_k}{O_k} \right]$$

### 2.4 Online GARCH(1,1) Recursive Tracking
$$\sigma_{t+1}^2 = \omega + \alpha r_t^2 + \beta \sigma_t^2, \quad \alpha + \beta < 1$$
Unconditional long-term variance:
$$\sigma_{\infty}^2 = \frac{\omega}{1 - \alpha - \beta}$$

---

## 3. Market Microstructure & Bid-Ask Spread Estimation

### 3.1 Corwin-Schultz Spread Estimator (from High-Low Prices)
Estimates effective bid-ask spread without requiring high-frequency L2 order book quotes:
$$\alpha = \frac{\sqrt{2\beta} - \sqrt{\beta}}{3 - 2\sqrt{2}} - \sqrt{\frac{\gamma}{3 - 2\sqrt{2}}}$$
$$\beta = \sum_{j=0}^1 \left( \ln \frac{H_{t-j}}{L_{t-j}} \right)^2, \quad \gamma = \left( \ln \frac{\max(H_t, H_{t-1})}{\min(L_t, L_{t-1})} \right)^2$$
$$\text{Spread}_{\text{CS}} = \frac{2 (e^\alpha - 1)}{1 + e^\alpha}$$

### 3.2 Amihud Illiquidity Ratio
Measures the absolute price impact per unit of dollar volume:
$$\text{ILLIQ}_t = \frac{1}{K} \sum_{k=1}^K \frac{|r_{t-k}|}{\text{Volume}_{t-k} \cdot P_{t-k}}$$

---

## 4. Fractal Memory & Hurst Exponent ($H$)

Financial return time series exhibit long-range memory and fractal scaling:
$$\mathbb{E}\left[ \frac{R(n)}{S(n)} \right] \propto C \cdot n^H \quad \text{as } n \to \infty$$
where $R(n)$ is the range of cumulative deviations and $S(n)$ is the standard deviation.

* **$H = 0.5$**: Standard Brownian motion (Random Walk, memoryless).
* **$0 < H < 0.5$**: Anti-persistent, mean-reverting process. Price reversals dominate.
* **$0.5 < H < 1.0$**: Persistent, trending process. Positive autocorrelation across time scales.

The diffusion scaling law for forward horizon $\tau$:
$$\sigma(\tau) = \sigma_0 \cdot \tau^H$$

---

## 5. Heavy Tails & Regime-Switching Dynamics

Crypto returns exhibit severe leptokurtosis (excess kurtosis $> 10$) and jump discontinuities.

### 5.1 Student-$t$ Distribution
Probability density with calibrated degrees of freedom $\nu$:
$$f(r \mid \mu, \sigma, \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\pi \nu} \sigma \, \Gamma\left(\frac{\nu}{2}\right)} \left[ 1 + \frac{1}{\nu}\left(\frac{r - \mu}{\sigma}\right)^2 \right]^{-\frac{\nu+1}{2}}$$

### 5.2 Two-State Gaussian Mixture Model (GMM)
$$f(r) = (1 - p_{\text{jump}}) \cdot \mathcal{N}(\mu_{\text{norm}}, \sigma_{\text{norm}}^2) + p_{\text{jump}} \cdot \mathcal{N}(\mu_{\text{jump}}, \sigma_{\text{jump}}^2)$$
where $\sigma_{\text{jump}} \gg \sigma_{\text{norm}}$. When the posterior jump probability $P(\text{State} = \text{Jump} \mid r_t) > \theta_{\text{threshold}}$, the risk engine triggers a volatility circuit breaker.

---

## 6. Portfolio Sizing & Capital Allocation

### 6.1 Volatility-Targeted Weighting
To equalize risk contribution across assets regardless of individual token volatility:
$$w_i = \text{Signal}_i \cdot \frac{\sigma_{\text{target}}}{\sigma_{i, \text{Garman-Klass}}}$$

### 6.2 Fractional Kelly Criterion
For an estimated directional drift $\hat{\mu}_i$ and variance $\hat{\sigma}_i^2$:
$$f_i^* = \kappa \cdot \frac{\hat{\mu}_i - r_f}{\hat{\sigma}_i^2}, \quad \kappa \in [0.1, 0.5] \quad (\text{Fractional Kelly})$$
Fractional Kelly guarantees exponential capital growth while avoiding the ruinous drawdowns of Full Kelly.

---

## 7. Realistic Transaction Costs & Slippage

Net return accounting for exchange fees and market impact:
$$r_{i, t}^{\text{net}} = r_{i, t}^{\text{gross}} - c_{\text{fee}} |\Delta w_{i, t}| - c_{\text{slip}} |\Delta w_{i, t}|^\alpha$$
* **$c_{\text{fee}}$**: Taker fee ($\approx 0.05\%$) or Maker fee ($\approx 0.02\%$).
* **$c_{\text{slip}}$**: Quadratic or sub-linear market impact coefficient proportional to $\frac{\text{OrderSize}}{\text{Liquidity}}$.
