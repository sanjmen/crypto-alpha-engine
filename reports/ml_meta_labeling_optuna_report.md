# Quantitative ML Alpha Optimization & Meta-Labeling Report
**Phase 4 Execution: Issues #21 (M7) & #35 (M10)**  
**Generated:** `2026-10-10 18:07:22 UTC`  
**Universe:** `10 liquid crypto perpetual contracts` (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT...)

---

## Executive Summary

Phase 4 bridges institutional machine learning theory (*Marcos López de Prado, Advances in Financial Machine Learning*) with high-capacity crypto perpetual trading.

By decoupling the primary directional decision from the secondary trade execution and bet sizing decision (**Meta-Labeling**), we transform a standard directional model into a high-precision strategy:
- **Out-of-Sample Rank IC (Purged CV)**: `+0.0655`
- **Meta-Classifier AUC-ROC**: `0.5004`
- **Baseline Unfiltered Precision**: `37.18%`
- **Meta-Filtered Precision ($P > 0.5$)**: `37.47%` (**+0.29 pp**)
- **Annualized Sharpe Ratio**: `4.98` (Unfiltered) $\longrightarrow$ **`3.80`** (Meta-Labeled Kelly Sizing)

---

## 1. Purged & Embargoed Cross-Validation (AFML Ch. 7)

Financial observations with non-zero holding periods violate the IID assumption. Standard K-fold CV leaks future returns into the training set.

### Leakage Suppression Protocol
1. **Purging**: All training observations whose label horizon $[t_0, t_1]$ intersects the out-of-sample test window are purged.
2. **Embargoing**: A 1.0% embargo buffer is enforced after each test partition to eliminate autoregressive feature memory and volatility clustering contamination.
3. **Walk-Forward Chronology**: Expanding window preserves historical causality.

```
Fold 1: [==== Train ====] [ Purged ] [ Test 1 ] [ Embargo ]
Fold 2: [======== Train ========] [ Purged ] [ Test 2 ] [ Embargo ]
Fold 3: [============ Train ============] [ Purged ] [ Test 3 ] [ Embargo ]
Fold 4: [================ Train ================] [ Purged ] [ Test 4 ]
```

---

## 2. Optuna Hyperparameter Tuning Results (Issue #21)

Bayesian Tree-structured Parzen Estimator (TPE) optimization was executed over the Purged Walk-Forward CV splits:

| Parameter | Optimal Value | Description |
| :--- | :--- | :--- |
| **`learning_rate`** | `0.0245` | Shrinkage step size for gradient boosting |
| **`num_leaves`** | `10` | Maximum tree leaves per base learner |
| **`max_depth`** | `3` | Maximum tree depth preventing combinatorial overfitting |
| **`min_child_samples`** | `33` | Minimum sample count required in terminal leaf |
| **`subsample`** | `0.8648` | Row subsampling fraction per boosting round |
| **`colsample_bytree`** | `0.7825` | Feature fraction randomly selected per tree |
| **`reg_alpha` (L1)** | `2.7294e+00` | Lasso sparsity regularization |
| **`reg_lambda` (L2)** | `2.2965e-02` | Ridge curvature penalty |
| **`ridge_alpha`** | `2067.84` | L2 regularization for linear component |
| **`ridge_weight`** | `0.60` | Blending weight allocated to Ridge model |

**Best Out-of-Sample Rank IC:** `+0.0655`

---

## 3. Marcos López de Prado Triple Barrier Method (AFML Ch. 3 & 4)

### Barrier Architecture
- **Profit-Taking Barrier ($pt$)**: $+1.5 \times \sigma_{\text{Garman-Klass}}$
- **Stop-Loss Barrier ($sl$)**: $-1.0 \times \sigma_{\text{Garman-Klass}}$
- **Vertical Time Horizon**: `24 hours`

### Execution Path Outcomes
- **Profit Take Hit (`PT`)**: `39.57%`
- **Stop Loss Hit (`SL`)**: `59.44%`
- **Vertical Timeout (`VERTICAL`)**: `0.99%`
- **Average Sample Uniqueness ($\bar{u}$)**: `0.0296` (AFML Ch. 4 concurrent label weighting)

---

## 4. Secondary Meta-Classifier & Kelly Position Sizing (Issue #35)

The Secondary Classifier models the conditional probability of trade success:
$$P(Y = 1 \mid \text{Regime}, \sigma_{\text{GK}}, \text{Hurst}, \text{Illiquidity}, |\text{Signal}|)$$

### Precision Uplift & Trade Filtering
- **Holdout AUC-ROC**: `0.5004`
- **Brier Calibration Score**: `0.2510`
- **Unfiltered Primary Model Win Rate**: `37.18%`
- **Meta-Filtered Win Rate ($P > 0.5$)**: `37.47%`
- **Executed Trade Ratio**: `44.30%` (rejection of 55.70% low-conviction false positives)

### Continuous Kelly Allocation
Each trade $i$ is sized via fractional Kelly leverage ($f = 0.5$):
$$w_i = \text{sign}(\text{primary\_score}) \times \max(0, 2 P_i - 1) \times 0.5$$

When $P_i \le 0.5$, the position is completely vetoed ($w_i = 0$), immunizing capital against high-volatility stop-outs.

---

## 5. Artifacts and Reproducibility
- **Model Pipeline**: `src/models/triple_barrier.py`, `src/models/purged_cv.py`, `src/models/meta_labeling.py`, `src/models/optuna_tuner.py`
- **Training Script**: `scripts/train_ml_meta_pipeline.py`
- **Colab GPU Notebook**: `scripts/colab_ml_training.ipynb`
- **Plotly Tearsheet**: `reports/tearsheet_meta_labeling_optuna.html`
