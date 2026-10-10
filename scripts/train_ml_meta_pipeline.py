"""
Production Machine Learning Training & Meta-Labeling Optimization Pipeline.
Issues #21 (M7) & #35 (M10).

Executes end-to-end quantitative ML workflow across liquid crypto perpetuals:
1. Multi-factor Feature Extraction (Momentum, Volatility, Microstructure, Fractal Hurst).
2. Purged & Embargoed Walk-Forward Cross-Validation (AFML Ch. 7).
3. Optuna Bayesian Hyperparameter Optimization for LightGBM & Ridge.
4. Marcos López de Prado's Triple Barrier Method (AFML Ch. 3 & 4) with dynamic Garman-Klass volatility.
5. Sample Uniqueness and Return-Attributed Sample Weighting.
6. Secondary GBDT Precision Meta-Classifier and Kelly Position Sizing Engine.
7. Publication-grade Markdown & Plotly HTML Tearsheet generation.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any

from src.features.hurst import estimate_hurst_rs
from src.features.microstructure import amihud_illiquidity
from src.features.volatility import garman_klass_volatility, parkinson_volatility
from src.models.meta_labeling import KellyPositionSizer, SecondaryMetaClassifier
from src.models.optuna_tuner import OptunaHyperparameterTuner, TuningResult
from src.models.triple_barrier import TripleBarrierConfig, TripleBarrierLabeler


def load_universe_bars(
    symbols: list[str],
    cache_dir: Path,
    max_bars: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Loads 1h parquet bar files from local/tmp cache."""
    data = {}
    for sym in symbols:
        file_path = cache_dir / "bars" / "1h" / f"{sym}.parquet"
        if not file_path.exists():
            continue
        df = pd.read_parquet(file_path)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
        if max_bars is not None and len(df) > max_bars:
            df = df.iloc[-max_bars:].reset_index(drop=True)
        df = df.set_index("datetime")
        data[sym] = df
    return data


def extract_features_and_targets(
    df: pd.DataFrame,
    horizon_bars: int = 24,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Computes multi-factor alpha features and forward returns for a single asset.
    """
    c = df["close"].values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values
    tb = df["taker_buy_vol"].values if "taker_buy_vol" in df.columns else v * 0.5

    n = len(c)
    idx = df.index

    # 1. Momentum factors
    log_ret = np.log(c / np.roll(c, 1))
    log_ret[0] = 0.0

    mom_6h = pd.Series(log_ret, index=idx).rolling(6).sum().values
    mom_12h = pd.Series(log_ret, index=idx).rolling(12).sum().values
    mom_24h = pd.Series(log_ret, index=idx).rolling(24).sum().values
    mom_72h = pd.Series(log_ret, index=idx).rolling(72).sum().values

    # 2. Volatility factors (Garman-Klass & Parkinson rolling 24h)
    gk_vol = np.zeros(n)
    park_vol = np.zeros(n)
    for i in range(24, n):
        gk_vol[i] = garman_klass_volatility(o[i-24:i], h[i-24:i], l[i-24:i], c[i-24:i])
        park_vol[i] = parkinson_volatility(h[i-24:i], l[i-24:i])

    safe_park = np.where(park_vol > 1e-6, park_vol, 1e-6)
    vol_ratio = np.where(park_vol > 1e-6, gk_vol / safe_park, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0, posinf=1.0, neginf=1.0)

    # 3. Microstructure & Order Flow Imbalance
    amihud = np.zeros(n)
    for i in range(24, n):
        amihud[i] = amihud_illiquidity(c[i-24:i], v[i-24:i])
    # Log transform illiquidity to stabilize distribution
    log_amihud = np.log1p(amihud * 1e6)

    taker_ratio = np.where(v > 1e-4, tb / v, 0.5)
    order_flow_imb = pd.Series(taker_ratio - 0.5, index=idx).rolling(12).mean().values

    # 4. Fractal persistence: Hurst exponent
    hurst_vals = np.full(n, 0.5)
    for i in range(72, n, 6):  # Compute every 6 bars for efficiency
        h_est = estimate_hurst_rs(c[i-72:i])
        hurst_vals[i:min(i+6, n)] = h_est

    # 5. Forward return target (24h forward return)
    fwd_ret = (np.roll(c, -horizon_bars) - c) / c
    fwd_ret[-horizon_bars:] = np.nan

    features = pd.DataFrame(
        {
            "mom_6h": mom_6h,
            "mom_12h": mom_12h,
            "mom_24h": mom_24h,
            "mom_72h": mom_72h,
            "vol_gk_24h": gk_vol,
            "vol_parkinson_24h": park_vol,
            "vol_ratio": vol_ratio,
            "log_amihud": log_amihud,
            "order_flow_imb": order_flow_imb,
            "hurst_h": hurst_vals,
        },
        index=idx,
    )

    targets = pd.Series(fwd_ret, index=idx, name="fwd_ret_24h")
    vol_series = pd.Series(gk_vol, index=idx, name="vol_gk")

    # Drop warm-up window
    warmup = 72
    return features.iloc[warmup:-horizon_bars], targets.iloc[warmup:-horizon_bars], vol_series.iloc[warmup:-horizon_bars]


def run_pipeline(
    cache_dir: Path,
    symbols: list[str],
    n_trials: int = 25,
    max_bars: int = 5000,
    output_dir: Path = PROJECT_ROOT / "reports",
):
    print("=" * 70)
    print("  CRYPTO QUANTITATIVE ML & META-LABELING OPTIMIZATION PIPELINE  ")
    print("=" * 70)
    print(f"[*] Universe: {len(symbols)} liquid perpetual symbols: {', '.join(symbols[:5])}...")
    print(f"[*] Lookback: {max_bars} 1h bars per symbol")
    print(f"[*] Optuna Bayesian Trials: {n_trials}")
    print(f"[*] Output Directory: {output_dir}")
    print("-" * 70)

    # 1. Load data
    print("[1/6] Loading cached market data...")
    raw_data = load_universe_bars(symbols, cache_dir, max_bars=max_bars)
    if not raw_data:
        raise RuntimeError(f"No market data found in {cache_dir}. Run sync first.")
    print(f"[+] Loaded {len(raw_data)} assets.")

    # 2. Extract multi-factor features across universe
    print("[2/6] Extracting quantitative multi-factor features...")
    all_features = []
    all_targets = []
    all_vols = []
    symbol_frames = {}

    for sym, df in raw_data.items():
        feat_df, targ_s, vol_s = extract_features_and_targets(df, horizon_bars=24)
        feat_df["symbol"] = sym
        symbol_frames[sym] = (df.loc[feat_df.index], feat_df, targ_s, vol_s)
        all_features.append(feat_df)
        all_targets.append(targ_s)
        all_vols.append(vol_s)

    combined_features = pd.concat(all_features, axis=0)
    combined_targets = pd.concat(all_targets, axis=0)
    print(f"[+] Feature matrix shape: {combined_features.shape}")

    # Cross-sectional target normalization (rank transform per timestamp)
    target_ranks = combined_targets.groupby(level=0).rank(pct=True) - 0.5

    feature_cols = [c for c in combined_features.columns if c != "symbol"]
    combined_features["target"] = target_ranks.values
    # Sort chronologically by datetime index so walk-forward CV steps forward in time
    combined_features = combined_features.sort_index()

    X_train_full = combined_features[feature_cols].copy()
    y_train_full = combined_features["target"].copy()

    # 3. Purged Walk-Forward Cross-Validation & Optuna Tuning for Alpha Predictor
    print("\n[3/6] Running Purged Walk-Forward CV & Optuna Hyperparameter Tuning (Issue #21)...")
    tuner = OptunaHyperparameterTuner(
        n_splits=4,
        min_train_pct=0.40,
        embargo_pct=0.01,
        random_state=42,
    )

    t0_tune = time.time()
    tuning_result = tuner.optimize_alpha_predictor(
        features=X_train_full,
        targets=y_train_full,
        metric="rank_ic",
        n_trials=n_trials,
    )
    t_tune_dur = time.time() - t0_tune

    print(f"[+] Optuna optimization completed in {t_tune_dur:.1f}s ({tuning_result.n_trials} trials)")
    print(f"[+] Best Out-of-Sample Rank IC: {tuning_result.best_score:.4f}")
    print("    Optimal Parameters:")
    for k, v in tuning_result.best_params.items():
        print(f"      • {k}: {v if isinstance(v, int) else f'{v:.4f}'}")

    # Build and fit tuned Primary Alpha Model
    primary_model = tuner.build_tuned_alpha_predictor(tuning_result)
    primary_model.fit(X_train_full, y_train_full)

    # 4. Marcos López de Prado Triple Barrier Labeling & Sample Uniqueness (AFML Ch. 3 & 4)
    print("\n[4/6] Executing Triple Barrier Method with Garman-Klass Volatility (AFML Ch. 3 & 4)...")
    tb_config = TripleBarrierConfig(
        pt=1.5,
        sl=1.0,
        holding_period_bars=24,
        min_ret=0.002,
        strict_pt=False,
        conservative_touch=True,
    )
    labeler = TripleBarrierLabeler(tb_config)

    meta_events_list = []
    meta_features_list = []

    for sym, (raw_df, feat_df, _, vol_s) in symbol_frames.items():
        sym_feat = feat_df[feature_cols]
        sym_preds = primary_model.predict_raw(sym_feat)

        # Primary model signals: enter long if pred > 0, short if pred < 0
        events_df = pd.DataFrame(
            {
                "side": np.sign(sym_preds),
                "primary_score": sym_preds,
            },
            index=sym_feat.index,
        )
        # Filter weak conviction (|score| < 0.1)
        events_df = events_df[events_df["primary_score"].abs() >= 0.1]

        if len(events_df) == 0:
            continue

        barriers = labeler.apply_barriers(
            close=raw_df["close"],
            events=events_df,
            high=raw_df["high"],
            low=raw_df["low"],
            volatility=vol_s,
        )

        if len(barriers) == 0:
            continue

        barriers["symbol"] = sym
        # Attach meta-features at event start
        meta_feat = sym_feat.loc[barriers.index].copy()
        meta_feat["primary_abs_score"] = events_df.loc[barriers.index, "primary_score"].abs()

        meta_events_list.append(barriers)
        meta_features_list.append(meta_feat)

    all_meta_events = pd.concat(meta_events_list, axis=0)
    all_meta_features = pd.concat(meta_features_list, axis=0)
    print(f"[+] Total Triple Barrier Trade Events: {len(all_meta_events)}")

    # Touch distribution
    touch_dist = all_meta_events["touch_type"].value_counts(normalize=True).to_dict()
    win_rate_unfiltered = float(all_meta_events["label"].mean())
    print(f"[+] Unfiltered Primary Model Win Rate: {win_rate_unfiltered:.2%}")
    print("    Barrier Touch Distribution:")
    for touch, pct in touch_dist.items():
        print(f"      • {touch.upper()}: {pct:.2%}")

    # Compute Sample Uniqueness and Return-Attributed Weights (AFML Ch. 4)
    price_master_idx = raw_data[symbols[0]].index
    sample_weights = TripleBarrierLabeler.compute_sample_weights(
        all_meta_events, price_master_idx, attribute_returns=True
    )
    uniqueness = TripleBarrierLabeler.compute_sample_uniqueness(all_meta_events, price_master_idx)
    avg_uniqueness = float(uniqueness.mean()) if len(uniqueness) > 0 else 1.0
    print(f"[+] Average Sample Uniqueness (AFML Ch. 4): {avg_uniqueness:.4f}")

    # 5. Train Secondary Meta-Classifier & Kelly Position Sizing (Issue #35)
    print("\n[5/6] Training Secondary GBDT Meta-Classifier & Kelly Sizing Engine (Issue #35)...")
    # Temporal train/test split: 75% train, 25% holdout
    meta_features_cols = list(all_meta_features.columns)
    combined_meta = pd.concat([all_meta_events.reset_index(), all_meta_features.reset_index(drop=True)], axis=1)
    combined_meta["sample_weight"] = sample_weights.values
    # Sort chronologically by t0
    combined_meta = combined_meta.sort_values("t0").reset_index(drop=True)

    split_idx = int(len(combined_meta) * 0.75)
    train_meta = combined_meta.iloc[:split_idx]
    test_meta = combined_meta.iloc[split_idx:]

    X_meta_train = train_meta[meta_features_cols]
    y_meta_train = train_meta["label"]
    w_meta_train = train_meta["sample_weight"]

    X_meta_test = test_meta[meta_features_cols]
    y_meta_test = test_meta["label"]

    meta_classifier = SecondaryMetaClassifier(kelly_fraction=0.5, max_leverage=1.0)
    meta_classifier.fit(X_meta_train, y_meta_train, sample_weights=w_meta_train)

    report = meta_classifier.evaluate_filter(X_meta_test, y_meta_test)
    print(f"[+] Meta-Classifier Out-of-Sample AUC-ROC: {report.auc_roc:.4f}")
    print(f"[+] Out-of-Sample Brier Score: {report.brier_score:.4f}")
    print(f"[+] Unfiltered Win Rate: {report.precision_unfiltered:.2%}")
    print(f"[+] Meta-Filtered Win Rate (P > 0.5): {report.precision_filtered:.2%}")
    print(f"[+] Precision Uplift: +{(report.precision_filtered - report.precision_unfiltered)*100:.2f} percentage points")
    print(f"[+] Trades Executed Ratio: {report.trades_executed_ratio:.2%}")
    print(f"[+] Theoretical Sharpe Uplift: {report.expected_sharpe_uplift:.2f}x")

    # Kelly Sizing Simulation on Holdout
    test_probs = meta_classifier.predict_proba(X_meta_test)
    test_sides = test_meta["side"].values
    test_rets = test_meta["ret"].values

    sizer = KellyPositionSizer(kelly_fraction=0.5, max_leverage=1.0)
    kelly_sizes = sizer.size_batch(test_probs, test_sides)

    # Strategy returns
    unfiltered_returns = test_sides * test_rets
    meta_kelly_returns = kelly_sizes * test_rets

    sharpe_unfiltered = (np.mean(unfiltered_returns) / (np.std(unfiltered_returns) + 1e-6)) * np.sqrt(24 * 365)
    sharpe_meta = (np.mean(meta_kelly_returns) / (np.std(meta_kelly_returns) + 1e-6)) * np.sqrt(24 * 365)
    print(f"[+] Realized Holdout Sharpe (Unfiltered): {sharpe_unfiltered:.2f}")
    print(f"[+] Realized Holdout Sharpe (Meta-Labeled Kelly): {sharpe_meta:.2f}")

    # 6. Generate Reports & Plotly Tearsheet
    print("\n[6/6] Exporting Markdown Report and Interactive Plotly Tearsheet...")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_md_path = output_dir / "ml_meta_labeling_optuna_report.md"
    tearsheet_html_path = output_dir / "tearsheet_meta_labeling_optuna.html"

    # Markdown Report
    generate_markdown_report(
        report_md_path,
        symbols=symbols,
        tuning_result=tuning_result,
        tb_config=tb_config,
        avg_uniqueness=avg_uniqueness,
        report=report,
        sharpe_unfiltered=sharpe_unfiltered,
        sharpe_meta=sharpe_meta,
        touch_dist=touch_dist,
    )
    print(f"[+] Saved report: {report_md_path}")

    # Interactive Plotly Tearsheet
    generate_plotly_tearsheet(
        tearsheet_html_path,
        tuning_result=tuning_result,
        test_events=test_meta,
        test_probs=test_probs,
        unfiltered_returns=unfiltered_returns,
        meta_kelly_returns=meta_kelly_returns,
    )
    print(f"[+] Saved tearsheet: {tearsheet_html_path}")

    print("\n" + "=" * 70)
    print("  PHASE 4 QUANTITATIVE ML PIPELINE SUCCESSFULLY EXECUTED  ")
    print("=" * 70)


def generate_markdown_report(
    path: Path,
    symbols: list[str],
    tuning_result: TuningResult,
    tb_config: TripleBarrierConfig,
    avg_uniqueness: float,
    report: Any,
    sharpe_unfiltered: float,
    sharpe_meta: float,
    touch_dist: dict[str, float],
):
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    p = tuning_result.best_params

    md_content = f"""# Quantitative ML Alpha Optimization & Meta-Labeling Report
**Phase 4 Execution: Issues #21 (M7) & #35 (M10)**  
**Generated:** `{timestamp_str}`  
**Universe:** `{len(symbols)} liquid crypto perpetual contracts` ({', '.join(symbols[:5])}...)

---

## Executive Summary

Phase 4 bridges institutional machine learning theory (*Marcos López de Prado, Advances in Financial Machine Learning*) with high-capacity crypto perpetual trading.

By decoupling the primary directional decision from the secondary trade execution and bet sizing decision (**Meta-Labeling**), we transform a standard directional model into a high-precision strategy:
- **Out-of-Sample Rank IC (Purged CV)**: `+{tuning_result.best_score:.4f}`
- **Meta-Classifier AUC-ROC**: `{report.auc_roc:.4f}`
- **Baseline Unfiltered Precision**: `{report.precision_unfiltered:.2%}`
- **Meta-Filtered Precision ($P > 0.5$)**: `{report.precision_filtered:.2%}` (**+{(report.precision_filtered - report.precision_unfiltered)*100:.2f} pp**)
- **Annualized Sharpe Ratio**: `{sharpe_unfiltered:.2f}` (Unfiltered) $\\longrightarrow$ **`{sharpe_meta:.2f}`** (Meta-Labeled Kelly Sizing)

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
| **`learning_rate`** | `{p.get('learning_rate', 0.03):.4f}` | Shrinkage step size for gradient boosting |
| **`num_leaves`** | `{p.get('num_leaves', 15)}` | Maximum tree leaves per base learner |
| **`max_depth`** | `{p.get('max_depth', 4)}` | Maximum tree depth preventing combinatorial overfitting |
| **`min_child_samples`** | `{p.get('min_child_samples', 20)}` | Minimum sample count required in terminal leaf |
| **`subsample`** | `{p.get('subsample', 0.8):.4f}` | Row subsampling fraction per boosting round |
| **`colsample_bytree`** | `{p.get('colsample_bytree', 0.8):.4f}` | Feature fraction randomly selected per tree |
| **`reg_alpha` (L1)** | `{p.get('reg_alpha', 0.1):.4e}` | Lasso sparsity regularization |
| **`reg_lambda` (L2)** | `{p.get('reg_lambda', 1.0):.4e}` | Ridge curvature penalty |
| **`ridge_alpha`** | `{p.get('ridge_alpha', 100.0):.2f}` | L2 regularization for linear component |
| **`ridge_weight`** | `{p.get('ridge_weight', 0.5):.2f}` | Blending weight allocated to Ridge model |

**Best Out-of-Sample Rank IC:** `+{tuning_result.best_score:.4f}`

---

## 3. Marcos López de Prado Triple Barrier Method (AFML Ch. 3 & 4)

### Barrier Architecture
- **Profit-Taking Barrier ($pt$)**: $+{tb_config.pt} \\times \\sigma_{{\\text{{Garman-Klass}}}}$
- **Stop-Loss Barrier ($sl$)**: $-{tb_config.sl} \\times \\sigma_{{\\text{{Garman-Klass}}}}$
- **Vertical Time Horizon**: `{tb_config.holding_period_bars} hours`

### Execution Path Outcomes
- **Profit Take Hit (`PT`)**: `{touch_dist.get('pt', 0.0):.2%}`
- **Stop Loss Hit (`SL`)**: `{touch_dist.get('sl', 0.0):.2%}`
- **Vertical Timeout (`VERTICAL`)**: `{touch_dist.get('vertical', 0.0):.2%}`
- **Average Sample Uniqueness ($\\bar{{u}}$)**: `{avg_uniqueness:.4f}` (AFML Ch. 4 concurrent label weighting)

---

## 4. Secondary Meta-Classifier & Kelly Position Sizing (Issue #35)

The Secondary Classifier models the conditional probability of trade success:
$$P(Y = 1 \\mid \\text{{Regime}}, \\sigma_{{\\text{{GK}}}}, \\text{{Hurst}}, \\text{{Illiquidity}}, |\\text{{Signal}}|)$$

### Precision Uplift & Trade Filtering
- **Holdout AUC-ROC**: `{report.auc_roc:.4f}`
- **Brier Calibration Score**: `{report.brier_score:.4f}`
- **Unfiltered Primary Model Win Rate**: `{report.precision_unfiltered:.2%}`
- **Meta-Filtered Win Rate ($P > 0.5$)**: `{report.precision_filtered:.2%}`
- **Executed Trade Ratio**: `{report.trades_executed_ratio:.2%}` (rejection of {1.0 - report.trades_executed_ratio:.2%} low-conviction false positives)

### Continuous Kelly Allocation
Each trade $i$ is sized via fractional Kelly leverage ($f = 0.5$):
$$w_i = \\text{{sign}}(\\text{{primary\\_score}}) \\times \\max(0, 2 P_i - 1) \\times 0.5$$

When $P_i \\le 0.5$, the position is completely vetoed ($w_i = 0$), immunizing capital against high-volatility stop-outs.

---

## 5. Artifacts and Reproducibility
- **Model Pipeline**: `src/models/triple_barrier.py`, `src/models/purged_cv.py`, `src/models/meta_labeling.py`, `src/models/optuna_tuner.py`
- **Training Script**: `scripts/train_ml_meta_pipeline.py`
- **Colab GPU Notebook**: `scripts/colab_ml_training.ipynb`
- **Plotly Tearsheet**: `reports/tearsheet_meta_labeling_optuna.html`
"""
    path.write_text(md_content, encoding="utf-8")


def generate_plotly_tearsheet(
    path: Path,
    tuning_result: TuningResult,
    test_events: pd.DataFrame,
    test_probs: np.ndarray,
    unfiltered_returns: np.ndarray,
    meta_kelly_returns: np.ndarray,
):
    """Generates an interactive 4-panel HTML tearsheet."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Cumulative Returns: Unfiltered vs Meta-Labeled Kelly Sizing",
            "Optuna Bayesian Hyperparameter Optimization History",
            "Predicted Trade Success Probability Distribution",
            "Win Rate by Meta-Classifier Confidence Decile",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # 1. Cumulative returns
    cum_unfiltered = np.cumprod(1.0 + unfiltered_returns) - 1.0
    cum_meta = np.cumprod(1.0 + meta_kelly_returns) - 1.0

    fig.add_trace(
        go.Scatter(
            y=cum_unfiltered * 100,
            mode="lines",
            name="Unfiltered Primary Model",
            line={"color": "#EF553B", "width": 2},
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=cum_meta * 100,
            mode="lines",
            name="Meta-Labeled Kelly Sized",
            line={"color": "#00CC96", "width": 2.5},
        ),
        row=1, col=1,
    )

    # 2. Optuna optimization history
    trials_df = tuning_result.trials_df
    if "value" in trials_df.columns:
        fig.add_trace(
            go.Scatter(
                x=trials_df["number"],
                y=trials_df["value"],
                mode="markers+lines",
                name="Trial Rank IC",
                marker={"color": "#636EFA", "size": 7},
                line={"color": "#AB63FA", "dash": "dot"},
            ),
            row=1, col=2,
        )

    # 3. Probability distribution
    fig.add_trace(
        go.Histogram(
            x=test_probs,
            nbinsx=30,
            name="P(Success)",
            marker_color="#FFA15A",
            opacity=0.75,
        ),
        row=2, col=1,
    )
    # Add vertical veto threshold line at 0.5
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", row=2, col=1)

    # 4. Calibration: Win rate by probability bucket
    bins = np.linspace(0.0, 1.0, 6)
    bin_indices = np.digitize(test_probs, bins) - 1
    y_true = test_events["label"].values
    bin_centers = []
    bin_win_rates = []
    for b in range(len(bins) - 1):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_centers.append(f"{bins[b]:.1f}-{bins[b+1]:.1f}")
            bin_win_rates.append(float(np.mean(y_true[mask]) * 100))

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=bin_win_rates,
            name="Empirical Win Rate %",
            marker_color="#19D3F3",
        ),
        row=2, col=2,
    )

    fig.update_layout(
        title="Institutional Machine Learning Alpha & Meta-Labeling Tearsheet (AFML Ch. 3 & 7)",
        template="plotly_dark",
        height=850,
        width=1200,
        showlegend=True,
    )

    fig.write_html(str(path))


def main():
    parser = argparse.ArgumentParser(description="Institutional ML & Meta-Labeling Pipeline")
    parser.add_argument("--cache-dir", type=str, default="/tmp/crypto_alpha_backtest_cache", help="Market data cache path")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,NEARUSDT", help="Comma-separated symbols")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna Bayesian trials")
    parser.add_argument("--max-bars", type=int, default=4000, help="Max 1h bars per symbol")
    parser.add_argument("--output-dir", type=str, default="reports", help="Reports output directory")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    cache_path = Path(args.cache_dir)
    out_path = Path(args.output_dir)

    run_pipeline(
        cache_dir=cache_path,
        symbols=symbols,
        n_trials=args.trials,
        max_bars=args.max_bars,
        output_dir=out_path,
    )


if __name__ == "__main__":
    main()
