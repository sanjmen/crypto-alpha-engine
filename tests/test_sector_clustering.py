"""
Unit tests for Sector Taxonomy and Statistical Clustering.
Issue #22.
"""

import numpy as np
import pandas as pd

from src.features.sector_clustering import SectorTaxonomy, StatisticalSectorClusterer


def test_sector_taxonomy_lookups():
    tax = SectorTaxonomy()

    # Core checks
    assert tax.get_sector("BTCUSDT") == "Layer1_Core"
    assert tax.get_sector("ETHUSDT") == "Layer1_Core"
    assert tax.get_sector("UNIUSDT") == "DeFi_DEX_Lending"
    assert tax.get_sector("DOGEUSDT") == "Meme_Speculative"
    assert tax.get_sector("RENDERUSDT") == "AI_Compute_Data"
    assert tax.get_sector("ARBUSDT") == "Modular_Infra_L2"
    assert tax.get_sector("GALAUSDT") == "Gaming_Metaverse"
    assert tax.get_sector("UNKNOWN_TOKEN") == "Other"

    # Universe partition
    universe = ["BTCUSDT", "ETHUSDT", "AAVEUSDT", "PEPEUSDT", "RANDOMUSDT"]
    partition = tax.filter_universe_by_sector(universe)

    assert "Layer1_Core" in partition
    assert "BTCUSDT" in partition["Layer1_Core"]
    assert "ETHUSDT" in partition["Layer1_Core"]
    assert "DeFi_DEX_Lending" in partition
    assert "AAVEUSDT" in partition["DeFi_DEX_Lending"]
    assert "Meme_Speculative" in partition
    assert "PEPEUSDT" in partition["Meme_Speculative"]


def test_statistical_sector_clusterer():
    np.random.seed(42)
    n_bars = 200
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")

    # Cluster 1: L1 factor
    l1_factor = np.random.normal(0, 0.02, n_bars)
    btc = l1_factor + np.random.normal(0, 0.005, n_bars)
    eth = l1_factor + np.random.normal(0, 0.005, n_bars)

    # Cluster 2: DeFi factor
    defi_factor = np.random.normal(0, 0.03, n_bars)
    uni = defi_factor + np.random.normal(0, 0.005, n_bars)
    aave = defi_factor + np.random.normal(0, 0.005, n_bars)

    returns = pd.DataFrame(
        {"BTC": btc, "ETH": eth, "UNI": uni, "AAVE": aave},
        index=dates,
    )

    clusterer = StatisticalSectorClusterer(n_clusters=2, linkage="ward")
    clusterer.fit(returns)

    assert clusterer.labels_ is not None
    assert len(clusterer.clusters_) == 2

    # BTC and ETH should be in the same cluster
    # UNI and AAVE should be in the same cluster
    btc_cluster = next(cid for cid, syms in clusterer.clusters_.items() if "BTC" in syms)
    eth_cluster = next(cid for cid, syms in clusterer.clusters_.items() if "ETH" in syms)
    assert btc_cluster == eth_cluster

    uni_cluster = next(cid for cid, syms in clusterer.clusters_.items() if "UNI" in syms)
    aave_cluster = next(cid for cid, syms in clusterer.clusters_.items() if "AAVE" in syms)
    assert uni_cluster == aave_cluster
    assert btc_cluster != uni_cluster

    pairs = clusterer.get_candidate_pairs()
    assert len(pairs) == 2  # (BTC, ETH) and (UNI, AAVE)
