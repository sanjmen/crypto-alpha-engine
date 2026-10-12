"""
Crypto Sector Clustering & Domain Taxonomy Engine.
Issue #22: Multi-Asset Sector Clustering for Statistical Arbitrage.

Categorizes the crypto perpetual universe into institutional fundamental sectors
(Layer 1s, DeFi, Memes, AI/Data, Modular/L2, Gaming, Storage) and provides
data-driven agglomerative clustering based on rolling correlation distance.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Curated institutional sector taxonomy across Binance perpetuals
CRYPTO_SECTOR_TAXONOMY: dict[str, list[str]] = {
    "Layer1_Core": [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT",
        "NEARUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "FTMUSDT",
        "SUIUSDT", "APTUSDT", "SEIUSDT", "INJUSDT", "KASUSDT", "ALGOUSDT",
        "XLMUSDT", "HBARUSDT", "ICPUSDT", "TRXUSDT", "TONUSDT"
    ],
    "DeFi_DEX_Lending": [
        "UNIUSDT", "AAVEUSDT", "LINKUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT",
        "LDOUSDT", "PENDLEUSDT", "DYDXUSDT", "COMPUSDT", "1INCHUSDT", "RUNEUSDT",
        "GMXUSDT", "SUSHIUSDT", "CAKEUSDT", "ENAUSDT", "JUPUSDT", "RAYUSDT"
    ],
    "Meme_Speculative": [
        "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT", "FLOKIUSDT",
        "MEMEUSDT", "BOMEUSDT", "POPCATUSDT", "1000SATSUSDT", "ORDIUSDT",
        "MYROUSDT", "MEWUSDT", "BRETTUSDT", "NEIROUSDT"
    ],
    "AI_Compute_Data": [
        "RENDERUSDT", "FETUSDT", "TAOUSDT", "GRTUSDT", "WLDUSDT", "ARKMUSDT",
        "IOUSDT", "THETAUSDT", "OCEANUSDT", "AGIXUSDT", "AIUSDT"
    ],
    "Modular_Infra_L2": [
        "ARBUSDT", "OPUSDT", "TIAUSDT", "STRKUSDT", "MANTAUSDT", "MATICUSDT",
        "POLUSDT", "ZKUSDT", "BLURUSDT", "PYTHUSDT", "WUSDT", "ALTUSDT"
    ],
    "Gaming_Metaverse": [
        "GALAUSDT", "AXSUSDT", "SANDUSDT", "MANAUSDT", "BEAMUSDT", "RONINUSDT",
        "PIXELUSDT", "PORTALUSDT", "ENJUSDT", "IMXUSDT"
    ],
    "Storage_Cloud": [
        "FILUSDT", "ARUSDT", "STORJUSDT", "SCUSDT"
    ],
}


class SectorTaxonomy:
    """
    Manages domain-specific crypto asset classification and lookups.
    """

    def __init__(self, taxonomy: dict[str, list[str]] | None = None):
        self.taxonomy = taxonomy or CRYPTO_SECTOR_TAXONOMY
        self._symbol_to_sector = {}
        for sector, symbols in self.taxonomy.items():
            for s in symbols:
                self._symbol_to_sector[s] = sector

    def get_sector(self, symbol: str) -> str:
        """Returns the fundamental sector for a given symbol, or 'Other' if unclassified."""
        return self._symbol_to_sector.get(symbol, "Other")

    def get_sector_symbols(self, sector: str) -> list[str]:
        """Returns all symbols belonging to a specific sector."""
        return self.taxonomy.get(sector, [])

    def filter_universe_by_sector(self, universe: list[str]) -> dict[str, list[str]]:
        """
        Partitions an input universe of symbols into their respective sectors.
        """
        sector_map: dict[str, list[str]] = {sec: [] for sec in self.taxonomy}
        sector_map["Other"] = []

        for sym in universe:
            sec = self.get_sector(sym)
            if sec in sector_map:
                sector_map[sec].append(sym)
            else:
                sector_map["Other"].append(sym)

        # Remove empty sectors
        return {sec: syms for sec, syms in sector_map.items() if len(syms) > 0}


class StatisticalSectorClusterer:
    """
    Data-driven sector clustering using correlation distance and Agglomerative Clustering.
    Discovers empirical statistical sectors directly from price co-movements.
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        max_clusters: int = 8,
        linkage: str = "ward",
    ):
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.linkage = linkage
        self.labels_: np.ndarray | None = None
        self.clusters_: dict[int, list[str]] = {}
        self.optimal_k_: int = 4

    def fit(self, returns: pd.DataFrame) -> "StatisticalSectorClusterer":
        """
        Fits agglomerative clustering on return correlation distance.

        Args:
            returns: DataFrame of asset returns (rows = time, columns = symbols).
        """
        clean_rets = returns.dropna(how="all").fillna(0.0)
        symbols = list(clean_rets.columns)
        n = len(symbols)

        if n < 2:
            self.clusters_ = {0: symbols}
            self.labels_ = np.zeros(n, dtype=int)
            return self

        # 1. Compute correlation matrix
        corr = clean_rets.corr().fillna(0.0).values
        # 2. Distance metric d_ij = sqrt(0.5 * (1 - rho_ij))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        # 3. Determine optimal number of clusters if not specified
        if self.n_clusters is not None:
            k = min(self.n_clusters, n)
        else:
            k = self._find_optimal_k(dist, n)

        self.optimal_k_ = k

        # 4. Fit Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean" if self.linkage == "ward" else "precomputed",
            linkage=self.linkage,
        )

        if self.linkage == "ward":
            # Ward requires Euclidean coordinates (correlation distance matrix columns)
            self.labels_ = clustering.fit_predict(dist)
        else:
            self.labels_ = clustering.fit_predict(dist)

        # 5. Group symbols by cluster label
        self.clusters_ = {}
        for idx, label in enumerate(self.labels_):
            self.clusters_.setdefault(int(label), []).append(symbols[idx])

        return self

    def _find_optimal_k(self, dist_matrix: np.ndarray, n_samples: int) -> int:
        """Finds k that maximizes silhouette score on correlation distance."""
        min_k = 2
        max_k = min(self.max_clusters, n_samples - 1)
        if max_k <= min_k:
            return min_k

        best_k = min_k
        best_score = -1.0

        for k in range(min_k, max_k + 1):
            clf = AgglomerativeClustering(
                n_clusters=k,
                metric="euclidean" if self.linkage == "ward" else "precomputed",
                linkage=self.linkage,
            )
            labels = clf.fit_predict(dist_matrix)
            if len(set(labels)) > 1:
                score = silhouette_score(dist_matrix, labels, metric="precomputed" if self.linkage != "ward" else "euclidean")
                if score > best_score:
                    best_score = score
                    best_k = k

        return best_k

    def get_candidate_pairs(self, min_pair_count: int = 1) -> list[tuple[str, str]]:
        """
        Generates intra-cluster pairs for cointegration testing.
        Only pairs within the same statistical cluster are evaluated.
        """
        pairs = []
        for syms in self.clusters_.values():
            if len(syms) >= 2:
                for i in range(len(syms)):
                    for j in range(i + 1, len(syms)):
                        pairs.append((syms[i], syms[j]))
        return pairs
