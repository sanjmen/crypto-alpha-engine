# Cloud Infrastructure & Distributed Compute Architecture

## 1. Inventory of Available User Resources (Audited)

An empirical audit of the local environment and connected Google Cloud/Drive accounts reveals an extraordinary institutional-tier infrastructure:

| Resource | Verified Capacity | Status | Optimal Function in Engine |
| :--- | :--- | :--- | :--- |
| **Google Drive** | **22.0 TiB Total (21.9 TiB Free)** | Connected via `rclone` (`gdrive:`) | Central repository for all raw and processed crypto market data (Parquet warehouse). Zero local disk consumed. |
| **Google Colab Ultra / Pro+** | **1,000 Compute Credits** | Active on Ultra Plan | Massive parallel data download (1 Gbps Google backbone), heavy feature engineering, and GPU model training (A100 / L4 / High-RAM 85GB). |
| **Google Cloud Platform (GCP)** | Project `trad-lab-499515` | Billing Enabled (`018FB5-B6CEE1`) | Cloud Run / Spot VMs / BigQuery for optional 24/7 live signal execution and alerts. |
| **Local Mac** | macOS Development Environment | Local Repository | Code authoring, architecture, git version control, and unit testing on small validation subsets. |

---

## 2. Distributed Workflow Architecture

```
                          [ GOOGLE CLOUD ECOSYSTEM ]
                                      |
         +----------------------------+----------------------------+
         |                                                         |
  [ GOOGLE COLAB (1000 CU) ]                              [ LOCAL MAC (Dev) ]
  - High-RAM (53-85 GB)                                   - Clean Architecture code
  - NVIDIA A100 / L4 GPUs                                 - Unit test suite (pytest)
  - 10 Gbps Google Backbone                               - Fast iteration & git
  - Heavy LightGBM & GMM training                                  |
         |                                                         |
         | (Mounts directly at Gigabit speed)                      | (Syncs via rclone)
         v                                                         v
  +-----------------------------------------------------------------------+
  |              GOOGLE DRIVE (22 TiB Free Storage)                       |
  |              Path: gdrive:trading/crypto-alpha-engine/data/            |
  |                                                                       |
  |  - /bars/{timeframe}/{symbol}.parquet                                 |
  |  - /metrics/{symbol}.parquet (5-min OI & Long/Short)                 |
  |  - /funding/{symbol}.parquet                                          |
  |  - /depth/{symbol}.parquet (L2 50 levels)                             |
  +-----------------------------------------------------------------------+
```

---

## 3. How Google Colab + Google Drive Saves 100% of Local Disk & Home Bandwidth

1. **Gigabit Ingestion without Home Bandwidth**:
   * If we download 100 GB of multi-year market data from Binance Vision at home, it consumes local WiFi bandwidth and hours of time.
   * **In Google Colab**, the download runs over Google's internal datacenter fiber at **300 to 800 MB/s**.
   * By mounting Google Drive (`drive.mount('/content/drive')`), Colab downloads the ZIPs from Binance Vision S3, decompresses them into memory, converts them to compressed Parquet with DuckDB, and writes them straight to **`My Drive/trading/crypto-alpha-engine/data/`**.
   * **Zero local disk space used. Zero home internet bandwidth consumed.**

2. **Accessing Data from Local Mac via `rclone`**:
   * We already verified that `/opt/homebrew/bin/rclone` is installed and has `gdrive:` authenticated.
   * To inspect or query files without downloading 50 GB:
     ```bash
     # List files in Google Drive storage
     rclone lsd gdrive:trading/crypto-alpha-engine/data/

     # Sync only a small test slice to local cache if needed
     rclone copy gdrive:trading/crypto-alpha-engine/data/bars/1h/BTC_USDT.parquet data/cache/bars/1h/
     ```

3. **Heavy Model Training with Colab 1000 Credits**:
   * **NVIDIA A100 (40GB/80GB VRAM)**: ~12-15 credits/hour $\approx$ 70+ hours of continuous A100 training.
   * **NVIDIA L4 (24GB VRAM)**: ~5 credits/hour $\approx$ 200 hours of L4 training.
   * **High-RAM CPU (85GB RAM)**: Ideal for cross-sectional matrix operations over 50 altcoins across 5 years.
   * **Background Execution**: Colab Ultra allows notebooks to run up to 24 hours in the cloud even when closing the browser window or putting the Mac to sleep.

---

## 4. Colab Ingestion Notebook Template

We provide a dedicated notebook: `scripts/colab_ingest_to_drive.ipynb`:
```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DATA_DIR = "/content/drive/MyDrive/trading/crypto-alpha-engine/data"
# Automated streaming from data.binance.vision directly into DRIVE_DATA_DIR
```
