#!/usr/bin/env bash
# ==============================================================================
# Crypto Alpha Engine — Zero-Trust Security, Docker & GitHub Actions Bootstrap
# Instance: crypto-alpha-engine | Static IP: 35.194.111.133 | Project: my-project-alpha-514420
#
# Comandos:
#   ./scripts/deploy_to_gcp.sh --bootstrap        # 1. Cierra puertos públicos (0.0.0.0/0) en Firewall
#                                                 #    dejando solo Túnel IAP + tu IP local.
#                                                 # 2. Instala Docker + Compose + Swap en la VM.
#                                                 # 3. Configura Service Account y GitHub Secrets (CI/CD).
#   ./scripts/deploy_to_gcp.sh --cutover-db       # Migra live_trading.db y shadow_trading.db a Tokio
#                                                 # haciendo WAL checkpoint previo y levanta contenedores.
#   ./scripts/deploy_to_gcp.sh --monitor-mainnet  # Abre el Dashboard en vivo de Mainnet por túnel IAP.
#   ./scripts/deploy_to_gcp.sh --monitor-shadow   # Abre el Dashboard en vivo de Shadow por túnel IAP.
#   ./scripts/deploy_to_gcp.sh --status           # Muestra estado de contenedores Docker y Firewall.
# ==============================================================================
set -euo pipefail

export CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity"

PROJECT_ID="my-project-alpha-514420"
ZONE="asia-northeast1-a"
INSTANCE="crypto-alpha-engine"
STATIC_IP="35.194.111.133"
SA_NAME="github-actions-deployer"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

MODE="${1:---status}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$MODE" == "--bootstrap" ]]; then
    echo "=================================================================="
    echo "  [1/4] Blindaje de Firewall GCP (Zero Public Open Ports)"
    echo "=================================================================="
    gcloud services enable iap.googleapis.com iam.googleapis.com --project="$PROJECT_ID" --quiet

    MY_IP="$(curl -4 -s https://ifconfig.me || echo "")"
    if [[ -n "$MY_IP" ]]; then
        ALLOWED_RANGES="35.235.240.0/20,${MY_IP}/32"
        echo "-> Cerrando puerto 22 público (0.0.0.0/0). Solo permitido:"
        echo "   - Túnel Zero-Trust Google IAP (35.235.240.0/20)"
        echo "   - Tu IP local actual (${MY_IP}/32)"
    else
        ALLOWED_RANGES="35.235.240.0/20"
        echo "-> Cerrando puerto 22 público (0.0.0.0/0). Solo permitido Túnel Google IAP (35.235.240.0/20)."
    fi

    gcloud compute firewall-rules update default-allow-ssh \
        --project="$PROJECT_ID" \
        --source-ranges="$ALLOWED_RANGES" \
        --quiet

    # Eliminar reglas públicas innecesarias (RDP / ICMP abiertas a 0.0.0.0/0)
    gcloud compute firewall-rules delete default-allow-rdp default-allow-icmp \
        --project="$PROJECT_ID" --quiet 2>/dev/null || true

    echo "=================================================================="
    echo "  [2/4] Instalando Docker Engine, Compose & 2GB Swap en Tokio"
    echo "=================================================================="
    gcloud compute ssh "$INSTANCE" \
        --zone="$ZONE" \
        --project="$PROJECT_ID" \
        --tunnel-through-iap \
        --quiet \
        --command="
            set -euo pipefail
            if [ ! -f /swapfile ]; then
                sudo fallocate -l 2G /swapfile
                sudo chmod 600 /swapfile
                sudo mkswap /swapfile
                sudo swapon /swapfile
                echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
            fi
            if ! command -v docker >/dev/null 2>&1; then
                sudo DEBIAN_FRONTEND=noninteractive apt-get update -y -qq
                sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates curl sqlite3 git
                curl -fsSL https://get.docker.com | sudo sh
                sudo usermod -aG docker \$USER
            fi
            mkdir -p ~/crypto-alpha-engine/data ~/crypto-alpha-engine/logs
            docker --version
            docker compose version
        "

    echo "=================================================================="
    echo "  [3/4] Creando Service Account de CI/CD para GitHub Actions"
    echo "=================================================================="
    if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" >/dev/null 2>&1; then
        gcloud iam service-accounts create "$SA_NAME" \
            --display-name="GitHub Actions CI/CD Deployer" \
            --project="$PROJECT_ID" \
            --quiet
    fi

    for ROLE in roles/compute.instanceAdmin.v1 roles/iap.tunnelResourceAccessor roles/iam.serviceAccountUser; do
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="$ROLE" \
            --condition=None \
            --quiet >/dev/null
    done

    KEY_FILE="/tmp/gcp-gh-deployer-key.json"
    if [[ ! -f "$KEY_FILE" ]]; then
        gcloud iam service-accounts keys create "$KEY_FILE" \
            --iam-account="$SA_EMAIL" \
            --project="$PROJECT_ID" \
            --quiet
    fi

    echo "=================================================================="
    echo "  [4/5] Configurando GitHub Actions Secrets (gh secret set)"
    echo "=================================================================="
    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
        gh secret set GCP_SA_KEY < "$KEY_FILE"
        if [[ -f ".env" ]]; then
            python3 -c '
import pathlib, subprocess
env_map = {}
for raw in pathlib.Path(".env").read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    if line.startswith("export "):
        line = line[7:].strip()
    k, v = line.split("=", 1)
    env_map[k.strip()] = v.strip().strip("\"").strip("\x27")
for key in ("BINANCE_API_KEY", "BINANCE_API_SECRET"):
    if env_map.get(key):
        subprocess.run(["gh", "secret", "set", key], input=env_map[key].encode(), check=True)
'
        fi
        rm -f "$KEY_FILE"
        echo "[OK] Secretos GCP_SA_KEY, BINANCE_API_KEY y BINANCE_API_SECRET cargados en GitHub!"
    else
        echo "[AVISO] 'gh' CLI no está autenticado. Tu clave JSON quedó temporalmente en: $KEY_FILE"
        echo "Ejecuta 'gh auth login' y vuelve a correr este paso, o súbela en GitHub -> Settings -> Secrets -> Actions."
    fi

    echo "=================================================================="
    echo "  [5/5] Sincronizando Bases de Datos SQLite (Hot Backup) y .env"
    echo "=================================================================="
    rm -f /tmp/live_trading.db /tmp/shadow_trading.db /tmp/dbs.tar.gz
    if [[ -f "data/live_trading.db" ]]; then
        sqlite3 data/live_trading.db ".backup '/tmp/live_trading.db'"
    fi
    if [[ -f "data/shadow_trading.db" ]]; then
        sqlite3 data/shadow_trading.db ".backup '/tmp/shadow_trading.db'"
    fi
    tar -C /tmp -czf /tmp/dbs.tar.gz live_trading.db shadow_trading.db

    gcloud compute scp /tmp/dbs.tar.gz \
        "$INSTANCE:/tmp/dbs.tar.gz" \
        --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet

    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet \
        --command="mkdir -p ~/crypto-alpha-engine/data ~/crypto-alpha-engine/logs && tar -xzf /tmp/dbs.tar.gz -C ~/crypto-alpha-engine/data/ && rm -f /tmp/dbs.tar.gz"

    if [[ -f ".env" ]]; then
        gcloud compute scp .env "$INSTANCE:~/crypto-alpha-engine/.env" \
            --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet
    fi
    rm -f /tmp/live_trading.db /tmp/shadow_trading.db /tmp/dbs.tar.gz

    echo ""
    echo "=================================================================="
    echo "  BOOTSTRAP COMPLETADO"
    echo "  Cada 'git push origin main' ahora ejecutará tests y desplegará"
    echo "  automáticamente los contenedores Docker en Tokio vía túnel IAP."
    echo "=================================================================="
    exit 0
fi

if [[ "$MODE" == "--sync-db-and-secrets" ]]; then
    KEY_FILE="/tmp/gcp-gh-deployer-key.json"
    if [[ -f "$KEY_FILE" ]] && command -v gh >/dev/null 2>&1; then
        gh secret set GCP_SA_KEY < "$KEY_FILE"
        rm -f "$KEY_FILE"
    fi
    if [[ -f ".env" ]] && command -v gh >/dev/null 2>&1; then
        python3 -c '
import pathlib, subprocess
env_map = {}
for raw in pathlib.Path(".env").read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    if line.startswith("export "):
        line = line[7:].strip()
    k, v = line.split("=", 1)
    env_map[k.strip()] = v.strip().strip("\"").strip("\x27")
for key in ("BINANCE_API_KEY", "BINANCE_API_SECRET"):
    if env_map.get(key):
        subprocess.run(["gh", "secret", "set", key], input=env_map[key].encode(), check=True)
'
        echo "[OK] Secretos BINANCE_API_KEY y BINANCE_API_SECRET cargados en GitHub!"
    fi

    echo "-> Subiendo backup comprimido de live_trading.db y shadow_trading.db a Tokio..."
    if [[ ! -f "/tmp/dbs.tar.gz" ]]; then
        sqlite3 data/live_trading.db ".backup '/tmp/live_trading.db'"
        sqlite3 data/shadow_trading.db ".backup '/tmp/shadow_trading.db'"
        tar -C /tmp -czf /tmp/dbs.tar.gz live_trading.db shadow_trading.db
    fi
    gcloud compute scp /tmp/dbs.tar.gz "$INSTANCE:/tmp/dbs.tar.gz" \
        --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet \
        --command="mkdir -p ~/crypto-alpha-engine/data ~/crypto-alpha-engine/logs && tar -xzf /tmp/dbs.tar.gz -C ~/crypto-alpha-engine/data/ && rm -f /tmp/dbs.tar.gz"
    if [[ -f ".env" ]]; then
        gcloud compute scp .env "$INSTANCE:~/crypto-alpha-engine/.env" \
            --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet
    fi
    rm -f /tmp/live_trading.db /tmp/shadow_trading.db /tmp/dbs.tar.gz
    echo "[OK] Bases de datos y .env sincronizados en Tokio!"
    exit 0
fi

if [[ "$MODE" == "--cutover-db" ]]; then
    echo "[1/4] Deteniendo procesos locales en Mac (Mainnet & Shadow) de forma segura..."
    pkill -INT -f "scripts/run_trading.py" || true
    sleep 3

    echo "[2/4] Ejecutando SQLite WAL Checkpoint (consolidando transacciones en disco)..."
    if [[ -f "data/live_trading.db" ]]; then
        sqlite3 data/live_trading.db "PRAGMA wal_checkpoint(TRUNCATE);"
    fi
    if [[ -f "data/shadow_trading.db" ]]; then
        sqlite3 data/shadow_trading.db "PRAGMA wal_checkpoint(TRUNCATE);"
    fi

    echo "[3/4] Subiendo bases de datos SQLite y .env a Tokio por túnel cifrado IAP..."
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet \
        --command="mkdir -p ~/crypto-alpha-engine/data ~/crypto-alpha-engine/logs && (cd ~/crypto-alpha-engine && docker compose stop || true) && rm -f ~/crypto-alpha-engine/data/*.db-wal ~/crypto-alpha-engine/data/*.db-shm"

    gcloud compute scp data/live_trading.db data/shadow_trading.db \
        "$INSTANCE:~/crypto-alpha-engine/data/" \
        --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet

    if [[ -f ".env" ]]; then
        gcloud compute scp .env "$INSTANCE:~/crypto-alpha-engine/.env" \
            --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet
    fi

    echo "[4/4] Levantando contenedores Docker (Mainnet & Shadow) en Tokio..."
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet \
        --command="cd ~/crypto-alpha-engine && chmod 600 .env && docker compose up -d && docker compose ps"

    echo "[OK] Bases de datos migradas y contenedores corriendo 24/7 en Tokio ($STATIC_IP)."
    exit 0
fi

if [[ "$MODE" == "--monitor-mainnet" ]]; then
    exec gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap -- -t \
        "docker exec -it crypto-mainnet python scripts/run_trading.py --monitor --mainnet"
fi

if [[ "$MODE" == "--monitor-shadow" ]]; then
    exec gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap -- -t \
        "docker exec -it crypto-shadow python scripts/run_trading.py --monitor --shadow"
fi

if [[ "$MODE" == "--diagnostics" ]]; then
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet \
        --command='
            set -euo pipefail
            mkdir -p ~/crypto-alpha-engine/data/cache ~/crypto-alpha-engine/logs
            echo "=== 1. RECURSOS DE LA INSTANCIA (CPU / RAM / SWAP / DISCO) ==="
            uptime
            free -h
            df -h /
            echo ""
            echo "=== 2. CONSUMO POR CONTENEDOR DOCKER ==="
            docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}"
            echo ""
            echo "=== 3. LATENCIA DESDE TOKIO A BINANCE FUTURES (fapi.binance.com) ==="
            curl -s -o /dev/null -w "Cold Handshake (TCP+TLS+HTTP): %{time_total}s\nWarm Keep-Alive RTT #1: " https://fapi.binance.com/fapi/v1/time \
                 -o /dev/null -w "%{time_total}s\nWarm Keep-Alive RTT #2: " https://fapi.binance.com/fapi/v1/time \
                 -o /dev/null -w "%{time_total}s\n" https://fapi.binance.com/fapi/v1/time
            echo ""
            echo "=== 4. ESTADO INTERNO DE MAINNET Y SHADOW (SQLITE + BINANCE LIVE) ==="
            docker exec -i crypto-mainnet python - << "PYEOF"
import os, sqlite3
from src.execution.live_broker import BinanceLiveBroker

def inspect_db(path, label):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    hb = dict(conn.execute("SELECT * FROM daemon_heartbeat WHERE id=1").fetchone() or {})
    eq = dict(conn.execute("SELECT * FROM accounts ORDER BY id DESC LIMIT 1").fetchone() or {})
    rb = dict(conn.execute("SELECT * FROM rebalance_snapshots ORDER BY id DESC LIMIT 1").fetchone() or {})
    pos = [dict(r) for r in conn.execute("SELECT symbol, side, quantity, entry_price, current_price, unrealized_pnl, strategy FROM positions WHERE quantity > 0").fetchall()]
    pend = [dict(r) for r in conn.execute("SELECT id, symbol, side, order_type, quantity, price, status FROM orders WHERE status = ?", ("PENDING",)).fetchall()]
    d_st = hb.get("status")
    w_st = hb.get("ws_state")
    tps = hb.get("ticks_per_sec", 0.0)
    reg = rb.get("regime_detected")
    upd = hb.get("timestamp")
    t_eq = eq.get("total_equity", 0.0)
    c_bal = eq.get("cash_balance", 0.0)
    upnl = eq.get("unrealized_pnl", 0.0)
    lev = eq.get("leverage", 0.0)
    print(f"[{label}] Heartbeat: status={d_st} | ws={w_st} ({tps:.1f} ticks/s) | regime={reg} | updated={upd}")
    print(f"[{label}] Equity: ${t_eq:.2f} | Cash: ${c_bal:.2f} | uPnL: ${upnl:+.2f} | Lev: {lev:.2f}x | Open Pos: {len(pos)} | Pending Orders: {len(pend)}")
    return pos

m_pos = inspect_db("data/live_trading.db", "MAINNET")
s_pos = inspect_db("data/shadow_trading.db", "SHADOW")

broker = BinanceLiveBroker(
    api_key=os.environ.get("BINANCE_API_KEY"),
    api_secret=os.environ.get("BINANCE_API_SECRET"),
    dry_run=False,
)
ex_pos = broker.reconcile_positions_from_exchange()
bal = broker.client.fetch_balance()
col = broker.compute_effective_margin_collateral(bal)
w_b = col.get("effective_wallet_usd", 0.0)
f_b = col.get("effective_free_usd", 0.0)
u_p = sum(float(p.unrealized_pnl) for p in ex_pos.values())
print(f"[BINANCE LIVE API] Wallet: ${w_b:.2f} | Free: ${f_b:.2f} | uPnL: ${u_p:+.2f} | Total Equity: ${(w_b + u_p):.2f} | Live Positions: {len(ex_pos)}")
local_map = {p["symbol"]: round(float(p["quantity"]), 6) for p in m_pos}
live_map = {s: round(abs(float(p.quantity)), 6) for s, p in ex_pos.items()}
diffs = {k: (local_map.get(k), live_map.get(k)) for k in set(local_map) | set(live_map) if local_map.get(k) != live_map.get(k)}
sync_msg = diffs if diffs else "100% SINCRONIZADO"
print(f"[SYNC CHECK] Discrepancias DB vs Binance Live: {len(diffs)} -> {sync_msg}")
PYEOF
            echo ""
            echo "=== 5. ARCHIVOS EN CACHE PARQUET (data/cache) ==="
            find ~/crypto-alpha-engine/data/cache -type f | wc -l
            echo ""
            echo "=== 6. ULTIMOS LOGS DE MAINNET Y SHADOW ==="
            cd ~/crypto-alpha-engine && docker compose logs --tail=20
        '
    exit 0
fi

# Default: --status
echo "=== Reglas de Firewall activas en $PROJECT_ID ==="
gcloud compute firewall-rules list --project="$PROJECT_ID" --format="table(name,direction,sourceRanges.list():label=SOURCE_RANGES,allowed[].map().firewall_rule().list():label=ALLOW)"
echo ""
echo "=== Contenedores Docker en $INSTANCE ($STATIC_IP) ==="
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --tunnel-through-iap --quiet \
    --command="cd ~/crypto-alpha-engine && docker compose ps && echo '' && docker compose logs --tail=15"
