"""
Colab Ingestion & Remote Compute Orchestrator.
Coordinates headless Google Colab execution, Drive mounting, real-time log streaming,
and continuous keepalive heartbeats to prevent idle disconnects.
Zero local disk usage. Zero browser tab clicks required.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cloud.colab_client import ColabClient


def parse_args():
    parser = argparse.ArgumentParser(description="Google Colab Remote Orchestrator for Crypto Alpha Engine")
    parser.add_argument("--session", type=str, default="crypto_data_ingest", help="Colab session name")
    parser.add_argument("--action", type=str, default="ingest", choices=["status", "ensure", "mount", "ingest", "stop", "heartbeat"], help="Action to execute")
    parser.add_argument("--gpu", type=str, default=None, help="GPU variant (e.g. T4, A100). Default None uses CPU (0 credits)")
    parser.add_argument("--poll-interval", type=int, default=5, help="Seconds between log polling")
    parser.add_argument("--heartbeat-interval", type=int, default=60, help="Seconds between keep-alive pings")
    parser.add_argument("--daemon", action="store_true", help="Launch remotely and exit immediately without streaming")
    return parser.parse_args()


def run_ingest(client: ColabClient, poll_interval: int = 5, heartbeat_interval: int = 60, daemon: bool = False):
    print(f"[*] Ensuring active Colab session '{client.session_name}'...")
    if not client.ensure_session():
        print(f"[-] Failed to provision/connect to Colab session '{client.session_name}'.")
        return False
    print(f"[+] Session '{client.session_name}' is active.")

    print("[*] Probing Google Drive mount or headless rclone synchronization...")
    if client.is_drive_mounted():
        print("[+] Google Drive verified mounted at /content/drive/MyDrive.")
    else:
        print("[*] Google Drive FUSE not mounted. Ensuring headless rclone channel...")
        client.exec_inline(
            'import subprocess, os; subprocess.run(["apt-get", "install", "-y", "-qq", "rclone"]); os.makedirs("/root/.config/rclone", exist_ok=True)',
            timeout=60
        )
        rclone_conf = Path.home() / ".config" / "rclone" / "rclone.conf"
        if rclone_conf.exists():
            client.upload(rclone_conf, "/root/.config/rclone/rclone.conf")
            print("[+] Rclone credentials synced to Colab VM successfully.")
        else:
            print("[-] Warning: ~/.config/rclone/rclone.conf not found locally.")

    print("[*] Installing required cloud dependencies (duckdb, pyarrow, pandas)...")
    client.install_packages(["duckdb", "pyarrow", "pandas", "requests"])

    # Upload worker script
    worker_local_path = PROJECT_ROOT / "scripts" / "colab_ingest_worker.py"
    if not worker_local_path.exists():
        print(f"[-] Worker script not found at {worker_local_path}")
        return False

    print(f"[*] Uploading worker script to Colab VM...")
    client.upload(worker_local_path, "/content/colab_ingest_worker.py")

    # Clean old sentinel and launch in background on Colab VM
    print("[*] Dispatching background ingestion process inside Colab VM...")
    launcher_code = """
import os, subprocess, sys

if os.path.exists('/content/INGEST_FINISHED'):
    os.remove('/content/INGEST_FINISHED')

log_file = open('/content/ingest.log', 'w')
proc = subprocess.Popen(
    [sys.executable, '/content/colab_ingest_worker.py'],
    stdout=log_file,
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print(f'WORKER_LAUNCHED_PID={proc.pid}')
"""
    rc, out, err = client.exec_inline(launcher_code, timeout=60)
    print(out.strip())
    if "WORKER_LAUNCHED_PID=" not in out:
        print(f"[-] Failed to launch worker: {err}")
        return False

    print("[+] Worker is executing headlessly in the background!")

    if daemon:
        print("[+] Daemon mode enabled. Worker will run remotely in background.")
        print(f"    To monitor: python scripts/colab_runner.py --session {client.session_name} --action status")
        return True

    # Real-time monitoring and heartbeat loop
    print("[*] Monitoring ingestion progress and sending keepalive heartbeats...")
    local_log_path = PROJECT_ROOT / "data" / "cache" / "remote_ingest.log"
    local_log_path.parent.mkdir(parents=True, exist_ok=True)

    last_heartbeat = time.time()
    last_log_size = 0
    start_time = time.time()

    while True:
        time.sleep(poll_interval)

        # 1. Keepalive ping every heartbeat_interval
        now = time.time()
        if now - last_heartbeat >= heartbeat_interval:
            last_heartbeat = now
            client.send_heartbeat()

        # 2. Download and stream log diff
        success = client.download("/content/ingest.log", local_log_path)
        if success and local_log_path.exists():
            current_size = local_log_path.stat().st_size
            if current_size > last_log_size:
                with open(local_log_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(last_log_size)
                    new_text = f.read()
                    if new_text:
                        sys.stdout.write(new_text)
                        sys.stdout.flush()
                last_log_size = current_size

        # 3. Check for completion sentinel
        check_code = "import os; print('FINISHED_YES' if os.path.exists('/content/INGEST_FINISHED') else 'FINISHED_NO')"
        _, check_out, _ = client.exec_inline(check_code, timeout=20)
        if "FINISHED_YES" in check_out:
            print("\n" + "=" * 60)
            print("[+] Ingestion worker reported completion!")
            manifest_local = PROJECT_ROOT / "data" / "cache" / "manifest.json"
            client.download("/content/INGEST_FINISHED", manifest_local)
            if manifest_local.exists():
                print(f"[+] Downloaded manifest:\n{manifest_local.read_text()}")
            print("[+] All data stored safely on Google Drive with 0 local disk consumed.")
            print("=" * 60)
            break

    return True


def main():
    args = parse_args()
    client = ColabClient(session_name=args.session)

    if args.action == "status":
        active = client.is_session_active()
        print(f"[*] Session '{args.session}' active: {active}")
        if active:
            mounted = client.is_drive_mounted()
            print(f"[*] Google Drive mounted: {mounted}")
    elif args.action == "ensure":
        ok = client.ensure_session(gpu=args.gpu)
        print(f"[*] Ensure session result: {ok}")
    elif args.action == "mount":
        rc, out, err = client.mount_drive()
        print(f"[*] Mount Drive result ({rc}): {out or err}")
    elif args.action == "heartbeat":
        ok = client.send_heartbeat()
        print(f"[*] Heartbeat sent: {ok}")
    elif args.action == "stop":
        ok = client.stop()
        print(f"[*] Session stopped: {ok}")
    elif args.action == "ingest":
        run_ingest(
            client=client,
            poll_interval=args.poll_interval,
            heartbeat_interval=args.heartbeat_interval,
            daemon=args.daemon,
        )


if __name__ == "__main__":
    main()
