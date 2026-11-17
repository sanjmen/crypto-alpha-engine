"""
Google Colab CLI Client for Remote Compute & Data Orchestration.
Encapsulates session provisioning, Drive mounting, file transfers,
remote script execution, keep-alive heartbeat, and clean lifecycle management.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile


class ColabClient:
    """
    Client for interacting with Google Colab runtimes via the official colab CLI.
    Enables headless operations, background execution, and zero-idle disconnects.
    """

    def __init__(self, session_name: str = "crypto_data_ingest", colab_bin: str = "colab") -> None:
        self.session_name = session_name
        self.colab_bin = shutil.which(colab_bin) or colab_bin

    def get_active_sessions(self) -> list[str]:
        """Returns list of active Colab session names."""
        try:
            res = subprocess.run([self.colab_bin, "sessions"], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return []
        lines = (res.stdout + res.stderr).splitlines()
        sessions: list[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if (
                line
                and not line.startswith("[colab]")
                and not line.startswith("Session")
                and not line.startswith("-")
                and not line.startswith("No active")
            ):
                parts = line.split()
                if parts:
                    name = parts[0].strip("[]")
                    if name:
                        sessions.append(name)
        return sessions

    def is_session_active(self) -> bool:
        """Checks if self.session_name is currently active on Colab."""
        return self.session_name in self.get_active_sessions()

    def ensure_session(self, gpu: str | None = None) -> bool:
        """
        Ensures an active session exists.
        If gpu is None, creates a CPU runtime (0 GPU credits consumed).
        If gpu is specified (e.g. 'T4', 'A100'), creates a GPU runtime.
        """
        if self.is_session_active():
            return True

        cmd = [self.colab_bin, "new", "-s", self.session_name]
        if gpu:
            cmd.extend(["--gpu", gpu])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def mount_drive(self, path: str = "/content/drive") -> tuple[int, str, str]:
        """Mounts Google Drive on the remote Colab session."""
        cmd = [self.colab_bin, "drivemount", "-s", self.session_name, path]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "TimeoutExpired: colab drivemount exceeded 120s"

    def is_drive_mounted(self, drive_path: str = "/content/drive/MyDrive") -> bool:
        """Checks if Google Drive is currently mounted inside the session."""
        code = f'import os\nprint("MOUNTED_TRUE" if os.path.exists({drive_path!r}) else "MOUNTED_FALSE")\n'
        _, out, _ = self.exec_inline(code, timeout=30)
        return "MOUNTED_TRUE" in out

    def install_packages(self, packages: list[str]) -> bool:
        """Installs Python packages on the Colab VM using colab install."""
        if not packages:
            return True
        cmd = [self.colab_bin, "install", "-s", self.session_name, *packages]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def upload(self, local_path: str | Path, remote_path: str) -> bool:
        """Uploads a local file to the Colab session."""
        resolved_path = Path(local_path).resolve()
        if not resolved_path.exists():
            return False
        try:
            res = subprocess.run(
                [self.colab_bin, "upload", "-s", self.session_name, str(resolved_path), remote_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def download(self, remote_path: str, local_path: str | Path) -> bool:
        """Downloads a remote file from the Colab session."""
        resolved_path = Path(local_path).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            res = subprocess.run(
                [self.colab_bin, "download", "-s", self.session_name, remote_path, str(resolved_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def exec_file(self, script_path: str | Path, timeout: int | None = 300) -> tuple[int, str, str]:
        """Executes a local Python script inside the remote Colab session."""
        resolved_path = Path(script_path).resolve()
        cmd = [self.colab_bin, "exec", "-s", self.session_name, "-f", str(resolved_path)]
        if timeout:
            cmd.extend(["--timeout", str(timeout)])
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 15 if timeout else None,
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"TimeoutExpired: colab exec exceeded {timeout}s"

    def exec_inline(self, code_str: str, timeout: int | None = 300) -> tuple[int, str, str]:
        """Executes Python code directly inside the remote Colab session via temporary file."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
            temp_file.write(code_str)
            temp_path = Path(temp_file.name)
        try:
            return self.exec_file(temp_path, timeout=timeout)
        finally:
            temp_path.unlink(missing_ok=True)

    def send_heartbeat(self) -> bool:
        """
        Sends an inline keepalive ping to the Colab session.
        Prevents idle disconnection and resets Colab's activity timeout timer.
        """
        code = "import time; print(f'💓 [HEARTBEAT] {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')"
        rc, _, _ = self.exec_inline(code, timeout=45)
        return rc == 0

    def stop(self) -> bool:
        """Stops the Colab session to release resources."""
        try:
            res = subprocess.run(
                [self.colab_bin, "stop", "-s", self.session_name],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False
