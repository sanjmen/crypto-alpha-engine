"""
Google Colab CLI Client for Remote Compute & Data Orchestration.
Encapsulates session provisioning, Drive mounting, file transfers,
remote script execution, keep-alive heartbeat, and clean lifecycle management.
"""

from pathlib import Path
import shutil
import subprocess
import time
from typing import List, Optional, Tuple, Union


class ColabClient:
    """
    Client for interacting with Google Colab runtimes via the official colab CLI.
    Enables headless operations, background execution, and zero-idle disconnects.
    """

    def __init__(self, session_name: str = "crypto_data_ingest", colab_bin: str = "colab"):
        self.session_name = session_name
        self.colab_bin = shutil.which(colab_bin) or colab_bin

    def get_active_sessions(self) -> List[str]:
        """Returns list of active Colab session names."""
        res = subprocess.run([self.colab_bin, "sessions"], capture_output=True, text=True)
        lines = (res.stdout + res.stderr).splitlines()
        sessions = []
        for line in lines:
            line = line.strip()
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

    def ensure_session(self, gpu: Optional[str] = None) -> bool:
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

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return False
        return True

    def mount_drive(self, path: str = "/content/drive") -> Tuple[int, str, str]:
        """Mounts Google Drive on the remote Colab session."""
        cmd = [self.colab_bin, "drivemount", "-s", self.session_name, path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode, res.stdout, res.stderr

    def is_drive_mounted(self, drive_path: str = "/content/drive/MyDrive") -> bool:
        """Checks if Google Drive is currently mounted inside the session."""
        code = f"""import sys
from pathlib import Path
sys.exit(0 if Path('{drive_path}').exists() else 1)
"""
        rc, _, _ = self.exec_inline(code, timeout=30)
        return rc == 0

    def install_packages(self, packages: List[str]) -> bool:
        """Installs Python packages on the Colab VM using colab install."""
        if not packages:
            return True
        cmd = [self.colab_bin, "install", "-s", self.session_name] + packages
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0

    def upload(self, local_path: Union[str, Path], remote_path: str) -> bool:
        """Uploads a local file to the Colab session."""
        local_path = Path(local_path).resolve()
        if not local_path.exists():
            return False
        res = subprocess.run(
            [self.colab_bin, "upload", "-s", self.session_name, str(local_path), remote_path],
            capture_output=True,
            text=True,
        )
        return res.returncode == 0

    def download(self, remote_path: str, local_path: Union[str, Path]) -> bool:
        """Downloads a remote file from the Colab session."""
        local_path = Path(local_path).resolve()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            [self.colab_bin, "download", "-s", self.session_name, remote_path, str(local_path)],
            capture_output=True,
            text=True,
        )
        return res.returncode == 0

    def exec_file(self, script_path: Union[str, Path], timeout: Optional[int] = 300) -> Tuple[int, str, str]:
        """Executes a local Python script inside the remote Colab session."""
        script_path = Path(script_path).resolve()
        cmd = [self.colab_bin, "exec", "-s", self.session_name, "-f", str(script_path)]
        if timeout:
            cmd.extend(["--timeout", str(timeout)])
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout + 15 if timeout else None
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"TimeoutExpired: colab exec exceeded {timeout}s"

    def exec_inline(self, code_str: str, timeout: Optional[int] = 300) -> Tuple[int, str, str]:
        """Executes Python code directly inside the remote Colab session via temporary file."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(code_str)
            tf_path = Path(tf.name)
        try:
            return self.exec_file(tf_path, timeout=timeout)
        finally:
            tf_path.unlink(missing_ok=True)

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
        res = subprocess.run(
            [self.colab_bin, "stop", "-s", self.session_name],
            capture_output=True,
            text=True,
        )
        return res.returncode == 0
