"""
Google Colab CLI Client for Remote Compute & Data Orchestration.
Encapsulates session provisioning, Drive mounting, file transfers,
remote script execution, keep-alive heartbeat, and clean lifecycle management.
"""

from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time


class ColabClient:
    """
    Client for interacting with Google Colab runtimes via the official colab CLI.
    Enables headless operations, background execution, and zero-idle disconnects.
    """

    def __init__(self, session_name: str = "crypto_data_ingest", colab_bin: str = "colab") -> None:
        self.session_name = session_name
        local_bin = Path.home() / ".local" / "bin" / "colab"
        self.colab_bin = shutil.which(colab_bin) or (str(local_bin) if local_bin.exists() else colab_bin)

    def get_active_sessions(self) -> list[str]:
        """Returns list of active Colab session names parsed from `colab status`."""
        try:
            res = subprocess.run([self.colab_bin, "status"], capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return []
        lines = (res.stdout + res.stderr).splitlines()
        sessions: list[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith("[") and "]" in line and not line.startswith("[colab]"):
                name = line[1 : line.index("]")].strip()
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
        _, out, _ = self.exec_inline(code, timeout=90)
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

    def _upload_single(self, resolved_path: Path, remote_path: str, retries: int = 3) -> bool:
        """Uploads a single file (`<= 6 MB`) to Colab with transient-error retries."""
        for attempt in range(max(1, retries)):
            try:
                res = subprocess.run(
                    [self.colab_bin, "upload", "-s", self.session_name, str(resolved_path), remote_path],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if res.returncode == 0:
                    return True
            except subprocess.TimeoutExpired:
                pass
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
        return False

    def upload(
        self,
        local_path: str | Path,
        remote_path: str,
        max_chunk_bytes: int = 6 * 1024 * 1024,
    ) -> bool:
        """
        Uploads a local file to the Colab session.
        Automatically splits files larger than `max_chunk_bytes` (`6 MB` default) into binary parts
        to avoid Jupyter `/api/contents` HTTP `413 Payload Too Large` proxy rejections, then
        reassembles and byte-verifies on the remote VM.
        """
        resolved_path = Path(local_path).resolve()
        if not resolved_path.exists():
            return False
        total_bytes = int(resolved_path.stat().st_size)
        if total_bytes <= max_chunk_bytes:
            return self._upload_single(resolved_path, remote_path)

        with tempfile.TemporaryDirectory(prefix="colab_up_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            part_remotes: list[str] = []
            with resolved_path.open("rb") as f_in:
                idx = 0
                while True:
                    buf = f_in.read(max_chunk_bytes)
                    if not buf:
                        break
                    part_local = tmp_root / f"part_{idx:04d}.bin"
                    part_local.write_bytes(buf)
                    part_remote = f"{remote_path}.part_{idx:04d}"
                    if not self._upload_single(part_local, part_remote):
                        return False
                    part_remotes.append(part_remote)
                    idx += 1

        join_py = (
            "import os\n"
            f"parts = {part_remotes!r}\n"
            f"dst = {remote_path!r}\n"
            "os.makedirs(os.path.dirname(dst) or '/', exist_ok=True)\n"
            "with open(dst, 'wb') as out_f:\n"
            "    for p in parts:\n"
            "        with open(p, 'rb') as in_f:\n"
            "            out_f.write(in_f.read())\n"
            "        os.remove(p)\n"
            "print(f'JOINED_BYTES={os.path.getsize(dst)}')\n"
        )
        rc, out, _ = self.exec_inline(join_py, timeout=180)
        return rc == 0 and f"JOINED_BYTES={total_bytes}" in out

    def sync_codebase(
        self,
        repo_root: str | Path,
        subdirs: tuple[str, ...] = ("src", "scripts"),
        remote_root: str = "/content",
    ) -> bool:
        """Bundles `src/` and `scripts/` into a single compressed tarball and extracts on the Colab VM."""
        root = Path(repo_root).resolve()
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            bundle_path = Path(tmp.name)
        try:
            with tarfile.open(bundle_path, "w:gz") as tf:
                for sub in subdirs:
                    sub_dir = root / sub
                    if not sub_dir.exists():
                        continue
                    for p in sub_dir.rglob("*"):
                        if "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
                            continue
                        if p.is_file():
                            tf.add(p, arcname=str(p.relative_to(root)))
            remote_tar = f"{remote_root.rstrip('/')}/_code_bundle.tar.gz"
            if not self.upload(bundle_path, remote_tar):
                return False
            untar_py = (
                "import os, tarfile; "
                f"os.makedirs({remote_root!r}, exist_ok=True); "
                f"tarfile.open({remote_tar!r}, 'r:gz').extractall({remote_root!r}); "
                f"os.remove({remote_tar!r}); "
                "print('BUNDLE_SYNC_OK')"
            )
            rc, out, _ = self.exec_inline(untar_py, timeout=120)
            return rc == 0 and "BUNDLE_SYNC_OK" in out
        finally:
            bundle_path.unlink(missing_ok=True)

    def _download_single(self, remote_path: str, resolved_path: Path, retries: int = 2) -> bool:
        """Downloads a single file (`<= 6 MB`) from Colab."""
        for attempt in range(max(1, retries)):
            try:
                res = subprocess.run(
                    [self.colab_bin, "download", "-s", self.session_name, remote_path, str(resolved_path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if res.returncode == 0 and resolved_path.exists():
                    return True
            except subprocess.TimeoutExpired:
                pass
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
        return False

    def download(
        self,
        remote_path: str,
        local_path: str | Path,
        max_chunk_bytes: int = 6 * 1024 * 1024,
    ) -> bool:
        """
        Downloads a remote file from the Colab session.
        Checks remote size first when above chunk threshold or falls back to remote `6 MB`
        chunked splitting so `20 MB+` Parquet artifacts download reliably without proxy truncation.
        """
        resolved_path = Path(local_path).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        probe_py = (
            "import os\n"
            f"p = {remote_path!r}\n"
            "print(f'RSIZE={os.path.getsize(p)}' if os.path.exists(p) else 'RSIZE=-1')\n"
        )
        rc_p, out_p, _ = self.exec_inline(probe_py, timeout=60)
        rsize = -1
        if rc_p == 0:
            for ln in out_p.splitlines():
                if ln.startswith("RSIZE="):
                    rsize = int(ln.split("=", 1)[1].strip())
        if rsize < 0:
            return False
        if rsize <= max_chunk_bytes:
            return self._download_single(remote_path, resolved_path, retries=3)

        split_py = (
            "import os\n"
            f"src = {remote_path!r}\n"
            f"csz = {int(max_chunk_bytes)}\n"
            "n_parts = 0\n"
            "with open(src, 'rb') as f_in:\n"
            "    while True:\n"
            "        b = f_in.read(csz)\n"
            "        if not b:\n"
            "            break\n"
            "        with open(f'{src}.dlpart_{n_parts:04d}', 'wb') as f_out:\n"
            "            f_out.write(b)\n"
            "        n_parts += 1\n"
            "print(f'N_PARTS={n_parts}')\n"
        )
        rc_s, out_s, _ = self.exec_inline(split_py, timeout=180)
        n_parts = 0
        if rc_s == 0:
            for ln in out_s.splitlines():
                if ln.startswith("N_PARTS="):
                    n_parts = int(ln.split("=", 1)[1].strip())
        if n_parts <= 0:
            return False

        try:
            with tempfile.TemporaryDirectory(prefix="colab_dl_") as tmp_dir:
                tmp_root = Path(tmp_dir)
                with resolved_path.open("wb") as f_out:
                    for idx in range(n_parts):
                        p_loc = tmp_root / f"dlpart_{idx:04d}.bin"
                        if not self._download_single(f"{remote_path}.dlpart_{idx:04d}", p_loc, retries=3):
                            return False
                        f_out.write(p_loc.read_bytes())
            return resolved_path.exists() and int(resolved_path.stat().st_size) == rsize
        finally:
            clean_py = (
                "import glob, os\n"
                f"for p in glob.glob({remote_path!r} + '.dlpart_*'):\n"
                "    try: os.remove(p)\n"
                "    except OSError: pass\n"
            )
            self.exec_inline(clean_py, timeout=60)

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
