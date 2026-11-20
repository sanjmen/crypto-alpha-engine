from unittest.mock import MagicMock, patch

from src.cloud.colab_client import ColabClient


def test_colab_client_init():
    client = ColabClient(session_name="test_session")
    assert client.session_name == "test_session"


@patch("subprocess.run")
def test_get_active_sessions(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="[colab] Active sessions:\nSession Name       GPU   Status\n--------------------------------\n[crypto_ingest]   None  Running\n[casmi_a100]      A100  Running\n",
        stderr="",
    )
    client = ColabClient()
    sessions = client.get_active_sessions()
    assert "crypto_ingest" in sessions
    assert "casmi_a100" in sessions
    assert len(sessions) == 2


@patch("subprocess.run")
def test_ensure_session_cpu(mock_run):
    # Case 1: already active
    mock_run.return_value = MagicMock(returncode=0, stdout="[my_session] None Running\n", stderr="")
    client = ColabClient(session_name="my_session")
    assert client.ensure_session() is True

    # Case 2: not active, create new CPU session
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="No active sessions found on server.\n", stderr=""),
        MagicMock(returncode=0, stdout="Session created\n", stderr=""),
    ]
    client2 = ColabClient(session_name="fresh_cpu")
    assert client2.ensure_session() is True
    assert mock_run.call_count == 3


@patch("subprocess.run")
def test_mount_drive(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="Mounted at /content/drive", stderr="")
    client = ColabClient(session_name="test_drive")
    rc, out, err = client.mount_drive()
    assert rc == 0
    assert "Mounted" in out


@patch("subprocess.run")
def test_send_heartbeat(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="💓 [HEARTBEAT] 2026-10-09 17:00:00", stderr="")
    client = ColabClient(session_name="test_hb")
    res = client.send_heartbeat()
    assert res is True


def test_chunked_upload_and_download_large_files(tmp_path):
    payload = b"0123456789ABCDEF" * 2048  # 32 KB synthetic binary
    src = tmp_path / "big.parquet"
    src.write_bytes(payload)

    remote_fs: dict[str, bytes] = {}
    client = ColabClient(session_name="chunk_test")

    def fake_upload_single(resolved_path, remote_path, retries=3):
        remote_fs[remote_path] = resolved_path.read_bytes()
        return True

    def fake_download_single(remote_path, resolved_path, retries=2):
        if remote_path not in remote_fs:
            return False
        resolved_path.write_bytes(remote_fs[remote_path])
        return True

    def fake_exec_inline(code_str, timeout=300):
        # Execute against a sandboxed temp root replacing '/content' with a local subdir
        rem_root = tmp_path / "remote_vm"
        rem_root.mkdir(exist_ok=True)
        for k, v in list(remote_fs.items()):
            rel = k.lstrip("/")
            fp = rem_root / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(v)
        patched_code = code_str.replace("'/content/", f"'{rem_root.as_posix()}/content/")
        from contextlib import redirect_stdout
        import io

        buf = io.StringIO()
        exec(patched_code, {}, {}) if False else None
        with redirect_stdout(buf):
            exec(compile(patched_code, "<colab>", "exec"), {})
        remote_fs.clear()
        for p in rem_root.rglob("*"):
            if p.is_file():
                rk = "/" + p.relative_to(rem_root).as_posix()
                remote_fs[rk] = p.read_bytes()
        return 0, buf.getvalue(), ""

    with (
        patch.object(client, "_upload_single", side_effect=fake_upload_single),
        patch.object(client, "_download_single", side_effect=fake_download_single),
        patch.object(client, "exec_inline", side_effect=fake_exec_inline),
    ):
        assert client.upload(src, "/content/p3_data/big.parquet", max_chunk_bytes=10_000) is True
        assert remote_fs["/content/p3_data/big.parquet"] == payload

        dst = tmp_path / "downloaded_big.parquet"
        assert client.download("/content/p3_data/big.parquet", dst, max_chunk_bytes=10_000) is True
        assert dst.read_bytes() == payload
