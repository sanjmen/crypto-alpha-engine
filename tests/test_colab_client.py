from unittest.mock import patch, MagicMock
from pathlib import Path
import subprocess
import pytest

from src.cloud.colab_client import ColabClient


def test_colab_client_init():
    client = ColabClient(session_name="test_session")
    assert client.session_name == "test_session"


@patch("subprocess.run")
def test_get_active_sessions(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="[colab] Active sessions:\nSession Name       GPU   Status\n--------------------------------\n[crypto_ingest]   None  Running\n[casmi_a100]      A100  Running\n",
        stderr=""
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
        MagicMock(returncode=0, stdout="Session created\n", stderr="")
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
