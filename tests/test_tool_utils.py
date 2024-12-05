"""General tests for the tool utilities."""

import os
from unittest.mock import AsyncMock

import mock
import pytest
from pytest_mock import MockerFixture

from freva_rest.tool_api.db import ToolState
from freva_rest.tool_api.utils import SSHClient


async def test_ssh_connection(mocker: MockerFixture) -> None:
    """Test ssh connections."""
    mock_connect = mocker.patch("asyncssh.connect", new_callable=AsyncMock)
    mock_scp = mocker.patch("asyncssh.scp", new_callable=AsyncMock)
    mock_conn = AsyncMock()
    # Mock the connection object returned by asyncssh.connect
    mock_connect.return_value.__aenter__.return_value = mock_conn
    mock_scp.return_value.__aenter__.return_value = mock_conn

    env = os.environ.copy()
    env["API_NO_SSH"] = "bar"
    with mock.patch.dict(os.environ, env, clear=True):
        async with SSHClient() as ssh:
            assert ssh._use_ssh is True
            mock_connect.assert_called_with(**ssh.connection_params)
            assert ssh.connection is not None
            await ssh.run_command("echo Hello world")
            await ssh.transfer_path("/tmp/foo", "/tmp/bar")
            assert mock_scp.call_count > 0
        ssh = SSHClient()
        with pytest.raises(ValueError):
            await ssh.run_command("echo Hello world")
        with pytest.raises(ValueError):
            await ssh.transfer_path("/tmp/foo", "/tmp/bar")


def test_tool_state() -> None:
    """Test the state of the tools."""
    assert ToolState.get_status_from_value(-1) == ToolState.UNKOWN
    with pytest.raises(ValueError):
        ToolState.get_status_from_value(42)
