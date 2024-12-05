"""General utilities for the tool api."""

import asyncio
import os
import shlex
import shutil
from datetime import datetime
from getpass import getuser
from pathlib import Path
from tempfile import mktemp
from typing import Any, AsyncIterator, Optional, Union

import asyncssh
import tomli
from aiohttp import ClientSession, ClientTimeout
from fastapi import HTTPException, status
from pydantic import ValidationError

from freva_rest.logger import logger
from freva_rest.rest import server_config

from .db import ToolConfig


class Stream:
    """
    A custom stream handler that can be used for stdout and stderr in asyncssh.run.
    It overrides the close method to do nothing, allowing the stream to remain open.
    """

    def __init__(self) -> None:
        max_size = int(os.getenv("QUEUE_MAX_SIZE", "512"))
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_size)
        self._stop_signal = asyncio.Event()

    async def write(self, data: str) -> None:
        """Put data to the qeueu"""
        if self._queue.full():  # pragma: no cover
            self._queue.get_nowait()
        self._queue.put_nowait(data)

    async def pipe_output(self, stream: Optional[AsyncIterator[bytes]]) -> None:
        """Pipe the output from the pipe into the queu."""
        if stream is not None:
            async for lines in stream:
                for line in lines.decode().splitlines():
                    await self.write(line)

    async def stream_content(self) -> AsyncIterator[str]:
        """Consume the content until the stop signal is set."""
        while not self._stop_signal.is_set():
            while not self._queue.empty():
                yield self._queue.get_nowait()
            await asyncio.sleep(1)
        while not self._queue.empty():  # pragma: no cover
            yield self._queue.get_nowait()

    def stop(self) -> None:
        """Stop the process."""
        self._stop_signal.set()

    async def close(self) -> None:
        """Mock closing a stream."""


class SSHClient:
    """
    A general-purpose SSH client for managing connections, transferring files,
    running commands, and handling lifecycle events.
    """

    def __init__(self) -> None:
        """Initialize the SSHClient."""

        self.connection_params = {
            "host": server_config.tool_host,
            "port": int(os.getenv("API_TOOL_SSH_PORT", "22")),
            "username": server_config.tool_admin_user or getuser(),
            "known_hosts": None,
        }
        if server_config.tool_admin_password:
            self.connection_params["password"] = server_config.tool_admin_password
        if server_config.tool_ssh_cert:
            ssh_cert = Path(server_config.tool_ssh_cert).expanduser().absolute()
            self.connection_params["client_keys"] = [str(ssh_cert)]
        self._proc = None
        self.connection: Optional[asyncssh.connection.SSHClientConnection] = None

    @property
    def _use_ssh(self) -> bool:
        no_ssh = os.getenv("API_NO_SSH", "")
        if no_ssh.isdigit():
            return bool(int(no_ssh)) is False
        return True

    async def __aenter__(self) -> "SSHClient":
        """
        Establish the SSH connection when entering the context.

        Returns
        -------
        SSHClient
            The initialized SSHClient instance.
        """
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """
        Close the SSH connection when exiting the context.
        """
        await self.close()

    async def connect(self) -> None:
        """
        Establish an SSH connection.
        """
        if self.connection is None and self._use_ssh:
            self.connection = await asyncssh.connect(**self.connection_params)

    async def close(self) -> None:
        """
        Close the SSH connection.
        """
        if self.connection is not None:
            self.connection.close()
            await self.connection.wait_closed()
            self.connection = None

    async def transfer_path(
        self,
        local_path: Union[str, Path],
        remote_path: Union[str, Path],
        recursive: bool = False,
    ) -> None:
        """
        Transfer a single file to the remote server.

        Parameters
        ----------
        local_path : Union[str, Path]
            Path to the local file to transfer.
        remote_path : str
            Path to the remote destination.
        recursive: bool, default: False
            Recursive file transfer
        """
        if self.connection is not None:
            await asyncssh.scp(
                local_path, (self.connection, remote_path), recurse=recursive
            )
        elif self._use_ssh is False:
            shutil.copytree(local_path, remote_path)

        else:
            raise ValueError("Either trun off ssh or ssh establish a connection.")

    async def run_command(
        self,
        command: str,
        stream: Optional[Stream] = None,
    ) -> None:
        """
        Run a command on the remote server and return its output.

        Parameters
        ----------
        command : str
            The command to execute.
        stream: io.TextIOWrapper
            The stdout/stderr to be captured

        Raises
        ------
        RuntimeError:
            if remote command fails.
        """
        if self.connection:
            await self.connection.run(
                command, stdout=stream, stderr=stream, check=True
            )
        elif self._use_ssh is False:
            stream = stream or Stream()
            process = await asyncio.create_subprocess_exec(
                *shlex.split(command),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_task = asyncio.create_task(stream.pipe_output(process.stdout))
            stderr_task = asyncio.create_task(stream.pipe_output(process.stderr))
            await process.wait()
            await stdout_task
            await stderr_task
        else:
            raise ValueError("Either trun off ssh or ssh establish a connection.")


async def clone_git_repo(
    repo_url: str, branch: str, target_dir: Union[Path, str]
) -> None:
    """Clone a Git repository of a given branch into a specified directory.

    Parameters
    ----------
    repo_url : str
        The URL of the Git repository to clone.
    branch : str
        The branch to clone.
    target_dir : Path
        The directory where the repository will be cloned.

    Raises
    ------
    RuntimeError
        If the cloning operation fails.
    """
    target_dir = Path(target_dir)

    # Ensure the target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Construct the git clone command
    cmd = [
        "git",
        "clone",
        "--recursive",
        "--branch",
        branch,
        "--single-branch",
        repo_url,
        str(target_dir),
    ]

    # Execute the command asynchronously
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(
            f"Git clone failed with exit code {process.returncode}.\n"
            f"stdout: {stdout.decode().strip()}\n"
            f"stderr: {stderr.decode().strip()}"
        )
    shutil.rmtree(target_dir / ".git")


async def download_tool(
    temp_dir: Union[str, Path],
    git_url: str,
    path: str,
    branch: str = "main",
    **kwargs: Any,
) -> ToolConfig:
    """Download the tool from a git repository and parse the tool content

    Parameters
    ----------
    temp_dir: str
        Temporary Directory where the tools content should be downloaded to.
    git_url: str,
        The git url that needs to be checked out.
    path: str,
        Path of the tool wihtin the git repository.
    branch: str, default: main
        The branch that is used for checkout.
    kwargs:
        Any extra information for the creation of the ToolConfig model.

    Returns
    -------
    ToolConfig:
        parsed tool configuration.
    """
    script_url = (
        "https://raw.githubusercontent.com/FREVA-CLINT/"
        "data-analysis-tools/refs/heads/add-example-tool/"
        "create_environment.py"
    )
    try:
        await clone_git_repo(git_url, branch, temp_dir)
    except RuntimeError as error:
        logger.error(error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not access repositroy: {error}",
        ) from None
    tool_path = Path(temp_dir) / path
    if tool_path.is_dir():
        for file in ("tool.toml", "pyproject.toml"):
            if (tool_path / file).exists():
                tool_path /= file
                break

    try:
        config = tomli.loads(tool_path.read_text())
    except tomli.TOMLDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading tool config: {error}",
        ) from None
    except (FileNotFoundError, IsADirectoryError):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not open config file: {tool_path.relative_to(temp_dir)}",
        )
    if not isinstance(config.get("tool"), dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The tool config must have a `tool` section",
        )
    tool_version = config.get("project", {}).get("version")
    tool_version = config["tool"].get("version") or tool_version
    async with ClientSession(timeout=ClientTimeout(5)) as session:
        async with session.get(script_url) as res:
            res.raise_for_status()
            (tool_path.parent / "create_environment.py").write_text(
                await res.text(), encoding="utf-8"
            )
    try:
        return ToolConfig(
            name=config["tool"].get("name", ""),
            version=tool_version or "",
            authors=config["tool"].get("authors", ""),
            summary=config["tool"].get("summary", ""),
            description=config["tool"].get("description", ""),
            title=config["tool"].get("title", ""),
            added=datetime.now(),
            parameters=[
                {"name": name, **param}  # type: ignore
                for name, param in config["tool"]
                .get("input_parameters", {})
                .items()
            ],
            command=config["tool"]["run"].get("command", ""),
            **kwargs,
        )
    except ValidationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from None


async def add_tool_on_remote_machine(
    tool: ToolConfig,
    stream: Stream,
    tool_dir: Union[str, Path],
    tool_path: Union[str, Path],
    *args: str,
) -> None:
    """Add the tool to the mongoDB.

    This methods adds a new (version of a) tool to the central tool DB as a
    background task.

    Parameters
    ----------
    tool: ToolConfig,
        The tool config that is added to the database.
    write_fd: int,
        The file descriptor for the write end of the pipe.
    tool_dir: str,
        The location where the tool is stored.
    *args: str
        Extra command line arguments
    """
    try:
        tool_dir = Path(tool_dir)
        remote_dir = Path(mktemp())
        tool_path = remote_dir / tool_path
        script_path = remote_dir / "create_environment.py"
        cmd = (
            f"{server_config.tool_python_path} {script_path} {tool_path} -p"
            f" {server_config.tool_conda_env_path} "
        ) + " ".join(args)
        conda_env = (
            Path(server_config.tool_conda_env_path)
            / tool.name
            / tool.version.strip("v")
        )
        async with SSHClient() as ssh:
            try:
                await stream.write(
                    f"Transferring {tool_dir} to {remote_dir}...\n"
                )
                await ssh.transfer_path(tool_dir, remote_dir, recursive=True)
                await stream.write(f"Executing script: {cmd}...\n")
                logger.info("Executing command %s on remote machine", cmd)
                await ssh.run_command(cmd, stream)
                tool.conda_env = str(conda_env)
                await tool.dump_to_db()
            except Exception as error:
                await stream.write("[ERROR]: could not register tool.")
                logger.error(error)
            else:
                await stream.write("Tool registering finished.")
            finally:
                try:
                    await ssh.run_command(f"rm -rf {remote_dir}", stream)
                except Exception as error:
                    logger.error(error)
                    logger.warning(f"Could not delete {remote_dir}")
    except Exception as error:
        logger.exception(error)
    finally:
        stream.stop()
