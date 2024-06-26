"""Script that runs the API server."""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
from base64 import b64encode
from pathlib import Path
from subprocess import Popen, run

REDIS_CONFIG = {
    "user": "redis",
    "passwd": "secret",
    "host": "redis://localhost:6379",
    "ssl_cert": "",
    "ssl_key": "",
    "scheduler_host": "localhost:4000",
    "cache_exp": 86000,
}

TEMP_DIR = Path(
    os.getenv("TEMP_DIR", os.path.join(tempfile.gettempdir(), "freva-nextgen"))
)
TEMP_DIR.mkdir(exist_ok=True, parents=True)
KEYCLOAK_URL = os.getenv("KEYCLOAK_HOST", "http://localhost:8080")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "freva")


def wait_for_keycloak(timeout: int = 500, time_increment: int = 10) -> None:
    """Wait for keycloak server an exit."""
    time_passed = 0
    url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/.well-known/openid-configuration"
    while time_passed < timeout:
        try:
            print(f"Trying {url}...", end="")
            conn = urllib.request.urlopen(url)
            print(f"{conn.status}")
            if conn.status == 200:
                return
        except Exception as error:
            print(error)
        time.sleep(time_increment)
        time_passed += time_increment
    raise SystemExit("Keycloak is not up.")


def kill_proc(proc: str) -> None:
    """Kill a potentially running process."""
    pid_file = TEMP_DIR / f"{proc}.pid"
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text())
            os.kill(pid, 15)
            time.sleep(2)
            if os.waitpid(pid, os.WNOHANG) == (0, 0):  # check running
                os.kill(pid, 9)
        except (ProcessLookupError, ValueError, ChildProcessError):
            pass


def prep_server(inp_dir: Path, install: bool = False) -> None:
    """Prepare the first server startup."""
    cert_dir = inp_dir / "certs"
    cert_dir.mkdir(exist_ok=True, parents=True)
    key_file = cert_dir / "client-key.pem"
    cert_file = cert_dir / "client-cert.pem"
    if not key_file.is_file() or not cert_file.is_file():
        if install:
            run(
                [sys.executable, "-m", "pip", "install", "cryptography"],
                check=True,
            )
        run([sys.executable, str(inp_dir / "keys.py")], check=True)
    REDIS_CONFIG["ssl_key"] = key_file.read_text()
    REDIS_CONFIG["ssl_cert"] = cert_file.read_text()
    config_file = TEMP_DIR / "data-portal-cluster-config.json"
    config_file.write_bytes(b64encode(json.dumps(REDIS_CONFIG).encode("utf-8")))


def start_server(inp_dir: Path, foreground: bool = False, *args: str) -> None:
    """Set up the server"""
    for proc in ("rest-server", "data-portal"):
        kill_proc(proc)
    prep_server(inp_dir, install=True)
    config_file = TEMP_DIR / "data-portal-cluster-config.json"
    args += ("--cert-dir", str(inp_dir.absolute() / "certs"))
    python_exe = sys.executable
    portal_pid = TEMP_DIR / "data-portal.pid"
    rest_pid = TEMP_DIR / "rest-server.pid"
    try:
        portal_proc = Popen(
            [
                python_exe,
                "-m",
                "data_portal_worker",
                "-v",
                "--dev",
                "-c",
                f"{config_file}",
            ]
        )
        rest_proc = Popen([python_exe, "-m", "freva_rest.cli"] + list(args))
        portal_pid.write_text(str(portal_proc.pid))
        rest_pid.write_text(str(rest_proc.pid))
        if foreground:
            portal_proc.communicate()
            rest_proc.communicate()
    except KeyboardInterrupt:
        portal_proc.kill()
        portal_pid.unlink()
        rest_proc.kill()
        rest_pid.unlink()


def cli() -> None:
    """Setup a cli."""
    parser = argparse.ArgumentParser(
        description=("Start the dev restAPI."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--script-dir",
        type=Path,
        help="The the input directory all scripts are located.",
        default=Path(__file__).parent / "dev-env",
    )
    parser.add_argument(
        "--gen-certs",
        action="store_true",
        help="Generate a pair of self-signed certificats.",
    )
    parser.add_argument(
        "--kill",
        "-k",
        help="Kill any running processes.",
        action="store_true",
    )
    parser.add_argument(
        "--foreground",
        "-f",
        help="Start service in the foreground",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--wait-for-keycloak",
        help="Wait for keycloak and exit.",
        action="store_true",
    )
    args, server_args = parser.parse_known_args()
    if args.gen_certs:
        prep_server(args.script_dir, False)
        return
    if args.wait_for_keycloak:
        wait_for_keycloak()
        return
    if args.kill:
        for proc in ("rest-server", "data-portal"):
            kill_proc(proc)
        return
    wait_for_keycloak()
    start_server(args.script_dir, args.foreground, *server_args)


if __name__ == "__main__":
    cli()
