"""Start a detached dashboard and publish its platform URL."""

import os
from pathlib import Path
from http.client import HTTPConnection
import socket
import subprocess
import sys
import time


DEFAULT_PORT_FIRST = 18736
DEFAULT_PORT_LAST = 18835


def dashboard_url(port, environment=None):
    environment = os.environ if environment is None else environment
    template = environment.get("VSCODE_PROXY_URI", "")
    if "{{port}}" in template:
        return template.replace("{{port}}", str(port))
    return f"http://127.0.0.1:{port}/"


def available_port(requested=0):
    if requested and not 1 <= requested <= 65535:
        raise ValueError("dashboard port must be in [1, 65535]")
    ports = (requested,) if requested else range(DEFAULT_PORT_FIRST, DEFAULT_PORT_LAST + 1)
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                candidate.bind(("0.0.0.0", port))
            except OSError:
                continue
        return port
    if requested:
        raise RuntimeError(f"dashboard port {requested} is already in use")
    raise RuntimeError(
        f"no dashboard port is available in [{DEFAULT_PORT_FIRST}, {DEFAULT_PORT_LAST}]"
    )


def start_dashboard(run, server_script, log_path, requested_port=0, timeout=10):
    run = Path(run).resolve()
    port = available_port(requested_port)
    command = [
        sys.executable,
        "-u",
        str(Path(server_script).resolve()),
        str(run),
        "--bind",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    log_path = Path(log_path).resolve()
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    health = f"http://127.0.0.1:{port}/api/health"
    deadline = time.monotonic() + timeout
    error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"dashboard failed to start; see {log_path}")
        try:
            connection = HTTPConnection("127.0.0.1", port, timeout=0.5)
            connection.request("GET", "/api/health")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return process, {
                    "pid": process.pid,
                    "port": port,
                    "url": dashboard_url(port),
                    "health_url": health,
                    "log": str(log_path),
                    "run": str(run),
                    "started_at": time.time(),
                }
        except OSError as exception:
            error = exception
            time.sleep(0.1)
    process.terminate()
    raise TimeoutError(f"dashboard startup timed out ({error!r}); see {log_path}")
