"""E2E fixtures: start the Streamlit app once per session (like Playwright's webServer).

Set E2E_BASE_URL to test an already running instance (for example the deployed Space)
instead of starting a local server. Local runs force LLM_PROVIDER=extractive so results are
deterministic and need no API key or Ollama.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]
STARTUP_TIMEOUT_S = 120


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def base_url() -> Iterator[str]:
    external = os.getenv("E2E_BASE_URL")
    if external:
        yield external.rstrip("/")
        return
    port = _free_port()
    env = os.environ | {"LLM_PROVIDER": "extractive", "HF_HUB_VERBOSITY": "error"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
         "--server.headless", "true", "--server.port", str(port),
         "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            try:
                if requests.get(f"{url}/_stcore/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            if proc.poll() is not None:
                raise RuntimeError("streamlit exited during startup")
            time.sleep(0.5)
        else:
            raise TimeoutError("streamlit did not become healthy")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture
def app(page, base_url):
    from tests.e2e.pages import CarePilotPage

    page.set_default_timeout(30_000)
    cp = CarePilotPage(page, base_url)
    cp.goto()
    return cp
