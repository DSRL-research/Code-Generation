"""Fixtures: a local source server on a free port, and the app with the policy switched on (`client`) or off
(`sandbox`, as the app runs by default).

The source server answers:
  /microgrids/<n>/readings?start=&end=   hourly readings of MG<n> (tests/resources/microgrids/), like the data service
  /microgrids/<n>/<name>                 tests/resources/microgrids/MG<n>/<name>.json, e.g. MG2's demand_forecast
  /files/<name>    tests/resources/sources/<name>.json
  /readings        2 days of 5-minute readings (tests/utils.py make_readings); raw in the test policy
  /readings-gap    the same without the reading of 2026-07-29T00:05:00
  /status/<code>   that status, with {"error": "refused by the test server"}
  /slow            replies after 2 s
  /redirect        302 to /files/battery_flex
  /not-json        text
  /many            1,001 records
  /big             a reply of about 300 KB
"""

import json
import threading
import time
from collections import Counter
from collections.abc import Iterator
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.main import create_app
from tests.utils import RESOURCES, make_readings

MICROGRIDS = RESOURCES / "microgrids"


class SourceServer:
    def __init__(self, server: ThreadingHTTPServer) -> None:
        self.base_url = f"http://127.0.0.1:{server.server_port}"
        self.requests: Counter[str] = Counter()

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"


def microgrid_reply(path: str, query: str) -> tuple[int, Any]:
    """/microgrids/<n>/readings?start=2026-07-15&end=2026-07-18 (days included), or /microgrids/<n>/<name>."""
    parts = path.strip("/").split("/")
    if len(parts) != 3:
        return 404, {"error": "use /microgrids/<n>/readings or /microgrids/<n>/<name>"}
    folder, name = MICROGRIDS / f"MG{parts[1]}", parts[2].removesuffix(".json")
    file = folder / f"{name}.json"
    if not file.exists():
        return 404, {"error": f"microgrid {parts[1]} has no {name}"}
    data = json.loads(file.read_text())
    if name != "readings":
        return 200, data
    days = {key: date.fromisoformat(values[0]) for key, values in parse_qs(query).items()}
    first, last = days.get("start", date.min), days.get("end", date.max)
    kept = [r for r in data["readings"] if first <= date.fromisoformat(r["timestamp"][:10]) <= last]
    return 200, {**data, "readings": kept}


def _handler(state: SourceServer) -> type[BaseHTTPRequestHandler]:
    readings = make_readings()
    readings_with_gap = make_readings(skip=lambda moment: moment.isoformat() == "2026-07-29T00:05:00")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            path = url.path
            state.requests[path] += 1
            if path.startswith("/microgrids/"):
                return self._reply(*microgrid_reply(path, url.query))
            if path.startswith("/files/"):
                file = RESOURCES / "sources" / f"{path.removeprefix('/files/')}.json"
                if not file.exists():
                    return self._reply(404, {"error": f"no file {file.name}"})
                return self._reply(200, json.loads(file.read_text()))
            if path == "/readings":
                return self._reply(200, readings)
            if path == "/readings-gap":
                return self._reply(200, readings_with_gap)
            if path.startswith("/status/"):
                return self._reply(int(path.removeprefix("/status/")), {"error": "refused by the test server"})
            if path == "/slow":
                time.sleep(2)
                return self._reply(200, [])
            if path == "/redirect":
                self.send_response(302)
                self.send_header("location", "/files/battery_flex")
                self.send_header("content-length", "0")
                self.end_headers()
                return None
            if path == "/not-json":
                return self._reply(200, None, body=b"plain text")
            if path == "/many":
                return self._reply(200, [{"n": n} for n in range(1001)])
            if path == "/big":
                return self._reply(200, [{"text": "x" * 1000} for _ in range(300)])
            return self._reply(404, {"error": "unknown path"})

        def _reply(self, status: int, data: Any, body: bytes | None = None) -> None:
            payload = body if body is not None else json.dumps(data).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # e.g. a broken pipe when the sandbox stops reading a reply that is too large: expected in the tests


@pytest.fixture(scope="session")
def source_server() -> Iterator[SourceServer]:
    server = _QuietServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    state = SourceServer(server)
    server.RequestHandlerClass = _handler(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state
    server.shutdown()
    server.server_close()


def _settings(tmp_path, policy_enabled: bool) -> Settings:
    return Settings(
        policy_enabled=policy_enabled,
        policy_file=RESOURCES / "policy.json",
        model_dir=tmp_path / "models",
        fetch_timeout_seconds=0.5,
        max_response_bytes=200_000,
        max_records_per_source=1000,
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    """The policy switched on: for the tests of the policy and of the error replies."""
    return _settings(tmp_path, policy_enabled=True)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:  # "with" runs the app's startup and shutdown
        yield test_client


@pytest.fixture
def sandbox(tmp_path) -> Iterator[TestClient]:
    """The app as it runs by default: the policy switched off. Its models go to a folder of the test."""
    settings = _settings(tmp_path, policy_enabled=False)
    settings = settings.model_copy(update={"max_records_per_source": 10_000})
    with TestClient(create_app(settings)) as test_client:
        yield test_client
