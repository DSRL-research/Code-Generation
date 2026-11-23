"""Two or more sandboxes in one test, each with its own model store, that reach each other by link: a node's links
are http://<node>/... (tests/test_model_sharing.py, tests/test_model_sharing_recipe.py)."""

from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from src.config import Settings


class Nodes(httpx.BaseTransport):
    """Sends the links of the other sandboxes (http://<node>/...) to their apps; any other link to the network."""

    def __init__(self, nodes: dict[str, TestClient]) -> None:
        self._nodes = nodes
        self._network = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        node = self._nodes.get(request.url.host)
        if node is None:
            return self._network.handle_request(request)
        reply = node.get(request.url.raw_path.decode("ascii"))
        return httpx.Response(reply.status_code, headers={"content-type": reply.headers["content-type"]},
                              content=reply.content)  # fmt: skip

    def close(self) -> None:
        self._network.close()


def node_settings(folder: Path) -> Settings:
    return Settings(policy_enabled=False, model_dir=folder, fetch_timeout_seconds=5, max_records_per_source=10_000)
