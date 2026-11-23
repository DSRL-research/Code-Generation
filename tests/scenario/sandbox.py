"""The sandbox as the orchestrator calls it: over HTTP."""

from typing import Any

from fastapi.testclient import TestClient


class SandboxRefused(Exception):
    def __init__(self, status: int, reply: dict[str, Any]) -> None:
        super().__init__(f"{status} {reply}")
        self.status = status
        self.reply = reply


class Sandbox:
    def __init__(self, http: TestClient) -> None:
        self._http = http
        self.runs: list[list[str]] = []  # the operators of each chain that was run, in order

    def run(self, chain: dict[str, Any]) -> list[dict[str, Any]]:
        self.runs.append(operators_of(chain))
        return self._post("/chains/run", chain)

    def check(self, chain: dict[str, Any]) -> dict[str, Any]:
        return self._post("/chains/check", chain)

    def operators(self) -> set[str]:
        return {entry["op"] for entry in self._http.get("/operators").json()}

    def model(self, name: str) -> dict[str, Any]:
        return self._http.get(f"/models/{name}").json()

    def _post(self, path: str, chain: dict[str, Any]) -> Any:
        reply = self._http.post(path, json=chain)
        if reply.status_code != 200:
            raise SandboxRefused(reply.status_code, reply.json())
        return reply.json()


def operators_of(chain: Any) -> list[str]:
    """Every "op" in a chain, chains under "with" included."""
    if isinstance(chain, dict):
        own = [chain["op"]] if "op" in chain else []
        return own + [op for value in chain.values() for op in operators_of(value)]
    if isinstance(chain, list):
        return [op for value in chain for op in operators_of(value)]
    return []
