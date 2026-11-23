"""ChainService: runs a chain written as JSON (ChainService.java, DESIGN.md section 2).

1. parse    JSON -> Chain                           ChainError (400)
2. check    Chain + policy, nothing fetched         PolicyViolation (403); NoPolicy allows everything
3. fetch    each link once                          ChainError (400) or SourceError (502)
4. execute  the steps in order                      ChainError (400); ML steps use the model store
5. reply    rows -> JSON values
"""

import logging
import time
from typing import Any

from src.clients import SourceClient
from src.operations import RunContext
from src.policy import ChainChecker, CheckReport
from src.services.chain_parser import ChainParser
from src.services.fetch_once import FetchOnce
from src.services.reply import to_reply
from src.store import ModelStore

logger = logging.getLogger(__name__)


class ChainService:
    def __init__(self, parser: ChainParser, checker: ChainChecker, client: SourceClient, models: ModelStore) -> None:
        self._parser = parser
        self._checker = checker
        self._client = client
        self._models = models

    def check(self, chain_json: Any) -> CheckReport:
        """Steps 1 and 2 only: whether the chain can be read and is allowed, and which links it would fetch."""
        return self._checker.check(self._parser.parse(chain_json))

    def run(self, chain_json: Any) -> list[dict[str, Any]]:
        """All the steps. Nothing is fetched until the chain has been parsed and has passed the policy."""
        started = time.perf_counter()
        chain = self._parser.parse(chain_json)
        self._checker.check(chain)
        rows = chain.execute(RunContext(fetch=FetchOnce(self._client.fetch), models=self._models))
        reply = to_reply(rows.items)
        logger.info("chain on %s: %d rows in %.0f ms", chain.source, len(reply), 1000 * (time.perf_counter() - started))
        return reply
