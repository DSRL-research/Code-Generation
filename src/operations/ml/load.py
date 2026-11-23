from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from src.exceptions import ChainError
from src.ml.sharing import model_from_record
from src.models.rows import Rows
from src.operations.base import CombinesChains, RunContext
from src.operations.ml.common import store_of

if TYPE_CHECKING:
    from src.models.chain import Chain


@dataclass(frozen=True)
class LoadOperation(CombinesChains):
    """Loads a model shared by another sandbox into the model store; the incoming rows pass through unchanged.

    JSON: {"op": "load", "from": {"source": "http://node12:8095/models/mg12_demand/export"}, "as": "mg12_demand"}
    The chain under "from" must give one record: a model as GET /models/{name}/export writes it (plain numbers, no
    code; it is checked before it is saved). Without "as", the model keeps the name in the record. A model of the
    same name is replaced. The next steps use it by name: predict, forecast, finetune.
    """

    name: ClassVar[str] = "load"
    chain_key: ClassVar[str] = "from"
    with_chain: Chain
    output: str | None = None

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        records = self.with_chain.execute(ctx).items
        if len(records) != 1:
            raise ChainError(f'the chain under "from" must give one record, a model, not {len(records)}')
        store_of(ctx).save(model_from_record(records[0], self.output))
        return rows
