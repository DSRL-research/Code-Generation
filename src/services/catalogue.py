"""The operator catalogue: what a code generator (an LLM, or the orchestrator) may use to write chains.

For each operator: its "op", its family (stream or ml), what it does (the first paragraph of its class
docstring, and its JSON example), and the JSON schema of its parameters. GET /operators returns it.
"""

from typing import Any

from src.models.chain_json import STEP_MODELS
from src.operations import ML_OPERATIONS, STREAM_OPERATIONS, Operation


def operator_catalogue() -> list[dict[str, Any]]:
    families = [("stream", STREAM_OPERATIONS), ("ml", ML_OPERATIONS)]
    return [_entry(operation, family) for family, operations in families for operation in operations]


def _entry(operation: type[Operation], family: str) -> dict[str, Any]:
    paragraphs = [part.strip() for part in (operation.__doc__ or "").split("\n\n")]
    return {
        "op": operation.name,
        "family": family,
        "description": " ".join(" ".join(paragraphs[:2]).split()),
        "schema": STEP_MODELS[operation.name].model_json_schema(by_alias=True),
    }
