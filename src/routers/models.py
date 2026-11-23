"""The trained models as assets: their cards, and, for the models that can be shared, their export.

The export is one record of plain numbers (src/models/model_record.py), never the model file. Its link is what
another sandbox gives to the load operator: {"op": "load", "from": {"source": "<link>"}}.
"""

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.dependencies import ModelStoreDep
from src.ml.sharing import export_record, shareable

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(request: Request, store: ModelStoreDep) -> list[dict[str, Any]]:
    """The card of every model: what it predicts, from which features, how good it was, where it came from."""
    return [_with_export_link(card, request) for card in store.cards()]


@router.get("/{name}")
def get_model(name: str, request: Request, store: ModelStoreDep) -> Any:
    card = store.card(name)
    if card is None:
        return _no_model(name)
    return _with_export_link(card, request)


@router.get("/{name}/export", name="export_model")
def export_model(name: str, store: ModelStoreDep) -> Any:
    """The model as a source of one record: its card and the numbers to predict with. 400 for a model that cannot
    be shared yet (gbr)."""
    if store.card(name) is None:
        return _no_model(name)
    return [export_record(store.load(name))]


def _with_export_link(card: dict[str, Any], request: Request) -> dict[str, Any]:
    """The card, with "export": the link of its export, if the model can be shared."""
    if not shareable(card):
        return card
    return {**card, "export": str(request.url_for("export_model", name=card["name"]))}


def _no_model(name: str) -> JSONResponse:
    return JSONResponse({"error": f"no model named {name}"}, status_code=404)
