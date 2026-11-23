from typing import Any

from fastapi import APIRouter

from src.services.catalogue import operator_catalogue

router = APIRouter(tags=["operators"])


@router.get("/operators")
def list_operators() -> list[dict[str, Any]]:
    """The operators a chain may use: what each does, and the JSON schema of its parameters."""
    return operator_catalogue()
