from typing import Annotated

from fastapi import Depends, Request

from src.services.chain_service import ChainService
from src.store import ModelStore


def get_chain_service(request: Request) -> ChainService:
    return request.app.state.chain_service


def get_model_store(request: Request) -> ModelStore:
    return request.app.state.model_store


# Use them as parameter types in a route, e.g. `service: ChainServiceDep`, and FastAPI passes the objects created
# at startup.
ChainServiceDep = Annotated[ChainService, Depends(get_chain_service)]
ModelStoreDep = Annotated[ModelStore, Depends(get_model_store)]
