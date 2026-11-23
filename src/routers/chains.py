"""POST /chains/run and POST /chains/check (ChainController.java, DESIGN.md section 7).

The routes only call the service. Errors become replies in routers/errors.py. The routes are plain `def`
functions: FastAPI runs them in a thread pool, so a run that fetches sources does not block other requests.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Body

from src.dependencies import ChainServiceDep

router = APIRouter(prefix="/chains", tags=["chains"])

ChainBody = Annotated[Any, Body(description='A chain: {"source": "http://...", "steps": [...]}')]


@router.post("/run")
def run_chain(service: ChainServiceDep, chain: ChainBody = None) -> list[dict[str, Any]]:
    """Checks the chain against the policy, fetches its sources, applies its steps, and returns the rows."""
    return service.run(chain)


@router.post("/check")
def check_chain(service: ChainServiceDep, chain: ChainBody = None) -> dict[str, Any]:
    """Checks the chain against the policy without fetching anything, and lists the links it would fetch."""
    return service.check(chain).to_json()
