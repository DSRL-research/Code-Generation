"""Errors -> HTTP replies (DESIGN.md section 7). A problem in a chain never gives 500."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.exceptions import SandboxError

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(SandboxError)
    async def sandbox_error(request: Request, error: SandboxError) -> JSONResponse:
        logger.info("%s %s -> %d %s", request.method, request.url.path, error.status, error)
        return JSONResponse(error.reply(), status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def unreadable_body(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": "the body must be a chain written as a JSON object"}, status_code=400)
