"""The errors a chain can end with (DESIGN.md section 7).

Each error knows its HTTP reply, so the routers only translate it:

| Error           | Status | Body                | When                                                     |
|-----------------|--------|---------------------|----------------------------------------------------------|
| ChainError      | 400    | {"error": "..."}    | the chain cannot be read or run, or a source refuses it  |
| PolicyViolation | 403    | {"rejected": "..."} | the chain breaks the owner's policy                      |
| SourceError     | 502    | {"error": "..."}    | a source cannot be reached, fails, or times out          |
"""

from typing import ClassVar, Self


class SandboxError(Exception):
    """Base class of the errors above.

    `where` names the part of the chain the error is about, for example "steps[1].with.steps[0] reduce", so the
    caller (often an LLM) can fix that part. It is set by the code that knows the position (see `at`).
    """

    status: ClassVar[int] = 400
    reply_key: ClassVar[str] = "error"

    def __init__(self, message: str, where: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.where = where

    def at(self, where: str) -> Self:
        """Names the part of the chain, unless a deeper part was named already. Returns the same error."""
        if self.where is None:
            self.where = where
        return self

    def __str__(self) -> str:
        return f"{self.where}: {self.message}" if self.where else self.message

    def reply(self) -> dict[str, str]:
        """The JSON body of the HTTP reply."""
        return {self.reply_key: str(self)}


class ChainError(SandboxError):
    """The chain cannot be read or run, or a source refused it (400)."""


class PolicyViolation(SandboxError):
    """The chain breaks the owner's policy (403). Raised before anything is fetched."""

    status = 403
    reply_key = "rejected"


class SourceError(SandboxError):
    """A source cannot be reached, replied 5xx, or did not reply in time (502)."""

    status = 502


class PolicyFileError(ValueError):
    """policy.json cannot be read. Raised at startup, never in reply to a chain."""
