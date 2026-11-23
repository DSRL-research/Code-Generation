"""The shape of policy.json. Policy.load checks the file against it at startup."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr


class SourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match: StrictStr
    raw: StrictBool


class OperatorEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    on_raw: Any = Field(default=False, alias="onRaw")  # false, true, or the operator's own rule (rules.py)


class PolicyFile(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    description: str | None = None
    sources: list[SourceEntry]
    time_fields: list[StrictStr] = Field(alias="timeFields")
    operators: dict[StrictStr, OperatorEntry]
