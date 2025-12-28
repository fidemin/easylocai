from __future__ import annotations

from abc import ABC
from typing import Any, AsyncIterator, Generic, TypeVar, Type

from pydantic import BaseModel

InModel = TypeVar("InModel", bound=BaseModel)
OutModel = TypeVar("OutModel", bound=BaseModel)


class Agent(ABC, Generic[InModel, OutModel]):
    input_model: Type[InModel]
    output_model: Type[OutModel]

    async def run(self, query: str | dict | InModel) -> OutModel:
        """Validate input -> run -> validate output."""
        input_ = self._validate_input(query)
        raw = await self._run(input_)
        return self._validate_output(raw)

    async def run_stream(self, query: str | dict | InModel) -> AsyncIterator[OutModel]:
        """Validate input -> stream raw chunks -> validate each output."""
        inp = self._validate_input(query)
        async for raw in self._run_stream(inp):
            yield self._validate_output(raw)

    # TODO: Make these abstractmethods once subclasses are updated.
    async def _run(self, query: InModel) -> Any: ...

    # TODO: Make these abstractmethods once subclasses are updated.
    async def _run_stream(self, query: InModel) -> AsyncIterator[Any]: ...

    def _validate_input(self, query: str | dict | InModel) -> InModel:
        if isinstance(query, str):
            # If your input model expects structured fields, pass {"query": "..."}
            return self.input_model.model_validate({"query": query})
        return self.input_model.model_validate(query)

    def _validate_output(self, raw: Any) -> OutModel:
        return self.output_model.model_validate(raw)
