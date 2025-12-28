from __future__ import annotations

from abc import ABC
from typing import AsyncIterator, Generic, TypeVar, Type

from pydantic import BaseModel

InModel = TypeVar("InModel", bound=BaseModel)
OutModel = TypeVar("OutModel", bound=BaseModel)


class Agent(ABC, Generic[InModel, OutModel]):
    input_model: Type[InModel]
    output_model: Type[OutModel]

    async def run(self, input_: InModel) -> OutModel:
        """Validate input -> run -> validate output."""
        return await self._run(input_)

    async def run_stream(self, input_: str | dict | InModel) -> AsyncIterator[OutModel]:
        """Validate input -> stream raw chunks -> validate each output."""
        async for output in self._run_stream(input_):
            yield output

    # TODO: Make these abstractmethods once subclasses are updated.
    async def _run(self, query: InModel) -> OutModel: ...

    # TODO: Make these abstractmethods once subclasses are updated.
    async def _run_stream(self, query: InModel) -> AsyncIterator[OutModel]: ...
