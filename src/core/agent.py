from abc import ABC, abstractmethod
from typing import AsyncIterator


class Agent(ABC):
    async def run(self, query: str | dict) -> str | dict:
        raise NotImplementedError("Subclasses must implement this method")

    async def run_stream(self, query: str | dict) -> AsyncIterator[str | dict]:
        raise NotImplementedError("Subclasses must implement this method")
