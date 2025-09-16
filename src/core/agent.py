from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    async def run(self, query: str | dict) -> str | dict:
        pass
