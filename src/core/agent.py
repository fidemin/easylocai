from abc import ABC, abstractmethod


class Agent(ABC):
    def chat(self, query: str | dict):
        return self._chat(query)

    @abstractmethod
    def _chat(self, query: str | dict):
        pass
