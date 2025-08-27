from abc import ABC, abstractmethod


class Agent(ABC):
    def chat(self, text: str):
        return self._chat(text)

    @abstractmethod
    def _chat(self, text: str):
        pass
