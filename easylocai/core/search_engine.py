from abc import ABCMeta, abstractmethod

from pydantic import BaseModel


class Record(BaseModel):
    id: str
    document: str
    metadata: dict | None = None


class SearchEngineCollection:
    @abstractmethod
    async def add(
        self,
        records: list[Record],
    ):
        pass

    @abstractmethod
    async def query(self, queries: list[str], *, top_k: int) -> list[list[Record]]:
        pass


class SearchEngine(metaclass=ABCMeta):
    @abstractmethod
    async def get_or_create_collection(self, name: str) -> SearchEngineCollection:
        pass
