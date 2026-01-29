from chromadb import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection

from easylocai.core.search_engine import SearchEngine, SearchEngineCollection, Record


class SemanticSearchEngineCollection(SearchEngineCollection):
    def __init__(self, chromadb_collection: AsyncCollection):
        self._chromadb_collection = chromadb_collection

    async def add(self, records: list[Record]):
        ids = [record.id for record in records]
        documents = [record.document for record in records]
        metadatas = [record.metadata for record in records]

        await self._chromadb_collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    async def query(self, queries: list[str], *, top_k: int) -> list[list[Record]]:
        results = await self._chromadb_collection.query(
            query_texts=queries,
            n_results=top_k,
        )

        list_of_records = []
        for i in range(len(queries)):
            records = []
            for j in range(len(results["ids"][i])):
                records.append(
                    Record(
                        id=results["ids"][i][j],
                        document=results["documents"][i][j],
                        metadata=results["metadatas"][i][j],
                    )
                )
            list_of_records.append(records)

        return list_of_records


class SemanticSearchEngine(SearchEngine):
    def __init__(self, chromadb_client: AsyncClientAPI):
        self._collections = {}
        self._chromadb_client = chromadb_client

    async def get_or_create_collection(self, name: str) -> SearchEngineCollection:
        if name not in self._collections:
            chromadb_collection = await self._chromadb_client.get_or_create_collection(
                name=name
            )
            self._collections[name] = SemanticSearchEngineCollection(
                chromadb_collection
            )
        return self._collections[name]
