from collections import defaultdict

from easylocai.core.search_engine import SearchEngine, SearchEngineCollection, Record
from easylocai.search_engines.keyword_search_engine import KeywordSearchEngine
from easylocai.search_engines.semantic_search_engine import SemanticSearchEngine


class AdvancedSearchEngineCollection(SearchEngineCollection):
    def __init__(
        self,
        keyword_collection: SearchEngineCollection,
        semantic_collection: SearchEngineCollection,
    ):
        self._keyword_collection = keyword_collection
        self._semantic_collection = semantic_collection

    async def add(self, records: list[Record]):
        await self._keyword_collection.add(records)
        await self._semantic_collection.add(records)

    async def query(
        self, queries: list[str], *, top_k: int, pool_multiplier: int = 3
    ) -> list[list[Record]]:
        local_top_k = max(top_k * pool_multiplier, 30)
        keyword_list_of_records = await self._keyword_collection.query(
            queries, top_k=local_top_k
        )
        semantic_list_of_records = await self._semantic_collection.query(
            queries, top_k=local_top_k
        )

        result = []

        for i in range(len(queries)):
            keyword_records = keyword_list_of_records[i]
            semantic_records = semantic_list_of_records[i]

            keyword_record_by_id = {record.id: record for record in keyword_records}

            semantic_record_by_id = {record.id: record for record in semantic_records}

            record_by_id = {**keyword_record_by_id, **semantic_record_by_id}

            scores_by_id = defaultdict(float)
            rrf_k_keyword = 60
            rrf_k_semantic = 60
            for rank, record in enumerate(keyword_records, 1):
                score = 1 / (rrf_k_keyword + rank)
                scores_by_id[record.id] += score

            for rank, record in enumerate(semantic_records, 1):
                score = 1 / (rrf_k_semantic + rank)
                scores_by_id[record.id] += score

            scores_sorted = sorted(
                scores_by_id.items(), key=lambda item: item[1], reverse=True
            )[:top_k]
            result.append([record_by_id[record_id] for record_id, _ in scores_sorted])

        return result


class AdvancedSearchEngine(SearchEngine):
    def __init__(self, chromadb_client):
        self._chromadb_client = chromadb_client
        self._keyword_se = KeywordSearchEngine()
        self._semantic_se = SemanticSearchEngine(chromadb_client)

    async def get_or_create_collection(
        self, name: str, **kwargs
    ) -> SearchEngineCollection:
        keyword_collection = await self._keyword_se.get_or_create_collection(
            name, **kwargs
        )
        semantic_collection = await self._semantic_se.get_or_create_collection(name)
        return AdvancedSearchEngineCollection(keyword_collection, semantic_collection)
