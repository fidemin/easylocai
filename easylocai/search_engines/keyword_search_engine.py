import re

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from easylocai.core.search_engine import SearchEngine, SearchEngineCollection, Record


class KeywordRecord(BaseModel):
    idx: int
    id: str
    document: str
    metadata: dict | None = None
    tokenized: list[str]


class KeywordSearchEngineCollection(SearchEngineCollection):
    def __init__(self):
        self._records: list[KeywordRecord] = []
        self._id_set = set()
        self._bm25 = None

    async def add(self, records: list[Record]):
        for idx, record in enumerate(records):
            id_ = records[idx].id

            if id_ in self._id_set:
                raise ValueError(f"Document with id {id_} already exists in the index.")
            self._id_set.add(id_)

            metadata = records[idx].metadata
            document = records[idx].document

            self._records.append(
                KeywordRecord(
                    idx=len(self._records),
                    id=id_,
                    document=document,
                    metadata=metadata,
                    tokenized=self._tokenize(document),
                )
            )

        # Rebuild BM25 index with all documents
        self._bm25 = BM25Okapi([r.tokenized for r in self._records])

    async def query(
        self, query_list: list[str], *, n_results: int
    ) -> list[list[Record]]:
        if self._bm25 is None:
            raise ValueError("The collection is empty. Add documents before querying.")

        list_of_records = []
        for query in query_list:
            tokenized_query = self._tokenize(query)
            scores = self._bm25.get_scores(tokenized_query)
            top_n_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:n_results]

            records = []
            for i in top_n_indices:
                records.append(
                    Record(
                        id=self._records[i].id,
                        document=self._records[i].document,
                        metadata=self._records[i].metadata,
                    )
                )
            list_of_records.append(records)
        return list_of_records

    def _tokenize(self, text):
        return re.sub(r"[^\w\s]", "", text.lower()).split()


class KeywordSearchEngine(SearchEngine):
    def __init__(self):
        self._collections = {}

    def get_or_create_collection(self, name: str) -> SearchEngineCollection:
        if name not in self._collections:
            self._collections[name] = SearchEngineCollection()
        return self._collections[name]
