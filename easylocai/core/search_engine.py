import re

from pydantic import BaseModel
from rank_bm25 import BM25Okapi


class SearchEngineData(BaseModel):
    idx: int
    id: str
    content: str
    metadata: dict | None = None
    tokenized: list[str]


class SearchResult(BaseModel):
    metadatas: list[list[dict | None]]
    documents: list[list[str]]
    ids: list[list[str]]


class SearchEngineCollection:
    def __init__(self):
        self._data: list[SearchEngineData] = []
        self._id_set = set()
        self._tokenized_corpus = []
        self._bm25 = None

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ):
        for idx, id_ in enumerate(ids):
            metadata = metadatas[idx] if metadatas else None
            if id_ in self._id_set:
                raise ValueError(f"Document with id {id_} already exists in the index.")

            self._id_set.add(id_)
            self._data.append(
                SearchEngineData(
                    idx=len(self._data),
                    id=id_,
                    content=documents[idx],
                    metadata=metadata,
                    tokenized=self._tokenize(documents[idx]),
                )
            )

        # Rebuild BM25 index with all documents
        self._bm25 = BM25Okapi([d.tokenized for d in self._data])

    def query(self, query_list: list[str], *, n_results: int) -> SearchResult:
        if self._bm25 is None:
            raise ValueError("The collection is empty. Add documents before querying.")

        list_of_metadatas = []
        list_of_ids = []
        list_of_documents = []
        for query in query_list:
            tokenized_query = self._tokenize(query)
            scores = self._bm25.get_scores(tokenized_query)
            top_n_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:n_results]
            top_n_data = [self._data[i] for i in top_n_indices]

            this_metadatas = [data.metadata for data in top_n_data]
            this_ids = [data.id for data in top_n_data]
            this_documents = [data.content for data in top_n_data]

            list_of_metadatas.append(this_metadatas)
            list_of_ids.append(this_ids)
            list_of_documents.append(this_documents)

        return SearchResult(
            metadatas=list_of_metadatas,
            documents=list_of_documents,
            ids=list_of_ids,
        )

    def _tokenize(self, text):
        return re.sub(r"[^\w\s]", "", text.lower()).split()


class SearchEngineClient:
    def __init__(self):
        self._collections = {}

    def get_or_create_collection(self, name: str) -> SearchEngineCollection:
        if name not in self._collections:
            self._collections[name] = SearchEngineCollection()
        return self._collections[name]
