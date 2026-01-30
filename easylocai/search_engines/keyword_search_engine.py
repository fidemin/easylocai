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
    def __init__(
        self,
        min_ngram: int | None = None,
        max_ngram: int | None = None,
    ):
        self._records: list[KeywordRecord] = []
        self._id_set = set()
        self._bm25 = None
        self._min_ngram = min_ngram
        self._max_ngram = max_ngram

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

    async def query(self, query_list: list[str], *, top_k: int) -> list[list[Record]]:
        if self._bm25 is None:
            raise ValueError("The collection is empty. Add documents before querying.")

        list_of_records = []
        for query in query_list:
            tokenized_query = self._tokenize(query)
            scores = self._bm25.get_scores(tokenized_query)
            top_n_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]

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

    def _tokenize(self, text: str) -> list[str]:
        """
        단어별 문자 N-gram 토큰화 (BM25를 위해 중복 보존)
        """
        # 1. 전처리: 마침표 분리 및 특수문자 제거
        clean_text = text.replace(".", " ")
        clean_text = re.sub(r"[^\w\s]", "", clean_text.lower())

        words = clean_text.split()

        all_tokens = []
        for word in words:
            # keep original word
            all_tokens.append(word)

            if self._min_ngram is None or self._max_ngram is None:
                continue

            if len(word) < self._min_ngram:
                continue

            for n in range(self._min_ngram, self._max_ngram + 1):
                for i in range(len(word) - n + 1):
                    ngram = word[i : i + n]
                    all_tokens.append(ngram)

        return all_tokens


class KeywordSearchEngine(SearchEngine):
    def __init__(self):
        self._collections = {}

    async def get_or_create_collection(
        self, name: str, **kwargs
    ) -> SearchEngineCollection:
        if name not in self._collections:
            self._collections[name] = KeywordSearchEngineCollection(
                kwargs.get("min_ngram"), kwargs.get("max_ngram")
            )
        return self._collections[name]
