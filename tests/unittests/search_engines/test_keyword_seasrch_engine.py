import logging
from unittest.mock import MagicMock, patch

import pytest

from easylocai.core.search_engine import Record
from easylocai.search_engines.keyword_search_engine import (
    KeywordSearchEngine,
    SearchEngineCollection,
    KeywordSearchEngineCollection,
)

logger = logging.getLogger(__name__)


class TestKeywordSearchEngineCollection:
    @pytest.fixture
    def mock_bm25(self):
        with patch("easylocai.search_engines.keyword_search_engine.BM25Okapi") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock, mock_instance

    @pytest.fixture
    def collection(self, mock_bm25):
        return KeywordSearchEngineCollection()

    async def test_add_single_document(self, collection, mock_bm25):
        """Test adding a single document to the collection."""
        mock_cls, mock_instance = mock_bm25

        await collection.add(
            [
                Record(
                    id="doc1",
                    document="This is a test document",
                    metadata=None,
                )
            ]
        )

        assert len(collection._records) == 1
        assert collection._records[0].id == "doc1"
        assert collection._records[0].document == "This is a test document"
        assert collection._records[0].idx == 0
        mock_cls.assert_called_once()

    async def test_add_multiple_documents(self, collection, mock_bm25):
        """Test adding multiple documents at once."""
        mock_cls, mock_instance = mock_bm25

        await collection.add(
            [
                Record(
                    id="doc1",
                    document="First document about Python programming",
                    metadata=None,
                ),
                Record(
                    id="doc2",
                    document="Second document about machine learning",
                    metadata=None,
                ),
                Record(
                    id="doc3",
                    document="Third document about data science",
                    metadata=None,
                ),
            ]
        )

        assert len(collection._records) == 3
        assert collection._records[0].id == "doc1"
        assert collection._records[1].id == "doc2"
        assert collection._records[2].id == "doc3"
        mock_cls.assert_called_once()

    async def test_add_documents_with_metadata(self, collection, mock_bm25):
        """Test adding documents with metadata."""
        await collection.add(
            [
                Record(
                    id="doc1", document="Document one", metadata={"author": "Alice"}
                ),
                Record(id="doc2", document="Document two", metadata={"author": "Bob"}),
            ]
        )

        assert collection._records[0].metadata == {"author": "Alice"}
        assert collection._records[1].metadata == {"author": "Bob"}

    async def test_add_documents_without_metadata(self, collection, mock_bm25):
        """Test adding documents without metadata results in None."""
        await collection.add(
            [Record(id="doc1", document="Document without metadata", metadata=None)]
        )

        assert collection._records[0].metadata is None

    @pytest.mark.parametrize(
        "first_records,second_records",
        [
            (
                [Record(id="doc1", document="First document", metadata=None)],
                [Record(id="doc1", document="Other document", metadata=None)],
            ),
            (
                [
                    Record(id="doc1", document="First document", metadata=None),
                    Record(id="doc2", document="Second document", metadata=None),
                ],
                [
                    Record(id="doc2", document="Another document", metadata=None),
                    Record(id="doc3", document="Third document", metadata=None),
                ],
            ),
        ],
    )
    async def test_add_duplicate_id_raises_error(
        self,
        collection,
        mock_bm25,
        first_records,
        second_records,
    ):
        await collection.add(first_records)

        with pytest.raises(ValueError) as exc_info:
            await collection.add(second_records)

        logger.info(f"Raised ValueError as expected: {exc_info.value}")

    async def test_add_duplicate_id_in_batch_raises_error(self, collection, mock_bm25):
        """Test that adding documents with duplicate id in same batch raises error."""
        await collection.add(
            [Record(id="doc1", document="First document", metadata=None)]
        )

        with pytest.raises(ValueError, match="Document with id doc1 already exists"):
            await collection.add(
                [
                    Record(id="doc2", document="Second document", metadata=None),
                    Record(id="doc1", document="Duplicate document", metadata=None),
                ]
            )

    async def test_query(self, collection, mock_bm25):
        """Test querying the collection after adding documents."""
        mock_cls, mock_instance = mock_bm25
        mock_instance.get_scores.return_value = [0.3, 0.9, 0.5]

        await collection.add(
            [
                Record(
                    id="doc1", document="Python programming language", metadata=None
                ),
                Record(
                    id="doc2", document="Machine learning with Python", metadata=None
                ),
                Record(id="doc3", document="Data science and analytics", metadata=None),
            ]
        )

        result = await collection.query(["Python"], top_k=2)

        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0].id == "doc2"
        assert result[0][1].id == "doc3"
        mock_instance.get_scores.assert_called_once()

    @pytest.mark.parametrize(
        "text,expected_tokens",
        [
            ("Hello World", ["hello", "world"]),
            ("Hello, World! How are you?", ["hello", "world", "how", "are", "you"]),
            ("UPPERCASE lowercase MixedCase", ["uppercase", "lowercase", "mixedcase"]),
        ],
    )
    def test_tokenize(self, collection, mock_bm25, text, expected_tokens):
        """Test the tokenization method with various inputs."""
        tokens = collection._tokenize(text)
        assert tokens == expected_tokens

    async def test_bm25_index_is_built(self, collection, mock_bm25):
        """Test that BM25 index is built after adding documents."""
        mock_cls, mock_instance = mock_bm25

        await collection.add(
            [
                Record(
                    id="doc1",
                    document="Python is a programming language",
                    metadata=None,
                ),
                Record(
                    id="doc2", document="Machine learning uses Python", metadata=None
                ),
            ]
        )

        mock_cls.assert_called_once_with(
            [
                ["python", "is", "a", "programming", "language"],
                ["machine", "learning", "uses", "python"],
            ]
        )

    async def test_incremental_add_rebuilds_index(self, collection, mock_bm25):
        """Test that adding documents incrementally rebuilds the BM25 index."""
        mock_cls, mock_instance = mock_bm25

        await collection.add(
            [Record(id="doc1", document="First document", metadata=None)]
        )

        await collection.add(
            [Record(id="doc2", document="Second document", metadata=None)]
        )

        assert mock_cls.call_count == 2
        assert len(collection._records) == 2


class TestKeywordSearchEngine:
    @pytest.fixture
    def search_engine(self):
        return KeywordSearchEngine()

    async def test_get_or_create_collection_creates_new(self, search_engine):
        collection = await search_engine.get_or_create_collection("test_collection")

        assert isinstance(collection, SearchEngineCollection)
        assert "test_collection" in search_engine._collections

    async def test_get_or_create_collection_returns_existing(self, search_engine):
        collection1 = await search_engine.get_or_create_collection("test_collection")
        collection2 = await search_engine.get_or_create_collection("test_collection")

        assert collection1 is collection2
        assert len(search_engine._collections) == 1

    async def test_multiple_collections(self, search_engine):
        collection1 = await search_engine.get_or_create_collection("collection1")
        collection2 = await search_engine.get_or_create_collection("collection2")

        assert collection1 is not collection2
        assert "collection1" in search_engine._collections
        assert "collection2" in search_engine._collections
        assert len(search_engine._collections) == 2
