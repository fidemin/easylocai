import logging
from unittest.mock import MagicMock, patch

import pytest

from easylocai.core.search_engine import SearchEngineClient, SearchEngineCollection

logger = logging.getLogger(__name__)


class TestSearchEngineCollection:
    @pytest.fixture
    def mock_bm25(self):
        with patch("easylocai.core.search_engine.BM25Okapi") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock, mock_instance

    @pytest.fixture
    def collection(self, mock_bm25):
        return SearchEngineCollection()

    def test_add_single_document(self, collection, mock_bm25):
        """Test adding a single document to the collection."""
        mock_cls, mock_instance = mock_bm25

        collection.add(
            ids=["doc1"],
            documents=["This is a test document"],
        )

        assert len(collection._data) == 1
        assert collection._data[0].id == "doc1"
        assert collection._data[0].content == "This is a test document"
        assert collection._data[0].idx == 0
        mock_cls.assert_called_once()

    def test_add_multiple_documents(self, collection, mock_bm25):
        """Test adding multiple documents at once."""
        mock_cls, mock_instance = mock_bm25

        collection.add(
            ids=["doc1", "doc2", "doc3"],
            documents=[
                "First document about Python programming",
                "Second document about machine learning",
                "Third document about data science",
            ],
        )

        assert len(collection._data) == 3
        assert collection._data[0].id == "doc1"
        assert collection._data[1].id == "doc2"
        assert collection._data[2].id == "doc3"
        mock_cls.assert_called_once()

    def test_add_documents_with_metadata(self, collection, mock_bm25):
        """Test adding documents with metadata."""
        collection.add(
            ids=["doc1", "doc2"],
            documents=["Document one", "Document two"],
            metadatas=[{"author": "Alice"}, {"author": "Bob"}],
        )

        assert collection._data[0].metadata == {"author": "Alice"}
        assert collection._data[1].metadata == {"author": "Bob"}

    def test_add_documents_without_metadata(self, collection, mock_bm25):
        """Test adding documents without metadata results in None."""
        collection.add(
            ids=["doc1"],
            documents=["Document without metadata"],
        )

        assert collection._data[0].metadata is None

    @pytest.mark.parametrize(
        "first_ids,first_documents,second_ids,second_documents",
        [
            (
                ["doc1"],
                ["First document"],
                ["doc1"],
                ["Other document"],
            ),
            (
                ["doc1", "doc2"],
                ["First document", "Second document"],
                ["doc2", "doc3"],
                ["Another document", "Third document"],
            ),
        ],
    )
    def test_add_duplicate_id_raises_error(
        self,
        collection,
        mock_bm25,
        first_ids,
        first_documents,
        second_ids,
        second_documents,
    ):
        collection.add(
            ids=first_ids,
            documents=first_documents,
        )

        with pytest.raises(ValueError) as exc_info:
            collection.add(
                ids=second_ids,
                documents=second_documents,
            )

        logger.info(f"Raised ValueError as expected: {exc_info.value}")

    def test_add_duplicate_id_in_batch_raises_error(self, collection, mock_bm25):
        """Test that adding documents with duplicate id in same batch raises error."""
        collection.add(
            ids=["doc1"],
            documents=["First document"],
        )

        with pytest.raises(ValueError, match="Document with id doc1 already exists"):
            collection.add(
                ids=["doc2", "doc1"],
                documents=["Second document", "Duplicate document"],
            )

    def test_query(self, collection, mock_bm25):
        """Test querying the collection after adding documents."""
        mock_cls, mock_instance = mock_bm25
        mock_instance.get_scores.return_value = [0.3, 0.9, 0.5]

        collection.add(
            ids=["doc1", "doc2", "doc3"],
            documents=[
                "Python programming language",
                "Machine learning with Python",
                "Data science and analytics",
            ],
        )

        result = collection.query(["Python"], n_results=2)

        assert len(result.ids) == 1
        assert len(result.ids[0]) == 2
        assert result.ids[0][0] == "doc2"
        assert result.ids[0][1] == "doc3"
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

    def test_bm25_index_is_built(self, collection, mock_bm25):
        """Test that BM25 index is built after adding documents."""
        mock_cls, mock_instance = mock_bm25

        collection.add(
            ids=["doc1", "doc2"],
            documents=[
                "Python is a programming language",
                "Machine learning uses Python",
            ],
        )

        mock_cls.assert_called_once_with(
            [
                ["python", "is", "a", "programming", "language"],
                ["machine", "learning", "uses", "python"],
            ]
        )

    def test_incremental_add_rebuilds_index(self, collection, mock_bm25):
        """Test that adding documents incrementally rebuilds the BM25 index."""
        mock_cls, mock_instance = mock_bm25

        collection.add(
            ids=["doc1"],
            documents=["First document"],
        )

        collection.add(
            ids=["doc2"],
            documents=["Second document"],
        )

        assert mock_cls.call_count == 2
        assert len(collection._data) == 2


class TestSearchEngineClient:
    @pytest.fixture
    def client(self):
        return SearchEngineClient()

    def test_get_or_create_collection_creates_new(self, client):
        collection = client.get_or_create_collection("test_collection")

        assert isinstance(collection, SearchEngineCollection)
        assert "test_collection" in client._collections

    def test_get_or_create_collection_returns_existing(self, client):
        collection1 = client.get_or_create_collection("test_collection")
        collection2 = client.get_or_create_collection("test_collection")

        assert collection1 is collection2
        assert len(client._collections) == 1

    def test_multiple_collections(self, client):
        collection1 = client.get_or_create_collection("collection1")
        collection2 = client.get_or_create_collection("collection2")

        assert collection1 is not collection2
        assert "collection1" in client._collections
        assert "collection2" in client._collections
        assert len(client._collections) == 2
