import pytest

from easylocai.core.search_engine import Record
from easylocai.search_engines.keyword_search_engine import KeywordSearchEngineCollection


class TestKeywordSearchEngineCollection:
    @pytest.fixture
    async def programming_docs_collection(self):
        """Create a collection with programming-related documents."""
        collection = KeywordSearchEngineCollection()
        await collection.add(
            [
                Record(
                    id="python-intro",
                    document="Python is a high-level programming language known for its simple syntax and readability. It is widely used for scripting and automation.",
                    metadata={"category": "language", "difficulty": "beginner"},
                ),
                Record(
                    id="python-web",
                    document="Django and Flask are popular Python web frameworks. Django provides a full-featured MVC architecture while Flask is a lightweight microframework.",
                    metadata={"category": "web", "difficulty": "intermediate"},
                ),
                Record(
                    id="java-intro",
                    document="Java is a statically typed object-oriented programming language. It runs on the Java Virtual Machine and is used for enterprise applications.",
                    metadata={"category": "language", "difficulty": "beginner"},
                ),
                Record(
                    id="rust-intro",
                    document="Rust is a systems programming language focused on safety and performance. It prevents memory errors through its ownership system.",
                    metadata={"category": "language", "difficulty": "advanced"},
                ),
                Record(
                    id="ml-basics",
                    document="Machine learning is a subset of artificial intelligence. Python libraries like scikit-learn and TensorFlow are commonly used for ML tasks.",
                    metadata=None,
                ),
            ]
        )
        return collection

    async def test_search_with_queries(self, programming_docs_collection):
        result = await programming_docs_collection.query(
            ["Java enterprise", "Rust memory safety", "machine learning"],
            n_results=2,
        )

        assert len(result) == 3
        assert result[0][0].id == "java-intro"
        assert result[0][0].metadata["category"] == "language"

        assert result[1][0].id == "rust-intro"
        assert result[1][0].metadata["category"] == "language"

        assert result[2][0].id == "ml-basics"
        assert result[2][0].metadata is None

    async def test_incremental_add_and_search(self):
        """Test adding documents incrementally and searching across all of them."""
        collection = KeywordSearchEngineCollection()

        await collection.add(
            [
                Record(
                    id="doc-1",
                    document="Introduction to databases and SQL queries",
                    metadata=None,
                )
            ]
        )

        await collection.add(
            [
                Record(
                    id="doc-2",
                    document="NoSQL databases like MongoDB store documents",
                    metadata=None,
                ),
                Record(
                    id="doc-3",
                    document="PostgreSQL is a powerful relational database",
                    metadata=None,
                ),
            ]
        )

        result = await collection.query(["SQL relational database"], n_results=3)

        assert len(result[0]) == 3
        assert result[0][0].id == "doc-3"
