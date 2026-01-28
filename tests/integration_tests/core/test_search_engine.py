import pytest

from easylocai.core.search_engine import SearchEngineCollection


class TestSearchEngineCollection:
    @pytest.fixture
    def programming_docs_collection(self):
        """Create a collection with programming-related documents."""
        collection = SearchEngineCollection()
        collection.add(
            ids=["python-intro", "python-web", "java-intro", "rust-intro", "ml-basics"],
            documents=[
                "Python is a high-level programming language known for its simple syntax and readability. It is widely used for scripting and automation.",
                "Django and Flask are popular Python web frameworks. Django provides a full-featured MVC architecture while Flask is a lightweight microframework.",
                "Java is a statically typed object-oriented programming language. It runs on the Java Virtual Machine and is used for enterprise applications.",
                "Rust is a systems programming language focused on safety and performance. It prevents memory errors through its ownership system.",
                "Machine learning is a subset of artificial intelligence. Python libraries like scikit-learn and TensorFlow are commonly used for ML tasks.",
            ],
            metadatas=[
                {"category": "language", "difficulty": "beginner"},
                {"category": "web", "difficulty": "intermediate"},
                {"category": "language", "difficulty": "beginner"},
                {"category": "language", "difficulty": "advanced"},
                None,
            ],
        )
        return collection

    def test_search_with_queries(self, programming_docs_collection):
        result = programming_docs_collection.query(
            ["Java enterprise", "Rust memory safety", "machine learning"],
            n_results=2,
        )

        assert len(result.ids) == 3
        assert result.ids[0][0] == "java-intro"
        assert result.metadatas[0][0]["category"] == "language"

        assert result.ids[1][0] == "rust-intro"
        assert result.metadatas[1][0]["category"] == "language"

        assert result.ids[2][0] == "ml-basics"
        assert result.metadatas[2][0] is None

    def test_incremental_add_and_search(self):
        """Test adding documents incrementally and searching across all of them."""
        collection = SearchEngineCollection()

        collection.add(
            ids=["doc-1"],
            documents=["Introduction to databases and SQL queries"],
        )

        collection.add(
            ids=["doc-2", "doc-3"],
            documents=[
                "NoSQL databases like MongoDB store documents",
                "PostgreSQL is a powerful relational database",
            ],
        )

        result = collection.query(["SQL relational database"], n_results=3)

        assert len(result.ids[0]) == 3
        assert "doc-3" == result.ids[0][0]
