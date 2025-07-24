"""
Unit tests for mini-rag core functionality.
"""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from mini_rag.core import ragify, search_similar, RagifiedData, _split_text_into_chunks


class TestTextChunking:
    """Test the text chunking functionality."""

    def test_split_text_basic(self):
        """Test basic text splitting into chunks."""
        text = "This is a test sentence with multiple words to split into chunks."
        chunks = _split_text_into_chunks(text, chunk_size=5, overlap=2)

        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_text_overlap(self):
        """Test that overlapping chunks share words."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = _split_text_into_chunks(text, chunk_size=4, overlap=2)

        # Should have overlapping words between chunks
        assert len(chunks) >= 2

    def test_split_text_empty(self):
        """Test splitting empty text."""
        chunks = _split_text_into_chunks("", chunk_size=5, overlap=2)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_split_text_single_chunk(self):
        """Test text shorter than chunk size."""
        text = "short text"
        chunks = _split_text_into_chunks(text, chunk_size=10, overlap=2)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestRagifiedData:
    """Test the RagifiedData class."""

    def setup_method(self):
        """Set up test data."""
        self.chunks = ["chunk1", "chunk2", "chunk3"]
        self.embeddings = np.random.rand(3, 384)  # Mock embeddings
        self.metadata = {
            "source": "test",
            "model_name": "test-model",
            "chunk_size": 100,
            "overlap": 20,
            "num_chunks": 3,
            "embedding_dim": 384
        }
        self.rag_data = RagifiedData(self.chunks, self.embeddings, self.metadata)

    def test_ragified_data_creation(self):
        """Test RagifiedData object creation."""
        assert self.rag_data.chunks == self.chunks
        assert np.array_equal(self.rag_data.embeddings, self.embeddings)
        assert self.rag_data.metadata == self.metadata

    def test_save_and_load(self):
        """Test saving and loading RagifiedData."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save data
            self.rag_data.save(tmp_path)
            assert os.path.exists(tmp_path)

            # Load data
            loaded_data = RagifiedData.load(tmp_path)

            # Verify loaded data
            assert loaded_data.chunks == self.chunks
            assert np.array_equal(loaded_data.embeddings, self.embeddings)
            assert loaded_data.metadata == self.metadata

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestRagify:
    """Test the ragify function."""

    def setup_method(self):
        """Set up test data."""
        self.sample_text = """
        This is a sample document for testing. It contains multiple sentences
        that will be split into chunks for embedding. The text should be long
        enough to create multiple chunks when processed by the ragify function.
        Machine learning and artificial intelligence are fascinating topics.
        """

    @patch('mini_rag.core.SentenceTransformer')
    def test_ragify_text_content(self, mock_transformer):
        """Test ragifying text content directly."""
        # Mock the transformer - we'll use side_effect to return embeddings matching actual chunks
        mock_model = MagicMock()
        def mock_encode(chunks, **kwargs):
            return np.random.rand(len(chunks), 384)
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        result = ragify(self.sample_text, is_file_path=False, chunk_size=20)

        assert isinstance(result, RagifiedData)
        assert len(result.chunks) > 0
        assert result.metadata['source'] == 'direct_text'
        assert result.metadata['chunk_size'] == 20
        mock_transformer.assert_called_once()
        mock_model.encode.assert_called_once()

    @patch('mini_rag.core.SentenceTransformer')
    def test_ragify_file_path(self, mock_transformer):
        """Test ragifying from file path."""
        # Mock the transformer
        mock_model = MagicMock()
        def mock_encode(chunks, **kwargs):
            return np.random.rand(len(chunks), 384)
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(self.sample_text)
            tmp_path = tmp_file.name

        try:
            result = ragify(tmp_path, is_file_path=True, chunk_size=30)

            assert isinstance(result, RagifiedData)
            assert len(result.chunks) > 0
            assert result.metadata['source'] == tmp_path
            assert result.metadata['chunk_size'] == 30

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @patch('mini_rag.core.SentenceTransformer')
    def test_ragify_list_content(self, mock_transformer):
        """Test ragifying list of text chunks."""
        # Mock the transformer
        mock_model = MagicMock()
        def mock_encode(chunks, **kwargs):
            return np.random.rand(len(chunks), 384)
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        text_list = ["First chunk of text", "Second chunk of text"]
        result = ragify(text_list, is_file_path=False)

        assert isinstance(result, RagifiedData)
        assert result.chunks == text_list
        assert result.metadata['source'] == 'direct_list'

    def test_ragify_file_not_found(self):
        """Test ragify with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ragify("non_existent_file.txt", is_file_path=True)

    def test_ragify_invalid_input_type(self):
        """Test ragify with invalid input types."""
        with pytest.raises(ValueError):
            ragify(123, is_file_path=False)  # Invalid type

        with pytest.raises(ValueError):
            ragify(["list"], is_file_path=True)  # List when expecting file path


class TestSearchSimilar:
    """Test the search_similar function."""

    def setup_method(self):
        """Set up test data."""
        self.chunks = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing deals with text and speech"
        ]
        self.embeddings = np.array([
            [0.1, 0.2, 0.3],  # ML embedding
            [0.4, 0.5, 0.6],  # DL embedding
            [0.7, 0.8, 0.9]   # NLP embedding
        ])
        self.metadata = {
            "source": "test",
            "model_name": "test-model",
            "chunk_size": 100,
            "overlap": 20,
            "num_chunks": 3,
            "embedding_dim": 3
        }
        self.rag_data = RagifiedData(self.chunks, self.embeddings, self.metadata)

    @patch('mini_rag.core.SentenceTransformer')
    def test_search_similar_basic(self, mock_transformer):
        """Test basic similarity search."""
        # Mock the transformer
        mock_model = MagicMock()
        query_embedding = np.array([[0.15, 0.25, 0.35]])  # Close to first chunk
        mock_model.encode.return_value = query_embedding
        mock_transformer.return_value = mock_model

        results = search_similar(self.rag_data, "machine learning query", top_k=2)

        assert len(results) == 2
        assert all(len(result) == 2 for result in results)  # (chunk, score) tuples
        assert all(isinstance(result[0], str) for result in results)  # chunk is string
        assert all(isinstance(result[1], float) for result in results)  # score is float

    @patch('mini_rag.core.SentenceTransformer')
    def test_search_similar_top_k_limit(self, mock_transformer):
        """Test top_k parameter limits results."""
        # Mock the transformer
        mock_model = MagicMock()
        query_embedding = np.array([[0.5, 0.5, 0.5]])
        mock_model.encode.return_value = query_embedding
        mock_transformer.return_value = mock_model

        results = search_similar(self.rag_data, "test query", top_k=1)

        assert len(results) == 1

    @patch('mini_rag.core.SentenceTransformer')
    def test_search_similar_custom_model(self, mock_transformer):
        """Test search with custom model name."""
        # Mock the transformer
        mock_model = MagicMock()
        query_embedding = np.array([[0.5, 0.5, 0.5]])
        mock_model.encode.return_value = query_embedding
        mock_transformer.return_value = mock_model

        results = search_similar(
            self.rag_data,
            "test query",
            top_k=1,
            model_name="custom-model"
        )

        mock_transformer.assert_called_with("custom-model")
        assert len(results) == 1

    @patch('mini_rag.core.SentenceTransformer')
    def test_search_similar_scores_descending(self, mock_transformer):
        """Test that results are returned in descending order of similarity."""
        # Mock the transformer to return a query that's most similar to the last chunk
        mock_model = MagicMock()
        query_embedding = np.array([[0.7, 0.8, 0.9]])  # Identical to last chunk
        mock_model.encode.return_value = query_embedding
        mock_transformer.return_value = mock_model

        results = search_similar(self.rag_data, "nlp query", top_k=3)

        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

        # First result should be the most similar (NLP chunk)
        assert "Natural language processing" in results[0][0]


class TestIntegration:
    """Integration tests combining multiple functions."""

    @patch('mini_rag.core.SentenceTransformer')
    def test_end_to_end_workflow(self, mock_transformer):
        """Test complete workflow from ragify to search."""
        # Mock the transformer
        mock_model = MagicMock()
        call_count = 0
        def mock_encode(input_data, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call for ragify
                return np.random.rand(len(input_data), 384)
            else:  # Second call for search
                return np.random.rand(1, 384)
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        # Sample text
        text = """
        Python is a programming language. Machine learning is popular in Python.
        Data science uses Python extensively. Web development can use Python frameworks.
        """

        # Ragify the text
        rag_data = ragify(text, is_file_path=False, chunk_size=10)

        # Search for similar content
        results = search_similar(rag_data, "programming", top_k=min(2, len(rag_data.chunks)))

        # Verify results
        assert len(results) <= 2
        assert len(results) > 0
        assert all(isinstance(chunk, str) and isinstance(score, float)
                  for chunk, score in results)

    @patch('mini_rag.core.SentenceTransformer')
    def test_save_load_search_workflow(self, mock_transformer):
        """Test workflow with save/load functionality."""
        # Mock the transformer
        mock_model = MagicMock()
        call_count = 0
        def mock_encode(input_data, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call for ragify
                return np.random.rand(len(input_data), 384)
            else:  # Second call for search
                return np.random.rand(1, 384)
        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        text = "Sample text for testing save and load functionality with search."

        # Create temporary file for saving
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Ragify and save
            rag_data = ragify(text, is_file_path=False, chunk_size=5)
            rag_data.save(tmp_path)

            # Load and search
            loaded_data = RagifiedData.load(tmp_path)
            results = search_similar(loaded_data, "test", top_k=1)

            # Verify
            assert len(results) >= 1
            assert isinstance(results[0][0], str)
            assert isinstance(results[0][1], float)

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)