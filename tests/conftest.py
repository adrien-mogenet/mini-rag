"""
Shared pytest fixtures and configuration for mini-rag tests.
"""

import pytest
import tempfile
import os
import numpy as np
from unittest.mock import MagicMock

from mini_rag.core import RagifiedData


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Machine learning is a subset of artificial intelligence that focuses on algorithms
    and statistical models. Deep learning uses neural networks with multiple layers
    to model complex patterns. Natural language processing enables computers to
    understand human language. Computer vision allows machines to interpret visual
    information from the world around us.
    """


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing enables computers to understand human language",
        "Computer vision allows machines to interpret visual information"
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return np.random.rand(4, 384)


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "source": "test_document.txt",
        "model_name": "all-MiniLM-L6-v2",
        "chunk_size": 100,
        "overlap": 20,
        "num_chunks": 4,
        "embedding_dim": 384
    }


@pytest.fixture
def sample_rag_data(sample_chunks, sample_embeddings, sample_metadata):
    """Sample RagifiedData object for testing."""
    return RagifiedData(sample_chunks, sample_embeddings, sample_metadata)


@pytest.fixture
def temp_file():
    """Temporary file fixture that cleans up after use."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        yield tmp_file.name

    # Cleanup
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture
def temp_pkl_file():
    """Temporary pickle file fixture that cleans up after use."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        yield tmp_file.name

    # Cleanup
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for testing."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(3, 384)
    return mock_model


@pytest.fixture
def test_file_with_content(sample_text, temp_file):
    """Create a temporary file with sample content."""
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    return temp_file