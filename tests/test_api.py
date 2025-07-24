"""
Tests for the public API of mini-rag package.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Test imports from main package
from mini_rag import ragify, search_similar
from mini_rag.core import RagifiedData


class TestPublicAPI:
    """Test the public API exposed by the package."""

    def test_package_imports(self):
        """Test that main functions can be imported from package root."""
        # These imports should work without error
        from mini_rag import ragify, search_similar

        # Functions should be callable
        assert callable(ragify)
        assert callable(search_similar)

    def test_package_attributes(self):
        """Test package attributes."""
        import mini_rag

        # Check version exists
        assert hasattr(mini_rag, '__version__')
        assert isinstance(mini_rag.__version__, str)

        # Check __all__ contains expected functions
        assert hasattr(mini_rag, '__all__')
        assert 'ragify' in mini_rag.__all__
        assert 'search_similar' in mini_rag.__all__

    @patch('mini_rag.core.SentenceTransformer')
    def test_api_workflow(self, mock_transformer):
        """Test the main API workflow."""
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

        # Test text
        text = "This is a test document. It contains some sample content for testing the API."

        # Test ragify function from main package
        rag_data = ragify(text, is_file_path=False)

        assert isinstance(rag_data, RagifiedData)
        assert len(rag_data.chunks) > 0

        # Test search_similar function from main package
        results = search_similar(rag_data, "test query")

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)


class TestDocstrings:
    """Test that functions have proper docstrings."""

    def test_ragify_docstring(self):
        """Test that ragify has a proper docstring."""
        assert ragify.__doc__ is not None
        assert len(ragify.__doc__.strip()) > 0
        assert "embedding" in ragify.__doc__.lower() or "ragify" in ragify.__doc__.lower()

    def test_search_similar_docstring(self):
        """Test that search_similar has a proper docstring."""
        assert search_similar.__doc__ is not None
        assert len(search_similar.__doc__.strip()) > 0
        assert "similar" in search_similar.__doc__.lower() or "search" in search_similar.__doc__.lower()


class TestParameterValidation:
    """Test parameter validation and error handling."""

    def test_ragify_parameter_defaults(self):
        """Test ragify with default parameters."""
        # Test that default parameters are reasonable
        from inspect import signature
        sig = signature(ragify)

        # Check default values
        assert sig.parameters['model_name'].default == "all-MiniLM-L6-v2"
        assert sig.parameters['chunk_size'].default == 500
        assert sig.parameters['overlap'].default == 50
        assert sig.parameters['is_file_path'].default == True

    def test_search_similar_parameter_defaults(self):
        """Test search_similar with default parameters."""
        from inspect import signature
        sig = signature(search_similar)

        # Check default values
        assert sig.parameters['top_k'].default == 5
        assert sig.parameters['model_name'].default is None