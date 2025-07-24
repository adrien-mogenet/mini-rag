"""
Core functionality for the mini-rag library.
"""

import os
import pickle
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RagifiedData:
    """Container for ragified document data."""

    def __init__(self, chunks: List[str], embeddings: np.ndarray, metadata: Dict):
        self.chunks = chunks
        self.embeddings = embeddings
        self.metadata = metadata

    def save(self, filepath: str) -> None:
        """Save ragified data to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'RagifiedData':
        """Load ragified data from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def _split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    # Handle empty or whitespace-only text
    if not text.strip():
        return [text]

    # Validate parameters
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)  # Ensure overlap is less than chunk_size

    words = text.split()
    chunks = []

    # Handle case where text has fewer words than chunk_size
    if len(words) <= chunk_size:
        return [text]

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)

        # Break if we've reached the end
        if i + chunk_size >= len(words):
            break

    return chunks


def ragify(
    input_data: Union[str, List[str]],
    model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 500,
    overlap: int = 50,
    is_file_path: bool = True
) -> RagifiedData:
    """
    Convert a large file or text into embeddings for efficient similarity search.

    Args:
        input_data: Either a file path (if is_file_path=True) or text content/list of texts
        model_name: Name of the sentence transformer model to use for embeddings
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        is_file_path: Whether input_data is a file path or direct text content

    Returns:
        RagifiedData: Object containing chunks, embeddings, and metadata

    Example:
        >>> rag_data = ragify("document.txt")
        >>> rag_data = ragify("This is some text content", is_file_path=False)
    """
    # Load the embedding model
    model = SentenceTransformer(model_name)

    # Process input data
    if is_file_path:
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"File not found: {input_data}")

            with open(input_data, 'r', encoding='utf-8') as file:
                text = file.read()

            # Split into chunks
            chunks = _split_text_into_chunks(text, chunk_size, overlap)
            source = input_data
        else:
            raise ValueError("When is_file_path=True, input_data must be a string file path")
    else:
        if isinstance(input_data, str):
            chunks = _split_text_into_chunks(input_data, chunk_size, overlap)
            source = "direct_text"
        elif isinstance(input_data, list):
            chunks = input_data  # Assume pre-chunked
            source = "direct_list"
        else:
            raise ValueError("input_data must be a string or list of strings when is_file_path=False")

    # Generate embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Create metadata
    metadata = {
        "source": source,
        "model_name": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1]
    }

    return RagifiedData(chunks, embeddings, metadata)


def search_similar(
    rag_data: RagifiedData,
    query: str,
    top_k: int = 5,
    model_name: Optional[str] = None
) -> List[Tuple[str, float]]:
    """
    Search for the most similar chunks to a given query.

    Args:
        rag_data: RagifiedData object containing chunks and embeddings
        query: Search query string
        top_k: Number of most similar chunks to return
        model_name: Model name for query embedding (uses same as ragify if None)

    Returns:
        List of tuples (chunk, score) ordered by similarity (highest first)

    Example:
        >>> results = search_similar(rag_data, "What is machine learning?", top_k=3)
        >>> for chunk, score in results:
        ...     print(f"Score: {score:.3f} - {chunk[:100]}...")
    """
    # Use the same model as used for ragification if not specified
    if model_name is None:
        model_name = rag_data.metadata["model_name"]

    # Load model and encode query
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Compute similarities
    similarities = cosine_similarity(query_embedding, rag_data.embeddings)[0]

    # Get top-k indices (limit to available chunks)
    num_chunks = len(rag_data.chunks)
    actual_top_k = min(top_k, num_chunks)
    top_indices = np.argsort(similarities)[::-1][:actual_top_k]

    # Return results as (chunk, score) tuples
    results = [
        (rag_data.chunks[idx], float(similarities[idx]))
        for idx in top_indices
    ]

    return results