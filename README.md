# Mini-RAG

A lightweight Python library for efficient knowledge base management in AI applications. Mini-RAG helps developers avoid passing large content directly to LLM prompts by providing efficient document embedding and similarity search capabilities, all this without operating any vector database! A local RAG system cna help you reduce the size of the prompt significantly, and thus:

- Reduce the latency, as smaller prompts are faster to process
- Reduce costs, as you need to send fewer tokens to the LLM
- Improve the general performances of your AI application, as larger prompts are more likely to generate hallucinations.


## Features

- **Simple API**: Just two main functions - `ragify()` and `search_similar()`
- **Flexible Input**: Support for files, text strings, or pre-chunked content
- **Efficient Search**: Fast similarity search using sentence transformers
- **Persistent Storage**: Save and load ragified data to disk
- **Configurable**: Customizable chunk sizes, overlap, and embedding models

## Installation

```bash
pip install mini-rag
```

Or install from source:

```bash
git clone https://github.com/yourusername/mini-rag.git
cd mini-rag
pip install -e .
```

## Quick Start

### Basic Usage

```python
from mini_rag import ragify, search_similar

# Ragify a document (convert to embeddings)
rag_data = ragify("path/to/your/document.txt")

# Search for similar content
results = search_similar(rag_data, "your search query", top_k=5)

# Display results
for chunk, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {chunk}")
```

### Working with Text Content

```python
# Ragify text content directly (without file)
text_content = """
Your large text content here...
This could be documentation, articles, books, etc.
"""

rag_data = ragify(text_content, is_file_path=False)
results = search_similar(rag_data, "specific topic", top_k=3)
```

### Saving and Loading Ragified Data

```python
# Save ragified data for later use
rag_data = ragify("document.txt")
rag_data.save("my_knowledge_base.pkl")

# Load ragified data
from mini_rag.core import RagifiedData
loaded_rag_data = RagifiedData.load("my_knowledge_base.pkl")

# Use loaded data for search
results = search_similar(loaded_rag_data, "query")
```

### Advanced Configuration

```python
# Custom chunking and model settings
rag_data = ragify(
    "document.txt",
    model_name="all-mpnet-base-v2",  # Different embedding model
    chunk_size=300,                   # Smaller chunks
    overlap=30                        # Less overlap between chunks
)

# Search with custom model (must be compatible)
results = search_similar(
    rag_data,
    "query",
    top_k=10,
    model_name="all-mpnet-base-v2"  # Same model as used for ragify
)
```

## API Reference

### `ragify(input_data, model_name="all-MiniLM-L6-v2", chunk_size=500, overlap=50, is_file_path=True)`

Convert a document or text into embeddings for efficient similarity search.

**Parameters:**
- `input_data` (str | List[str]): File path or text content
- `model_name` (str): Sentence transformer model name
- `chunk_size` (int): Number of words per chunk
- `overlap` (int): Overlapping words between chunks
- `is_file_path` (bool): Whether input_data is a file path

**Returns:** `RagifiedData` object containing chunks, embeddings, and metadata

### `search_similar(rag_data, query, top_k=5, model_name=None)`

Search for the most similar chunks to a query.

**Parameters:**
- `rag_data` (RagifiedData): Ragified document data
- `query` (str): Search query
- `top_k` (int): Number of results to return
- `model_name` (str, optional): Model for query embedding

**Returns:** List of (chunk, similarity_score) tuples

### `RagifiedData` Class

Container for ragified document data with methods:
- `save(filepath)`: Save to disk
- `load(filepath)`: Load from disk (class method)

## Performance Tips

1. **Model Selection**: Use lightweight models like `"all-MiniLM-L6-v2"` for speed, or larger models like `"all-mpnet-base-v2"` for better accuracy
2. **Chunk Size**: Smaller chunks (200-300 words) for precise search, larger chunks (500-800 words) for more context
3. **Persistence**: Save ragified data to avoid re-embedding on every run
4. **Batch Processing**: Ragify multiple documents and combine for comprehensive search

## Use Cases

- **Documentation Search**: Index and search through large codebases or documentation
- **Content Recommendation**: Find similar articles, papers, or content pieces
- **Question Answering**: Retrieve relevant context for LLM-based Q&A systems
- **Research Assistant**: Search through research papers and academic content
- **Customer Support**: Find relevant help articles based on user queries

## Requirements

- Python >= 3.7
- sentence-transformers >= 2.2.0
- scikit-learn >= 1.3.0
- numpy >= 1.21.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.