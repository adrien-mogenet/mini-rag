"""
Mini-RAG: A lightweight library for efficient knowledge base management in AI applications.

This library provides tools to convert large documents into embeddings and perform
similarity search, helping developers avoid passing large content directly to LLM prompts.
"""

from .core import ragify, search_similar

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = ["ragify", "search_similar"]