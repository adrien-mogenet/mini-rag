#!/usr/bin/env python3
"""
Example script demonstrating the mini-rag library usage.

This script shows how to:
1. Ragify text content
2. Search for similar content
3. Save and load ragified data
"""

from mini_rag import ragify, search_similar
from mini_rag.core import RagifiedData

def main():
    print("🚀 Mini-RAG Example Script")
    print("=" * 50)

    # Sample text content (simulating a large knowledge base)
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience.

    Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in areas such as image recognition, natural language processing, and speech recognition.

    Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human language in a valuable way.

    Computer vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world. It involves developing algorithms and techniques to extract meaningful information from digital images or videos.

    Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. It's inspired by behavioral psychology and has been successfully applied to game playing, robotics, and autonomous systems.

    Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, mathematics, programming, and domain expertise.
    """

    print("\n1. Ragifying sample text content...")
    print("-" * 30)

    # Ragify the sample text
    rag_data = ragify(sample_text, is_file_path=False, chunk_size=100, overlap=20)

    print(f"✅ Successfully ragified text into {rag_data.metadata['num_chunks']} chunks")
    print(f"📊 Embedding dimensions: {rag_data.metadata['embedding_dim']}")
    print(f"🤖 Model used: {rag_data.metadata['model_name']}")

    print("\n2. Sample chunks:")
    print("-" * 30)
    for i, chunk in enumerate(rag_data.chunks[:3]):  # Show first 3 chunks
        print(f"Chunk {i+1}: {chunk[:100]}...")

    print("\n3. Searching for similar content...")
    print("-" * 30)

    # Example queries
    queries = [
        "What is deep learning?",
        "How do computers understand images?",
        "Tell me about AI for games"
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = search_similar(rag_data, query, top_k=2)

        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n  Result {i} (Score: {score:.3f}):")
            print(f"  {chunk[:150]}...")

    print("\n4. Saving ragified data...")
    print("-" * 30)

    # Save ragified data to disk
    save_path = "example_rag_data.pkl"
    rag_data.save(save_path)
    print(f"✅ Saved ragified data to {save_path}")

    print("\n5. Loading ragified data...")
    print("-" * 30)

    # Load ragified data from disk
    loaded_rag_data = RagifiedData.load(save_path)
    print(f"✅ Loaded ragified data with {len(loaded_rag_data.chunks)} chunks")

    # Test search with loaded data
    test_query = "artificial intelligence algorithms"
    print(f"\n🔍 Testing loaded data with query: '{test_query}'")
    results = search_similar(loaded_rag_data, test_query, top_k=1)

    for chunk, score in results:
        print(f"\n  Best match (Score: {score:.3f}):")
        print(f"  {chunk}")

    print("\n6. Performance demonstration...")
    print("-" * 30)

    # Demonstrate different chunk sizes
    print("Testing different chunk sizes:")

    for chunk_size in [50, 100, 200]:
        rag_data_test = ragify(sample_text, is_file_path=False, chunk_size=chunk_size)
        results = search_similar(rag_data_test, "machine learning", top_k=1)
        best_score = results[0][1] if results else 0
        print(f"  Chunk size {chunk_size:3d}: {rag_data_test.metadata['num_chunks']:2d} chunks, best score: {best_score:.3f}")

    print("\n✨ Mini-RAG example completed successfully!")
    print(f"💡 Tip: You can now use the saved file '{save_path}' in your own applications!")

if __name__ == "__main__":
    main()