from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mini-rag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A lightweight library for efficient knowledge base management in AI applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mini-rag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    keywords="rag, embeddings, similarity-search, nlp, ai, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/mini-rag/issues",
        "Source": "https://github.com/yourusername/mini-rag",
    },
)