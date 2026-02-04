# GG QA - PDF Question Answering System

A Python library for querying PDFs using vector search and LLM integration.

## Features

- **PDF Text Extraction**: Extract text from PDFs at page level using PyMuPDF
- **Text Search**: Search PDF content with optional fuzzy matching
- **Vector Embeddings**: Embed PDF pages for semantic search
- **In-Memory Vector Store**: NumPy-based similarity search (stand-in for production search API)
- **Embedding Cache**: Cache embeddings to disk for efficiency
- **LLM Integration**: Query PDFs using PydanticAI with OpenRouter

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from gg_qa import PDFDocument, EmbeddingModel, VectorStore, QAEngine

# Load a PDF
doc = PDFDocument.from_file("document.pdf")

# Create embeddings
embedder = EmbeddingModel()
embeddings = embedder.embed_pages(doc.pages)

# Build vector store
store = VectorStore()
store.add(doc.pages, embeddings)

# Query with LLM
engine = QAEngine(vector_store=store, document=doc)
answer = await engine.query("What is the main topic of this document?")
print(answer)
```

## Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key for LLM queries
- `EMBEDDING_MODEL`: (Optional) Embedding model to use (default: text-embedding-3-small)

## License

MIT
