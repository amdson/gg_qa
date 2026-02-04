"""Embedding generation and caching module."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from openai import OpenAI

from gg_qa.pdf import PDFPage


@dataclass
class EmbeddingResult:
    """Container for embedding results."""
    
    embeddings: np.ndarray  # Shape: (n_texts, embedding_dim)
    texts: list[str]
    model: str
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 0
    
    def __len__(self) -> int:
        return len(self.texts)


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts."""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class EmbeddingModel(BaseEmbeddingModel):
    """OpenAI-compatible embedding model using OpenRouter or OpenAI API."""
    
    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        dimension: int = 1536
    ):
        """
        Initialize the embedding model.
        
        Args:
            model: Model identifier (default: OpenAI text-embedding-3-small via OpenRouter)
            api_key: API key (falls back to OPENROUTER_API_KEY or OPENAI_API_KEY env vars)
            base_url: API base URL (default: OpenRouter)
            dimension: Expected embedding dimension
        """
        self._model = model
        self._dimension = dimension
        
        # Resolve API key
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "No API key provided. Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable, "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        response = self.client.embeddings.create(
            model=self._model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def embed_texts(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """
        Embed multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            
        Returns:
            Array of embeddings with shape (len(texts), dimension)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self._model,
                input=batch
            )
            
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [d.embedding for d in sorted_data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def embed_pages(self, pages: Sequence[PDFPage]) -> EmbeddingResult:
        """
        Embed PDF pages.
        
        Args:
            pages: Sequence of PDFPage objects
            
        Returns:
            EmbeddingResult containing embeddings and metadata
        """
        texts = [page.text for page in pages]
        embeddings = self.embed_texts(texts)
        
        return EmbeddingResult(
            embeddings=embeddings,
            texts=texts,
            model=self._model
        )


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock embedding model for testing without API calls."""
    
    def __init__(self, dimension: int = 384, seed: int | None = None):
        """
        Initialize mock embedding model.
        
        Args:
            dimension: Embedding dimension
            seed: Random seed for reproducibility
        """
        self._dimension = dimension
        self._seed = seed
        self._rng = np.random.default_rng(seed)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return "mock-embedding-model"
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a deterministic pseudo-random embedding based on text content."""
        # Use hash of text for deterministic results
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = np.random.default_rng(text_hash % (2**32))
        embedding = rng.standard_normal(self._dimension).astype(np.float32)
        # Normalize to unit length
        return embedding / np.linalg.norm(embedding)
    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        return np.vstack([self.embed_text(t) for t in texts])
    
    def embed_pages(self, pages: Sequence[PDFPage]) -> EmbeddingResult:
        """Embed PDF pages."""
        texts = [page.text for page in pages]
        embeddings = self.embed_texts(texts)
        
        return EmbeddingResult(
            embeddings=embeddings,
            texts=texts,
            model=self.model_name
        )


class EmbeddingCache:
    """Cache for storing and retrieving embeddings from disk."""
    
    def __init__(self, cache_dir: str | Path = ".embedding_cache"):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.cache_dir / "metadata.json"
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load cache metadata from disk."""
        if self._metadata_file.exists():
            with open(self._metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self._metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2)
    
    def _get_cache_key(self, document_path: str, model_name: str) -> str:
        """Generate a cache key for a document and model combination."""
        combined = f"{document_path}:{model_name}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_content_hash(self, texts: list[str]) -> str:
        """Generate a hash of the text content for cache validation."""
        combined = "".join(texts)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(
        self,
        document_path: str,
        model_name: str,
        texts: list[str]
    ) -> np.ndarray | None:
        """
        Retrieve cached embeddings if available and valid.
        
        Args:
            document_path: Path to the source document
            model_name: Name of the embedding model
            texts: List of texts (for validation)
            
        Returns:
            Cached embeddings array or None if not found/invalid
        """
        cache_key = self._get_cache_key(document_path, model_name)
        
        if cache_key not in self._metadata:
            return None
        
        meta = self._metadata[cache_key]
        content_hash = self._get_content_hash(texts)
        
        # Validate content hash matches
        if meta.get("content_hash") != content_hash:
            return None
        
        # Load embeddings from disk
        embedding_file = self.cache_dir / f"{cache_key}.npy"
        if not embedding_file.exists():
            return None
        
        return np.load(embedding_file)
    
    def put(
        self,
        document_path: str,
        model_name: str,
        texts: list[str],
        embeddings: np.ndarray
    ):
        """
        Store embeddings in the cache.
        
        Args:
            document_path: Path to the source document
            model_name: Name of the embedding model
            texts: List of texts that were embedded
            embeddings: Embedding array to cache
        """
        cache_key = self._get_cache_key(document_path, model_name)
        content_hash = self._get_content_hash(texts)
        
        # Save embeddings
        embedding_file = self.cache_dir / f"{cache_key}.npy"
        np.save(embedding_file, embeddings)
        
        # Update metadata
        self._metadata[cache_key] = {
            "document_path": document_path,
            "model_name": model_name,
            "content_hash": content_hash,
            "num_embeddings": len(embeddings),
            "dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else 0
        }
        self._save_metadata()
    
    def invalidate(self, document_path: str, model_name: str):
        """Remove cached embeddings for a document/model combination."""
        cache_key = self._get_cache_key(document_path, model_name)
        
        if cache_key in self._metadata:
            del self._metadata[cache_key]
            self._save_metadata()
            
            embedding_file = self.cache_dir / f"{cache_key}.npy"
            if embedding_file.exists():
                embedding_file.unlink()
    
    def clear(self):
        """Clear all cached embeddings."""
        for file in self.cache_dir.glob("*.npy"):
            file.unlink()
        self._metadata = {}
        self._save_metadata()


class CachedEmbeddingModel:
    """Wrapper that adds caching to any embedding model."""
    
    def __init__(
        self,
        model: BaseEmbeddingModel,
        cache: EmbeddingCache | None = None
    ):
        """
        Initialize cached embedding model.
        
        Args:
            model: The underlying embedding model
            cache: Optional cache instance (creates default if not provided)
        """
        self.model = model
        self.cache = cache or EmbeddingCache()
    
    @property
    def dimension(self) -> int:
        return self.model.dimension
    
    @property
    def model_name(self) -> str:
        return self.model.model_name
    
    def embed_pages(
        self,
        pages: Sequence[PDFPage],
        document_path: str | None = None
    ) -> EmbeddingResult:
        """
        Embed PDF pages with caching.
        
        Args:
            pages: Sequence of PDFPage objects
            document_path: Path to use for caching (uses page's document_path if not provided)
            
        Returns:
            EmbeddingResult with embeddings
        """
        texts = [page.text for page in pages]
        
        # Determine document path for caching
        if document_path is None:
            document_path = pages[0].document_path if pages and pages[0].document_path else "unknown"
        
        # Try to get from cache
        cached = self.cache.get(document_path, self.model_name, texts)
        if cached is not None:
            return EmbeddingResult(
                embeddings=cached,
                texts=texts,
                model=self.model_name
            )
        
        # Generate new embeddings
        embeddings = self.model.embed_texts(texts)
        
        # Cache the results
        self.cache.put(document_path, self.model_name, texts, embeddings)
        
        return EmbeddingResult(
            embeddings=embeddings,
            texts=texts,
            model=self.model_name
        )
