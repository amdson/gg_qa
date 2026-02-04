"""In-memory vector store for similarity search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from gg_qa.pdf import PDFPage


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search."""
    
    page: PDFPage
    score: float  # Similarity score (higher is better)
    index: int  # Index in the store


@dataclass
class VectorStore:
    """
    In-memory vector store using NumPy for similarity search.
    
    This is a stand-in implementation that performs brute-force all-to-all
    similarity computation. Designed to be replaced with a production
    vector database or search API in the future.
    """
    
    embeddings: np.ndarray | None = None
    pages: list[PDFPage] = field(default_factory=list)
    _normalized: bool = False
    
    def add(
        self,
        pages: Sequence[PDFPage],
        embeddings: np.ndarray,
        normalize: bool = True
    ):
        """
        Add pages and their embeddings to the store.
        
        Args:
            pages: PDF pages to add
            embeddings: Corresponding embeddings array (n_pages, dim)
            normalize: Whether to L2-normalize embeddings for cosine similarity
        """
        if len(pages) != len(embeddings):
            raise ValueError(
                f"Number of pages ({len(pages)}) must match number of embeddings ({len(embeddings)})"
            )
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
        
        # Add to store
        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
            self.pages = list(pages)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])
            self.pages.extend(pages)
        
        self._normalized = normalize
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float | None = None
    ) -> list[VectorSearchResult]:
        """
        Search for the most similar pages to a query embedding.
        
        Args:
            query_embedding: Query vector (1D array)
            k: Number of results to return
            threshold: Minimum similarity score (optional)
            
        Returns:
            List of VectorSearchResult sorted by similarity (descending)
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # Ensure query is 1D
        query = query_embedding.flatten().astype(np.float32)
        
        # Normalize query if store is normalized
        if self._normalized:
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
        
        # Compute similarities (dot product for normalized vectors = cosine similarity)
        similarities = np.dot(self.embeddings, query)
        
        # Get top-k indices
        if k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Use argpartition for efficiency with large arrays
            partition_idx = np.argpartition(similarities, -k)[-k:]
            top_indices = partition_idx[np.argsort(similarities[partition_idx])[::-1]]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            
            if threshold is not None and score < threshold:
                continue
            
            results.append(VectorSearchResult(
                page=self.pages[idx],
                score=score,
                index=int(idx)
            ))
        
        return results[:k]
    
    def search_multi(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
        threshold: float | None = None
    ) -> list[list[VectorSearchResult]]:
        """
        Search for multiple queries at once.
        
        Args:
            query_embeddings: Query vectors (n_queries, dim)
            k: Number of results per query
            threshold: Minimum similarity score (optional)
            
        Returns:
            List of result lists, one per query
        """
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        return [
            self.search(q, k=k, threshold=threshold)
            for q in query_embeddings
        ]
    
    def remove(self, indices: list[int]):
        """
        Remove items from the store by index.
        
        Args:
            indices: List of indices to remove
        """
        if self.embeddings is None:
            return
        
        # Sort indices in descending order to avoid index shifting issues
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.pages):
                del self.pages[idx]
        
        # Create mask for embeddings
        mask = np.ones(len(self.embeddings), dtype=bool)
        for idx in indices:
            if 0 <= idx < len(mask):
                mask[idx] = False
        
        self.embeddings = self.embeddings[mask]
    
    def clear(self):
        """Remove all items from the store."""
        self.embeddings = None
        self.pages = []
    
    def __len__(self) -> int:
        """Return the number of items in the store."""
        return len(self.pages)
    
    @property
    def dimension(self) -> int | None:
        """Return the embedding dimension, or None if empty."""
        if self.embeddings is None:
            return None
        return self.embeddings.shape[1]


class HybridSearcher:
    """
    Combines vector search with text search for hybrid retrieval.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        vector_weight: float = 0.7
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            vector_store: VectorStore instance
            vector_weight: Weight for vector search scores (text weight = 1 - vector_weight)
        """
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.text_weight = 1 - vector_weight
    
    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 5,
        use_fuzzy: bool = True,
        fuzzy_threshold: float = 50.0
    ) -> list[VectorSearchResult]:
        """
        Perform hybrid search combining vector and text similarity.
        
        Args:
            query_embedding: Query vector for semantic search
            query_text: Query text for keyword/fuzzy search
            k: Number of results to return
            use_fuzzy: Whether to use fuzzy text matching
            fuzzy_threshold: Minimum fuzzy match score
            
        Returns:
            Combined and re-ranked results
        """
        from rapidfuzz import fuzz
        
        # Get vector search results (more than k for re-ranking)
        vector_results = self.vector_store.search(query_embedding, k=k * 2)
        
        # Score each result with text matching
        scored_results = []
        query_lower = query_text.lower()
        
        for vr in vector_results:
            page_text = vr.page.text.lower()
            
            # Text score
            if query_lower in page_text:
                text_score = 1.0
            elif use_fuzzy:
                text_score = fuzz.partial_ratio(query_lower, page_text) / 100.0
            else:
                text_score = 0.0
            
            # Combined score
            combined_score = (
                self.vector_weight * vr.score +
                self.text_weight * text_score
            )
            
            scored_results.append(VectorSearchResult(
                page=vr.page,
                score=combined_score,
                index=vr.index
            ))
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        
        return scored_results[:k]
