"""PDF handling module for text extraction and search."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pymupdf
from rapidfuzz import fuzz, process


@dataclass
class PDFPage:
    """Represents a single page from a PDF document."""
    
    page_number: int  # 1-indexed
    text: str
    document_path: str | None = None
    
    @property
    def char_count(self) -> int:
        """Return the character count of the page text."""
        return len(self.text)
    
    @property
    def word_count(self) -> int:
        """Return the word count of the page text."""
        return len(self.text.split())
    
    def contains(self, query: str, case_sensitive: bool = False) -> bool:
        """Check if the page contains the query string."""
        text = self.text if case_sensitive else self.text.lower()
        query = query if case_sensitive else query.lower()
        return query in text
    
    def to_context_string(self) -> str:
        """Format page for LLM context."""
        return f"[Page {self.page_number}]\n{self.text}"


@dataclass
class SearchResult:
    """Result from a text search operation."""
    
    page: PDFPage
    score: float  # 0-100 for fuzzy, 1.0 for exact match
    matched_text: str | None = None
    

@dataclass
class PDFDocument:
    """Represents a PDF document with text extraction and search capabilities."""
    
    pages: list[PDFPage] = field(default_factory=list)
    path: str | None = None
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, path: str | Path) -> PDFDocument:
        """Load a PDF document from a file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        
        pages = []
        with fitz.open(path) as doc:
            metadata = dict(doc.metadata) if doc.metadata else {}
            
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                pages.append(PDFPage(
                    page_number=page_num,
                    text=text,
                    document_path=str(path)
                ))
        
        return cls(pages=pages, path=str(path), metadata=metadata)
    
    @classmethod
    def from_bytes(cls, data: bytes, filename: str = "document.pdf") -> PDFDocument:
        """Load a PDF document from bytes."""
        pages = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            metadata = dict(doc.metadata) if doc.metadata else {}
            
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                pages.append(PDFPage(
                    page_number=page_num,
                    text=text,
                    document_path=filename
                ))
        
        return cls(pages=pages, path=filename, metadata=metadata)
    
    @property
    def page_count(self) -> int:
        """Return the number of pages in the document."""
        return len(self.pages)
    
    @property
    def full_text(self) -> str:
        """Return the full text of the document."""
        return "\n\n".join(page.text for page in self.pages)
    
    def get_page(self, page_number: int) -> PDFPage:
        """Get a specific page by number (1-indexed)."""
        if page_number < 1 or page_number > len(self.pages):
            raise IndexError(f"Page {page_number} out of range (1-{len(self.pages)})")
        return self.pages[page_number - 1]
    
    def search_exact(
        self, 
        query: str, 
        case_sensitive: bool = False,
        max_results: int | None = None
    ) -> list[SearchResult]:
        """
        Search for exact text matches in the document.
        
        Args:
            query: The text to search for
            case_sensitive: Whether to perform case-sensitive search
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects for pages containing the query
        """
        results = []
        search_query = query if case_sensitive else query.lower()
        
        for page in self.pages:
            text = page.text if case_sensitive else page.text.lower()
            if search_query in text:
                results.append(SearchResult(
                    page=page,
                    score=1.0,
                    matched_text=query
                ))
                
                if max_results and len(results) >= max_results:
                    break
        
        return results
    
    def search_regex(
        self,
        pattern: str,
        flags: int = re.IGNORECASE,
        max_results: int | None = None
    ) -> list[SearchResult]:
        """
        Search for regex pattern matches in the document.
        
        Args:
            pattern: Regex pattern to search for
            flags: Regex flags (default: case-insensitive)
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects for pages with matches
        """
        results = []
        compiled = re.compile(pattern, flags)
        
        for page in self.pages:
            match = compiled.search(page.text)
            if match:
                results.append(SearchResult(
                    page=page,
                    score=1.0,
                    matched_text=match.group()
                ))
                
                if max_results and len(results) >= max_results:
                    break
        
        return results
    
    def search_fuzzy(
        self,
        query: str,
        threshold: float = 70.0,
        max_results: int | None = None
    ) -> list[SearchResult]:
        """
        Search for fuzzy text matches using rapidfuzz.
        
        This performs sentence-level fuzzy matching within each page.
        
        Args:
            query: The text to search for
            threshold: Minimum similarity score (0-100) to include in results
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        results = []
        
        for page in self.pages:
            # Split page into sentences/chunks for more granular matching
            sentences = self._split_into_chunks(page.text)
            
            if not sentences:
                continue
            
            # Find best matching sentence in the page
            best_match = process.extractOne(
                query,
                sentences,
                scorer=fuzz.partial_ratio
            )
            
            if best_match and best_match[1] >= threshold:
                results.append(SearchResult(
                    page=page,
                    score=best_match[1],
                    matched_text=best_match[0]
                ))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        
        if max_results:
            results = results[:max_results]
        
        return results
    
    def _split_into_chunks(self, text: str, min_length: int = 20) -> list[str]:
        """Split text into sentence-like chunks for fuzzy matching."""
        # Split on sentence boundaries
        chunks = re.split(r'(?<=[.!?])\s+', text)
        # Filter out very short chunks
        return [c.strip() for c in chunks if len(c.strip()) >= min_length]
    
    def __iter__(self) -> Iterator[PDFPage]:
        """Iterate over pages in the document."""
        return iter(self.pages)
    
    def __len__(self) -> int:
        """Return the number of pages."""
        return len(self.pages)
    
    def to_llm_context(
        self,
        pages: list[int] | None = None,
        max_chars: int | None = None
    ) -> str:
        """
        Format document content for LLM context.
        
        Args:
            pages: Specific page numbers to include (1-indexed), or None for all
            max_chars: Maximum character limit for the context
            
        Returns:
            Formatted string suitable for LLM input
        """
        if pages:
            selected = [self.get_page(p) for p in pages]
        else:
            selected = self.pages
        
        context_parts = []
        total_chars = 0
        
        for page in selected:
            page_context = page.to_context_string()
            
            if max_chars:
                if total_chars + len(page_context) > max_chars:
                    # Truncate if needed
                    remaining = max_chars - total_chars
                    if remaining > 100:  # Only add if meaningful space remains
                        context_parts.append(page_context[:remaining] + "...")
                    break
            
            context_parts.append(page_context)
            total_chars += len(page_context)
        
        return "\n\n".join(context_parts)
