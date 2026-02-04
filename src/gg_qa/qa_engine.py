"""QA Engine for LLM-based question answering over PDFs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from gg_qa.embeddings import BaseEmbeddingModel
from gg_qa.pdf import PDFDocument, PDFPage
from gg_qa.vector_store import VectorStore, VectorSearchResult


@dataclass
class QAResult:
    """Result from a QA query."""
    
    answer: str
    source_pages: list[PDFPage]
    query: str
    context_used: str


class QAEngine:
    """
    Question-answering engine that combines retrieval with LLM generation.
    
    Uses PydanticAI for LLM interaction, designed for OpenRouter.
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Instructions:
- Answer questions using ONLY the information provided in the context below
- If the context doesn't contain enough information to answer, say so clearly
- Cite specific page numbers when referencing information
- Be concise but thorough in your answers
- If asked about something not in the context, explain that you can only answer based on the provided document"""
    
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        document: PDFDocument | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
        llm_model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        system_prompt: str | None = None,
        retrieval_k: int = 5,
        max_context_chars: int = 8000
    ):
        """
        Initialize the QA engine.
        
        Args:
            vector_store: VectorStore for semantic retrieval
            document: PDFDocument for text search fallback
            embedding_model: Model for embedding queries
            llm_model: LLM model identifier for OpenRouter
            api_key: API key (falls back to OPENROUTER_API_KEY env var)
            base_url: API base URL
            system_prompt: Custom system prompt (uses default if not provided)
            retrieval_k: Number of pages to retrieve for context
            max_context_chars: Maximum characters for context window
        """
        self.vector_store = vector_store
        self.document = document
        self.embedding_model = embedding_model
        self.retrieval_k = retrieval_k
        self.max_context_chars = max_context_chars
        
        # Resolve API key
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError(
                "No API key provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize LLM
        self.llm = OpenAIModel(
            llm_model,
            base_url=base_url,
            api_key=api_key
        )
        
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Create agent
        self.agent = Agent(
            self.llm,
            system_prompt=self.system_prompt
        )
    
    def _retrieve_pages(
        self,
        query: str,
        k: int | None = None
    ) -> list[PDFPage]:
        """
        Retrieve relevant pages for a query.
        
        Args:
            query: User query
            k: Number of pages to retrieve
            
        Returns:
            List of relevant PDFPage objects
        """
        k = k or self.retrieval_k
        
        # Try vector search first
        if self.vector_store and self.embedding_model:
            query_embedding = self.embedding_model.embed_text(query)
            results = self.vector_store.search(query_embedding, k=k)
            if results:
                return [r.page for r in results]
        
        # Fall back to text search
        if self.document:
            # Try exact search first
            text_results = self.document.search_exact(query, max_results=k)
            if text_results:
                return [r.page for r in text_results]
            
            # Try fuzzy search
            fuzzy_results = self.document.search_fuzzy(query, max_results=k)
            if fuzzy_results:
                return [r.page for r in fuzzy_results]
            
            # Last resort: return first k pages
            return self.document.pages[:k]
        
        return []
    
    def _build_context(
        self,
        pages: Sequence[PDFPage],
        max_chars: int | None = None
    ) -> str:
        """
        Build context string from retrieved pages.
        
        Args:
            pages: Pages to include in context
            max_chars: Maximum characters
            
        Returns:
            Formatted context string
        """
        max_chars = max_chars or self.max_context_chars
        
        context_parts = []
        total_chars = 0
        
        for page in pages:
            page_context = page.to_context_string()
            
            if total_chars + len(page_context) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    context_parts.append(page_context[:remaining] + "...")
                break
            
            context_parts.append(page_context)
            total_chars += len(page_context)
        
        return "\n\n---\n\n".join(context_parts)
    
    async def query(
        self,
        question: str,
        additional_context: str | None = None,
        retrieval_k: int | None = None
    ) -> QAResult:
        """
        Answer a question using retrieval-augmented generation.
        
        Args:
            question: User question
            additional_context: Extra context to include (e.g., from other documents)
            retrieval_k: Override number of pages to retrieve
            
        Returns:
            QAResult with answer and source information
        """
        # Retrieve relevant pages
        pages = self._retrieve_pages(question, k=retrieval_k)
        
        # Build context
        context = self._build_context(pages)
        
        if additional_context:
            context = f"{additional_context}\n\n---\n\n{context}"
        
        # Build prompt
        prompt = f"""Context from document:

{context}

---

Question: {question}

Please provide a detailed answer based on the context above."""
        
        # Query LLM
        result = await self.agent.run(prompt)
        
        return QAResult(
            answer=result.data,
            source_pages=list(pages),
            query=question,
            context_used=context
        )
    
    def query_sync(
        self,
        question: str,
        additional_context: str | None = None,
        retrieval_k: int | None = None
    ) -> QAResult:
        """
        Synchronous version of query().
        
        Args:
            question: User question
            additional_context: Extra context to include
            retrieval_k: Override number of pages to retrieve
            
        Returns:
            QAResult with answer and source information
        """
        import asyncio
        return asyncio.run(self.query(question, additional_context, retrieval_k))
    
    async def query_with_pages(
        self,
        question: str,
        page_numbers: list[int]
    ) -> QAResult:
        """
        Answer a question using specific pages as context.
        
        Args:
            question: User question
            page_numbers: Specific page numbers to use (1-indexed)
            
        Returns:
            QAResult with answer
        """
        if not self.document:
            raise ValueError("Document required for page-specific queries")
        
        pages = [self.document.get_page(p) for p in page_numbers]
        context = self._build_context(pages)
        
        prompt = f"""Context from document (pages {', '.join(map(str, page_numbers))}):

{context}

---

Question: {question}

Please provide a detailed answer based on the context above."""
        
        result = await self.agent.run(prompt)
        
        return QAResult(
            answer=result.data,
            source_pages=pages,
            query=question,
            context_used=context
        )


class MultiDocQAEngine:
    """
    QA engine for querying across multiple documents.
    """
    
    def __init__(
        self,
        documents: dict[str, PDFDocument],
        vector_stores: dict[str, VectorStore] | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
        llm_model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize multi-document QA engine.
        
        Args:
            documents: Dictionary mapping document names to PDFDocument objects
            vector_stores: Optional dictionary of vector stores per document
            embedding_model: Model for embedding queries
            llm_model: LLM model identifier
            api_key: API key
            base_url: API base URL
        """
        self.documents = documents
        self.vector_stores = vector_stores or {}
        self.embedding_model = embedding_model
        
        # Create unified engine for generation
        self._engine = QAEngine(
            llm_model=llm_model,
            api_key=api_key,
            base_url=base_url
        )
    
    def _retrieve_from_all(
        self,
        query: str,
        k_per_doc: int = 3
    ) -> dict[str, list[PDFPage]]:
        """Retrieve relevant pages from all documents."""
        results = {}
        
        for name, doc in self.documents.items():
            store = self.vector_stores.get(name)
            
            if store and self.embedding_model:
                query_embedding = self.embedding_model.embed_text(query)
                search_results = store.search(query_embedding, k=k_per_doc)
                results[name] = [r.page for r in search_results]
            else:
                # Fall back to text search
                text_results = doc.search_fuzzy(query, max_results=k_per_doc)
                results[name] = [r.page for r in text_results]
        
        return results
    
    async def query(
        self,
        question: str,
        k_per_doc: int = 3,
        max_context_chars: int = 12000
    ) -> QAResult:
        """
        Query across all documents.
        
        Args:
            question: User question
            k_per_doc: Pages to retrieve per document
            max_context_chars: Maximum context size
            
        Returns:
            QAResult with answer
        """
        # Retrieve from all docs
        doc_pages = self._retrieve_from_all(question, k_per_doc)
        
        # Build combined context
        context_parts = []
        all_pages = []
        total_chars = 0
        
        for doc_name, pages in doc_pages.items():
            if not pages:
                continue
            
            doc_context = f"[Document: {doc_name}]\n"
            for page in pages:
                page_str = page.to_context_string()
                
                if total_chars + len(doc_context) + len(page_str) > max_context_chars:
                    break
                
                doc_context += page_str + "\n\n"
                all_pages.append(page)
                total_chars += len(page_str)
            
            context_parts.append(doc_context)
        
        context = "\n---\n\n".join(context_parts)
        
        # Query
        prompt = f"""Context from multiple documents:

{context}

---

Question: {question}

Please provide a detailed answer based on the context above. Reference which document(s) the information comes from."""
        
        result = await self._engine.agent.run(prompt)
        
        return QAResult(
            answer=result.data,
            source_pages=all_pages,
            query=question,
            context_used=context
        )
