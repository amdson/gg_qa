"""PDF handling module for text extraction and search."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pymupdf
from rapidfuzz import fuzz, process

@dataclass
class DocChunk:
    """Represents text extracted from a pdf document"""
    document_path: str | None = None
    chunk_number: int
    text_index: int # index of first char of chunk in composite text
    text: str
    
    @property
    def char_count(self) -> int:
        """Return the character count of the chunk text."""
        return len(self.text)
    
    def to_context_string(self) -> str:
        """Format chunk for LLM context."""
        return f"[Chunk {self.page_number}]\n{self.text}"

@dataclass
class SearchResult:
    """Result from a text search operation."""
    
    page: DocChunk
    score: float  # 0-100 for fuzzy, 1.0 for exact match
    matched_text: str | None = None

#block size is measured in characters, not tokens
split_patterns = ['\n', '\t', '. ', '-', ' ']
import regex as re
def split_block(block):
    assert len(block) >= 3
    #find the inner third boundaries
    third = len(block) // 3
    inner_start = third
    inner_end = 2 * third
    center = len(block) // 2
    
    #try each split pattern in priority order
    best_split = None
    for pattern in split_patterns:
        matches = []
        idx = inner_start
        while idx < inner_end:
            pos = block.find(pattern, idx)
            if pos == -1 or pos >= inner_end:
                break
            matches.append(pos + len(pattern))
            idx = pos + 1
        
        if matches:
            best_split = min(matches, key=lambda x: abs(x - center))
            break
    
    #if no pattern matched, split in the middle
    if best_split is None:
        best_split = center
    #add both halves back to process
    return (block[:best_split], block[best_split:])

# chunks should be greater than max_chunk_size / 2 and less than max_chunk_size
# chunks should overlap by at least chunk_overlap characters and at most 2*chunk_overlap characters
def chunk_stream(source_doc, block_stream, max_chunk_size, chunk_overlap):
    assert chunk_overlap < max_chunk_size / 4 #make sure chunks cannot overlap with >1 chunk ahead or behind them. 
    prev_len = 0 #Num char before current chunk
    buffer_len = 0 #total chars within stack
    block_buffer = [] #buffer of blocks. First block must be included in next chunk returned
    chunk_number = 0

    for block in block_stream:
        block_buffer.append(block)
        buffer_len += len(block)

        if buffer_len >= max_chunk_size:
            #greedily pull chunks from the bottom of the stack until the chunk size is > max_chunk_size / 2
            #if a block would cause the chunk to be > max_chunk_size, split it and put both halves back on the stack
            chunk_blocks = []
            curr_len = 0
            while block_buffer and curr_len < max_chunk_size / 2:
                next_block = block_buffer.pop(0)
                if curr_len + len(next_block) <= max_chunk_size:
                    buffer_len -= len(next_block)
                    chunk_blocks.append(next_block)
                    curr_len += len(next_block)
                else:
                    first_half, second_half = split_block(next_block)
                    block_buffer.insert(0, second_half)
                    block_buffer.insert(0, first_half)
            yield DocChunk(document_path=source_doc, chunk_number=chunk_number, 
                           text_index=prev_len, text=''.join(chunk_blocks))
            prev_len += curr_len
            chunk_number += 1

            to_keep = []
            keep_len = 0
            while keep_len < chunk_overlap and chunk_blocks:
                b = chunk_blocks.pop()
                if keep_len + len(b) <= chunk_overlap:
                    to_keep.append(b)
                    keep_len += len(b)
                elif keep_len + len(b) > 2*chunk_overlap:
                    first_half, second_half = split_block(b)
                    chunk_blocks.extend([first_half, second_half])
                else:
                    to_keep.append(b)
                    keep_len += len(b)
                    break
            to_keep.reverse()
            block_buffer = to_keep + block_buffer
            buffer_len = sum(len(b) for b in block_buffer)
    yield DocChunk(document_path=source_doc, chunk_number=chunk_number, 
                   text_index=prev_len, text=''.join(block_buffer))
                
def merge_chunks(c1: DocChunk, c2: DocChunk):
    assert c1.text_index + c1.char_count > c2.text_index and c1.document_path == c2.document_path
    return DocChunk(document_path=c1.document_path, chunk_number=-1, text_index=c1.text_index, 
                    text=c1.text + c2.text[c1.text_index + c1.char_count - c2.text_index:])

def chunk_pdf(path: str | Path, chunk_size=1024, chunk_overlap=128):
    def pymu_generator(path):
        with pymupdf.open(path) as doc:
            # Iterate through each page
            for page in doc:
                # Extract text using get_text()
                blocks = page.get_text('blocks')
                for block in blocks:
                    yield block
    blocks = pymu_generator(path)
    return list(chunk_stream(path, blocks, chunk_size, chunk_overlap))

@dataclass
class PDFDocument:
    """Represents a PDF document with text extraction and search capabilities."""
    
    chunks: list[DocChunk] = field(default_factory=list)
    path: str | None = None
    metadata: dict = field(default_factory=dict)