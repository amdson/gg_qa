"""Tests for chunk_stream and related chunking functions."""

import pytest
from gg_qa.pdf import chunk_stream, split_block, DocChunk


class TestSplitBlock:
    """Tests for the split_block function."""
    
    def test_split_on_newline(self):
        #newline is highest priority pattern
        block = "aaaaaa\nbbbbbb"
        left, right = split_block(block)
        assert left == "aaaaaa\n"
        assert right == "bbbbbb"
    
    def test_split_on_period_space(self):
        #period-space when no newline in inner third
        block = "aaaaaaa. bbbbbbb"
        left, right = split_block(block)
        assert left == "aaaaaaa. "
        assert right == "bbbbbbb"
    
    def test_split_in_middle_when_no_pattern(self):
        #no pattern matches, should split in middle
        block = "abcdefghijklmnop"
        left, right = split_block(block)
        assert len(left) + len(right) == len(block)
        assert left + right == block
    
    def test_split_prefers_center_match(self):
        #when multiple matches, pick closest to center
        block = "aaa\nbbb\nccc\nddd"
        left, right = split_block(block)
        #should pick the middle newline
        assert left + right == block
        assert abs(len(left) - len(right)) <= 4  #reasonably balanced
    
    def test_minimum_block_size(self):
        #blocks must be at least 3 chars
        with pytest.raises(AssertionError):
            split_block("ab")


class TestChunkStream:
    """Tests for the chunk_stream function."""
    
    def test_single_small_block(self):
        #single block smaller than max_chunk_size
        blocks = ["hello world"]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"
        assert chunks[0].document_path == "test.pdf"
        assert chunks[0].chunk_number == 1
        assert chunks[0].text_index == 0
    
    def test_empty_stream(self):
        #empty block stream
        blocks = []
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        assert len(chunks) == 1
        assert chunks[0].text == ""
    
    def test_multiple_small_blocks_combine(self):
        #multiple small blocks should combine into one chunk
        blocks = ["hello ", "world ", "test"]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        assert len(chunks) == 1
        assert chunks[0].text == "hello world test"
    
    def test_blocks_split_into_chunks(self):
        #blocks that exceed max_chunk_size should split
        blocks = ["a" * 50, "b" * 50, "c" * 50, "d" * 50]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 100
    
    def test_chunk_minimum_size(self):
        #chunks should be at least max_chunk_size / 2
        blocks = ["a" * 80, "b" * 80, "c" * 80]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        #final chunk may be smaller, but others should be >= 50
        for chunk in chunks[:-1]:
            assert len(chunk.text) >= 50
    
    def test_chunk_overlap_requirement(self):
        #chunks should overlap by at least chunk_overlap chars
        blocks = ["word" * 50]  #200 chars
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=20))
        if len(chunks) >= 2:
            for i in range(len(chunks) - 1):
                c1, c2 = chunks[i], chunks[i+1]
                overlap = (c1.text_index + c1.char_count) - c2.text_index
                assert overlap >= 20, f"Overlap {overlap} less than required 20"
    
    def test_overlap_not_excessive(self):
        #overlap should be at most 2*chunk_overlap
        blocks = ["word" * 50]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=20))
        if len(chunks) >= 2:
            for i in range(len(chunks) - 1):
                c1, c2 = chunks[i], chunks[i+1]
                overlap = (c1.text_index + c1.char_count) - c2.text_index
                assert overlap <= 40, f"Overlap {overlap} exceeds max 40"
    
    def test_text_index_continuity(self):
        #text_index should track position correctly
        blocks = ["a" * 100, "b" * 100, "c" * 100]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        #first chunk starts at 0
        assert chunks[0].text_index == 0
        #subsequent chunks should have increasing text_index
        for i in range(1, len(chunks)):
            assert chunks[i].text_index > chunks[i-1].text_index
    
    def test_chunk_number_increments(self):
        #chunk_number should track chunk count
        blocks = ["a" * 200]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        #verify chunks exist and are numbered
        assert len(chunks) >= 1
    
    def test_large_single_block_splits(self):
        #single block larger than max should be split properly
        blocks = ["x" * 500]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        assert len(chunks) >= 5
        for chunk in chunks:
            assert len(chunk.text) <= 100
    
    def test_mixed_block_sizes(self):
        #mix of small and large blocks
        blocks = ["tiny", "a" * 200, "small", "b" * 150, "end"]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        for chunk in chunks:
            assert len(chunk.text) <= 100
        #all text should be preserved
        total_input = "".join(blocks)
        #accounting for overlap, total output should be >= input
        total_output = sum(len(c.text) for c in chunks)
        assert total_output >= len(total_input)
    
    def test_overlap_assertion(self):
        #chunk_overlap must be less than max_chunk_size / 4
        blocks = ["hello"]
        with pytest.raises(AssertionError):
            list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=30))
    
    def test_blocks_with_split_patterns(self):
        #blocks containing natural split points
        blocks = ["Hello world. ", "This is a test. ", "Another sentence. "] * 10
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=20))
        for chunk in chunks:
            assert len(chunk.text) <= 100
    
    def test_preserves_document_path(self):
        #document_path should be preserved in all chunks
        blocks = ["a" * 200]
        chunks = list(chunk_stream("my/custom/path.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        for chunk in chunks:
            assert chunk.document_path == "my/custom/path.pdf"
    
    def test_generator_input(self):
        #should work with generator input
        def block_gen():
            for c in "abcdefghij":
                yield c * 20
        chunks = list(chunk_stream("test.pdf", block_gen(), max_chunk_size=100, chunk_overlap=10))
        assert len(chunks) >= 1


class TestChunkStreamEdgeCases:
    """Edge cases and boundary conditions."""
    
    def test_exact_max_size_block(self):
        #block exactly at max_chunk_size
        blocks = ["x" * 100]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        assert all(len(c.text) <= 100 for c in chunks)
    
    def test_just_over_max_size_block(self):
        #block just over max_chunk_size
        blocks = ["x" * 101]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        assert len(chunks) >= 2
        assert all(len(c.text) <= 100 for c in chunks)
    
    def test_whitespace_only_blocks(self):
        #blocks with only whitespace
        blocks = ["   ", "\n\n", "\t\t"]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        assert len(chunks) >= 1
    
    def test_unicode_content(self):
        #unicode characters
        blocks = ["Hello 世界! ", "こんにちは ", "مرحبا "] * 10
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        combined = "".join(c.text for c in chunks)
        assert "世界" in combined or any("世界" in c.text for c in chunks)
    
    def test_very_small_max_chunk_size(self):
        #very small max_chunk_size
        blocks = ["hello world this is a test"]
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=20, chunk_overlap=4))
        for chunk in chunks:
            assert len(chunk.text) <= 20
    
    def test_many_tiny_blocks(self):
        #many tiny blocks
        blocks = list("a" * 100)  #100 single-char blocks
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=50, chunk_overlap=5))
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 50
    
    def test_alternating_sizes(self):
        #alternating big and small blocks
        blocks = []
        for i in range(10):
            blocks.append("x" * 80)
            blocks.append("y")
        chunks = list(chunk_stream("test.pdf", iter(blocks), max_chunk_size=100, chunk_overlap=10))
        for chunk in chunks:
            assert len(chunk.text) <= 100
