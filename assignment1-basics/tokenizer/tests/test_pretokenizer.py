import os
import tempfile

import pytest
import regex as re
from src.pretokenizer import (
    PRETOKENIZE_REGEX,
    Pretokenizer,
    find_chunk_boundaries,
    pretokenize_chunk,
)

EOD: bytes = b"<|endoftext|>"


class TestPretokenizeRegex:
    """Test the GPT-2 style regex pattern"""

    def test_contractions(self):
        text = "I'm don't won't can't"
        tokens = re.findall(PRETOKENIZE_REGEX, text)
        assert "'m" in tokens
        assert "'t" in tokens

    def test_words_with_leading_space(self):
        text = "hello world"
        tokens = re.findall(PRETOKENIZE_REGEX, text)
        assert "hello" in tokens
        assert " world" in tokens

    def test_numbers(self):
        text = "test 123 456"
        tokens = re.findall(PRETOKENIZE_REGEX, text)
        assert " 123" in tokens
        assert " 456" in tokens

    def test_punctuation(self):
        text = "hello, world!"
        tokens = re.findall(PRETOKENIZE_REGEX, text)
        assert "," in tokens
        assert "!" in tokens

    def test_unicode_letters(self):
        text = "hello 你好 world"
        tokens = re.findall(PRETOKENIZE_REGEX, text)
        assert " 你好" in tokens

    def test_trailing_whitespace(self):
        text = "hello   \n"
        tokens = re.findall(PRETOKENIZE_REGEX, text)
        # Trailing whitespace not followed by non-whitespace
        assert any(t.isspace() for t in tokens)


class TestFindChunkBoundaries:
    """Test file chunking logic"""

    def test_basic_chunking(self):
        content = b"hello<|endoftext|>world<|endoftext|>foo"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            with open(path, "rb") as f:
                boundaries = find_chunk_boundaries(f, 2, EOD)
            assert len(boundaries) >= 2
            assert boundaries[0] == 0
            assert boundaries[-1] == len(content)
        finally:
            os.unlink(path)

    def test_single_chunk(self):
        content = b"hello world"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            with open(path, "rb") as f:
                boundaries = find_chunk_boundaries(f, 1, EOD)
            assert boundaries == [0, len(content)]
        finally:
            os.unlink(path)

    def test_no_special_token(self):
        content = b"hello world no token here"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            with open(path, "rb") as f:
                boundaries = find_chunk_boundaries(f, 3, EOD)
            # Should still return valid boundaries
            assert boundaries[0] == 0
            assert boundaries[-1] == len(content)
        finally:
            os.unlink(path)


class TestPretokenizeChunk:
    """Test chunk pretokenization"""

    def test_basic_chunk(self):
        content = b"hello world"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokens = pretokenize_chunk(path, 0, len(content))
            assert "hello" in tokens
            assert " world" in tokens
        finally:
            os.unlink(path)

    def test_partial_chunk(self):
        content = b"hello world foo bar"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            # Read only "world"
            tokens = pretokenize_chunk(path, 6, 11)
            assert "world" in tokens
            assert "hello" not in tokens
        finally:
            os.unlink(path)


class TestPretokenizer:
    """Test full pretokenizer"""

    def test_pretokenize_file(self):
        content = b"hello world<|endoftext|>foo bar"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            results = Pretokenizer.pretokenize(path, EOD)
            # Flatten results from all chunks
            all_tokens = [t for chunk in results for t in chunk]
            assert "hello" in all_tokens
            assert " world" in all_tokens
        finally:
            os.unlink(path)
