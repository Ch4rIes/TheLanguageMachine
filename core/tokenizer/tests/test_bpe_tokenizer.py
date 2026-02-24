import os
import tempfile

import pytest
from src.tokenizer import BPETokenizer


class TestBPETokenizerInit:
    def test_init(self):
        tokenizer = BPETokenizer()
        assert tokenizer.vocab == {}
        assert tokenizer.merges == []


class TestBPETokenizerTrain:
    def test_vocab_starts_with_256_bytes(self):
        """Vocab should include all 256 single bytes"""
        content = b"hello world"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=256, special_tokens=[])

            # Should have exactly 256 single-byte tokens
            assert len(vocab) == 256
            for i in range(256):
                assert vocab[i] == bytes([i])
        finally:
            os.unlink(path)

    def test_special_tokens_added_to_vocab(self):
        """Special tokens should be added after the 256 bytes"""
        content = b"hello<|endoftext|>world"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            special_tokens = [b"<|endoftext|>", b"<|pad|>"]
            vocab, merges = tokenizer.train(path, vocab_size=258, special_tokens=special_tokens)

            # Special tokens at indices 256 and 257
            assert vocab[256] == b"<|endoftext|>"
            assert vocab[257] == b"<|pad|>"
        finally:
            os.unlink(path)

    def test_simple_merge(self):
        """Test that the most frequent pair gets merged"""
        # "aa" repeated should cause 'a'+'a' to be merged
        content = b"aa aa aa"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=257, special_tokens=[])

            # Should have one merge
            assert len(merges) == 1
            # The merge should be (b'a', b'a') -> b'aa'
            assert merges[0] == (b"a", b"a")
            assert vocab[256] == b"aa"
        finally:
            os.unlink(path)

    def test_multiple_merges(self):
        """Test multiple merges in sequence"""
        # "abab abab" - 'ab' should be merged, then possibly 'abab'
        content = b"abab abab abab"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=258, special_tokens=[])

            # Should have at least one merge
            assert len(merges) >= 1
            # First merge should be the most frequent pair
            assert len(vocab) == 258
        finally:
            os.unlink(path)

    def test_vocab_size_limit(self):
        """Training should stop when vocab_size is reached"""
        content = b"abcabc abcabc abcabc abcabc"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=260, special_tokens=[])

            assert len(vocab) == 260
            assert len(merges) == 4  # 260 - 256 = 4 merges
        finally:
            os.unlink(path)

    def test_unicode_text(self):
        """Test with unicode characters (multi-byte)"""
        content = "你好你好你好".encode("utf-8")
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=257, special_tokens=[])

            # Should have one merge (most frequent byte pair in the repeated text)
            assert len(merges) == 1
            assert len(vocab) == 257
        finally:
            os.unlink(path)

    def test_merges_are_bytes(self):
        """Merges should be tuples of bytes"""
        content = b"hello hello hello"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=258, special_tokens=[])

            for merge in merges:
                assert isinstance(merge, tuple)
                assert len(merge) == 2
                assert isinstance(merge[0], bytes)
                assert isinstance(merge[1], bytes)
        finally:
            os.unlink(path)

    def test_vocab_values_are_bytes(self):
        """All vocab values should be bytes"""
        content = b"test content here"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=260, special_tokens=[b"<|endoftext|>"])

            for token_id, token_bytes in vocab.items():
                assert isinstance(token_id, int)
                assert isinstance(token_bytes, bytes)
        finally:
            os.unlink(path)

    def test_merged_token_is_concatenation(self):
        """A merged token should be the concatenation of its pair"""
        content = b"xy xy xy xy xy"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=257, special_tokens=[])

            # The merge (b'x', b'y') should produce vocab entry b'xy'
            assert len(merges) == 1
            token1, token2 = merges[0]
            assert vocab[256] == token1 + token2
        finally:
            os.unlink(path)

    def test_empty_file(self):
        """Handle empty file gracefully"""
        content = b""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=256, special_tokens=[])

            # Should still have base vocab, no merges
            assert len(vocab) == 256
            assert len(merges) == 0
        finally:
            os.unlink(path)


class TestBPEEdgeCases:
    """Test edge cases and tricky scenarios"""

    def test_overlapping_pairs_aaa(self):
        """
        Test 'aaa' where (a,a) appears at overlapping positions.
        'aaa' has bytes [97, 97, 97]
        Pairs: (97,97) at position 0 and position 1
        After merging position 0: [97,97] + [97] -> should become (aa, a)
        """
        content = b"aaa"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=257, special_tokens=[])

            # First merge should be (a, a) -> aa
            assert len(merges) >= 1
            assert merges[0] == (b"a", b"a")
            assert vocab[256] == b"aa"
        finally:
            os.unlink(path)

    def test_overlapping_pairs_aaaa(self):
        """
        Test 'aaaa' - even more overlapping.
        After first merge of (a,a): could become [aa, aa] or [aa, a, a] depending on order
        """
        content = b"aaaa"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=258, special_tokens=[])

            # Should complete without error
            assert len(merges) >= 1
            # First merge is (a, a)
            assert merges[0] == (b"a", b"a")
        finally:
            os.unlink(path)

    def test_overlapping_pairs_multiple_words(self):
        """
        Test multiple words with overlapping pairs.
        'aaa aaa' - each word has overlapping (a,a) pairs
        """
        content = b"aaa aaa aaa"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=257, special_tokens=[])

            # Should handle without crashing
            assert len(merges) >= 1
        finally:
            os.unlink(path)

    def test_abab_pattern(self):
        """
        Test 'abababab' - after merging (a,b), pairs shift
        """
        content = b"abababab"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=258, special_tokens=[])

            # (a,b) should be merged first (appears 4 times)
            # Then (ab, ab) could be merged
            assert merges[0] == (b"a", b"b")
            assert vocab[256] == b"ab"
        finally:
            os.unlink(path)

    def test_chain_merges(self):
        """
        Test that chain merges work correctly.
        'abcd abcd abcd abcd' - after (a,b) merge, (ab,c) should be possible
        """
        content = b"abcd abcd abcd abcd"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            tokenizer = BPETokenizer()
            vocab, merges = tokenizer.train(path, vocab_size=260, special_tokens=[])

            # Should do 4 merges without error
            assert len(merges) == 4
            # All merged tokens should be valid bytes
            for token_id, token_bytes in vocab.items():
                assert isinstance(token_bytes, bytes)
        finally:
            os.unlink(path)
