import pytest
from src.tokenizer import TokenIndex


class TestTokenNode:
    def test_create_node(self):
        node = TokenIndex.TokenNode(("a",))
        assert node.tokens == ("a",)
        assert node.prev is None
        assert node.next is None

    def test_link_nodes(self):
        node1 = TokenIndex.TokenNode(("a",))
        node2 = TokenIndex.TokenNode(("b",))
        node1.next = node2
        node2.prev = node1
        assert node1.next == node2
        assert node2.prev == node1


class TestTokenLinkedList:
    def test_create_empty_list(self):
        ll = TokenIndex.TokenLinkedList()
        assert ll.head.tokens is None
        assert ll.tail.tokens is None
        assert ll.head.next == ll.tail
        assert ll.tail.prev == ll.head


class TestTokenIndex:
    def test_init(self):
        index = TokenIndex()
        assert len(index.token_sequence) == 0
        assert len(index.pairs_to_token_node) == 0
        assert len(index.token_pair_count) == 0

    def test_populate_single_word(self):
        index = TokenIndex()
        chunks = [["ab"]]
        index.populate_words(chunks)

        # Check word_list has one entry
        assert len(index.token_sequence) == 1

        # Check linked list structure: head -> 97 ('a') -> 98 ('b') -> tail
        ll = index.token_sequence[0]
        node_a = ll.head.next
        node_b = node_a.next

        assert node_a.tokens == (97,)  # ord('a') = 97
        assert node_b.tokens == (98,)  # ord('b') = 98
        assert node_b.next == ll.tail

    def test_populate_pair_counts(self):
        index = TokenIndex()
        chunks = [["ab"]]
        index.populate_words(chunks)

        # Should have one pair: (97,) -> (98,)
        pair = ((97,), (98,))
        assert index.token_pair_count[pair] == 1
        assert len(index.pairs_to_token_node[pair]) == 1

    def test_populate_repeated_pairs(self):
        index = TokenIndex()
        chunks = [["aa"]]
        index.populate_words(chunks)

        pair = ((97,), (97,))
        assert index.token_pair_count[pair] == 1

    def test_populate_multiple_words(self):
        index = TokenIndex()
        chunks = [["ab", "ab"]]
        index.populate_words(chunks)

        # Two words, same content
        assert len(index.token_sequence) == 2

        pair = ((97,), (98,))
        assert index.token_pair_count[pair] == 2

    def test_populate_multiple_chunks(self):
        index = TokenIndex()
        chunks = [["ab"], ["cd"]]
        index.populate_words(chunks)

        assert len(index.token_sequence) == 2
        assert index.token_pair_count[((97,), (98,))] == 1  # a-b
        assert index.token_pair_count[((99,), (100,))] == 1  # c-d

    def test_populate_longer_word(self):
        index = TokenIndex()
        chunks = [["abc"]]
        index.populate_words(chunks)

        # Pairs: a-b, b-c
        assert index.token_pair_count[((97,), (98,))] == 1
        assert index.token_pair_count[((98,), (99,))] == 1
        assert len(index.token_pair_count) == 2

    def test_get_merged_token(self):
        node1 = TokenIndex.TokenNode((97,))
        node2 = TokenIndex.TokenNode((98,))
        merged = TokenIndex.get_merged_token(node1, node2)
        assert merged == (97, 98)

    def test_get_merged_token_multi_char(self):
        node1 = TokenIndex.TokenNode((97, 98))
        node2 = TokenIndex.TokenNode((99,))
        merged = TokenIndex.get_merged_token(node1, node2)
        assert merged == (97, 98, 99)


class TestMergeTokens:
    def test_simple_merge(self):
        """Test merging 'a' + 'b' in word 'ab'"""
        index = TokenIndex()
        chunks = [["ab"]]
        index.populate_words(chunks)

        pair = ((97,), (98,))
        index.merge_tokens(pair)

        # After merge: head -> (97, 98) -> tail
        ll = index.token_sequence[0]
        merged_node = ll.head.next
        assert merged_node.tokens == (97, 98)
        assert merged_node.next == ll.tail
        assert merged_node.prev == ll.head

    def test_merge_updates_counts(self):
        """Test that pair counts are updated after merge"""
        index = TokenIndex()
        chunks = [["abc"]]
        index.populate_words(chunks)

        # Before: a-b, b-c
        pair_ab = ((97,), (98,))

        index.merge_tokens(pair_ab)

        # After merging a+b: ab-c
        # Old pairs should decrease, new pair should exist
        pair_abc = ((97, 98), (99,))
        assert index.token_pair_count[pair_abc] == 1
