from collections import defaultdict

from pretokenizer import Pretokenizer
from sortedcontainers import SortedList


class Location:
    def __init__(self, chunk_id, start, end):
        self.chunk_id = chunk_id
        self.start = start
        self.end = end


class TokenIndex:
    class TokenNode:
        def __init__(self, tokens, freq=1):
            self.prev: TokenIndex.TokenNode | None = None
            self.next: TokenIndex.TokenNode | None = None
            self.tokens = tokens
            self.freq = freq  # frequency of the word this node belongs to

    class TokenLinkedList:
        def __init__(self):
            self.head: TokenIndex.TokenNode = TokenIndex.TokenNode(None)
            self.tail: TokenIndex.TokenNode = TokenIndex.TokenNode(None)
            self.head.next = self.tail
            self.tail.prev = self.head

    def __init__(self):
        self.token_sequence = defaultdict(lambda: TokenIndex.TokenLinkedList())
        self.pairs_to_token_node = defaultdict(set)  # starting node for each pair
        self.token_pair_count = defaultdict(int)
        self.count_list = SortedList()

    def populate_words(self, chunks):
        vocab = set()

        # Count word frequencies across all chunks — only store unique words
        word_freq = defaultdict(int)
        for chunk in chunks:
            for word in chunk:
                word_freq[word] += 1

        def insert_word(word, word_id, freq):
            linked_list = self.token_sequence[word_id]
            nd = linked_list.head

            for byte in word.encode("utf-8"):
                vocab.add(byte)
                new_token_node = TokenIndex.TokenNode((byte,), freq=freq)
                nd.next = new_token_node
                new_token_node.prev = nd
                nd = new_token_node

                if nd.prev != linked_list.head and nd.prev:
                    pair = (nd.prev.tokens, nd.tokens)
                    self.pairs_to_token_node[pair].add(nd.prev)
                    self.token_pair_count[pair] += freq  # weighted by frequency

            nd.next = linked_list.tail
            linked_list.tail.prev = nd

        for word_id, (word, freq) in enumerate(word_freq.items()):
            insert_word(word, word_id, freq)

        # Populate count_list from token_pair_count
        for pair, cnt in self.token_pair_count.items():
            self.count_list.add((cnt, pair))

        return vocab

    @staticmethod
    def get_merged_token(*args):
        if not args:
            return tuple()
        merged_tokens = []
        for token_node in args:
            merged_tokens += token_node.tokens
        return tuple(merged_tokens)

    def mutate_pair_count(self, pair, delta):
        cur_count = self.token_pair_count[pair]
        self.count_list.discard((cur_count, pair))
        self.count_list.add((cur_count + delta, pair))
        self.token_pair_count[pair] += delta

    def merge_tokens(self, merge_pair):
        starting_nodes = self.pairs_to_token_node[merge_pair]

        def get_pairs_delta(starting_node, new_node):
            delta = {}
            freq = starting_node.freq
            # Left neighbor: check prev exists and has real tokens (not head sentinel)
            if starting_node.prev and starting_node.prev.tokens is not None:
                delta[(starting_node.prev.tokens, starting_node.tokens)] = -freq
                delta[(starting_node.prev.tokens, new_node.tokens)] = freq
            # Right neighbor: check next.next exists and has real tokens (not tail sentinel)
            if starting_node.next.next and starting_node.next.next.tokens is not None:
                delta[(starting_node.next.tokens, starting_node.next.next.tokens)] = -freq
                delta[(new_node.tokens, starting_node.next.next.tokens)] = freq
            return delta

        for starting_node in list(starting_nodes):
            prev = starting_node.prev
            y_node = starting_node.next
            after_next = starting_node.next.next
            new_node = TokenIndex.TokenNode(
                TokenIndex.get_merged_token(starting_node, starting_node.next),
                freq=starting_node.freq,
            )

            # handle count delta
            for pair, delta in get_pairs_delta(starting_node, new_node).items():
                self.mutate_pair_count(pair, delta)

            # update pairs_to_token_node for left neighbor
            if prev.tokens is not None:
                self.pairs_to_token_node[(prev.tokens, starting_node.tokens)].discard(prev)
                self.pairs_to_token_node[(prev.tokens, new_node.tokens)].add(prev)

            # update pairs_to_token_node for right neighbor
            if after_next.tokens is not None:
                self.pairs_to_token_node[(y_node.tokens, after_next.tokens)].discard(y_node)
                self.pairs_to_token_node[(new_node.tokens, after_next.tokens)].add(new_node)

            # revise linked list
            prev.next = new_node
            new_node.prev = prev
            after_next.prev = new_node
            new_node.next = after_next

        # Clean up the merged pair from data structures
        del self.pairs_to_token_node[merge_pair]
        self.count_list.discard((self.token_pair_count[merge_pair], merge_pair))
        del self.token_pair_count[merge_pair]


class BPETokenizer:
    def __init__(self):
        self.merges: list[tuple[bytes, bytes]] = []
        self.vocab: dict[int, bytes] = {}
        self.index = TokenIndex()

    def train(self, input_path, vocab_size, special_tokens: list[bytes], log_interval: int = 100):
        for i in range(256):
            self.vocab[i] = bytes([i])

        for token in special_tokens:
            self.vocab[len(self.vocab)] = token

        special_token = special_tokens[0] if special_tokens else b"<|endoftext|>"
        pretokenized_chunks = Pretokenizer.pretokenize(input_path, special_token)
        self.index.populate_words(pretokenized_chunks)

        target_merges = vocab_size - len(self.vocab)
        merge_count = 0

        while len(self.vocab) < vocab_size:
            merged_pair = self.__merge()
            if merged_pair is None:
                break
            new_token = merged_pair[0] + merged_pair[1]
            self.vocab[len(self.vocab)] = new_token
            self.merges.append(merged_pair)
            merge_count += 1

            if merge_count % log_interval == 0:
                progress = merge_count / target_merges * 100
                token_display = new_token.decode("utf-8", errors="replace")
                print(f"[{merge_count}/{target_merges}] ({progress:.1f}%) merged: {repr(token_display)}")

        print(f"Training complete: {len(self.vocab)} tokens, {len(self.merges)} merges")
        return self.vocab, self.merges

    def __merge(self):
        if not self.index.count_list:
            return None
        # Query most frequent pair
        freq, pair = self.index.count_list[-1]
        if freq <= 0:
            return None
        self.index.merge_tokens(pair)
        # Convert tuple tokens to bytes
        token1 = b"".join(p.encode("utf-8") if isinstance(p, str) else bytes([p]) for p in pair[0])
        token2 = b"".join(p.encode("utf-8") if isinstance(p, str) else bytes([p]) for p in pair[1])
        return (token1, token2)
