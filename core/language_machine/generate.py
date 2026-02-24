import heapq
from typing import List, Optional

import regex as re
import torch
from torch import Tensor

from language_machine.transformer.utils.softmax import softmax

_PRETOKENIZE_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """
    Scale logits by temperature.

    Args:
        logits: (vocab_size,) unnormalized logits
        temperature: Temperature value.
            - temperature=1.0: no change
            - temperature<1.0: sharper distribution (more deterministic)
            - temperature>1.0: flatter distribution (more random)

    Returns:
        Scaled logits
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature


def top_p_filtering(logits: Tensor, top_p: float) -> Tensor:
    """
    Apply top-p (nucleus) filtering to logits.

    Keeps the smallest set of tokens whose cumulative probability >= top_p,
    setting all other logits to -inf.

    Args:
        logits: (vocab_size,) unnormalized logits
        top_p: Cumulative probability threshold (0.0 to 1.0)
            - top_p=1.0: no filtering (keep all tokens)
            - top_p=0.9: keep tokens comprising top 90% probability mass

    Returns:
        Filtered logits with unlikely tokens set to -inf
    """
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("top_p must be positive")

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Compute cumulative probabilities
    probs = softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    # Find cutoff index: first position where cumulative prob exceeds top_p
    # We shift by 1 to keep at least one token
    sorted_mask = cumulative_probs > top_p
    sorted_mask[1:] = sorted_mask[:-1].clone()
    sorted_mask[0] = False

    # Set filtered tokens to -inf
    sorted_logits[sorted_mask] = float("-inf")

    # Unsort to restore original order
    filtered_logits = torch.zeros_like(logits)
    filtered_logits.scatter_(0, sorted_indices, sorted_logits)

    return filtered_logits


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    """
    Sample a single token from logits with temperature and top-p filtering.

    Args:
        logits: (vocab_size,) unnormalized logits for next token
        temperature: Softmax temperature
        top_p: Nucleus sampling threshold

    Returns:
        Sampled token ID
    """
    # Apply temperature
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)

    # Apply top-p filtering
    if top_p < 1.0:
        logits = top_p_filtering(logits, top_p)

    # Sample from distribution
    probs = softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()

    return token_id


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_tokens: List[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
) -> List[int]:
    """
    Generate tokens from a language model given a prompt.

    Args:
        model: TransformerLM model
        prompt_tokens: List of token IDs for the prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Softmax temperature for sampling
            - 1.0: standard sampling
            - <1.0: more deterministic
            - >1.0: more random
        top_p: Nucleus sampling threshold (0.0-1.0)
            - 1.0: no filtering
            - 0.9: sample from top 90% probability mass
        eos_token_id: Stop generation when this token is sampled (e.g., <|endoftext|>)
        device: Device to run on

    Returns:
        List of all tokens (prompt + generated)
    """
    model.eval()

    # Convert to tensor
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Get context length from model (assume it's stored or infer from position embeddings)
    # For now, we'll track and truncate if needed
    context_length = getattr(model, "context_length", None)

    generated = list(prompt_tokens)

    for _ in range(max_new_tokens):
        # Truncate to context length if needed
        if context_length and tokens.shape[1] > context_length:
            tokens = tokens[:, -context_length:]

        # Forward pass
        logits = model(tokens)  # (1, seq_len, vocab_size)

        # Get logits for the last position
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # Sample next token
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
        )

        # Check for EOS
        if eos_token_id is not None and next_token == eos_token_id:
            break

        # Append to sequence
        generated.append(next_token)
        tokens = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)

    return generated


def encode(text: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> List[int]:
    """
    Encode text to token IDs using BPE vocab and merges.

    Args:
        text: Input text string
        vocab: Mapping from token ID to bytes
        merges: List of BPE merge pairs in order of creation

    Returns:
        List of token IDs
    """
    bytes_to_id = {v: k for k, v in vocab.items()}
    # O(1) rank lookup: lower rank = earlier merge = higher priority
    merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    def encode_word(word: str) -> List[bytes]:
        # Start as individual bytes
        tokens = [bytes([b]) for b in word.encode("utf-8")]
        if len(tokens) == 1:
            return tokens

        # Build a min-heap of (rank, position) for every adjacent pair
        # that exists in the merge table. Stale entries are lazily filtered.
        heap = []
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in merge_rank:
                heapq.heappush(heap, (merge_rank[pair], i))

        # Track which positions are still valid (not yet consumed by a merge)
        removed = set()

        while heap:
            rank, i = heapq.heappop(heap)
            # Skip stale entries
            if i in removed:
                continue
            # Find the actual right neighbour (skip removed slots)
            j = i + 1
            while j in removed:
                j += 1
            if j >= len(tokens):
                continue
            pair = (tokens[i], tokens[j])
            if merge_rank.get(pair) != rank:
                continue  # stale: pair changed since pushed

            merged = pair[0] + pair[1]
            tokens[i] = merged
            removed.add(j)

            # Check new left pair (predecessor of i → i)
            li = i - 1
            while li in removed:
                li -= 1
            if li >= 0:
                new_pair = (tokens[li], merged)
                if new_pair in merge_rank:
                    heapq.heappush(heap, (merge_rank[new_pair], li))

            # Check new right pair (i → successor of j)
            rj = j + 1
            while rj in removed:
                rj += 1
            if rj < len(tokens):
                new_pair = (merged, tokens[rj])
                if new_pair in merge_rank:
                    heapq.heappush(heap, (merge_rank[new_pair], i))

        return [t for idx, t in enumerate(tokens) if idx not in removed]

    words = re.findall(_PRETOKENIZE_REGEX, text)
    token_ids = []
    for word in words:
        for tok in encode_word(word):
            token_ids.append(bytes_to_id[tok])
    return token_ids


def decode(token_ids: List[int], vocab: dict[int, bytes]) -> str:
    """
    Decode token IDs to text.

    Args:
        token_ids: List of token IDs
        vocab: Mapping from token ID to bytes

    Returns:
        Decoded text string
    """
    # Concatenate all token bytes
    text_bytes = b"".join(vocab[token_id] for token_id in token_ids)

    # Decode to string (handle invalid UTF-8 gracefully)
    return text_bytes.decode("utf-8", errors="replace")


def get_eos_token_id(vocab: dict[int, bytes], eos_token: str = "<|endoftext|>") -> Optional[int]:
    """
    Find the token ID for the end-of-sequence token.

    In standard BPE vocab layout:
    - 0-255: raw byte tokens
    - 256: first special token (typically <|endoftext|>)
    - 257+: additional special tokens, then merges

    Args:
        vocab: Mapping from token ID to bytes
        eos_token: The EOS token string to search for

    Returns:
        Token ID if found, None otherwise
    """
    eos_bytes = eos_token.encode("utf-8")

    # Fast path: check index 256 first (standard location for first special token)
    if 256 in vocab and vocab[256] == eos_bytes:
        return 256

    # Fallback: search entire vocab
    for token_id, token_bytes in vocab.items():
        if token_bytes == eos_bytes:
            return token_id
    return None


def generate_text(
    model: torch.nn.Module,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str = "<|endoftext|>",
    device: str = "cpu",
) -> str:
    """
    Generate text from a prompt string.

    Args:
        model: TransformerLM model
        vocab: BPE vocabulary mapping token ID -> bytes
        merges: BPE merge pairs in order
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        eos_token: End of sequence token string
        device: Device to run on

    Returns:
        Generated text (including prompt)
    """
    # Encode prompt
    prompt_tokens = encode(prompt, vocab, merges)

    # Get EOS token ID
    eos_token_id = get_eos_token_id(vocab, eos_token)

    # Generate
    output_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device,
    )

    # Decode
    return decode(output_tokens, vocab)
