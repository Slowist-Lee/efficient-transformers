from __future__ import annotations

import collections
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import regex
import tiktoken


class _Tokenizer:
    def __init__(self, enc: tiktoken.Encoding, special_token_to_id: dict[str, int]):
        self._enc = enc
        self._special_token_to_id = special_token_to_id
        self._special_id_to_token = {v: k for k, v in special_token_to_id.items()}
        self._sorted_special_tokens = sorted(special_token_to_id.keys(), key=len, reverse=True)
        if self._sorted_special_tokens:
            escaped = [re.escape(token) for token in self._sorted_special_tokens]
            self._special_pattern = re.compile("|".join(escaped))
        else:
            self._special_pattern = None

    def _encode_without_special(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special=set())

    def encode(self, text: str) -> list[int]:
        if not self._special_pattern:
            return self._encode_without_special(text)

        ids: list[int] = []
        cursor = 0
        for match in self._special_pattern.finditer(text):
            if match.start() > cursor:
                ids.extend(self._encode_without_special(text[cursor : match.start()]))
            ids.append(self._special_token_to_id[match.group(0)])
            cursor = match.end()

        if cursor < len(text):
            ids.extend(self._encode_without_special(text[cursor:]))
        return ids

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        normal_ids: list[int] = []

        def flush_normal_ids() -> None:
            if normal_ids:
                pieces.append(self._enc.decode(normal_ids))
                normal_ids.clear()

        for token_id in ids:
            special_token = self._special_id_to_token.get(token_id)
            if special_token is not None:
                flush_normal_ids()
                pieces.append(special_token)
            else:
                normal_ids.append(token_id)

        flush_normal_ids()
        return "".join(pieces)

    def encode_iterable(self, iterable: Iterable[str]):
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id


def gpt2_bytes_to_unicode_local() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_stats(token_sequences: list[list[str]]) -> collections.Counter:
    pair_counts = collections.Counter()
    for sequence in token_sequences:
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            pair_counts[pair] += 1
    return pair_counts


def merge_pair_in_sequences(
    token_sequences: list[list[str]],
    pair_to_merge: tuple[str, str],
    new_token_representation: str,
) -> list[list[str]]:
    new_overall_sequences: list[list[str]] = []
    (p1, p2) = pair_to_merge
    for sequence in token_sequences:
        new_sequence: list[str] = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and sequence[i] == p1 and sequence[i + 1] == p2:
                new_sequence.append(new_token_representation)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        new_overall_sequences.append(new_sequence)
    return new_overall_sequences


def merge_token_sequence(token_seq: tuple[bytes, ...], best_pair: tuple[bytes, bytes], new_token: bytes) -> tuple[bytes, ...]:
    new_seq: list[bytes] = []
    i = 0
    while i < len(token_seq):
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == best_pair:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(token_seq[i])
            i += 1
    return tuple(new_seq)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    del merges  # tiktoken uses mergeable ranks from vocab token ids.

    gpt2_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    special_tokens = special_tokens or []
    special_token_bytes = {s.encode("utf-8"): s for s in special_tokens}

    mergeable_ranks: dict[bytes, int] = {}
    special_tokens_map: dict[str, int] = {}
    for token_id, token_bytes in vocab.items():
        special_str = special_token_bytes.get(token_bytes)
        if special_str is not None:
            special_tokens_map[special_str] = token_id
        else:
            mergeable_ranks[token_bytes] = token_id

    encoding = tiktoken.Encoding(
        name="cs336-custom",
        pat_str=gpt2_pat,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens_map,
    )

    return _Tokenizer(encoding, special_tokens_map)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    del kwargs

    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    current_next_id = 256

    token_frequency_table: dict[tuple[bytes, ...], int] = defaultdict(int)
    existing_byte_values = set(vocab.values())

    for special_token in special_tokens:
        if len(vocab) >= vocab_size:
            break
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in existing_byte_values:
            vocab[current_next_id] = token_bytes
            existing_byte_values.add(token_bytes)
            current_next_id += 1

    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    if special_tokens:
        split_pattern = "|".join(map(regex.escape, special_tokens))
        chunks = regex.split(split_pattern, text)
    else:
        chunks = [text]

    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        for word in regex.findall(pat, chunk):
            word_bytes = word.encode("utf-8")
            bytes_list = [bytes([byte_value]) for byte_value in word_bytes]
            token_frequency_table[tuple(bytes_list)] += 1

    merges: list[tuple[bytes, bytes]] = []

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for token in token_frequency_table:
        freq = token_frequency_table[token]
        for i in range(len(token) - 1):
            pair_counts[(token[i], token[i + 1])] += freq

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        max_count = max(pair_counts.values())
        candidates = [pair for pair, count in pair_counts.items() if count == max_count]
        best_pair = max(candidates)
        merges.append(best_pair)

        new_token_bytes = best_pair[0] + best_pair[1]
        vocab[current_next_id] = new_token_bytes
        current_next_id += 1

        affected_tokens: list[tuple[tuple[bytes, ...], int]] = []
        for token, freq in token_frequency_table.items():
            if any(token[i : i + 2] == best_pair for i in range(len(token) - 1)):
                affected_tokens.append((token, freq))

        for token, freq in affected_tokens:
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            merged_token = merge_token_sequence(token, best_pair, new_token_bytes)

            for i in range(len(merged_token) - 1):
                pair = (merged_token[i], merged_token[i + 1])
                pair_counts[pair] += freq

            del token_frequency_table[token]
            token_frequency_table[merged_token] += freq

    return vocab, merges
