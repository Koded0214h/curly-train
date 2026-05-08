from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable


class SentencePieceTokenizer:
    """
    A small SentencePiece-style tokenizer.

    It learns subword pieces over whitespace-normalized text using BPE merges
    with the SentencePiece leading-space marker (U+2581).
    """

    def __init__(self, vocab_size: int = 2000, unk_token: str = "<unk>"):
        self.target_vocab_size = vocab_size
        self.unk_token = unk_token
        self.special_tokens = [unk_token]

        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}
        self.piece_to_id: dict[str, int] = {}
        self.id_to_piece: dict[int, str] = {}
        self.vocab: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.piece_to_id)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.replace("\n", " ").replace("\t", " ").split())

    @staticmethod
    def _word_to_symbols(word: str) -> tuple[str, ...]:
        return tuple(["▁"] + list(word))

    @staticmethod
    def _merge_once(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        merged = pair[0] + pair[1]
        output: list[str] = []
        i = 0

        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                output.append(merged)
                i += 2
            else:
                output.append(symbols[i])
                i += 1

        return tuple(output)

    def _apply_merges(self, symbols: list[str]) -> list[str]:
        if not self.merge_ranks:
            return symbols

        while True:
            best_rank = None
            best_index = None

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = i

            if best_index is None:
                break

            merged_piece = symbols[best_index] + symbols[best_index + 1]
            symbols = symbols[:best_index] + [merged_piece] + symbols[best_index + 2 :]

        return symbols

    def _build_vocab(self, initial_symbols: Iterable[str]) -> None:
        pieces: list[str] = []

        for token in self.special_tokens:
            if token not in pieces:
                pieces.append(token)

        for token in sorted(set(initial_symbols)):
            if token not in pieces:
                pieces.append(token)

        for left, right in self.merges:
            merged = left + right
            if merged not in pieces:
                pieces.append(merged)

        self.piece_to_id = {piece: idx for idx, piece in enumerate(pieces)}
        self.id_to_piece = {idx: piece for piece, idx in self.piece_to_id.items()}
        self.vocab = self.piece_to_id

    def train(self, text: str) -> None:
        normalized = self._normalize(text)
        word_counts = Counter(normalized.split())

        if not word_counts:
            raise ValueError("Tokenizer training text is empty.")

        corpus = Counter()
        initial_symbols: list[str] = [self.unk_token]

        for word, freq in word_counts.items():
            symbols = self._word_to_symbols(word)
            corpus[symbols] += freq
            initial_symbols.extend(symbols)

        while len(set(initial_symbols)) + len(self.merges) + len(self.special_tokens) < self.target_vocab_size:
            pair_counts = Counter()

            for symbols, freq in corpus.items():
                for left, right in zip(symbols, symbols[1:]):
                    pair_counts[(left, right)] += freq

            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < 2:
                break

            self.merge_ranks[best_pair] = len(self.merges)
            self.merges.append(best_pair)

            merged_corpus = Counter()
            for symbols, freq in corpus.items():
                merged_corpus[self._merge_once(symbols, best_pair)] += freq
            corpus = merged_corpus

        self._build_vocab(initial_symbols)

    def encode(self, text: str) -> list[int]:
        normalized = self._normalize(text)
        if not normalized:
            return []

        tokens: list[int] = []

        for word in normalized.split():
            pieces = self._apply_merges(list(self._word_to_symbols(word)))
            for piece in pieces:
                token_id = self.piece_to_id.get(piece, self.piece_to_id[self.unk_token])
                tokens.append(token_id)

        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        pieces: list[str] = []

        for token in tokens:
            piece = self.id_to_piece.get(int(token), self.unk_token)
            if piece == self.unk_token:
                continue
            pieces.append(piece)

        text = "".join(pieces).replace("▁", " ").strip()
        return text

    def save(self, path: str | Path) -> None:
        data = {
            "target_vocab_size": self.target_vocab_size,
            "unk_token": self.unk_token,
            "special_tokens": self.special_tokens,
            "merges": self.merges,
            "piece_to_id": self.piece_to_id,
        }

        path = Path(path)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SentencePieceTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(
            vocab_size=data["target_vocab_size"],
            unk_token=data["unk_token"],
        )
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer.merge_ranks = {pair: idx for idx, pair in enumerate(tokenizer.merges)}
        tokenizer.piece_to_id = {piece: int(idx) for piece, idx in data["piece_to_id"].items()}
        tokenizer.id_to_piece = {idx: piece for piece, idx in tokenizer.piece_to_id.items()}
        tokenizer.vocab = tokenizer.piece_to_id
        return tokenizer
