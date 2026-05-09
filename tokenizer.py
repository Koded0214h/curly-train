from pathlib import Path

from sentencepiece_tokenizer import SentencePieceTokenizer


ROOT = Path(__file__).resolve().parent
TEXT_PATH = ROOT / "processed" / "dialogue.txt"
MODEL_PATH = ROOT / "tokenizer.model.json"
DEFAULT_VOCAB_SIZE = 2000

_tokenizer: SentencePieceTokenizer | None = None
vocab_size = 0


def build_tokenizer(
    force_retrain: bool = False,
    target_vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> SentencePieceTokenizer:
    global _tokenizer, vocab_size

    should_retrain = force_retrain or not MODEL_PATH.exists()

    if not should_retrain and TEXT_PATH.exists():
        should_retrain = TEXT_PATH.stat().st_mtime > MODEL_PATH.stat().st_mtime

    if should_retrain:
        text = TEXT_PATH.read_text(encoding="utf-8")
        tokenizer = SentencePieceTokenizer(vocab_size=target_vocab_size)
        tokenizer.train(text)
        tokenizer.save(MODEL_PATH)
    else:
        tokenizer = SentencePieceTokenizer.load(MODEL_PATH)

    _tokenizer = tokenizer
    vocab_size = tokenizer.vocab_size
    return tokenizer


def get_tokenizer() -> SentencePieceTokenizer:
    global _tokenizer

    if _tokenizer is None:
        _tokenizer = build_tokenizer(force_retrain=False)

    return _tokenizer


def encode(text: str) -> list[int]:
    return get_tokenizer().encode(text)


def decode(tokens) -> str:
    return get_tokenizer().decode(tokens)


def get_vocab_size() -> int:
    return get_tokenizer().vocab_size


build_tokenizer(force_retrain=False)
