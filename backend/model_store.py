from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import glob
import json
import re
from pathlib import Path

import torch

import tokenizer as tok
from model import TransformerLanguageModel


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATTERN = str(ROOT / "model_step_*.pt")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    embed_size: int
    block_size: int
    num_layers: int
    num_heads: int


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _checkpoint_sort_key(path: Path) -> tuple[int, float]:
    try:
        ckpt = torch.load(path, map_location="cpu")
        step = int(ckpt.get("step", -1))
    except Exception:
        step = -1
    return step, path.stat().st_mtime


def latest_checkpoint_path() -> Path:
    candidates = [Path(path) for path in glob.glob(CHECKPOINT_PATTERN)]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found. Expected files matching {CHECKPOINT_PATTERN!r}."
        )
    return max(candidates, key=_checkpoint_sort_key)


def infer_config(state_dict: dict[str, torch.Tensor]) -> ModelConfig:
    vocab_size, embed_size = state_dict["token_embedding.weight"].shape
    block_size = state_dict["position_embedding.weight"].shape[0]

    layer_ids: set[int] = set()
    head_ids: set[int] = set()

    pattern = re.compile(r"^blocks\.(\d+)\.sa\.heads\.(\d+)\.key\.weight$")
    for key in state_dict:
        match = pattern.match(key)
        if match:
            layer_ids.add(int(match.group(1)))
            head_ids.add(int(match.group(2)))

    if not layer_ids or not head_ids:
        raise ValueError("Unable to infer transformer depth from checkpoint state.")

    return ModelConfig(
        vocab_size=vocab_size,
        embed_size=embed_size,
        block_size=block_size,
        num_layers=max(layer_ids) + 1,
        num_heads=max(head_ids) + 1,
    )


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, object]:
    tokenizer = tok.build_tokenizer()
    checkpoint_path = latest_checkpoint_path()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state"]
    config = infer_config(state_dict)

    device = get_device()
    model = TransformerLanguageModel(
        vocab_size=config.vocab_size,
        embed_size=config.embed_size,
        block_size=config.block_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "tokenizer": tokenizer,
        "model": model,
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "config": config,
        "device": device,
    }


def generate_text(
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
) -> dict[str, object]:
    bundle = load_bundle()
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    checkpoint_path = bundle["checkpoint_path"]
    config = bundle["config"]

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    generated_ids = generated[0].tolist()
    full_text = tokenizer.decode(generated_ids)
    continuation_ids = generated_ids[len(prompt_ids) :]
    continuation_text = tokenizer.decode(continuation_ids)

    return {
        "checkpoint": checkpoint_path.name,
        "config": asdict(config),
        "prompt_token_count": len(prompt_ids),
        "generated_token_count": len(generated_ids),
        "prompt": prompt,
        "continuation_text": continuation_text,
        "full_text": full_text,
        "token_ids": generated_ids,
        "device": str(device),
    }


def stream_generation(
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
):
    bundle = load_bundle()
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    checkpoint_path = bundle["checkpoint_path"]
    config = bundle["config"]

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = list(prompt_ids)
    prompt_text = tokenizer.decode(prompt_ids)
    previous_text = prompt_text

    yield {
        "type": "start",
        "checkpoint": checkpoint_path.name,
        "config": asdict(config),
        "prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "device": str(device),
    }

    for next_idx, idx in model.generate_stream(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    ):
        generated_ids.append(int(next_idx.item()))
        full_text = tokenizer.decode(generated_ids)
        delta = full_text[len(previous_text) :]
        previous_text = full_text

        yield {
            "type": "token",
            "token_id": int(next_idx.item()),
            "generated_token_count": len(generated_ids),
            "full_text": full_text,
            "delta": delta,
            "continuation_text": full_text[len(prompt_text) :],
            "token_ids": generated_ids.copy(),
        }

    yield {
        "type": "done",
        "checkpoint": checkpoint_path.name,
        "config": asdict(config),
        "generated_token_count": len(generated_ids),
        "full_text": tokenizer.decode(generated_ids),
        "token_ids": generated_ids,
    }
