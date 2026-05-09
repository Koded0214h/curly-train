# Curly Train

A from-scratch autoregressive transformer language model trained on *Classroom of the Elite* anime dialogue, with a streaming web interface and a live 3D token graph visualiser.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Quick Start](#5-quick-start)
6. [Manual Setup](#6-manual-setup)
   - 6.1 [Python environment](#61-python-environment)
   - 6.2 [Frontend dependencies](#62-frontend-dependencies)
   - 6.3 [Running the backend](#63-running-the-backend)
   - 6.4 [Running the frontend](#64-running-the-frontend)
7. [Data Pipeline](#7-data-pipeline)
   - 7.1 [Raw subtitle format](#71-raw-subtitle-format)
   - 7.2 [Cleaning the data](#72-cleaning-the-data)
   - 7.3 [Adding more episodes](#73-adding-more-episodes)
8. [Tokenizer](#8-tokenizer)
9. [Training the Model](#9-training-the-model)
   - 9.1 [Hyperparameters](#91-hyperparameters)
   - 9.2 [Running training](#92-running-training)
   - 9.3 [Monitoring training](#93-monitoring-training)
   - 9.4 [Checkpoints](#94-checkpoints)
10. [Web Interface](#10-web-interface)
    - 10.1 [Controls](#101-controls)
    - 10.2 [Streaming output](#102-streaming-output)
    - 10.3 [3D Token Graph](#103-3d-token-graph)
11. [API Reference](#11-api-reference)
12. [Configuration](#12-configuration)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project Overview

Curly Train is a self-contained generative AI project built entirely from first principles:

- **No pre-trained weights** — the model is trained from zero on a small, focused dialogue corpus.
- **Custom BPE tokenizer** — a SentencePiece-style byte-pair encoding implementation written in pure Python; no external tokenizer libraries required.
- **GPT-style transformer** — multi-head causal self-attention, pre-norm residual blocks, GELU activations, dropout regularisation, and cosine learning-rate decay.
- **Streaming API** — the FastAPI backend emits server-sent events (SSE) so the browser receives tokens one at a time as they are sampled.
- **3D visualiser** — a Three.js scene that places each token as a node on a fibonacci sphere and draws sequential and vocabulary-cluster edges between them.

The corpus is nine episodes of *Classroom of the Elite* (COTE) English subtitle files (`.srt`). After cleaning, this yields roughly **3,177 lines of dialogue** — a deliberately tiny dataset that makes a full training run finish in under an hour on a MacBook GPU (MPS) and in minutes on a discrete GPU.

---

## 2. Architecture

### Model

| Component | Detail |
|---|---|
| Type | Decoder-only transformer (GPT-style) |
| Layers | 4 transformer blocks |
| Attention heads | 8 per block |
| Embedding dimension | 256 |
| Feed-forward dimension | 1 024 (4× embed) |
| Context window | 128 tokens |
| Activation | GELU |
| Normalisation | Pre-LayerNorm on attention and FFN inputs |
| Regularisation | Dropout (0.1) on attention weights, FFN output, and residual connections |
| Positional encoding | Learned absolute embeddings |
| Parameters | ~13 M |

### Tokenizer

| Component | Detail |
|---|---|
| Algorithm | Byte-pair encoding (BPE) with SentencePiece leading-space marker (`▁`) |
| Vocabulary size | 2 000 subword pieces |
| Special tokens | `<unk>` |
| Storage format | JSON (merge rules + piece→id mapping) |

### Web stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI 0.115, Uvicorn 0.34 |
| Frontend | React 18, Vite 5, Three.js 0.169 |
| Transport | REST (`/api/generate`) + SSE (`/api/generate/stream`) |
| Dev proxy | Vite proxies `/api/*` → `localhost:8000` — no CORS issues |

---

## 3. Project Structure

```
curly-train/
│
├── run.sh                    # One-command launcher (backend + frontend)
│
├── data/                     # Raw .srt subtitle files (source corpus)
│   ├── e01.srt
│   ├── e02.srt
│   └── ...
│
├── processed/
│   └── dialogue.txt          # Cleaned dialogue lines (output of clean_data.py)
│
├── clean_data.py             # SRT → dialogue.txt pipeline
├── tokenizer.py              # Tokenizer entry point (build / load / encode / decode)
├── sentencepiece_tokenizer.py# BPE implementation
├── tokenizer.model.json      # Trained tokenizer (auto-generated)
│
├── self_attention.py         # SelfAttentionHead + MultiHeadAttention
├── model.py                  # TransformerLanguageModel
├── train.py                  # Training loop with eval, checkpointing, loss curve
│
├── model_step_*.pt           # Checkpoints saved every eval_interval steps
├── output.log                # Training log (loss + generated samples)
├── loss_curve.png            # Loss curve saved after training
│
├── server_app.py             # FastAPI app (all routes)
├── requirements.txt          # Python dependencies
│
├── backend/
│   ├── __init__.py
│   ├── main.py               # Uvicorn entry point (re-exports app from server_app)
│   └── model_store.py        # Model loading, inference, streaming helpers
│
└── frontend/
    ├── vite.config.js        # Vite config with /api proxy
    ├── package.json
    └── src/
        ├── main.jsx          # React root
        ├── App.jsx           # Main UI (streaming, controls, layout)
        ├── Visualizer.jsx    # Three.js 3D token graph
        ├── api.js            # fetch + SSE helpers
        └── styles.css        # Full stylesheet
```

---

## 4. Prerequisites

### Python

- **Python 3.10 or higher** (project uses 3.12)
- Check: `python3 --version`

### Node.js

- **Node.js 18 or higher**
- Check: `node --version`

### GPU (optional but recommended)

| Platform | Backend | Notes |
|---|---|---|
| Mac (M-series) | MPS | Automatically detected — roughly 5–10× faster than CPU |
| Linux / Windows with NVIDIA | CUDA | Automatically detected |
| Any | CPU | Works, but training is slow (~3–4× slower than MPS on M2) |

The web server runs fine on CPU only; GPU matters for training speed.

---

## 5. Quick Start

Clone the repo, then from the project root:

```bash
sh run.sh
```

The script will:

1. Create a `.venv` Python virtual environment if one does not already exist.
2. Activate the venv and install all Python dependencies from `requirements.txt`.
3. Run `npm install` in `frontend/` if `node_modules` is missing.
4. Start the **backend** on `http://localhost:8000`.
5. Start the **frontend** on `http://localhost:5173`.

Open `http://localhost:5173` in your browser. Press **Ctrl+C** in the terminal to stop both servers cleanly.

> **Pre-trained checkpoint included.** The repository ships with `model_step_2700.pt` (the furthest trained checkpoint). The server loads it automatically — you do not need to train first to use the web interface.

---

## 6. Manual Setup

Use this if you want finer control over each component.

### 6.1 Python environment

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 6.3 Running the backend

From the **project root** (not from inside `backend/`):

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- `--reload` enables hot-reload on file changes (useful during development; omit in production).
- The server loads the latest checkpoint on startup. First load may take a few seconds while the model is read from disk and mapped to the GPU.
- Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

### 6.4 Running the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies all `/api/*` requests to the backend on port 8000, so there are no cross-origin issues.

#### Production build

To serve the frontend as static files from FastAPI (single-server deployment):

```bash
cd frontend
npm run build      # outputs to frontend/dist/
cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

FastAPI automatically mounts `frontend/dist/` at `/` if the directory exists, so the entire app is served from one port.

---

## 7. Data Pipeline

### 7.1 Raw subtitle format

Source files live in `data/` as standard `.srt` subtitle files:

```
1
00:00:14,070 --> 00:00:16,990
I see. So this is
a special privilege.

2
00:00:17,800 --> 00:00:20,490
It's a school where you can gain
a bright future.
```

### 7.2 Cleaning the data

Run the cleaning script to convert `.srt` files into a flat dialogue corpus:

```bash
source .venv/bin/activate
python clean_data.py
```

This script:
- Strips subtitle numbering, timestamps, and HTML tags (`<i>`, `<b>`, etc.).
- Removes sound effect annotations (`[applause]`, `[crowd cheering]`, etc.).
- Filters lines that contain ranking/score keywords (`average`, `points`, `1st`, etc.) which are not useful dialogue.
- Drops lines that are more than 30 % numeric characters.
- Joins multi-line subtitle blocks into single lines.
- Writes the result to `processed/dialogue.txt`.

Output: **one dialogue line per subtitle block**, no timestamps, no tags.

### 7.3 Adding more episodes

1. Download or obtain `.srt` subtitle files for additional episodes.
2. Place them in `data/` with any filename ending in `.srt`.
3. Re-run the cleaning script:
   ```bash
   python clean_data.py
   ```
4. Re-train the tokenizer and model (see §8 and §9). The tokenizer auto-retrains when `dialogue.txt` is newer than `tokenizer.model.json`.

> More episodes = larger, more diverse corpus = better generalisation. The current 9-episode corpus is intentionally small so training is fast and reproducible.

---

## 8. Tokenizer

The tokenizer is a from-scratch BPE implementation (`sentencepiece_tokenizer.py`) that mirrors the SentencePiece convention:

- Every word is prefixed with `▁` (Unicode U+2581, a lower-one-eighth block) to mark word boundaries.
- Characters are the initial symbols; the BPE training loop greedily merges the most frequent adjacent pairs until the vocabulary reaches the target size.
- The trained tokenizer is saved to `tokenizer.model.json` as a JSON file containing the merge list and piece→id mapping.

The tokenizer is loaded automatically by both the training script and the web server. You do not need to train it manually — `tokenizer.py` checks whether `tokenizer.model.json` is older than `dialogue.txt` and retrains if necessary.

To force a retrain:

```bash
python -c "import tokenizer; tokenizer.build_tokenizer(force_retrain=True)"
```

To inspect the vocabulary:

```python
from tokenizer import build_tokenizer
t = build_tokenizer()
print(t.vocab_size)            # 1999
print(t.encode("I understand"))
print(t.decode([123, 456]))
```

---

## 9. Training the Model

### 9.1 Hyperparameters

All hyperparameters are at the top of `train.py`:

| Parameter | Default | Notes |
|---|---|---|
| `block_size` | 128 | Context window in tokens |
| `batch_size` | 32 | Halved to 16 automatically on CPU |
| `max_iters` | 3 000 | Total training steps |
| `eval_interval` | 300 | Steps between evaluations and checkpoints |
| `eval_iters` | 64 | Mini-batches averaged for stable val loss |
| `embed_size` | 256 | Embedding / hidden dimension |
| `num_layers` | 4 | Number of transformer blocks |
| `num_heads` | 8 | Attention heads per block |
| `dropout` | 0.1 | Applied to attention, FFN, and residuals |
| `lr` | 3e-4 | Peak learning rate (AdamW) |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `temperature` | 0.8 | Sampling temperature for training samples |
| `top_k` | 50 | Top-k sampling for training samples |

The learning rate follows a **cosine decay schedule** with a **100-step linear warmup**: it rises linearly from 0 to `lr` over the first 100 steps, then decays with a cosine curve down to 10 % of peak by the final step.

### 9.2 Running training

```bash
source .venv/bin/activate
python train.py
```

Training will print a status line every `eval_interval` steps:

```
STEP    0 | train loss 7.5912 | val loss 7.6031 | lr 0.00e+00
STEP  300 | train loss 4.2801 | val loss 4.5113 | lr 3.00e-04
STEP  600 | train loss 3.7244 | val loss 4.1882 | lr 2.99e-04
...
```

Both **training loss** and **validation loss** are reported (averaged over 64 mini-batches each, measured in `model.eval()` mode so dropout is disabled during evaluation).

When training finishes:
- A final 500-token generation sample is printed to the console and appended to `output.log`.
- A **loss curve** is saved to `loss_curve.png` (train and val lines).

Expected training times on the default 9-episode corpus:

| Hardware | Time for 3 000 steps |
|---|---|
| Apple M2 (MPS) | ~25–40 minutes |
| NVIDIA RTX 3080 (CUDA) | ~5–8 minutes |
| CPU only | ~2–3 hours |

### 9.3 Monitoring training

**Live log tail:**

```bash
tail -f output.log
```

The log records every evaluation step with a generated text sample so you can see how the model's language quality improves over training.

**Loss curve** (after training):

```bash
open loss_curve.png   # macOS
# or just view it in any image viewer
```

### 9.4 Checkpoints

A checkpoint is saved at every `eval_interval` step as `model_step_<N>.pt` in the project root. Each checkpoint contains:

```python
{
    "model_state":     ...,   # model weights
    "optimizer_state": ...,   # AdamW state (allows resuming)
    "scheduler_state": ...,   # LR scheduler state
    "step":            ...,   # training step number
    "val_loss":        ...,   # validation loss at this checkpoint
}
```

The web server (`model_store.py`) automatically finds and loads the **highest-step checkpoint** at startup using the `step` field stored inside each file — so whichever checkpoint has the largest step number is used, regardless of filename sort order.

To load a specific checkpoint in Python:

```python
import torch
from model_store import infer_config
from model import TransformerLanguageModel

ckpt = torch.load("model_step_1800.pt", map_location="cpu")
cfg  = infer_config(ckpt["model_state"])
model = TransformerLanguageModel(**cfg.__dict__)
model.load_state_dict(ckpt["model_state"])
model.eval()
```

To resume training from a checkpoint, load both `model_state` and `optimizer_state` and set `step` as the starting iteration (edit `train.py` accordingly).

---

## 10. Web Interface

### 10.1 Controls

| Control | Range | Effect |
|---|---|---|
| **Prompt** | Free text | The text the model continues from |
| **Temperature** | 0.20 – 1.80 | Higher = more random / creative; lower = more repetitive / focused |
| **Top-k** | 5 – 200 | Only the top-k most likely tokens are considered at each step; lower = safer, higher = more varied |
| **Max tokens** | 32 – 512 | Maximum number of new tokens to generate |

Click **Generate** to start streaming. Click **Stop** (the same button, which changes label while generating) to abort mid-generation. Click **Copy** to copy the full text to the clipboard.

### 10.2 Streaming output

Generation uses the `/api/generate/stream` SSE endpoint. Tokens appear in the output panel one by one as they are sampled — you do not wait for the full sequence before seeing output. A blinking cursor shows that generation is in progress; a green **Done** badge appears when it completes.

The prompt text is shown in a dimmed colour; the continuation is shown in full brightness, making it easy to see where the model's output begins.

### 10.3 3D Token Graph

The visualiser in the centre panel renders all tokens in the current text as a **3D graph** using Three.js:

- **Nodes** — one sphere per token, sized proportionally to token length, coloured by a deterministic hash of the token string.
- **Sequential edges** — semi-transparent lines connecting each token to the next in sequence (the natural reading order path through the graph).
- **Vocabulary cluster edges** — cyan edges connecting tokens that share the same first three characters, highlighting repeated roots or prefixes in the vocabulary.
- **Labels** — billboard text sprites above each node showing the token string, tinted to match the node colour.
- **Camera** — orbits the graph continuously on a slow circular path, rising and falling gently. Nodes breathe with a subtle scale pulse.

The graph updates every time a new generation run completes. During streaming, the graph reflects the growing token sequence in real time.

---

## 11. API Reference

The backend exposes a JSON REST API and an SSE stream. Interactive docs are at `http://localhost:8000/docs`.

### `GET /api/health`

Returns server status and loaded model configuration.

```json
{
  "ok": true,
  "checkpoint": "model_step_2700.pt",
  "device": "mps",
  "config": {
    "vocab_size": 1999,
    "embed_size": 256,
    "block_size": 128,
    "num_layers": 4,
    "num_heads": 8
  }
}
```

### `GET /api/model`

Returns model info without the `ok` field. Used by the frontend to populate the header badges.

### `POST /api/generate`

Generates a full continuation synchronously and returns when complete.

**Request body:**

```json
{
  "prompt": "Ayanokoji-kun,",
  "max_new_tokens": 120,
  "temperature": 0.8,
  "top_k": 50
}
```

**Response:**

```json
{
  "checkpoint": "model_step_2700.pt",
  "config": { ... },
  "prompt": "Ayanokoji-kun,",
  "prompt_token_count": 4,
  "generated_token_count": 124,
  "continuation_text": " what are you thinking about?",
  "full_text": "Ayanokoji-kun, what are you thinking about?",
  "token_ids": [12, 34, ...],
  "device": "mps"
}
```

### `GET /api/generate/stream`

Streams generation as **server-sent events (SSE)**. Parameters are passed as query strings.

```
GET /api/generate/stream?prompt=Hello&max_new_tokens=120&temperature=0.8&top_k=50
```

Event types:

| Type | When | Key fields |
|---|---|---|
| `start` | Once, before first token | `checkpoint`, `config`, `prompt_token_count` |
| `token` | Once per generated token | `delta`, `full_text`, `generated_token_count` |
| `done` | Once, after last token | `full_text`, `token_ids`, `generated_token_count` |
| `error` | On exception | `detail` |

Each event is a standard SSE frame: `data: <json>\n\n`.

---

## 12. Configuration

### Backend

| Environment variable | Default | Effect |
|---|---|---|
| `VITE_API_URL` | `""` (relative) | Override the backend URL the frontend connects to |

In production, set `VITE_API_URL` at **build time**:

```bash
VITE_API_URL=https://your-backend.example.com npm run build
```

### Training

Edit the constants at the top of `train.py`. Key decisions:

- **`block_size`** — increasing this allows longer-range dependencies but increases memory quadratically in the attention layer.
- **`embed_size` / `num_layers` / `num_heads`** — scale up for a larger model; scale down for faster iteration. Keep `embed_size` divisible by `num_heads`.
- **`dropout`** — 0.0 turns off regularisation (useful for very small datasets where underfitting is the problem); 0.1–0.2 is standard.

> **Note:** Changing `embed_size`, `num_layers`, or `num_heads` changes the model architecture. Existing checkpoints trained with the old architecture will not load into the new model — you must start training from scratch.

---

## 13. Troubleshooting

### Backend returns 500 / frontend shows "Generation failed"

1. Check the backend terminal for the Python traceback.
2. Most common cause: no checkpoint file found. Make sure at least one `model_step_*.pt` file exists in the project root.
3. If you changed the model architecture (`embed_size`, `num_layers`, `num_heads`), old checkpoints are incompatible. Delete them and retrain.

### CORS error in browser console (`status: null`)

This means the browser could not reach the backend at all (connection refused). Verify:

1. The backend is running: `curl http://localhost:8000/api/health`
2. You started the backend **before** or at the same time as the frontend.
3. The Vite proxy config in `frontend/vite.config.js` points to the correct backend port.

### `ModuleNotFoundError: No module named 'model_store'`

Run uvicorn from the **project root**, not from inside `backend/`:

```bash
# Correct
uvicorn backend.main:app

# Wrong
cd backend && uvicorn main:app   # breaks the path setup
```

### MPS / CUDA not detected

PyTorch 2.2 requires macOS 12.3+ for MPS. Check:

```python
import torch
print(torch.backends.mps.is_available())   # True on Apple Silicon / AMD Mac
print(torch.cuda.is_available())            # True on NVIDIA
```

If neither is available the model runs on CPU — slower for training, but identical results.

### Tokenizer retrains every time the server starts

The tokenizer retrains when `dialogue.txt` is newer than `tokenizer.model.json`. If you do not want retraining, touch `tokenizer.model.json` to update its timestamp:

```bash
touch tokenizer.model.json
```

### Training loss is not going down

- Verify the data was cleaned correctly: `head -20 processed/dialogue.txt`
- Try lowering the learning rate to `1e-4`.
- Try increasing `batch_size` (if GPU memory allows).
- Check `output.log` for signs of numerical instability (NaN loss).

### Out of memory during training

Reduce `batch_size` or `block_size`. On MPS, MPS memory is shared with system RAM, so closing other applications helps.

### Frontend build error

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

---

## Licence

MIT — do whatever you want with it.
