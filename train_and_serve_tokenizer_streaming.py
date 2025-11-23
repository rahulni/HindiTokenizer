#!/usr/bin/env python3
"""
Streaming-friendly BPE tokenizer trainer for large Hindi corpus (1.23GB)
- Does NOT load full file into memory
- Shows progress
- Chunked training for HuggingFace `tokenizers`
- Includes API service (FastAPI) and evaluation
"""

import os
import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import Sequence, NFKC
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tqdm import tqdm
import regex as re
import json

# ----------------------
# CONFIG
# ----------------------
CORPUS_PATH = Path("clean_corpus/hindi_corpus_clean.txt")
TOKENIZER_DIR = Path("tokenizer_hindi_bpe_8k_stream")
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
HF_DIR = TOKENIZER_DIR / "hf"

DEFAULT_VOCAB = 8000
DEFAULT_MIN_FREQ = 2


# ----------------------
# Helper: read chunk
# ----------------------
def yield_chunks(corpus_path: Path, chunk_size: int = 50000):
    """
    Yields blocks of lines so the tokenizer can train chunk-by-chunk.
    """
    buf = []
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            buf.append(line.strip())
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf


# ----------------------
# Train BPE tokenizer in streaming mode
# ----------------------
def train_bpe_streaming(
    corpus_path: Path,
    output_json: Path,
    vocab_size: int = DEFAULT_VOCAB,
    min_frequency: int = DEFAULT_MIN_FREQ,
    chunk_size: int = 50000,
    log_interval: int = 200000
):
    """
    Streaming BPE training: feeds corpus chunks gradually to trainer.

    HuggingFace tokenizers does NOT have streaming BPE, but we can
    emulate it by feeding repeated chunks using a special trainer option.
    """

    print(f"[TRAIN] Streaming BPE tokenizer")
    print(f"[TRAIN] Corpus: {corpus_path}")
    print(f"[TRAIN] Vocab size: {vocab_size}")
    print(f"[TRAIN] Min frequency: {min_frequency}")
    print(f"[TRAIN] Chunk size: {chunk_size:,} lines per batch")
    print("-----------------------------------------------------")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=False,     # we show custom progress
    )

    # NOTE:
    # tokenizers library requires .train(files) OR .train_from_iterator(iterator)
    # We will use train_from_iterator() and feed chunks safely.
    # This avoids loading entire corpus at once.

    def line_iterator():
        total = 0
        for chunk in yield_chunks(corpus_path, chunk_size):
            for line in chunk:
                total += 1
                if total % log_interval == 0:
                    print(f"[TRAIN] Processed ~{total:,} lines...")
                yield line

    tokenizer.train_from_iterator(line_iterator(), trainer=trainer)

    # Save final tokenizer
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_json))
    print(f"[TRAIN] Saved tokenizer → {output_json}")

    return tokenizer


# ----------------------
# Build HF wrapper
# ----------------------
def save_hf_tokenizer(tokenizer_json: Path, hf_dir: Path):
    hf_dir.mkdir(parents=True, exist_ok=True)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

    hf_tok.save_pretrained(str(hf_dir))
    print(f"[HF] Saved HuggingFace tokenizer → {hf_dir}")

    return hf_tok

#########################################
# PART 2 — Evaluation + API + Main Flow
#########################################

def clean_eval_text(t: str):
    return re.sub(r"\s+", " ", t.strip())


def evaluate_tokenizer(tok, corpus_path: Path, sample: int = 1000):
    lines = []
    for i, line in enumerate(corpus_path.open("r", encoding="utf-8", errors="ignore")):
        if len(lines) >= sample:
            break
        clean = clean_eval_text(line)
        if clean:
            lines.append(clean)

    total_chars = 0
    total_tokens = 0
    unk = tok.unk_token

    unk_count = 0

    for s in lines:
        total_chars += len(s.replace(" ", ""))
        ids = tok(s, add_special_tokens=False)["input_ids"]
        tokens = tok.convert_ids_to_tokens(ids)
        total_tokens += len(ids)
        unk_count += sum(1 for t in tokens if t == unk)

    print("\n[EVAL] Tokenizer Evaluation")
    print("------------------------------")
    print(f"Sentences: {len(lines)}")
    print(f"Characters (no space): {total_chars}")
    print(f"Tokens: {total_tokens}")
    print(f"Avg tokens per sentence: {total_tokens/len(lines):.3f}")
    print(f"Avg tokens per word: {total_tokens/max(1, sum(len(s.split()) for s in lines)):.3f}")
    print(f"Compression ratio: {total_chars/max(1, total_tokens):.3f}")
    print(f"UNK rate: {unk_count/max(1, total_tokens):.6f}\n")


# ----------------------
# FastAPI App
# ----------------------
def build_api(hf_tok):
    app = FastAPI(title="Hindi BPE Tokenizer (Streaming)")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def home():
        return {"message": "Hindi BPE Tokenizer", "vocab": hf_tok.vocab_size}

    @app.get("/tokenize")
    def tokenize_api(text: str):
        enc = hf_tok(text, add_special_tokens=False)
        tokens = hf_tok.convert_ids_to_tokens(enc["input_ids"])
        return {"tokens": tokens, "ids": enc["input_ids"]}

    @app.get("/decode")
    def decode_api(ids: str):
        arr = [int(x) for x in ids.split(",") if x.strip()]
        return {"text": hf_tok.decode(arr)}

    return app


# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(CORPUS_PATH))
    parser.add_argument("--vocab", default=DEFAULT_VOCAB, type=int)
    parser.add_argument("--minfreq", default=DEFAULT_MIN_FREQ, type=int)
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--chunk", default=50000, type=int)
    parser.add_argument("--log_interval", default=200000, type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)

    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    TOKENIZER_DIR.mkdir(exist_ok=True)

    # Train if needed
    if not TOKENIZER_JSON.exists() or args.force:
        tokenizer = train_bpe_streaming(
            corpus_path=corpus_path,
            output_json=TOKENIZER_JSON,
            vocab_size=args.vocab,
            min_frequency=args.minfreq,
            chunk_size=args.chunk,
            log_interval=args.log_interval,
        )
        hf_tok = save_hf_tokenizer(TOKENIZER_JSON, HF_DIR)
    else:
        print("[INFO] Loading existing tokenizer...")
        hf_tok = PreTrainedTokenizerFast(
            tokenizer_file=str(TOKENIZER_JSON),
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

    # Evaluate
    evaluate_tokenizer(hf_tok, corpus_path)

    # Train Only?
    if args.train_only:
        print("[DONE] Training completed.")
        exit(0)

    # Serve API
    if args.serve:
        app = build_api(hf_tok)
        print(f"[INFO] Serving API at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

