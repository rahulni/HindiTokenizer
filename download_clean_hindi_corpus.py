# #!/usr/bin/env python3
# """
# train_and_serve_tokenizer.py

# End-to-end:
#  - Train BPE tokenizer (from tokenizers) on a Hindi corpus (UTF-8)
#  - Evaluate tokenizer (UNK rate, avg tokens/word, compression ratio)
#  - Wrap as HuggingFace PreTrainedTokenizerFast and save
#  - Start FastAPI app to serve tokenization/decode/stats endpoints

# Requirements (example):
# pip install tokenizers transformers fastapi uvicorn tqdm regex

# Author: ChatGPT (Rahul requested)
# """

# import os
# import argparse
# from pathlib import Path
# from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
# from tokenizers.normalizers import Sequence, NFKC
# from tokenizers.pre_tokenizers import Whitespace
# from tokenizers.implementations import BPE
# from transformers import PreTrainedTokenizerFast
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# from tqdm import tqdm
# import regex as re
# import json
# import math

# # -----------------------
# # Config / Defaults
# # -----------------------
# CORPUS_PATH = Path("clean_corpus/hindi_corpus_clean.txt")
# TOKENIZER_OUTPUT_DIR = Path("tokenizer_hindi_bpe_8k")
# JSON_TOKENIZER_PATH = TOKENIZER_OUTPUT_DIR / "tokenizer.json"
# HF_TOKENIZER_DIR = TOKENIZER_OUTPUT_DIR / "hf"
# DEFAULT_VOCAB_SIZE = 8000
# MIN_FREQ = 2
# SAMPLE_LINES_FOR_QUICK_TRAIN = None  # set to int for quick tests, else None to use entire corpus

# # -----------------------
# # Utilities
# # -----------------------
# def ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)

# def read_corpus_lines(corpus_path: Path, max_lines: int = None):
#     with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:
#         if max_lines is None:
#             for line in f:
#                 yield line.rstrip("\n")
#         else:
#             for i, line in enumerate(f):
#                 if i >= max_lines:
#                     break
#                 yield line.rstrip("\n")

# # Simple normalization for evaluation input (do not alter tokenizer training normalization)
# def clean_for_eval(s: str):
#     s = s.strip()
#     s = re.sub(r"\s+", " ", s)
#     return s

# # -----------------------
# # Train Tokenizer
# # -----------------------
# def train_bpe_tokenizer(
#     corpus_path: Path,
#     output_json: Path,
#     vocab_size: int = DEFAULT_VOCAB_SIZE,
#     min_frequency: int = MIN_FREQ,
#     sample_lines: int = None
# ):
#     """
#     Trains a BPE tokenizer using HuggingFace tokenizers (Rust engine).
#     Saves tokenizer JSON to output_json.
#     """
#     print(f"[TRAIN] Training BPE tokenizer on: {corpus_path}")
#     print(f"[TRAIN] vocab_size={vocab_size}, min_frequency={min_frequency}, sample_lines={sample_lines}")

#     # Initialize tokenizer
#     tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

#     # Normalizer: minimal unicode normalization (NFKC). You can expand if needed.
#     tokenizer.normalizer = Sequence([NFKC()])

#     # Pre-tokenizer: whitespace (we let BPE learn subwords)
#     tokenizer.pre_tokenizer = Whitespace()

#     # Trainer
#     trainer = trainers.BpeTrainer(
#         vocab_size=vocab_size,
#         min_frequency=min_frequency,
#         special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
#     )

#     # Prepare corpus files list (tokenizers.train accepts files)
#     # To support sampling we will write a temporary sample file if needed
#     if sample_lines is not None:
#         sample_file = output_json.parent / "temp_sample_corpus.txt"
#         print(f"[TRAIN] Creating sample file with {sample_lines} lines at {sample_file}")
#         with sample_file.open("w", encoding="utf-8") as fw:
#             for line in read_corpus_lines(corpus_path, max_lines=sample_lines):
#                 fw.write(line + "\n")
#         files = [str(sample_file)]
#     else:
#         files = [str(corpus_path)]

#     tokenizer.train(files, trainer)

#     # Save tokenizer JSON
#     ensure_dir(output_json.parent)
#     tokenizer.save(str(output_json))
#     print(f"[TRAIN] Tokenizer saved -> {output_json}")

#     # cleanup sample if created
#     if sample_lines is not None and sample_file.exists():
#         sample_file.unlink()

#     return tokenizer

# # -----------------------
# # Create HF PreTrainedTokenizerFast
# # -----------------------
# def build_and_save_hf_tokenizer(tokenizer_json_path: Path, hf_dir: Path):
#     """
#     Wrap tokenizers tokenizer json using PreTrainedTokenizerFast and save to hf_dir.
#     """
#     print(f"[HF] Building HuggingFace PreTrainedTokenizerFast from {tokenizer_json_path}")

#     ensure_dir(hf_dir)

#     # Create HF tokenizer out of the tokenizers JSON file
#     hf_tokenizer = PreTrainedTokenizerFast(
#         tokenizer_file=str(tokenizer_json_path),
#         unk_token="[UNK]",
#         pad_token="[PAD]",
#         cls_token="[CLS]",
#         sep_token="[SEP]",
#         mask_token="[MASK]",
#     )

#     # Save huggingface-style tokenizer (config + files)
#     hf_tokenizer.save_pretrained(str(hf_dir))
#     print(f"[HF] Saved HF tokenizer to {hf_dir}")

#     return hf_tokenizer

# # -----------------------
# # Evaluation Metrics
# # -----------------------
# def evaluate_tokenizer_on_sample(hf_tokenizer: PreTrainedTokenizerFast, sample_texts, verbose: bool = True):
#     """
#     Evaluate UNK rate, avg tokens per word, compression ratio on a list of strings.
#     Returns dictionary with metrics.
#     """
#     total_chars_no_space = 0
#     total_tokens = 0
#     total_words = 0
#     total_unk_tokens = 0

#     token_lengths = []

#     for s in sample_texts:
#         s_clean = clean_for_eval(s)
#         if not s_clean:
#             continue
#         # remove spaces for char count
#         chars_no_space = len(s_clean.replace(" ", ""))
#         enc = hf_tokenizer(s_clean, add_special_tokens=False)
#         ids = enc["input_ids"]
#         tokens = hf_tokenizer.convert_ids_to_tokens(ids)

#         total_chars_no_space += chars_no_space
#         total_tokens += len(ids)
#         token_lengths.append(len(ids))

#         # word count
#         words = s_clean.split()
#         total_words += max(1, len(words))

#         # UNK count: tokenizers often produce special token for unknown; we use unk_token id test
#         unk_token = hf_tokenizer.unk_token
#         # tokens could be like '▁मैं' etc.
#         total_unk_tokens += sum(1 for t in tokens if t == unk_token)

#     avg_tokens_per_sentence = (sum(token_lengths) / len(token_lengths)) if token_lengths else 0
#     avg_tokens_per_word = (total_tokens / total_words) if total_words else 0
#     compression_ratio = (total_chars_no_space / total_tokens) if total_tokens else 0
#     unk_rate = (total_unk_tokens / total_tokens) if total_tokens else 0

#     metrics = {
#         "total_sentences": len(token_lengths),
#         "total_chars_no_space": total_chars_no_space,
#         "total_tokens": total_tokens,
#         "avg_tokens_per_sentence": avg_tokens_per_sentence,
#         "avg_tokens_per_word": avg_tokens_per_word,
#         "compression_ratio": compression_ratio,
#         "unk_rate": unk_rate
#     }

#     if verbose:
#         print("[EVAL] Samples:", metrics["total_sentences"])
#         print(f"[EVAL] Total chars (no spaces): {metrics['total_chars_no_space']}")
#         print(f"[EVAL] Total tokens: {metrics['total_tokens']}")
#         print(f"[EVAL] Avg tokens/sentence: {metrics['avg_tokens_per_sentence']:.3f}")
#         print(f"[EVAL] Avg tokens/word: {metrics['avg_tokens_per_word']:.3f}")
#         print(f"[EVAL] Compression ratio (chars_no_space / tokens): {metrics['compression_ratio']:.3f}")
#         print(f"[EVAL] UNK rate: {metrics['unk_rate']:.6f}")

#     return metrics

# # -----------------------
# # Utility: sample from corpus for evaluation
# # -----------------------
# def sample_corpus_lines(corpus_path: Path, n: int = 1000):
#     lines = []
#     for i, line in enumerate(read_corpus_lines(corpus_path)):
#         if line and len(lines) < n:
#             lines.append(line)
#         elif len(lines) >= n:
#             break
#     return lines

# # -----------------------
# # FastAPI App
# # -----------------------
# def create_app(hf_tokenizer: PreTrainedTokenizerFast):
#     app = FastAPI(title="Hindi BPE Tokenizer API")

#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=["*"],
#         allow_methods=["*"],
#         allow_headers=["*"],
#         allow_credentials=True,
#     )

#     @app.get("/")
#     def root():
#         return {"message": "Hindi BPE Tokenizer API", "vocab_size": hf_tokenizer.vocab_size}

#     @app.get("/tokenize")
#     def api_tokenize(text: str, return_ids: bool = True):
#         enc = hf_tokenizer(text, add_special_tokens=False)
#         tokens = hf_tokenizer.convert_ids_to_tokens(enc["input_ids"])
#         if return_ids:
#             return {"tokens": tokens, "ids": enc["input_ids"]}
#         else:
#             return {"tokens": tokens}

#     @app.get("/decode")
#     def api_decode(ids: str):
#         # ids: comma separated ints
#         id_list = [int(x) for x in ids.split(",") if x.strip()]
#         text = hf_tokenizer.decode(id_list, clean_up_tokenization_spaces=True)
#         return {"text": text}

#     @app.get("/stats")
#     def api_stats(sample: int = 500):
#         samples = sample_corpus_lines(CORPUS_PATH, n=sample)
#         metrics = evaluate_tokenizer_on_sample(hf_tokenizer, samples, verbose=False)
#         return metrics

#     return app

# # -----------------------
# # Main: CLI
# # -----------------------
# def main():
#     parser = argparse.ArgumentParser(description="Train BPE (8k) tokenizer from scratch and serve via FastAPI")
#     parser.add_argument("--corpus", type=str, default=str(CORPUS_PATH), help="Path to cleaned Hindi corpus (one sentence per line)")
#     parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
#     parser.add_argument("--min_freq", type=int, default=MIN_FREQ)
#     parser.add_argument("--sample_lines", type=int, default=SAMPLE_LINES_FOR_QUICK_TRAIN,
#                         help="If set, train quickly on first N lines (useful for tests)")
#     parser.add_argument("--train_only", action="store_true", help="Only train and exit (no API server)")
#     parser.add_argument("--serve", action="store_true", help="Train (if needed) and serve API (uvicorn)")
#     parser.add_argument("--port", type=int, default=8000)
#     parser.add_argument("--host", type=str, default="127.0.0.1")
#     parser.add_argument("--force_retrain", action="store_true", help="Force re-training even if tokenizer exists")
#     args = parser.parse_args()

#     corpus_path = Path(args.corpus)
#     if not corpus_path.exists():
#         raise FileNotFoundError(f"Corpus not found: {corpus_path}. Run the cleaning script first.")

#     ensure_dir(TOKENIZER_OUTPUT_DIR)
#     ensure_dir(HF_TOKENIZER_DIR)

#     # If tokenizer json exists and not force_retrain, load and wrap HF tokenizer; else train
#     if JSON_TOKENIZER_PATH.exists() and not args.force_retrain:
#         print(f"[INFO] Found existing tokenizer JSON: {JSON_TOKENIZER_PATH}. Loading.")
#         hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(JSON_TOKENIZER_PATH),
#                                                unk_token="[UNK]",
#                                                pad_token="[PAD]",
#                                                cls_token="[CLS]",
#                                                sep_token="[SEP]",
#                                                mask_token="[MASK]")
#     else:
#         # Train
#         trained_tok = train_bpe_tokenizer(
#             corpus_path=corpus_path,
#             output_json=JSON_TOKENIZER_PATH,
#             vocab_size=args.vocab_size,
#             min_frequency=args.min_freq,
#             sample_lines=(args.sample_lines if args.sample_lines and args.sample_lines > 0 else None)
#         )
#         hf_tokenizer = build_and_save_hf_tokenizer(JSON_TOKENIZER_PATH, HF_TOKENIZER_DIR)

#     # Quick evaluation on small sample and print metrics
#     print("\n[INFO] Evaluating tokenizer on sample sentences:")
#     sample_sents = sample_corpus_lines(corpus_path, n=500)
#     evaluate_tokenizer_on_sample(hf_tokenizer, sample_sents)

#     if args.train_only:
#         print("[INFO] Training complete. Exiting (--train_only).")
#         return

#     if args.serve:
#         print("[INFO] Starting FastAPI server with tokenizer endpoints.")
#         app = create_app(hf_tokenizer)
#         uvicorn.run(app, host=args.host, port=args.port, log_level="info")
#     else:
#         print("[INFO] Done. You can start server with --serve to serve the tokenizer via HTTP.")

# if __name__ == "__main__":
#     main()
import os
import bz2
import re
import urllib.request
import shutil
from pathlib import Path
import json
from huggingface_hub import snapshot_download

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path("clean_corpus")
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_FILE = BASE_DIR / "hindi_corpus_clean.txt"
FILTERED_FILE = BASE_DIR / "hindi_corpus_clean_filtered.txt"

# Wikipedia XML dump
WIKI_URL = "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2"

# Tokenizers to Download (32K)
TOKENIZER_MODELS = {
    "indictrans2": "ai4bharat/indictrans2-en-indic-1B",
    "muril": "google/muril-base-cased",
    "gpt_hindi": "surajp/gpt-hindi"
}

# ============================================================
# HELPERS
# ============================================================
def setup():
    BASE_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    print("[INFO] Directory structure ready...")

def download_file(url, output_path):
    print(f"[INFO] Downloading: {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"[OK] Download complete → {output_path}")

def clean_line(text):
    text = text.strip()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\[\[File:.*?\]\]", " ", text)
    text = re.sub(r"\{\{.*?\}\}", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^ऀ-ॿa-zA-Z0-9 ।.!?,:;()\"'’“”\-\/]", " ", text)
    return text.strip()

# ============================================================
# Extract Wikipedia (PURE PYTHON SAFE FOR WINDOWS)
# ============================================================
def extract_wiki(xml_file, out_file):
    print("[INFO] Extracting Wikipedia pages (Python XML parser)...")
    count = 0
    
    with bz2.open(xml_file, "rt", encoding="utf-8", errors="ignore") as f, \
         open(out_file, "w", encoding="utf-8") as out:
        
        page = []
        recording = False
        
        for line in f:
            if "<page>" in line:
                page = []
                recording = True
                continue

            if "</page>" in line:
                recording = False
                full_text = "\n".join(page)
                cleaned = clean_line(full_text)
                if len(cleaned) > 30:
                    out.write(cleaned + "\n")
                    count += 1
                continue

            if recording:
                page.append(line)

    print(f"[OK] Extracted {count} cleaned wiki pages → {out_file}")

# ============================================================
# Filter corpus (remove short or noisy lines)
# ============================================================
def filter_corpus(input_file, output_file, min_len=30):
    print("[INFO] Filtering corpus for long lines > 30 chars...")
    kept = 0

    with open(input_file, "r", encoding="utf-8", errors="ignore") as fi, \
         open(output_file, "w", encoding="utf-8") as fo:
        
        for line in fi:
            l = line.strip()
            if len(l) >= min_len:
                fo.write(l + "\n")
                kept += 1

    print(f"[OK] Filtered corpus saved → {output_file} ({kept} lines kept)")

# ============================================================
# Download 32K tokenizers
# ============================================================
def download_tokenizers():
    TOK_DIR = BASE_DIR / "tokenizers"
    TOK_DIR.mkdir(exist_ok=True)

    print("\n=== Downloading 32K Hindi Tokenizers ===")

    for name, repo in TOKENIZER_MODELS.items():
        dest = TOK_DIR / name
        print(f"[INFO] Downloading {name} from {repo}")
        snapshot_download(repo, local_dir=str(dest), local_dir_use_symlinks=False)
        print(f"[OK] Saved → {dest}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    setup()

    wiki_file = TEMP_DIR / "wiki_dump.xml.bz2"

    # Download Wikipedia dump
    if not wiki_file.exists():
        download_file(WIKI_URL, wiki_file)

    # Extract Wikipedia
    extract_wiki(wiki_file, OUTPUT_FILE)

    # Filter corpus (important for compression ratio)
    filter_corpus(OUTPUT_FILE, FILTERED_FILE, min_len=30)

    # Download tokenizers
    download_tokenizers()

    print("\n🎉 DONE — Corpus extracted, cleaned, filtered, and tokenizers downloaded.")
    print(f"➡ Clean corpus: {OUTPUT_FILE}")
    print(f"➡ Filtered corpus: {FILTERED_FILE}")
    print(f"➡ Tokenizers stored under: clean_corpus/tokenizers/")
