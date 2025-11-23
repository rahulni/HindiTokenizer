import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import PreTrainedTokenizerFast
import os

# --------------------------------------
# LOAD TOKENIZER
# --------------------------------------

TOKENIZER_JSON = "tokenizer_hindi_bpe_8k_stream/tokenizer.json"
HF_DIR = "tokenizer_hindi_bpe_8k_stream/hf"

if os.path.exists(HF_DIR):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(HF_DIR)
elif os.path.exists(TOKENIZER_JSON):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_JSON)
else:
    raise ValueError("Tokenizer not found!")

print("Tokenizer loaded: vocab =", tokenizer.vocab_size)

# --------------------------------------
# ENCODE / DECODE FUNCTIONS
# --------------------------------------

def encode_text(text: str):
    """Basic encode: returns token IDs as CSV, token count, and compression ratio."""
    enc = tokenizer(text, add_special_tokens=False)
    token_ids = enc["input_ids"]
    token_count = len(token_ids)
    csv_ids = ",".join(str(x) for x in token_ids)
    
    # Calculate compression ratio (characters per token)
    char_count = len(text)
    compression_ratio = char_count / token_count if token_count > 0 else 0.0
    
    return csv_ids, token_count, f"{compression_ratio:.2f}"

def decode_ids(ids: str):
    """Decode from comma-separated IDs to text."""
    try:
        arr = [int(x) for x in ids.split(",") if x.strip()]
        return tokenizer.decode(arr)
    except:
        return "❌ Invalid ID list"

# --------------------------------------
# FASTAPI REST BACKEND
# --------------------------------------

api = FastAPI(title="Hindi Tokenizer API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@api.get("/")
def home():
    return {
        "message": "Hindi Tokenizer API",
        "vocab_size": tokenizer.vocab_size
    }

@api.get("/tokenize")
def tokenize_endpoint(text: str):
    enc = tokenizer(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    return {"tokens": tokens, "ids": enc["input_ids"]}

@api.get("/decode")
def decode_endpoint(ids: str):
    try:
        arr = [int(x) for x in ids.split(",") if x.strip()]
        return {"text": tokenizer.decode(arr)}
    except:
        return {"error": "Invalid id list"}

# --------------------------------------
# GRADIO FRONTEND
# --------------------------------------

with gr.Blocks(title="Hindi Tokenizer") as demo:
    gr.Markdown("## 🔡 Hindi BPE Tokenizer — Encode / Decode")

    with gr.Tab("Encode"):
        text_in = gr.Textbox(label="Enter text", lines=3)
        
        with gr.Row():
            token_count_out = gr.Number(label="Token Count", precision=0)
            compression_ratio_out = gr.Textbox(label="Compression Ratio (chars/token)", interactive=False)
        
        ids_out = gr.Textbox(label="Token IDs", lines=8, max_lines=20)
        btn = gr.Button("Encode")
        btn.click(encode_text, text_in, [ids_out, token_count_out, compression_ratio_out])

    with gr.Tab("Decode"):
        ids_in = gr.Textbox(label="Comma-separated token IDs", lines=4)
        text_out = gr.Textbox(label="Decoded text", lines=8, max_lines=20)
        btn3 = gr.Button("Decode")
        btn3.click(decode_ids, ids_in, text_out)

# Mount FastAPI + Gradio

if "app" not in globals():
    app = gr.mount_gradio_app(api, demo, path="/gradio")

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
