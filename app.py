import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import PreTrainedTokenizerFast
import os
import json
import random
import hashlib
import re

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

def get_color_for_token(token_id, seed=None):
    """Generate a consistent color for a token ID."""
    if seed is not None:
        random.seed(seed)
    # Generate a hash-based color
    hash_obj = hashlib.md5(str(token_id).encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    # Use HSL for better color distribution
    hue = hash_int % 360
    saturation = 60 + (hash_int % 30)
    lightness = 75 + (hash_int % 15)
    return f"hsl({hue}, {saturation}%, {lightness}%)"

def encode_text(text: str):
    """Basic encode: returns token IDs as CSV, token count, compression ratio, and color-coded HTML."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    offsets = enc.get("offset_mapping", [])
    
    token_count = len(token_ids)
    csv_ids = ",".join(str(x) for x in token_ids)
    
    # Calculate compression ratio (characters per token)
    char_count = len(text)
    compression_ratio = char_count / token_count if token_count > 0 else 0.0
    
    # First, build token-to-word mapping using offsets
    token_ranges = []
    for idx, (start, end) in enumerate(offsets):
        if start is not None and end is not None:
            token_ranges.append((idx, start, end))
        else:
            token_ranges.append((idx, None, None))
    
    # Get word positions for mapping
    words_with_positions = []
    for match in re.finditer(r'\S+', text):
        word = match.group()
        word_start = match.start()
        word_end = match.end()
        words_with_positions.append((word, word_start, word_end))
    
    # Build token-to-word mapping
    token_to_words_map = {}
    for token_idx, token_start, token_end in token_ranges:
        if token_start is not None and token_end is not None:
            token_to_words_map[token_idx] = []
            for word_idx, (word, word_start, word_end) in enumerate(words_with_positions):
                if token_start < word_end and token_end > word_start:
                    token_to_words_map[token_idx].append(word_idx)
    
    # Store token data for potential future use
    token_data = []
    for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
        token_data.append({
            "idx": i,
            "token": token,
            "id": token_id
        })
    
    # Include JavaScript for highlighting in the HTML
    highlight_script = """
    <script>
    if (typeof window.highlightFunctions === 'undefined') {
        window.highlightFunctions = {};
        window.currentHighlighted = null;
        
        window.clearHighlights = function() {
            if (window.currentHighlighted) {
                window.currentHighlighted.forEach(el => {
                    el.classList.remove('highlighted');
                    el.style.borderColor = 'transparent';
                    el.style.boxShadow = 'none';
                    if (el.style.transform) el.style.transform = 'scale(1)';
                });
            }
            window.currentHighlighted = null;
        };
        
        window.highlightInputWord = function(tokenIndicesStr) {
            window.clearHighlights();
            const tokenIndices = tokenIndicesStr.split(',');
            const highlighted = [];
            
            // Highlight input words
            document.querySelectorAll(`.input-word-tag[data-word-tokens="${tokenIndicesStr}"]`).forEach(el => {
                el.classList.add('highlighted');
                el.style.borderColor = '#ff0000';
                el.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
                highlighted.push(el);
            });
            
            // Highlight corresponding token IDs
            tokenIndices.forEach(idx => {
                document.querySelectorAll(`.token-id-tag[data-token-idx="${idx}"]`).forEach(el => {
                    el.classList.add('highlighted');
                    el.style.borderColor = '#ff0000';
                    el.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
                    el.style.transform = 'scale(1.1)';
                    highlighted.push(el);
                });
            });
            
            window.currentHighlighted = highlighted;
        };
        
        window.highlightTokenId = function(tokenIdx) {
            window.clearHighlights();
            const tokenIdEl = document.querySelector(`.token-id-tag[data-token-idx="${tokenIdx}"]`);
            if (!tokenIdEl) return;
            
            const highlighted = [tokenIdEl];
            tokenIdEl.classList.add('highlighted');
            tokenIdEl.style.borderColor = '#ff0000';
            tokenIdEl.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
            tokenIdEl.style.transform = 'scale(1.1)';
            
            // Find input words that contain this token
            const tokenId = tokenIdEl.getAttribute('data-token-id');
            document.querySelectorAll('.input-word-tag').forEach(wordEl => {
                const tokenIndices = wordEl.getAttribute('data-word-tokens');
                if (tokenIndices) {
                    const tokenList = tokenIndices.split(',');
                    if (tokenList.includes(tokenIdx.toString())) {
                        wordEl.classList.add('highlighted');
                        wordEl.style.borderColor = '#ff0000';
                        wordEl.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
                        highlighted.push(wordEl);
                    }
                }
            });
            
            window.currentHighlighted = highlighted;
        };
    }
    </script>
    """
    
    token_json = json.dumps(token_data)
    
    # Create clickable HTML for input text (mirror of textbox) - uses same words_with_positions
    input_word_html_parts = []
    for word, word_start, word_end in words_with_positions:
        word_escaped = word.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
        
        # Find tokens whose character ranges overlap with this word
        word_token_indices = []
        for token_idx, token_start, token_end in token_ranges:
            if token_start is not None and token_end is not None:
                if token_start < word_end and token_end > word_start:
                    word_token_indices.append(token_idx)
        
        if word_token_indices:
            token_id_for_word = token_ids[word_token_indices[0]]
            color = get_color_for_token(token_id_for_word, seed=42)
            token_indices_str = ",".join(map(str, word_token_indices))
            input_word_html_parts.append(
                f'<span class="input-word-tag" data-word-tokens="{token_indices_str}" '
                f'style="background-color: {color}; padding: 2px 6px; margin: 2px; '
                f'border-radius: 4px; cursor: pointer; display: inline-block; '
                f'border: 2px solid transparent; transition: all 0.2s;" '
                f'onclick="highlightInputWord(\'{token_indices_str}\')" '
                f'onmouseover="this.style.borderColor=\'#333\'" '
                f'onmouseout="if(!document.querySelector(\'.input-word-tag.highlighted\')) this.style.borderColor=\'transparent\'">{word_escaped}</span>'
            )
        else:
            input_word_html_parts.append(f'<span style="padding: 2px 6px; margin: 2px;">{word_escaped}</span>')
    
    input_html = '<div style="line-height: 2; padding: 10px; background: #ffffff; border: 2px solid #e0e0e0; border-radius: 8px; min-height: 60px;">' + " ".join(input_word_html_parts) + '</div>'
    
    # Create token IDs display with labels for highlighting
    token_ids_html_parts = []
    for i, token_id in enumerate(token_ids):
        color = get_color_for_token(token_id, seed=42)
        # Find which words contain this token
        word_indices = token_to_words_map.get(i, [])
        word_labels = [words_with_positions[idx][0] for idx in word_indices]
        word_label = ", ".join(word_labels[:2]) if word_labels else ""  # Show first 2 words as label
        
        token_ids_html_parts.append(
            f'<div class="token-id-tag" data-token-idx="{i}" data-token-id="{token_id}" '
            f'style="background-color: {color}; padding: 6px 10px; margin: 4px; '
            f'border-radius: 6px; cursor: pointer; display: inline-block; vertical-align: top; '
            f'border: 2px solid transparent; transition: all 0.2s; text-align: center; min-width: 60px;" '
            f'onclick="highlightTokenId({i})" '
            f'onmouseover="this.style.borderColor=\'#333\'" '
            f'onmouseout="if(!document.querySelector(\'.token-id-tag[data-token-idx=\'{i}\'].highlighted\')) this.style.borderColor=\'transparent\'">'
            f'<div style="font-family: monospace; font-weight: bold; font-size: 14px; color: #000;">{token_id}</div>'
            f'<div style="font-size: 10px; color: #555; margin-top: 2px; word-break: break-word; max-width: 80px;">{word_label if word_label else "&nbsp;"}</div>'
            f'</div>'
        )
    
    token_ids_html = '<div style="padding: 10px; background: #f8f9fa; border-radius: 8px; margin-top: 10px;">' + "".join(token_ids_html_parts) + '</div>'
    
    return csv_ids, token_count, f"{compression_ratio:.2f}", token_ids_html, token_json, input_html

def decode_ids(ids: str):
    """Decode from comma-separated IDs to text with color-coded HTML."""
    try:
        arr = [int(x) for x in ids.split(",") if x.strip()]
        decoded_text = tokenizer.decode(arr, skip_special_tokens=False)
        
        # Re-encode with offsets to map tokens to words accurately
        enc_with_offsets = tokenizer(decoded_text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(arr)
        offsets = enc_with_offsets.get("offset_mapping", [])
        
        # Build token-to-character-range mapping
        token_ranges = []
        for idx, (start, end) in enumerate(offsets):
            if start is not None and end is not None:
                token_ranges.append((idx, start, end))
            else:
                token_ranges.append((idx, None, None))
        
        # Get word positions for mapping
        words_with_positions = []
        for match in re.finditer(r'\S+', decoded_text):
            word = match.group()
            word_start = match.start()
            word_end = match.end()
            words_with_positions.append((word, word_start, word_end))
        
        # Create color-coded HTML for decoded text
        word_html_parts = []
        
        for word, word_start, word_end in words_with_positions:
            word_escaped = word.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
            
            # Find tokens whose character ranges overlap with this word
            word_token_indices = []
            for token_idx, token_start, token_end in token_ranges:
                if token_start is not None and token_end is not None:
                    # Check if token overlaps with word
                    if token_start < word_end and token_end > word_start:
                        word_token_indices.append(token_idx)
            
            if word_token_indices and word_token_indices[0] < len(arr):
                token_id_for_word = arr[word_token_indices[0]]
                color = get_color_for_token(token_id_for_word, seed=42)
                token_indices_str = ",".join(map(str, word_token_indices))
                word_html_parts.append(
                    f'<span class="decode-word-tag" data-word-tokens="{token_indices_str}" '
                    f'style="background-color: {color}; padding: 2px 6px; margin: 2px; '
                    f'border-radius: 4px; cursor: pointer; display: inline-block; '
                    f'border: 2px solid transparent; transition: all 0.2s;" '
                    f'onclick="highlightDecodeWord(\'{token_indices_str}\')" '
                    f'onmouseover="this.style.borderColor=\'#333\'" '
                    f'onmouseout="if(!document.querySelector(\'.decode-word-tag.highlighted\')) this.style.borderColor=\'transparent\'">{word_escaped}</span>'
                )
            else:
                word_html_parts.append(f'<span style="padding: 2px 6px; margin: 2px;">{word_escaped}</span>')
        
        # Build token-to-word mapping for decode
        token_to_words_map = {}
        for token_idx, token_start, token_end in token_ranges:
            if token_start is not None and token_end is not None:
                token_to_words_map[token_idx] = []
                for word_idx, (word, word_start, word_end) in enumerate(words_with_positions):
                    if token_start < word_end and token_end > word_start:
                        token_to_words_map[token_idx].append(word_idx)
        
        decoded_html = '<div style="line-height: 2; padding: 10px; background: #ffffff; border: 2px solid #e0e0e0; border-radius: 8px; min-height: 60px;">' + " ".join(word_html_parts) + '</div>'
        
        # Create token IDs display with labels for decode (similar to encode)
        decode_token_ids_html_parts = []
        for i, token_id in enumerate(arr):
            color = get_color_for_token(token_id, seed=42)
            # Find which words contain this token
            word_indices = token_to_words_map.get(i, [])
            word_labels = [words_with_positions[idx][0] for idx in word_indices if idx < len(words_with_positions)]
            word_label = ", ".join(word_labels[:2]) if word_labels else ""  # Show first 2 words as label
            
            decode_token_ids_html_parts.append(
                f'<div class="decode-token-id-tag" data-token-idx="{i}" data-token-id="{token_id}" '
                f'style="background-color: {color}; padding: 6px 10px; margin: 4px; '
                f'border-radius: 6px; cursor: pointer; display: inline-block; vertical-align: top; '
                f'border: 2px solid transparent; transition: all 0.2s; text-align: center; min-width: 60px;" '
                f'onclick="highlightDecodeTokenId({i})" '
                f'onmouseover="this.style.borderColor=\'#333\'" '
                f'onmouseout="if(!document.querySelector(\'.decode-token-id-tag[data-token-idx=\'{i}\'].highlighted\')) this.style.borderColor=\'transparent\'">'
                f'<div style="font-family: monospace; font-weight: bold; font-size: 14px; color: #000;">{token_id}</div>'
                f'<div style="font-size: 10px; color: #555; margin-top: 2px; word-break: break-word; max-width: 80px;">{word_label if word_label else "&nbsp;"}</div>'
                f'</div>'
            )
        
        decode_token_ids_html = '<div style="padding: 10px; background: #f8f9fa; border-radius: 8px; margin-top: 10px;">' + "".join(decode_token_ids_html_parts) + '</div>'
        
        return decoded_html, decode_token_ids_html, decoded_text
        
    except Exception as e:
        error_msg = f"❌ Invalid ID list: {str(e)}"
        return f"<div style='padding: 10px; color: red;'>{error_msg}</div>", "", error_msg

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

# JavaScript for interactive highlighting
highlight_js = """
<script>
let currentHighlighted = null;

function clearHighlights() {
    if (currentHighlighted) {
        currentHighlighted.forEach(el => {
            el.classList.remove('highlighted');
            el.style.borderColor = 'transparent';
            el.style.boxShadow = 'none';
        });
    }
    currentHighlighted = null;
}

function highlightToken(tokenIdx) {
    clearHighlights();
    const tokenEl = document.querySelector(`.token-tag[data-token-idx="${tokenIdx}"]`);
    if (!tokenEl) return;
    
    const tokenId = tokenEl.getAttribute('data-token-id');
    const highlighted = [tokenEl];
    
    // Highlight all tokens with same ID
    document.querySelectorAll(`.token-tag[data-token-id="${tokenId}"]`).forEach(el => {
        el.classList.add('highlighted');
        el.style.borderColor = '#ff0000';
        el.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
        highlighted.push(el);
    });
    
    // Highlight corresponding words
    document.querySelectorAll('.word-tag').forEach(wordEl => {
        const tokenIndices = wordEl.getAttribute('data-word-tokens').split(',');
        if (tokenIndices.includes(tokenIdx.toString())) {
            wordEl.classList.add('highlighted');
            wordEl.style.borderColor = '#ff0000';
            wordEl.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
            highlighted.push(wordEl);
        }
    });
    
    currentHighlighted = highlighted;
}

function highlightWord(tokenIndicesStr) {
    clearHighlights();
    const tokenIndices = tokenIndicesStr.split(',');
    const highlighted = [];
    
    // Highlight words
    document.querySelectorAll(`.word-tag[data-word-tokens="${tokenIndicesStr}"]`).forEach(el => {
        el.classList.add('highlighted');
        el.style.borderColor = '#ff0000';
        el.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
        highlighted.push(el);
    });
    
    // Highlight corresponding tokens
    tokenIndices.forEach(idx => {
        document.querySelectorAll(`.token-tag[data-token-idx="${idx}"]`).forEach(el => {
            el.classList.add('highlighted');
            el.style.borderColor = '#ff0000';
            el.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
            highlighted.push(el);
        });
    });
    
    currentHighlighted = highlighted;
}

function highlightDecodeToken(tokenIdx) {
    clearHighlights();
    const tokenEl = document.querySelector(`.decode-token-tag[data-token-idx="${tokenIdx}"]`);
    if (!tokenEl) return;
    
    const highlighted = [tokenEl];
    tokenEl.classList.add('highlighted');
    tokenEl.style.borderColor = '#ff0000';
    tokenEl.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
    
    // Highlight corresponding words in decoded text
    document.querySelectorAll('.decode-word-tag').forEach(wordEl => {
        const tokenIndices = wordEl.getAttribute('data-word-tokens').split(',');
        if (tokenIndices.includes(tokenIdx.toString())) {
            wordEl.classList.add('highlighted');
            wordEl.style.borderColor = '#ff0000';
            wordEl.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
            highlighted.push(wordEl);
        }
    });
    
    currentHighlighted = highlighted;
}

function highlightInputWord(tokenIndicesStr) {
    clearHighlights();
    const tokenIndices = tokenIndicesStr.split(',');
    const highlighted = [];
    
    // Highlight input words
    document.querySelectorAll(`.input-word-tag[data-word-tokens="${tokenIndicesStr}"]`).forEach(el => {
        el.classList.add('highlighted');
        el.style.borderColor = '#ff0000';
        el.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
        highlighted.push(el);
    });
    
    // Highlight corresponding token IDs
    tokenIndices.forEach(idx => {
        document.querySelectorAll(`.token-id-tag[data-token-idx="${idx}"]`).forEach(el => {
            el.classList.add('highlighted');
            el.style.borderColor = '#ff0000';
            el.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
            el.style.transform = 'scale(1.1)';
            highlighted.push(el);
        });
    });
    
    currentHighlighted = highlighted;
}

function highlightTokenId(tokenIdx) {
    clearHighlights();
    const tokenIdEl = document.querySelector(`.token-id-tag[data-token-idx="${tokenIdx}"]`);
    if (!tokenIdEl) return;
    
    const highlighted = [tokenIdEl];
    tokenIdEl.classList.add('highlighted');
    tokenIdEl.style.borderColor = '#ff0000';
    tokenIdEl.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
    tokenIdEl.style.transform = 'scale(1.1)';
    
    // Find input words that contain this token
    document.querySelectorAll('.input-word-tag').forEach(wordEl => {
        const tokenIndices = wordEl.getAttribute('data-word-tokens');
        if (tokenIndices) {
            const tokenList = tokenIndices.split(',');
            if (tokenList.includes(tokenIdx.toString())) {
                wordEl.classList.add('highlighted');
                wordEl.style.borderColor = '#ff0000';
                wordEl.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
                highlighted.push(wordEl);
            }
        }
    });
    
    currentHighlighted = highlighted;
}

function highlightDecodeWord(tokenIndicesStr) {
    clearHighlights();
    const tokenIndices = tokenIndicesStr.split(',');
    const highlighted = [];
    
    // Highlight words
    document.querySelectorAll(`.decode-word-tag[data-word-tokens="${tokenIndicesStr}"]`).forEach(el => {
        el.classList.add('highlighted');
        el.style.borderColor = '#ff0000';
        el.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
        highlighted.push(el);
    });
    
    // Highlight corresponding token IDs
    tokenIndices.forEach(idx => {
        document.querySelectorAll(`.decode-token-id-tag[data-token-idx="${idx}"]`).forEach(el => {
            el.classList.add('highlighted');
            el.style.borderColor = '#ff0000';
            el.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
            el.style.transform = 'scale(1.1)';
            highlighted.push(el);
        });
    });
    
    currentHighlighted = highlighted;
}

function highlightDecodeTokenId(tokenIdx) {
    clearHighlights();
    const tokenIdEl = document.querySelector(`.decode-token-id-tag[data-token-idx="${tokenIdx}"]`);
    if (!tokenIdEl) return;
    
    const highlighted = [tokenIdEl];
    tokenIdEl.classList.add('highlighted');
    tokenIdEl.style.borderColor = '#ff0000';
    tokenIdEl.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
    tokenIdEl.style.transform = 'scale(1.1)';
    
    // Find decoded words that contain this token
    document.querySelectorAll('.decode-word-tag').forEach(wordEl => {
        const tokenIndices = wordEl.getAttribute('data-word-tokens');
        if (tokenIndices) {
            const tokenList = tokenIndices.split(',');
            if (tokenList.includes(tokenIdx.toString())) {
                wordEl.classList.add('highlighted');
                wordEl.style.borderColor = '#ff0000';
                wordEl.style.boxShadow = '0 0 8px rgba(255,0,0,0.5)';
                highlighted.push(wordEl);
            }
        }
    });
    
    currentHighlighted = highlighted;
}
</script>
"""

with gr.Blocks(title="Hindi Tokenizer", head=highlight_js) as demo:
    gr.Markdown("## 🔡 Hindi BPE Tokenizer — Encode / Decode")
    
    # Hidden component to store token data
    token_data_store = gr.State(value="")

    with gr.Tab("Encode"):
        text_in = gr.Textbox(label="Enter text", lines=3)
        
        gr.Markdown("### 📝 Input Text (Click words to highlight token IDs)")
        input_html_out = gr.HTML(label="Clickable Input Text", value="<div style='padding: 10px; color: #666; font-style: italic;'>Enter text above and click Encode to see clickable words</div>")
        
        with gr.Row():
            token_count_out = gr.Number(label="Token Count", precision=0)
            compression_ratio_out = gr.Textbox(label="Compression Ratio (chars/token)", interactive=False)
        
        gr.Markdown("### Token IDs (Click to highlight words)")
        token_ids_html_out = gr.HTML(label="Token IDs with Labels")
        
        ids_out = gr.Textbox(label="Token IDs (CSV)", lines=4, max_lines=10, interactive=False)
        btn = gr.Button("Encode", variant="primary")
        btn.click(encode_text, text_in, [ids_out, token_count_out, compression_ratio_out, token_ids_html_out, token_data_store, input_html_out])

    with gr.Tab("Decode"):
        ids_in = gr.Textbox(label="Comma-separated token IDs", lines=4)
        
        gr.Markdown("### 📝 Decoded Text (Click words to highlight token IDs)")
        decoded_text_html_out = gr.HTML(label="Clickable Decoded Text", value="<div style='padding: 10px; color: #666; font-style: italic;'>Enter token IDs above and click Decode to see clickable words</div>")
        
        gr.Markdown("### Token IDs (Click to highlight words)")
        decode_token_ids_html_out = gr.HTML(label="Token IDs with Labels")
        
        decoded_text_out = gr.Textbox(label="Decoded Text", lines=4, max_lines=10, interactive=False)
        
        btn3 = gr.Button("Decode", variant="primary")
        btn3.click(decode_ids, ids_in, [decoded_text_html_out, decode_token_ids_html_out, decoded_text_out])

# Mount FastAPI + Gradio

if "app" not in globals():
    app = gr.mount_gradio_app(api, demo, path="/gradio")

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
