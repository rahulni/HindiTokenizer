# 🔡 Hindi BPE Tokenizer

A streaming-based Byte Pair Encoding (BPE) tokenizer trained on a large Hindi corpus, featuring a modern web UI and REST API for tokenization and decoding tasks.

## 📋 Overview

This project implements a BPE tokenizer specifically trained for Hindi text using HuggingFace's `tokenizers` library. The tokenizer is designed with memory efficiency in mind, using streaming techniques to handle large corpora without loading everything into memory at once.

### Key Features

- ✅ **Streaming Training**: Processes large corpus files without memory overflow
- ✅ **BPE Algorithm**: Byte Pair Encoding with configurable vocabulary size
- ✅ **Web UI**: Interactive Gradio interface for encoding/decoding
- ✅ **REST API**: FastAPI backend for programmatic access
- ✅ **Real-time Metrics**: Token count, compression ratio, and more
- ✅ **HuggingFace Compatible**: Full integration with `transformers` library

## 🎯 Main Functions

### Encoding
- Convert Hindi text into token IDs
- Calculate token count and compression ratio
- Export token IDs as comma-separated values (CSV)

### Decoding
- Convert token IDs back to Hindi text
- Handle invalid inputs gracefully

### Evaluation Metrics
- **Compression Ratio**: Average characters per token (higher = better)
- **Token Count**: Number of tokens generated
- **UNK Rate**: Percentage of unknown tokens

## 📊 Tokenizer Specifications

### Vocabulary
- **Vocabulary Size**: 8,000 tokens (default, configurable)
- **Special Tokens**: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- **Algorithm**: Byte Pair Encoding (BPE)
- **Normalization**: NFKC normalization
- **Pre-tokenization**: Whitespace-based

### Training Configuration
- **Min Frequency**: 2 (minimum occurrence for a merge to be considered)
- **Chunk Size**: 50,000 lines per batch (streaming)
- **Corpus**: Large Hindi text corpus (~1.23GB)

## 📈 Performance Metrics

The tokenizer evaluation provides the following metrics:

| Metric | Description |
|--------|-------------|
| **Compression Ratio** | Characters per token (typically 3-5 for Hindi) |
| **Avg Tokens/Sentence** | Average number of tokens per sentence |
| **Avg Tokens/Word** | Average number of tokens per word |
| **UNK Rate** | Percentage of unknown tokens (lower is better) |

### Example Metrics (from evaluation)
```
Sentences: 1000
Characters (no space): ~50,000
Tokens: ~15,000
Avg tokens per sentence: ~15.000
Avg tokens per word: ~1.500
Compression ratio: ~3.333
UNK rate: ~0.000001
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Navigate to Project
```bash
cd s11
```

### Step 2: Install Dependencies

For running the web app:
```bash
pip install -r requirements.txt
```

For training the tokenizer:
```bash
pip install -r requirements_train.txt
```

### Step 3: Download and Prepare Corpus

Run the corpus download and cleaning script:
```bash
python download_clean_hindi_corpus.py
```

This will:
- Download Hindi corpus data
- Clean and preprocess the text
- Save to `clean_corpus/hindi_corpus_clean.txt`

### Step 4: Train the Tokenizer

Train a new tokenizer (optional if pre-trained model exists):
```bash
python train_and_serve_tokenizer_streaming.py --vocab 8000 --minfreq 2
```

Options:
- `--vocab`: Vocabulary size (default: 8000)
- `--minfreq`: Minimum frequency threshold (default: 2)
- `--chunk`: Chunk size for streaming (default: 50000)
- `--force`: Force retrain even if tokenizer exists
- `--train_only`: Train only, don't serve API

## 💻 Usage

### Web UI (Recommended)

Launch the Gradio web interface:

```bash
python app.py
```

Then open your browser to:
```
http://localhost:7860
```

#### UI Features

**Encode Tab:**
- Enter Hindi text in the input box
- View token count and compression ratio
- Get token IDs as comma-separated values

**Decode Tab:**
- Enter comma-separated token IDs
- View decoded Hindi text

### REST API

The FastAPI backend provides the following endpoints:

#### Base URL
```
http://localhost:7860/gradio
```

#### Endpoints

**1. Home**
```
GET /
```
Returns basic info and vocabulary size.

**2. Tokenize**
```
GET /tokenize?text=नमस्ते
```
Returns tokens and token IDs for the input text.

**Response:**
```json
{
  "tokens": ["▁नमस्ते"],
  "ids": [1234, 5678]
}
```

**3. Decode**
```
GET /decode?ids=1234,5678
```
Returns decoded text from token IDs.

**Response:**
```json
{
  "text": "नमस्ते"
}
```

### Python API

```python
from transformers import PreTrainedTokenizerFast

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_hindi_bpe_8k_stream/hf")

# Encode
text = "नमस्ते"
encoded = tokenizer(text, add_special_tokens=False)
token_ids = encoded["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(token_ids)

print(f"Token IDs: {token_ids}")
print(f"Tokens: {tokens}")

# Decode
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")

# Calculate compression ratio
char_count = len(text)
token_count = len(token_ids)
compression_ratio = char_count / token_count
print(f"Compression Ratio: {compression_ratio:.2f}")
```

## 📁 Project Structure

```
s11/
├── app.py                              # Main Gradio UI application
├── train_and_serve_tokenizer_streaming.py  # Tokenizer training script
├── download_clean_hindi_corpus.py      # Corpus download and cleaning
├── requirements.txt                    # App dependencies
├── requirements_train.txt              # Training dependencies
├── clean_corpus/
│   └── hindi_corpus_clean.txt         # Processed Hindi corpus
└── tokenizer_hindi_bpe_8k_stream/
    ├── tokenizer.json                 # Tokenizer model file
    └── hf/                            # HuggingFace format
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── special_tokens_map.json
```

## 🔧 Configuration

### Tokenizer Parameters

You can customize the tokenizer during training:

```python
# In train_and_serve_tokenizer_streaming.py
DEFAULT_VOCAB = 8000        # Vocabulary size
DEFAULT_MIN_FREQ = 2        # Minimum merge frequency
```

### UI Configuration

```python
# In app.py
TOKENIZER_JSON = "tokenizer_hindi_bpe_8k_stream/tokenizer.json"
HF_DIR = "tokenizer_hindi_bpe_8k_stream/hf"
```

## 📝 Example Workflows

### Workflow 1: Quick Start with Pre-trained Model

1. Install dependencies: `pip install -r requirements.txt`
2. Launch app: `python app.py`
3. Use the web UI at `http://localhost:7860`

### Workflow 2: Train Custom Tokenizer

1. Prepare corpus: `python download_clean_hindi_corpus.py`
2. Train tokenizer: `python train_and_serve_tokenizer_streaming.py --vocab 10000`
3. Launch app: `python app.py`

### Workflow 3: API Integration

1. Start the app: `python app.py`
2. Use REST endpoints programmatically:
   ```python
   import requests
   response = requests.get("http://localhost:7860/gradio/tokenize?text=नमस्ते")
   print(response.json())
   ```

## 🎓 Technical Details

### BPE Algorithm
- Implements the standard Byte Pair Encoding algorithm
- Merges most frequent byte pairs iteratively
- Builds vocabulary from corpus statistics

### Streaming Implementation
- Processes corpus in chunks (default: 50,000 lines)
- Uses iterator-based training to avoid memory issues
- Progress logging every 200,000 lines

### Normalization
- NFKC normalization for consistent character representation
- Handles various Unicode forms

## 🐛 Troubleshooting

### Issue: Tokenizer not found
**Solution**: Train the tokenizer first using `train_and_serve_tokenizer_streaming.py`

### Issue: Out of memory during training
**Solution**: Reduce chunk size with `--chunk 10000`

### Issue: Port already in use
**Solution**: Change port in `app.py`: `demo.launch(server_port=7861)`

## 📄 License

This project is part of the ERA (End-to-End Real AI) course.

## 👤 Author

Created as part of Session 11 - Tokenizer Training and Deployment.

## 🙏 Acknowledgments

- HuggingFace for the `tokenizers` and `transformers` libraries
- Gradio for the web UI framework
- FastAPI for the REST API framework

---

**Note**: This tokenizer is specifically optimized for Hindi text. Performance on other languages may vary.

