# NER Dataset Generation

This repository contains tools for training dual encoder models on Named Entity Recognition (NER) tasks with span-based binary cross-entropy loss.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
ner-dataset-generation/
├── train.py              # Main training script (run from project root)
├── requirements.txt      # Python dependencies
├── configs/              # JSON configuration files
│   ├── default.json     # Standard configuration
│   ├── quick_test.json  # Fast config for testing
│   ├── large_model.json # Config for large models
│   └── README.md        # Config documentation
├── src/
│   ├── __init__.py      # Package initialization
│   ├── model.py         # SpanModel (dual encoder) definition
│   ├── collator.py      # InBatchDataCollator for batching
│   ├── data.py          # Data processing utilities
│   ├── metrics.py       # Evaluation metrics and span prediction
│   ├── config.py        # SpanModelConfig configuration
│   ├── logger.py        # Logging utilities
│   ├── trainer.py       # Training and evaluation loops
│   └── args.py          # Model and data training arguments
└── scripts/
    └── evaluate_gliner.py
```

## Training

### Using Config Files (Recommended)

Train using a JSON config file:

```bash
# Use configuration file
python train.py configs/conll.json

# The config file should contain all training parameters
```

### Config File Format

The config file should be a JSON file containing all training parameters. Example:

```json
{
  "run_name": "my-experiment",
  "token_encoder": "bert-base-uncased",
  "type_encoder": "bert-base-uncased",
  "dataset_name": "conll2003",
  "annotation_format": "text",
  "train_file": "path/to/train.jsonl",
  "validation_file": "path/to/validation.jsonl",
  "test_file": "path/to/test.jsonl",
  "output_dir": "./outputs",
  "per_device_train_batch_size": 16,
  "per_device_eval_batch_size": 16,
  "learning_rate": 3e-05,
  "max_steps": 3000,
  "eval_steps": 500,
  "max_seq_length": 512,
  "warmup_steps": 300,
  "seed": 42,
  "max_span_length": 30,
  "dropout": 0.1,
  "linear_hidden_size": 128,
  "init_temperature": 0.03,
  "start_loss_weight": 1.0,
  "end_loss_weight": 1.0,
  "span_loss_weight": 3.0,
  "do_train": true,
  "do_eval": true,
  "do_predict": true
}
```

See `configs/conll.json` for a complete example.

## Training Features

- **Comprehensive Logging**: All training events logged to both console and `run.log` file with timestamps
- **JSON Config Files**: Manage training configurations with JSON files for reproducibility and easy experimentation
- **Step-based training**: Train for a specified number of steps rather than epochs
- **Periodic evaluation**: Automatically evaluates on validation set every N steps
- **Final test evaluation**: Evaluates on test set after training and saves results to JSON
- **Span-based predictions**: Predicts spans using start, end, and span logits with BCE loss and thresholding
- **Overlap resolution**: Automatically resolves overlapping spans by keeping highest confidence spans
- **Comprehensive metrics**: Computes macro-averaged precision, recall, F1 score, and accuracy
- **Checkpoint saving**: Saves checkpoints at each evaluation interval with full model state
- **Best model tracking**: Automatically saves the best model based on validation F1 score
- **Batch-specific entity types**: Only encodes entity types present in each batch (efficient for large label sets)

## Model Architecture

The SpanModel uses a dual encoder architecture:

- **Token Encoder**: Pre-trained transformer (e.g., BERT) that encodes input text tokens
- **Type Encoder**: Pre-trained transformer that encodes entity type labels
- **Span Module**: Computes span representations by concatenating start/end token embeddings with span width embeddings
- **Similarity Computation**: Uses dot product similarity with learnable temperature scaling

The model predicts:
- **Start logits**: `[batch_size, num_types, seq_len]` - probability that a token is the start of an entity of a given type
- **End logits**: `[batch_size, num_types, seq_len]` - probability that a token is the end of an entity of a given type  
- **Span logits**: `[batch_size, num_types, seq_len, seq_len]` - probability that a span (start, end) is an entity of a given type

**Training Loss**: Binary cross-entropy with logits (BCE with logits) combining:
- Start position loss (weighted)
- End position loss (weighted)
- Span-level loss (weighted)

**Prediction**: A span is predicted if all three logits (start, end, span) exceed the threshold (default 0.5) after sigmoid.

## Dataset Format

The training script expects JSONL files where each line is a JSON object with:

```json
{
  "id": "0",
  "tokens": ["SOCCER", "-", "JAPAN", "GET", "LUCKY", "WIN", ",", "CHINA", "IN", "SURPRISE", "DEFEAT", "."],
  "text": "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .",
  "token_spans": [
    {"start": 2, "end": 3, "label": "LOC"},
    {"start": 7, "end": 8, "label": "PER"}
  ],
  "char_spans": [
    {"start": 9, "end": 14, "label": "LOC"},
    {"start": 31, "end": 36, "label": "PER"}
  ]
}
```

**Fields**:
- `id`: Unique identifier for the example
- `tokens`: List of token strings
- `text`: Full text as a space-separated string
- `token_spans`: List of spans with exclusive end indices `[start, end)` and labels
- `char_spans`: List of character-level spans with exclusive end indices and labels

The collator can work with either `token_spans` (when `annotation_format: "tokens"`) or `char_spans` (when `annotation_format: "text"`).

## Output Files

After training, the following files are saved to the output directory:

```
outputs/
├── run.log                  # Complete training log with timestamps
├── config.json              # Training configuration used
├── checkpoint-500/          # Checkpoint at step 500
│   ├── config.json
│   ├── token_encoder/       # Token encoder model
│   ├── type_encoder/        # Type encoder model
│   └── model.pt             # Span model weights
├── checkpoint-1000/         # Checkpoint at step 1000
│   ├── config.json
│   ├── token_encoder/
│   ├── type_encoder/
│   └── model.pt
├── best/                    # Best model based on validation F1
│   ├── config.json
│   ├── token_encoder/
│   ├── type_encoder/
│   └── model.pt
├── final/                   # Final model after all training steps
│   ├── config.json
│   ├── token_encoder/
│   ├── type_encoder/
│   └── model.pt
└── test_results.json        # Final test set evaluation results
```

### `run.log`
Complete training log with timestamps for all events:
```
2025-10-29 14:30:15 - INFO - Using device: cuda
2025-10-29 14:30:20 - INFO - Loading dataset: conll2003
2025-10-29 14:30:45 - INFO - Training samples: 14041
2025-10-29 14:30:50 - INFO - Starting training for 10000 steps...
2025-10-29 14:32:10 - INFO - Step 200/10000
2025-10-29 14:32:10 - INFO - Training Loss: 0.8234
...
```

### `test_results.json`
Final test set evaluation results:
```json
{
  "test_metrics": {
    "loss": 0.1234,
    "precision": 0.8542,
    "recall": 0.8321,
    "f1": 0.8430,
    "accuracy": 0.8430
  },
  "config": { ... },
  "final_step": 10000
}
```

## Loading Saved Models

You can load a saved model for inference:

```python
from src.model import SpanModel

# Load model from checkpoint
model = SpanModel.from_pretrained("./outputs/best")
model.eval()

# Use the model for inference
# (see model.forward() for input format)
```

## Evaluation

For evaluation using GLiNER, see `scripts/evaluate_gliner.py`.

## Utilities

### Dataset Formatting

The `conll_dataset_formatter.py` script can convert CoNLL2003 format datasets to the required JSONL format:

```bash
python conll_dataset_formatter.py
```

This will transform datasets to include tokens, text, token spans, and character spans with exclusive end indices.