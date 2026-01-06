# NER Dataset Generation

## Training

### Bi-Encoder Models

**BCE Loss:**
```bash
accelerate launch train_bi_encoder.py configs/bi_encoder.json
```

**Contrastive Loss:**
```bash
accelerate launch train_bi_encoder_contrastive.py configs/bi_encoder_contrastive.json
```

### Cross-Encoder Models

**BCE Loss:**
```bash
accelerate launch train_cross_encoder.py configs/cross_encoder.json
```

**Contrastive Loss:**
```bash
accelerate launch train_cross_encoder_contrastive.py configs/cross_encoder_contrastive.json
```

### Customizing Training Data

To use multiple training files (e.g., all finerweb files), modify the config:

```json
{
  "train_file": "data/finerweb_splitted/*.jsonl",
  "validation_file": "data/conll2003/validation.jsonl",
  "test_file": "data/conll2003/test.jsonl"
}
```

The `train_file` field supports glob patterns, so `*.jsonl` will match all JSONL files in the directory.

To change the test dataset, simply update the `test_file` path in the config to point to your desired evaluation dataset.

## Evaluation

### Bi-Encoder Evaluation

```bash
python evaluate_bi_encoder.py \
  --pretrained_model_name_or_path models/bi_encoder/checkpoint-5000 \
  --evaluation_file data/conll2003/test.jsonl \
  --threshold 0.5 \
  --evaluation_format tokens
```

### Cross-Encoder Evaluation

```bash
python evaluate_cross_encoder.py \
  --pretrained_model_name_or_path models/cross_encoder/checkpoint-5000 \
  --evaluation_file data/conll2003/test.jsonl \
  --threshold 0.5 \
  --evaluation_format tokens
```

### Evaluation File Formats

The `--evaluation_file` argument accepts:
- **JSONL files**: Path to a `.jsonl` file (e.g., `data/conll2003/test.jsonl`)
- **HuggingFace DatasetDict**: Path to a directory containing a saved DatasetDict (e.g., `data/eval_data/panx/en`)

The script automatically detects the format and loads the appropriate split (`test` or `dev`).

### Evaluation Format

- `--evaluation_format text`: Uses character-level spans (`char_spans`) from the dataset
- `--evaluation_format tokens`: Uses token-level spans (`token_spans`) from the dataset

### Threshold

- For **BCE models**: Pass a float value (e.g., `0.5`)
- For **Contrastive models**: Pass either `"cls"` or `"label_token"` as a string
