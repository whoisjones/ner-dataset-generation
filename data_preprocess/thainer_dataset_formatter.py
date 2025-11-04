import json
import os
from datasets import load_dataset

def tokens_to_char_spans(tokens):
    """
    Convert tokens to character offsets by concatenating tokens directly.
    Tokens may already contain spaces, so we don't add extra spaces.
    Returns a list of (start_char, end_char) for each token.
    """
    char_spans = []
    current_pos = 0
    
    for token in tokens:
        start = current_pos
        end = current_pos + len(token)
        char_spans.append((start, end))
        current_pos = end  # No extra space, tokens are concatenated directly
    
    return char_spans

def transform_bio_to_spans(tokens, ner_tags, id2label):
    """
    Convert BIO format annotations to span-based format with exclusive end indices.
    
    Returns:
        token_spans: List of dicts with keys: start, end (exclusive), label
        char_spans: List of dicts with keys: start, end (exclusive), label
    """
    token_spans = []
    char_spans = []
    
    # Get character positions for each token
    token_char_positions = tokens_to_char_spans(tokens)
    
    start_idx = None
    ent_type = None
    
    for idx, tag_id in enumerate(ner_tags):
        tag = id2label[tag_id]
        if tag == 'O':
            if start_idx is not None:
                # Close previous entity (exclusive end)
                end_idx = idx
                token_spans.append({
                    "start": start_idx,
                    "end": end_idx,
                    "label": ent_type
                })
                
                # Character span (exclusive end)
                char_start = token_char_positions[start_idx][0]
                char_end = token_char_positions[idx - 1][1]
                char_spans.append({
                    "start": char_start,
                    "end": char_end,
                    "label": ent_type
                })
                
                start_idx = None
                ent_type = None
        elif tag.startswith('B-'):
            if start_idx is not None:
                # Close previous entity before starting a new one (exclusive end)
                end_idx = idx
                token_spans.append({
                    "start": start_idx,
                    "end": end_idx,
                    "label": ent_type
                })
                
                char_start = token_char_positions[start_idx][0]
                char_end = token_char_positions[idx - 1][1]
                char_spans.append({
                    "start": char_start,
                    "end": char_end,
                    "label": ent_type
                })
            
            start_idx = idx
            ent_type = tag[2:]  # Remove 'B-'
        elif tag.startswith('I-'):
            if start_idx is None:
                # I- without B- before, treat as B-
                start_idx = idx
                ent_type = tag[2:]
            # else: continue the current entity
    
    # Handle last span if exists (exclusive end)
    if start_idx is not None:
        end_idx = len(ner_tags)
        token_spans.append({
            "start": start_idx,
            "end": end_idx,
            "label": ent_type
        })
        
        char_start = token_char_positions[start_idx][0]
        char_end = token_char_positions[-1][1]
        char_spans.append({
            "start": char_start,
            "end": char_end,
            "label": ent_type
        })
    
    return token_spans, char_spans

def transform_dataset(dataset, output_file):
    """
    Transform test.jsonl to include tokens, text, char spans, and token spans.
    """
    transformed_data = []
    id2label = {i: label for i, label in enumerate(dataset.features["ner"].feature.names)}
    
    for idx, sample in enumerate(dataset):
        tokens = sample['words']
        ner_tags = sample['ner']
        
        # Concatenate tokens directly (tokens may already contain spaces)
        text = ''.join(tokens)
        
        # Skip data points where text consists of empty strings
        if not text.strip():
            continue
        
        token_spans, char_spans = transform_bio_to_spans(tokens, ner_tags, id2label)
        
        transformed_sample = {
            'id': idx,
            'tokens': tokens,
            'text': text,
            'token_spans': token_spans,
            'char_spans': char_spans
        }
        
        transformed_data.append(transformed_sample)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for sample in transformed_data:
            f.write(json.dumps(sample) + '\n')
    
if __name__ == '__main__':
    dataset = load_dataset('pythainlp/thainer-corpus-v2.2', revision='convert/parquet')
    transform_dataset(dataset['train'], '/vol/tmp/goldejon/ner/data/thainer/train.jsonl')
    transform_dataset(dataset['validation'], '/vol/tmp/goldejon/ner/data/thainer/validation.jsonl')
    transform_dataset(dataset['test'], '/vol/tmp/goldejon/ner/data/thainer/test.jsonl')