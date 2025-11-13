import json
import glob
import os
from tqdm import tqdm
from pythainlp.tokenize import word_tokenize

def char_offsets_to_token_offsets(char_spans, tokens, text):
    """
    Convert character-level spans to token-level spans.
    
    Args:
        char_spans: List of dicts with {"start": int, "end": int, "label": str}
        tokens: List of token strings
        text: Original text string
    
    Returns:
        List of dicts with {"start": int, "end": int, "label": str} where
        start/end are token indices (exclusive end)
    """
    # First, build character position map for each token
    token_char_positions = []
    char_pos = 0
    
    for token in tokens:
        # Find token in text starting from current position
        remaining_text = text[char_pos:]
        token_idx = remaining_text.find(token)
        
        if token_idx != -1:
            token_start = char_pos + token_idx
            token_end = token_start + len(token)
            token_char_positions.append((token_start, token_end))
            char_pos = token_end
        else:
            # Fallback: assume space-separated
            token_start = char_pos
            token_end = token_start + len(token)
            token_char_positions.append((token_start, token_end))
            char_pos = token_end + 1
    
    # Now map each char_span to token indices
    token_spans = []
    
    for char_span in char_spans:
        span_start = int(char_span["start"])
        span_end = int(char_span["end"])
        label = char_span["label"]
        
        # Find first token that overlaps with span start
        token_start_idx = None
        for i, (token_start, token_end) in enumerate(token_char_positions):
            # Token overlaps if it starts before span ends and ends after span starts
            if token_start < span_end and token_end > span_start:
                token_start_idx = i
                break
        
        if token_start_idx is None:
            # No token found for this span, skip it
            continue
        
        # Find last token that overlaps with span end
        token_end_idx = token_start_idx
        for i in range(token_start_idx, len(token_char_positions)):
            token_start, token_end = token_char_positions[i]
            if token_start < span_end and token_end > span_start:
                token_end_idx = i
            else:
                # We've passed the span, stop
                break
        
        # token_end_idx is inclusive, but we need exclusive end
        token_spans.append({
            "start": token_start_idx,
            "end": token_end_idx + 1,  # Exclusive end
            "label": label
        })
    
    return token_spans

if __name__ == "__main__":
    for file in glob.glob("/vol/tmp/goldejon/ner/data/finerweb_splitted/tha.jsonl"):
        with open(file, "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]

        for sample in tqdm(dataset):
            sample["tokens"] = [x for x in word_tokenize(sample["text"]) if x.strip()]
            sample["token_spans"] = char_offsets_to_token_offsets(
                sample["char_spans"], 
                sample["tokens"],
                sample["text"]
            )

        with open(file, "w") as f:
            for sample in dataset:
                f.write(json.dumps(sample) + "\n")
