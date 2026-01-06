from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import stanza
import re
from src.paths import get_paths

def all_spans_mask(input_ids, sequence_ids, max_span_length = 30):
    text_start_index = 0
    while sequence_ids[text_start_index] == None:
        text_start_index += 1

    text_end_index = len(input_ids) - 1
    while sequence_ids[text_end_index] == None:
        text_end_index -= 1

    start_mask, end_mask = [], []
    for sequence_id in sequence_ids:
        start_mask.append(1 if sequence_id is not None else 0)
        end_mask.append(1 if sequence_id is not None else 0)

    span_mask = [
        [
            (j - i >= 0 and j - i < max_span_length) * s * e for j, e in enumerate(end_mask)
        ]
        for i, s in enumerate(start_mask)
    ]

    return text_start_index, text_end_index, start_mask, end_mask, span_mask

def subwords_mask(input_ids, word_ids, max_span_length = 30):
    text_start_index = 0
    while word_ids[text_start_index] == None:
        text_start_index += 1

    text_end_index = len(input_ids) - 1
    while word_ids[text_end_index] == None:
        text_end_index -= 1

    start_mask, end_mask = [], []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            start_mask.append(1 if word_id != previous_word_id else 0)
            end_mask.append(1 if (idx + 1 == len(word_ids) or word_ids[idx + 1] != word_id) else 0)
        else:
            start_mask.append(0)
            end_mask.append(0)
        previous_word_id = word_id

    span_mask = [
        [
            (j - i >= 0 and j - i < max_span_length) * s * e for j, e in enumerate(end_mask)
        ]
        for i, s in enumerate(start_mask)
    ]

    return text_start_index, text_end_index, start_mask, end_mask, span_mask


def char_spans_to_token_spans(char_spans, token_char_positions):
    """
    Convert character spans to token-level spans.
    
    Args:
        char_spans: List of dicts with {"start": int, "end": int, "label": str} (character positions)
        token_char_positions: List of (start_char, end_char) tuples for each token
    
    Returns:
        token_spans: List of dicts with {"start": int, "end": int, "label": str} (token indices, exclusive end)
    """
    token_spans = []
    
    for char_span in char_spans:
        span_start = char_span["start"]
        span_end = char_span["end"]
        label = char_span["label"]
        
        # Find first token that overlaps with span
        token_start_idx = None
        for i, (tok_start, tok_end) in enumerate(token_char_positions):
            # Token overlaps if it starts before span ends and ends after span starts
            if tok_start < span_end and tok_end > span_start:
                token_start_idx = i
                break
        
        if token_start_idx is None:
            # No token found for this span, skip it
            continue
        
        # Find last token that overlaps with span
        token_end_idx = token_start_idx
        for i in range(token_start_idx, len(token_char_positions)):
            tok_start, tok_end = token_char_positions[i]
            if tok_start < span_end and tok_end > span_start:
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


def whitespace_tokenize_and_adjust_spans(text, char_spans):
    """
    Tokenize text by whitespace and adjust character spans to align with token boundaries.
    The text remains whitespace-separated.
    
    Args:
        text: Original text string
        char_spans: List of dicts with {"start": int, "end": int, "label": str}
    
    Returns:
        tokens: List of token strings
        adjusted_char_spans: List of dicts with adjusted character spans aligned to token boundaries
        token_spans: List of dicts with token-level spans (token indices)
    """
    # Split by whitespace to get tokens
    tokens = text.split()
    
    # Compute token boundaries in the original text
    token_char_positions = []
    current_pos = 0
    
    for token in tokens:
        # Find the token in the text starting from current_pos
        token_start = text.find(token, current_pos)
        if token_start == -1:
            # Fallback: assume sequential
            token_start = current_pos
        token_end = token_start + len(token)
        token_char_positions.append((token_start, token_end))
        # Move past this token and any whitespace
        current_pos = token_end
        # Skip whitespace
        while current_pos < len(text) and text[current_pos].isspace():
            current_pos += 1
    
    # Adjust char_spans to align with token boundaries
    adjusted_char_spans = []
    for span in char_spans:
        span_start = span["start"]
        span_end = span["end"]
        label = span["label"]
        
        # Find tokens that overlap with this span
        token_start_idx = None
        token_end_idx = None
        
        for i, (tok_start, tok_end) in enumerate(token_char_positions):
            # Check if token overlaps with span (inclusive start, exclusive end)
            if tok_start < span_end and tok_end > span_start:
                if token_start_idx is None:
                    token_start_idx = i
                token_end_idx = i
        
        if token_start_idx is not None and token_end_idx is not None:
            # Use token boundaries
            final_start = token_char_positions[token_start_idx][0]
            final_end = token_char_positions[token_end_idx][1]
            adjusted_char_spans.append({
                "start": final_start,
                "end": final_end,
                "label": label
            })
    
    # Convert to token spans
    token_spans = char_spans_to_token_spans(adjusted_char_spans, token_char_positions)
    
    return tokens, adjusted_char_spans, token_spans


def stanza_tokenize_and_adjust_spans(text, char_spans, nlp):
    """
    Remove whitespace, tokenize with Stanza, and adjust character spans.
    
    Args:
        text: Original text string (may contain whitespace)
        char_spans: List of dicts with {"start": int, "end": int, "label": str}
        nlp: Stanza Pipeline object
    
    Returns:
        tokens: List of token strings
        adjusted_char_spans: List of dicts with adjusted character spans
        token_spans: List of dicts with token-level spans (token indices)
    """
    # Remove all whitespace
    text_no_whitespace = re.sub(r'\s+', '', text)
    
    # Build mapping from original char positions to new positions (without whitespace)
    original_to_new = {}
    new_pos = 0
    
    for char_idx, char in enumerate(text):
        if not char.isspace():
            original_to_new[char_idx] = new_pos
            new_pos += 1
    
    # Tokenize with Stanza
    doc = nlp(text_no_whitespace)
    tokens = [word.text for sent in doc.sentences for word in sent.words]
    
    # Get token character positions in the new text (without whitespace)
    # Find each token sequentially in the text
    token_char_positions = []
    current_pos = 0
    for token in tokens:
        # Find token starting from current position
        token_start = text_no_whitespace.find(token, current_pos)
        if token_start == -1:
            # Fallback: assume sequential (shouldn't happen with proper tokenization)
            token_start = current_pos
        token_end = token_start + len(token)
        token_char_positions.append((token_start, token_end))
        current_pos = token_end
    
    # Adjust char_spans: first map to new text positions, then to token boundaries
    adjusted_char_spans = []
    for span in char_spans:
        start = span["start"]
        end = span["end"]
        label = span["label"]
        
        # Map to positions in text without whitespace
        new_start = None
        new_end = None
        
        # Find first non-whitespace character at or after start
        for i in range(start, min(end, len(text))):
            if not text[i].isspace() and i in original_to_new:
                new_start = original_to_new[i]
                break
        
        # Find last non-whitespace character before end
        for i in range(end - 1, start - 1, -1):
            if i < len(text) and not text[i].isspace() and i in original_to_new:
                new_end = original_to_new[i] + 1  # Exclusive end
                break
        
        if new_start is None or new_end is None or new_start >= new_end:
            continue
        
        # Find tokens that overlap with this span
        token_start_idx = None
        token_end_idx = None
        
        for i, (tok_start, tok_end) in enumerate(token_char_positions):
            # Check if token overlaps with span
            if (tok_start < new_end and tok_end > new_start):
                if token_start_idx is None:
                    token_start_idx = i
                token_end_idx = i
        
        if token_start_idx is not None and token_end_idx is not None:
            # Use token boundaries
            final_start = token_char_positions[token_start_idx][0]
            final_end = token_char_positions[token_end_idx][1]
            adjusted_char_spans.append({
                "start": final_start,
                "end": final_end,
                "label": label
            })
    
    # Convert to token spans
    token_spans = char_spans_to_token_spans(adjusted_char_spans, token_char_positions)
    
    return tokens, adjusted_char_spans, token_spans


def process_language_dataset(language, dataset):
    """
    Process dataset for a given language and return list with text and adjusted char_spans.
    
    Args:
        language: Language code ('eng', 'swa', 'tha', 'cmn')
        dataset: HuggingFace dataset
    
    Returns:
        List of dicts with 'text' and 'char_spans' keys
    """
    processed_data = []
    
    # Initialize Stanza for Thai and Chinese if needed
    nlp = None
    if language in ['tha', 'cmn']:
        lang_code = 'th' if language == 'tha' else 'zh'
        try:
            nlp = stanza.Pipeline(lang_code, processors='tokenize', use_gpu=False)
        except Exception as e:
            print(f"Warning: Could not initialize Stanza for {language}: {e}")
            print("Falling back to character-level tokenization")
    
    for sample in tqdm(dataset["train"], desc=f"Processing {language}"):
        text = sample.get("text", "")
        char_spans = sample.get("char_spans", [])
        
        if not text:
            continue
        
        if language in ['eng', 'swh']:
            # Whitespace tokenize (text remains whitespace-separated)
            tokens, adjusted_char_spans, token_spans = whitespace_tokenize_and_adjust_spans(text, char_spans)
            # Keep original text (it's already whitespace-separated)
            processed_text = text
        elif language in ['tha', 'cmn']:
            # Remove whitespace and use Stanza
            if nlp is not None:
                tokens, adjusted_char_spans, token_spans = stanza_tokenize_and_adjust_spans(text, char_spans, nlp)
                # Rebuild text without whitespace (tokens concatenated)
                processed_text = ''.join(tokens)
            else:
                # Fallback: just remove whitespace, keep original spans adjusted
                processed_text = re.sub(r'\s+', '', text)
                # Adjust spans by removing whitespace
                original_to_new = {}
                new_pos = 0
                for char_idx, char in enumerate(text):
                    if not char.isspace():
                        original_to_new[char_idx] = new_pos
                        new_pos += 1
                
                adjusted_char_spans = []
                for span in char_spans:
                    new_start = original_to_new.get(span["start"])
                    new_end = original_to_new.get(span["end"] - 1)
                    if new_start is not None and new_end is not None:
                        adjusted_char_spans.append({
                            "start": new_start,
                            "end": new_end + 1,
                            "label": span["label"]
                        })
                tokens = list(processed_text)  # Character-level fallback
                # Compute token positions for fallback
                token_char_positions = [(i, i+1) for i in range(len(tokens))]
                token_spans = char_spans_to_token_spans(adjusted_char_spans, token_char_positions)
        else:
            # Unknown language, skip
            continue
        
        processed_data.append({
            "id": sample.get("id", ""),
            "text": processed_text,
            "char_spans": adjusted_char_spans,
            "token_spans": token_spans,
            "tokens": tokens
        })
    
    return processed_data


def compute_span_masks_for_sample(sample, tokenizer, format_type, loss_masking, max_span_length=30):
    """
    Compute span masks for a single sample.
    
    Args:
        sample: Dict with 'text', 'char_spans', 'tokens', 'token_spans'
        tokenizer: HuggingFace tokenizer
        format_type: 'text' (uses char_spans) or 'tokens' (uses token_spans)
        loss_masking: 'none' (all_spans_mask) or 'subwords' (subwords_mask)
        max_span_length: Maximum span length
    
    Returns:
        Dict with span mask information
    """
    if format_type == 'text':
        # Use text and char_spans with all_spans_mask (sequence_ids)
        text = sample['text']
        encodings = tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None,
            return_offsets_mapping=True
        )
        input_ids = encodings['input_ids']
        sequence_ids = encodings.sequence_ids(0)
        
        # For text format, always use all_spans_mask
        text_start_idx, text_end_idx, start_mask, end_mask, span_mask = all_spans_mask(
            input_ids, sequence_ids, max_span_length
        )
    
    elif format_type == 'tokens':
        # Use tokens and token_spans with subwords_mask (word_ids)
        tokens = sample['tokens']
        encodings = tokenizer(
            tokens,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None,
            is_split_into_words=True
        )
        input_ids = encodings['input_ids']
        word_ids = encodings.word_ids(0)
        
        # For tokens format, always use subwords_mask
        text_start_idx, text_end_idx, start_mask, end_mask, span_mask = subwords_mask(
            input_ids, word_ids, max_span_length
        )
    
    # Count valid spans (span_mask is a 2D list: span_mask[i][j] = 1 if span from i to j is valid)
    num_valid_spans = sum(sum(row) for row in span_mask)
    
    # Count number of labels (positive spans) - should be same for both formats
    if format_type == 'text':
        num_labels = len(sample.get('char_spans', []))
    else:  # tokens format
        num_labels = len(sample.get('token_spans', []))
    
    return {
        'num_valid_spans': num_valid_spans,
        'num_labels': num_labels
    }


def main():
    # First task: Get processed data for each language
    language_data = {}
    
    paths = get_paths()
    for language in ['eng', 'swh', 'tha', 'cmn']:
        dataset_path = paths.finerweb_splitted_dir / f"{language}.jsonl"
        dataset = load_dataset("json", data_files={"train": str(dataset_path)})
        dataset['train'] = dataset["train"].select(range(1000))
        
        # Process the dataset
        processed_data = process_language_dataset(language, dataset)
        language_data[language] = processed_data
        
        print(f"Processed {len(processed_data)} samples for {language}")
    
    # Second task: Loop over tokenizers and compute span masks
    tokenizers = [
        'google-bert/bert-base-multilingual-cased',
        'FacebookAI/xlm-roberta-base',
        'microsoft/mdeberta-v3-base',
        'jhu-clsp/mmBERT-base',
        'google/rembert',
        'google/mt5-base'
    ]
    
    # Format and loss_masking combinations:
    # - text format: uses all_spans_mask with sequence_ids (loss_masking='none' is just for naming)
    # - tokens format: uses subwords_mask with word_ids (loss_masking='subwords')
    format_configs = [
        ('text', 'none'),  # text format with char_spans -> all_spans_mask
        ('tokens', 'subwords')  # tokens format with token_spans -> subwords_mask
    ]
    
    # Single output file for all results
    output_file = 'span_ratios.jsonl'
    # Clear output file at the start
    with open(output_file, 'w') as f:
        pass  # Clear file
    
    results = []
    
    for tokenizer_name in tqdm(tokenizers, desc="Processing tokenizers"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
            continue
        
        for language in ['eng', 'swh', 'tha', 'cmn']:
            if language not in language_data:
                continue
            
            for format_type, loss_masking in format_configs:
                # Compute span masks for all samples and collect lists
                valid_spans_list = []
                labels_list = []
                
                for sample in tqdm(language_data[language], desc=f"{tokenizer_name} {language} {format_type} {loss_masking}", leave=False):
                    try:
                        mask_info = compute_span_masks_for_sample(
                            sample, tokenizer, format_type, loss_masking, max_span_length=30
                        )
                        valid_spans_list.append(mask_info['num_valid_spans'])
                        labels_list.append(mask_info['num_labels'])
                    except Exception as e:
                        print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
                        continue
                
                if len(valid_spans_list) > 0:
                    result = {
                        'tokenizer': tokenizer_name,
                        'language': language,
                        'format': format_type,
                        'loss_masking': loss_masking,
                        'valid_spans': valid_spans_list,
                        'labels': labels_list,
                        'num_samples': len(valid_spans_list)
                    }
                    results.append(result)
                    
                    # Write to single joint file incrementally
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(result) + '\n')
    
    print(f"\nCompleted processing. Total results: {len(results)}")
    return results


if __name__ == "__main__":
    main()

