from tqdm import tqdm


def transform_bio_to_span(sample, id2label):
    """
    Convert BIO format annotations to span-based format.
    
    Returns:
        spans: List of (start, end) tuples
        labels: List of entity types
    """
    spans = []
    labels = []
    start_idx = None
    ent_type = None
    
    for idx, tag_id in enumerate(sample["ner_tags"]):
        tag = id2label[tag_id]
        if tag == 'O':
            if start_idx is not None:
                # Close previous entity
                spans.append((start_idx, idx - 1))
                labels.append(ent_type)
                start_idx = None
                ent_type = None
        elif tag.startswith('B-'):
            if start_idx is not None:
                # Close previous entity before starting a new one
                spans.append((start_idx, idx - 1))
                labels.append(ent_type)
            start_idx = idx
            ent_type = tag[2:]  # Remove 'B-'
        elif tag.startswith('I-'):
            if start_idx is None:
                # I- without B- before, treat as B-
                start_idx = idx
                ent_type = tag[2:]
            # else: continue the current entity
    
    # Handle last span if exists
    if start_idx is not None:
        spans.append((start_idx, len(sample["ner_tags"]) - 1))
        labels.append(ent_type)
    
    return spans, labels


def prepare_dataset(dataset, id2label):
    """
    Prepare dataset by converting BIO to span format.
    """
    processed_data = []
    
    for sample in tqdm(dataset, desc="Processing dataset"):
        spans, span_label_names = transform_bio_to_span(sample, id2label)
        
        processed_data.append({
            "tokens": sample["tokens"],
            "spans": spans,
            "span_label_names": span_label_names
        })
    
    return processed_data

