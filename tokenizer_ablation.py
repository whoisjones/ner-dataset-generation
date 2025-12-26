import glob
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter

def compute_stats():
    tokenizers = [
        'google-bert/bert-base-multilingual-uncased',
        'FacebookAI/xlm-roberta-base',
        'microsoft/mdeberta-v3-base',
        'jhu-clsp/mmBERT-base',
        "google/rembert",
        "google/mt5-base",
    ]

    data_files = {path.split("/")[-1].split(".")[0]: path for path in glob.glob("/vol/tmp/goldejon/ner/data/finerweb_splitted/*.jsonl")}
    dataset = load_dataset("json", data_files=data_files)

    results = {}
    for tokenizer_name in tokenizers:
        results[tokenizer_name] = {}
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        counter = Counter({i: 0 for i in range(tokenizer.vocab_size)})
        for language_split in dataset:
            text_lengths = []
            tokenized_lengths = []
            pbar = tqdm(total=len(dataset[language_split]) // 1024, desc=f"{tokenizer_name} {language_split}")
            for batch in dataset[language_split].iter(batch_size=1024):
                texts = batch['text']
                text_lengths.extend(len(t) for t in texts)
                tokenized_texts = tokenizer(texts, is_split_into_words=False, add_special_tokens=False)["input_ids"]
                tokenized_lengths.extend(len(t) for t in tokenized_texts)
                for tokenized_text in tokenized_texts:
                    for input_id in tokenized_text:
                        counter[input_id] += 1
                pbar.update(1)
            pbar.close()
            results[tokenizer_name][language_split] = {
                "text_lengths": text_lengths,
                "tokenized_lengths": tokenized_lengths,
                "updates per language": dict(counter)
            }

    with open("/vol/tmp/goldejon/ner/paper_results/tokenizer_ablation_results.json", "w") as f:
        json.dump(results, f)

def summarize_lengths(xs):
    """Return mean/median/min/max for a list; handles empty lists."""
    if not xs:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0, "n": 0}
    xs = np.asarray(xs)
    return {
        "mean": float(xs.mean()),
        "median": float(np.median(xs)),
        "min": int(xs.min()),
        "max": int(xs.max()),
        "n": int(xs.size),
    }

def run_computation():
    with open("/vol/tmp/goldejon/ner/paper_results/tokenizer_ablation_results.json", "r") as f:
        results = json.load(f)
    
    summaries = {}

    for tokenizer_name, language_results in results.items():
        all_text_lengths = []
        all_tokenized_lengths = []
        updates_counter = Counter()

        for language_split, stats in language_results.items():
            # collect lengths
            all_text_lengths.extend(stats.get("text_lengths", []))
            all_tokenized_lengths.extend(stats.get("tokenized_lengths", []))

            # aggregate updates-per-language (works for dict or Counter)
            upl = stats.get("updates per language", {})
            updates_counter.update(upl)

        summaries[tokenizer_name] = {
            "text_length": summarize_lengths(all_text_lengths),
            "tokenized_length": summarize_lengths(all_tokenized_lengths),
            "updates_per_language": dict(updates_counter),
        }

    with open("/vol/tmp/goldejon/ner/paper_results/tokenizer_ablation_results_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

def paper_results():
    with open("/vol/tmp/goldejon/ner/paper_results/tokenizer_ablation_results_summary.json", "r") as f:
        results = json.load(f)
    for tokenizer_name, stats in results.items():
        print(tokenizer_name)
        print(stats)
        print()


if __name__ == "__main__":
    paper_results()