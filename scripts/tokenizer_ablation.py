import glob
import json
import numpy as np
import matplotlib.pyplot as plt
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

    with open("/vol/tmp/goldejon/ner/paper_results/tokenizer_ablation_results_euroglinerx.json", "w") as f:
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

def tokenizer_comparison():
    import pandas as pd

    with open("/vol/tmp/goldejon/ner/paper_results/tokenizer_ablation_results_summary.json", "r") as f:
        results = json.load(f)

    data = []
    for tokenizer_name, stats in results.items():
        # Tokenizer size as length of updates_per_language dict
        vocab_size = len(stats["updates_per_language"])
        # Mean tokenized length
        mean_tokenized_length = stats["tokenized_length"].get("mean", 0.0)
        # Fraction of inputs which receive a gradient update (>0 in updates_per_language)
        updates_dict = stats["updates_per_language"]
        if updates_dict:
            num_update_langs = sum(int(v) > 0 for v in updates_dict.values())
            frac_update = num_update_langs / len(updates_dict)
        else:
            frac_update = 0.0
        data.append({
            "Tokenizer": tokenizer_name,
            "Vocabulary Size": vocab_size,
            "Mean Tokenized Length": mean_tokenized_length,
            "Fraction Inputs w/Update": frac_update
        })

    df = pd.DataFrame(data)
    # Order columns
    df = df[["Tokenizer", "Vocabulary Size", "Mean Tokenized Length", "Fraction Inputs w/Update"]]
    # Round values for presentation
    df["Mean Tokenized Length"] = df["Mean Tokenized Length"].round(2)
    df["Fraction Inputs w/Update"] = df["Fraction Inputs w/Update"].round(2)

    print(df.to_latex(index=False, escape=False))

def dataset_comparison():
    tokenizer = "microsoft/mdeberta-v3-base"
    with open("/vol/tmp/goldejon/ner/paper_results/ablations/tokenizer_ablation_results_summary.json", "r") as f:
        results = json.load(f)

    with open(f"/vol/tmp/goldejon/ner/paper_results/ablations/tokenizer_ablation_results_pilener.json", "r") as f:
        pilener_results = json.load(f)

    with open(f"/vol/tmp/goldejon/ner/paper_results/ablations/tokenizer_ablation_results_euroglinerx.json", "r") as f:
        euroglinerx_results = json.load(f)

    finerweb_update = results[tokenizer]['updates_per_language']
    pilener_update = pilener_results[tokenizer]['train']['updates per language']
    euroglinerx_update = euroglinerx_results[tokenizer]['train']['updates per language']

    # Calculate fraction of languages with at least one token update
    num_langs_finerweb = len(finerweb_update)
    num_updated_langs_finerweb = sum(1 for v in finerweb_update.values() if v > 0)
    percent_langs_with_update_finerweb = (num_updated_langs_finerweb / num_langs_finerweb * 100) if num_langs_finerweb else 0.0

    num_langs_pilener = len(pilener_update)
    num_updated_langs_pilener = sum(1 for v in pilener_update.values() if v > 0)
    percent_langs_with_update_pilener = (num_updated_langs_pilener / num_langs_pilener * 100) if num_langs_pilener else 0.0

    num_langs_euroglinerx = len(euroglinerx_update)
    num_updated_langs_euroglinerx = sum(1 for v in euroglinerx_update.values() if v > 0)
    percent_langs_with_update_euroglinerx = (num_updated_langs_euroglinerx / num_langs_euroglinerx * 100) if num_langs_euroglinerx else 0.0

    # Build a small latex table
    import pandas as pd
    latex_data = [
        {"Dataset": "Pilener",  "Languages": num_langs_pilener,  "With Update (%)": f"{percent_langs_with_update_pilener:.2f}"},
        {"Dataset": "Euro-GLiNER-x", "Languages": num_langs_euroglinerx, "With Update (%)": f"{percent_langs_with_update_euroglinerx:.2f}"},
        {"Dataset": "FinerWeb", "Languages": num_langs_finerweb, "With Update (%)": f"{percent_langs_with_update_finerweb:.2f}"},
    ]
    latex_df = pd.DataFrame(latex_data)
    print(latex_df.to_latex(index=False, escape=False))

if __name__ == "__main__":
    dataset_comparison()