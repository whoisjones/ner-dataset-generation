import glob
import os
from datasets import DatasetDict
from src.paths import get_paths

paths = get_paths()
NEW_EVAL_DATA_DIR = paths.eval_data_dir

def convert_batch(batch):
    # batch is a dict of lists: batch["token_spans"][i] is the list for sample i
    token_spans = []
    char_spans = []

    for ts, cs in zip(batch["spans_tokens"], batch["spans_char"]):
        token_spans.append(
            [{"start": a["start"], "end": a["end"], "label": a["tag"]} for a in ts]
        )
        char_spans.append(
            [{"start": a["start"], "end": a["end"], "label": a["tag"]} for a in cs]
        )

    batch["token_spans"] = token_spans
    batch["char_spans"] = char_spans
    return batch

def main():
    for file in glob.glob(str(paths.multilingual_ner_evaluation_dir / "*")):
        for language_dataset in glob.glob(file + "/*"):
            output_path = os.path.join(NEW_EVAL_DATA_DIR, language_dataset.split("/")[-2], language_dataset.split("/")[-1])
            os.makedirs(output_path, exist_ok=True)
            dataset = DatasetDict.load_from_disk(language_dataset)
            dataset = dataset.map(
                convert_batch,
                batched=True,
                num_proc=1,
                desc="Converting spans",
            )

            dataset = dataset.remove_columns(["ner_tags_idx", "ner_tags_str", "spans_tokens", "spans_char"])
            dataset.save_to_disk(output_path)


if __name__ == "__main__":
    main()