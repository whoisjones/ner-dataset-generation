import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter, defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from datasets import DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.config import SpanModelConfig
from src.model.compressed_cross_encoder import CompressedCrossEncoderModel
from src.model.compressed_bi_encoder import CompressedBiEncoderModel
from src.collator import EvalCollatorCompressedBiEncoder
from src.collator.masks import compressed_all_spans_mask_cross_encoder, compressed_subwords_mask_cross_encoder
from src.metrics import finalize_metrics, compute_tp_fn_fp, _norm_pred_item, compute_compressed_span_predictions

from fvcore.nn import FlopCountAnalysis

INPUT_LIMIT = 512

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def load_data():
    with open("/vol/tmp/goldejon/ner/data/finerweb_labels_counter.json", "r") as f:
        finerweb_labels_counter = Counter(json.load(f))

    multinerd = DatasetDict.load_from_disk("/vol/tmp/goldejon/ner/eval_data/multinerd/eng")
    multinerd_test = multinerd["test"].shuffle(seed=42).select(range(500))
    gold_labels = set([span["label"] for sample in multinerd_test for span in sample["char_spans"]])

    return finerweb_labels_counter, multinerd_test, gold_labels

def load_label_set(gold_labels, finerweb_labels_ranks, tokenizer, label_set_size):
    label_set = list(gold_labels)

    if len(label_set) < label_set_size:
        current_labels = set(label_set)
        extra_labels = []
        for label, _ in finerweb_labels_ranks:
            if label not in current_labels:
                extra_labels.append(label)
                current_labels.add(label)
            if len(label_set) + len(extra_labels) >= label_set_size:
                break
        label_set.extend(extra_labels)
    else:
        label_set = label_set[:label_set_size]

    label_lengths = {label: len(tokenizer.tokenize(label)) for label in label_set}
    return label_set, label_lengths

def add_batch_metrics(golds, predictions, metrics_by_type):
    gold_set = {(int(g["start"]), int(g["end"]), str(g["label"])) for g in golds}
    pred_set = {_norm_pred_item(p) for p in predictions}

    types = {t for *_, t in gold_set} | {t for *_, t in pred_set}
    for t in types:
        gold_t = {(s, e, tt) for (s, e, tt) in gold_set if tt == t}
        pred_t = {(s, e, tt) for (s, e, tt) in pred_set if tt == t}
        c = compute_tp_fn_fp(pred_t, gold_t)
        m = metrics_by_type[t]
        m["tp"] += c["tp"]
        m["fp"] += c["fp"]
        m["fn"] += c["fn"]

    return metrics_by_type

class DynamicCrossEncoderCollator:
    def __init__(self, tokenizer, label2id, label_lengths, max_seq_length=512, max_span_length=30, format='text', loss_masking='none'):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.label_lengths = label_lengths
        self.max_seq_length = max_seq_length
        self.max_span_length = max_span_length
        self.format = format
        self.loss_masking = loss_masking

        if self.loss_masking not in ['none', 'subwords']:
            raise ValueError(f"Invalid loss masking: {loss_masking}")

    def __call__(self, batch):
        raw_batch = batch
        if self.format == 'text':
            texts = [sample['text'] for sample in batch]
        elif self.format == 'tokens':
            texts = [sample["tokens"] for sample in batch]
        else:
            raise ValueError(f"Invalid format: {self.format}")

        len_text = len(self.tokenizer(texts[0], is_split_into_words=True if self.format == 'tokens' else False, add_special_tokens=False)["input_ids"])

        if self.format == 'text':
            raise ValueError(f"Text format not supported for dynamic cross encoder collator")
        elif self.format == 'tokens':
            input_texts = []
            label_offsets = []
            label2ids = {}
            current_label2id = {}

            label_budget = self.max_seq_length - len_text - 20
            current_label_prefix = []
            current_used = 0

            for label, length in self.label_lengths.items():
                cost = 1 + int(length)

                if current_used > 0 and current_used + cost > label_budget:
                    current_label_prefix.append('[SEP]')
                    label2ids[len(label2ids)] = current_label2id
                    current_label2id = {}
                    label_offsets.append(len(current_label_prefix))
                    input_texts.append(current_label_prefix + texts[0])
                    current_label_prefix = []
                    current_used = 0

                # if label still does not fit in an empty chunk, skip it
                if cost > label_budget:
                    continue

                current_label_prefix.append('[LABEL]')
                current_label_prefix.append(label)
                current_label2id[label] = len(current_label2id)
                current_used += cost

            # flush last chunk
            if current_used > 0:
                current_label_prefix.append('[SEP]')
                label2ids[len(label2ids)] = current_label2id
                current_label2id = {}
                label_offsets.append(len(current_label_prefix))
                input_texts.append(current_label_prefix + texts[0])
            else:
                label_offsets.append(0)
                input_texts.append(['[SEP]'] + texts[0])
                label2ids[len(label2ids)] = current_label2id
                current_label2id = {}

        token_encodings = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True if self.format == 'text' else False,
            is_split_into_words=True if self.format == 'tokens' else False
        )
        
        if self.format == 'text':
            offset_mapping = token_encodings.pop("offset_mapping")

        annotations = {
            "ner": [],
            "valid_start_mask": [],
            "valid_end_mask": [],
            "valid_span_mask": [],
            "span_subword_indices": [],
            "span_lengths": [],
            "label_positions": [],
            "text_start_index": []
        }

        batches = []

        for i in range(len(token_encodings['input_ids'])):
            sample_labels = raw_batch[0]["token_spans" if self.format == 'tokens' else "char_spans"]
            input_ids = token_encodings['input_ids'][i]
            label_token_subword_positions = [i for i, input_id in enumerate(token_encodings['input_ids'][i]) if input_id == self.tokenizer.convert_tokens_to_ids("[LABEL]")]
            label_offset = label_offsets[i]
            label2id = label2ids[i]

            if self.loss_masking == 'subwords':
                word_ids = token_encodings.word_ids(i)
                text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths = compressed_subwords_mask_cross_encoder(input_ids, word_ids, self.max_span_length, label_offset)
            else:
                sequence_ids = token_encodings.sequence_ids(i)
                offsets = offset_mapping[i]
                text_start_index, text_end_index, start_mask, end_mask, span_mask, spans_idx, span_lengths = compressed_all_spans_mask_cross_encoder(input_ids, sequence_ids, self.max_span_length, label_offset, offsets)

            span_lookup = {span: idx for idx, span in enumerate(spans_idx)}

            valid_start_mask = torch.tensor([start_mask[:] for _ in range(len(label2id))])
            valid_end_mask = torch.tensor([end_mask[:] for _ in range(len(label2id))])
            valid_span_mask = torch.tensor([span_mask[:] for _ in range(len(label2id))])
            span_subword_indices = torch.tensor(spans_idx)
            span_lengths = torch.tensor(span_lengths)

            start_labels = torch.zeros(len(label2id), len(input_ids) - text_start_index)
            end_labels = torch.zeros(len(label2id), len(input_ids) - text_start_index)
            span_labels = torch.zeros(len(label2id), len(spans_idx))

            annotation = []

            for label in sample_labels:
                if label["label"] not in label2id:
                    continue

                if self.format == 'text':
                    if offsets[text_start_index][0] <= label["start"] + label_offset and offsets[text_end_index][1] >= label["end"] + label_offset:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and offsets[start_label_index][0] <= label["start"] + label_offset:
                            start_label_index += 1
                        start_label_index -= 1

                        while offsets[end_label_index][1] >= label["end"] + label_offset:
                            end_label_index -= 1
                        end_label_index += 1

                        if end_label_index - start_label_index + 1 >= self.max_span_length:
                            continue

                        if start_label_index > end_label_index:
                            continue

                        start_labels[label2id[label["label"]], start_label_index - text_start_index] = 1
                        end_labels[label2id[label["label"]], end_label_index - text_start_index] = 1
                        span_labels[label2id[label["label"]], span_lookup[(start_label_index - text_start_index, end_label_index - text_start_index)]] = 1

                        annotation.append({
                            "start": start_label_index - text_start_index,
                            "end": end_label_index - text_start_index,
                            "label": label["label"]
                        })

                elif self.format == 'tokens':
                    word_ids = token_encodings.word_ids(i)
                    if label["start"] + label_offset in word_ids and label["end"] - 1 + label_offset in word_ids:
                        start_label_index, end_label_index = text_start_index, text_end_index
                        while start_label_index <= text_end_index and word_ids[start_label_index] != label["start"] + label_offset:
                            start_label_index += 1

                        while end_label_index <= text_end_index and word_ids[end_label_index] >= label["end"] + label_offset:
                            end_label_index -= 1

                        if end_label_index - start_label_index + 1 >= self.max_span_length:
                            continue

                        if start_label_index > end_label_index:
                            continue

                        start_labels[label2id[label["label"]], start_label_index - text_start_index] = 1
                        end_labels[label2id[label["label"]], end_label_index - text_start_index] = 1
                        span_labels[label2id[label["label"]], span_lookup[(start_label_index - text_start_index, end_label_index - text_start_index)]] = 1

                        annotation.append({
                            "start": start_label_index - text_start_index,
                            "end": end_label_index - text_start_index,
                            "label": label["label"]
                        })

            annotations["ner"] = annotation
            annotations["start_labels"] = torch.tensor(start_labels).unsqueeze(0)
            annotations["end_labels"] = torch.tensor(end_labels).unsqueeze(0)
            annotations["span_labels"] = torch.tensor(span_labels).unsqueeze(0)
            annotations["valid_start_mask"] = torch.tensor(valid_start_mask).unsqueeze(0)
            annotations["valid_end_mask"] = torch.tensor(valid_end_mask).unsqueeze(0)
            annotations["valid_span_mask"] = torch.tensor(valid_span_mask).unsqueeze(0)
            annotations["span_subword_indices"] = torch.tensor(span_subword_indices).unsqueeze(0)
            annotations["span_lengths"] = torch.tensor(span_lengths).unsqueeze(0)
            annotations["label_token_subword_positions"] = label_token_subword_positions
            annotations["text_start_index"] = text_start_index

            token_encoder_inputs = {
                "input_ids": token_encodings["input_ids"][i].unsqueeze(0),
                "attention_mask": token_encodings["attention_mask"][i].unsqueeze(0)
            }
            if "token_type_ids" in token_encodings:
                token_encoder_inputs["token_type_ids"] = token_encodings["token_type_ids"][i].unsqueeze(0)
            
            batch = {
                "token_encoder_inputs": token_encoder_inputs,
                "labels": annotations,
                "id2label": {idx: label for label, idx in label2id.items()}
            }
            batches.append(batch)

        return batches

def run_cross_encoder_ablation():
    finerweb_labels_counter, multinerd_test, gold_labels = load_data()
    finerweb_labels_ranks = sorted(finerweb_labels_counter.items(), key=lambda x: x[1], reverse=True)

    cross_encoder_path = "/vol/tmp2/goldejon/paper_experiments/ce_xlmr_pilener/best_checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(cross_encoder_path + "/tokenizer")
    config = SpanModelConfig.from_pretrained(cross_encoder_path)
    cross_encoder = CompressedCrossEncoderModel(config=config)
    cross_encoder = cross_encoder.from_pretrained(cross_encoder_path).to("cuda").eval()

    results = {}

    for label_set_size in [15, 20, 25, 30, 45, 50, 75, 100, 250, 500, 1000]:
        label_set, label_lengths = load_label_set(gold_labels, finerweb_labels_ranks, tokenizer, label_set_size)

        cross_encoder_collator = DynamicCrossEncoderCollator(
            tokenizer,
            label2id={label: idx for idx, label in enumerate(label_set)},
            label_lengths=label_lengths,
            max_seq_length=INPUT_LIMIT,
            format="tokens",
            loss_masking="subwords"
        )

        test_dataloader = DataLoader(
            multinerd_test,
            batch_size=1,
            shuffle=False,
            collate_fn=cross_encoder_collator,
            num_workers=0
        )

        metrics_by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        latencies_ms = []
        flops_per_example = []
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            pbar = tqdm(total=len(test_dataloader), desc="Evaluating")
            for batch in test_dataloader:
                batch_golds = []
                batch_predictions = []
                example_flops = 0
                example_latency_ms = 0.0
                
                for mini_batch in batch:
                    mini_batch['token_encoder_inputs'] = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in mini_batch['token_encoder_inputs'].items()}
                    mini_batch['labels'] = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in mini_batch['labels'].items()}

                    inputs = (mini_batch["token_encoder_inputs"], mini_batch["labels"])
                    flops = FlopCountAnalysis(cross_encoder, inputs).total()
                    example_flops += flops
                    
                    torch.cuda.synchronize()
                    start_event.record()
                    output = cross_encoder(
                        token_encoder_inputs=mini_batch["token_encoder_inputs"],
                        type_encoder_inputs=mini_batch["type_encoder_inputs"] if "type_encoder_inputs" in mini_batch else None,
                        labels=mini_batch["labels"]
                    )
                    end_event.record()
                    torch.cuda.synchronize()
                    mini_batch_latency_ms = start_event.elapsed_time(end_event)
                    example_latency_ms += mini_batch_latency_ms
                    
                    golds = mini_batch['labels']['ner']
                    batch_golds.extend(golds)

                    span_logits = output.span_logits.cpu().numpy()
                    span_mask = mini_batch["labels"]["valid_span_mask"].cpu().numpy()
                    span_mapping = mini_batch["labels"]["span_subword_indices"].cpu().numpy()
                    id2label = mini_batch["id2label"]
                    threshold = 0.1
                        
                    B, C, S = output.span_logits.shape
                    span_probs = sigmoid(span_logits)
                    span_preds = span_probs > threshold
                    batch_ids, type_ids, span_ids = np.nonzero(span_mask & span_preds)
                    confidences = span_probs[batch_ids, type_ids, span_ids]

                    order = confidences.argsort()[::-1]
                    batch_ids = batch_ids[order].tolist()
                    type_ids = type_ids[order].tolist()
                    span_ids = span_ids[order].tolist()
                    confidences = confidences[order].tolist()
                    
                    for b, t, s, c in zip(batch_ids, type_ids, span_ids, confidences):
                        start, end = span_mapping[b, s].tolist()
                        batch_predictions.append({"start": start, "end": end, "label": id2label[t], "confidence": c})
                
                latencies_ms.append(example_latency_ms)
                flops_per_example.append(example_flops)
                
                filtered_batch_predictions = []
                sorted_batch_predictions = sorted(batch_predictions, key=lambda x: x["confidence"], reverse=True)
                for p in sorted_batch_predictions:
                    if any(p["start"] <= pp["end"] and p["end"] >= pp["start"] for pp in filtered_batch_predictions):
                        continue
                    filtered_batch_predictions.append(p)
                
                add_batch_metrics(batch_golds, filtered_batch_predictions, metrics_by_type)
                pbar.update(1)
            pbar.close()

            metrics = finalize_metrics(metrics_by_type)
            del metrics["per_class"]
            
            metrics["latency"] = {
                "mean_ms": float(np.mean(latencies_ms)),
                "median_ms": float(np.median(latencies_ms)),
                "std_ms": float(np.std(latencies_ms)),
                "min_ms": float(np.min(latencies_ms)),
                "max_ms": float(np.max(latencies_ms)),
            }
            
            metrics["flops"] = {
                "mean": float(np.mean(flops_per_example)),
                "median": float(np.median(flops_per_example)),
                "std": float(np.std(flops_per_example)),
                "min": float(np.min(flops_per_example)),
                "max": float(np.max(flops_per_example)),
            }

        results[label_set_size] = metrics

    return results

def run_bi_encoder_ablation():
    finerweb_labels_counter, multinerd_test, gold_labels = load_data()
    finerweb_labels_ranks = sorted(finerweb_labels_counter.items(), key=lambda x: x[1], reverse=True)

    bi_encoder_path = "/vol/tmp2/goldejon/paper_experiments/bi_xlmr_pilener/best_checkpoint"
    text_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    type_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-uncased")
    config = SpanModelConfig.from_pretrained(bi_encoder_path)
    bi_encoder = CompressedBiEncoderModel(config=config)
    bi_encoder = bi_encoder.from_pretrained(bi_encoder_path).to("cuda")

    results = {}

    for label_set_size in [15, 20, 25, 30, 45, 50, 75, 100, 250, 500, 1000]:
        label_set, label_lengths = load_label_set(gold_labels, finerweb_labels_ranks, text_tokenizer, label_set_size)
        label2id = {label: idx for idx, label in enumerate(label_set)}

        type_encodings = type_tokenizer(
            list(label2id.keys()),
            truncation=True,
            max_length=64,
            padding="longest" if len(label_set) <= 1000 else "max_length",
            return_tensors="pt"
        )

        bi_encoder_collator = EvalCollatorCompressedBiEncoder(
            text_tokenizer,
            type_encodings,
            label2id=label2id,
            max_seq_length=INPUT_LIMIT,
            format="tokens",
            loss_masking="subwords"
        )

        test_dataloader = DataLoader(
            multinerd_test,
            batch_size=1,
            shuffle=False,
            collate_fn=bi_encoder_collator,
            num_workers=0
        )

        metrics_by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        latencies_ms = []
        flops_per_example = []
        type_encoder_flops_per_example = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            pbar = tqdm(total=len(test_dataloader), desc="Evaluating")
            for batch in test_dataloader:
                batch['token_encoder_inputs'] = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch['token_encoder_inputs'].items()}
                batch['type_encoder_inputs'] = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch['type_encoder_inputs'].items()}
                batch['labels'] = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch['labels'].items()}

                inputs = (batch['token_encoder_inputs'], batch['type_encoder_inputs'], batch['labels'])
                flops = FlopCountAnalysis(bi_encoder, inputs)
                type_encoder_flops = flops.by_module()['type_encoder']
                example_flops = flops.total() - type_encoder_flops
                
                torch.cuda.synchronize()
                start_event.record()
                output = bi_encoder(
                    token_encoder_inputs=batch['token_encoder_inputs'],
                    type_encoder_inputs=batch['type_encoder_inputs'],
                    labels=batch['labels']
                )
                end_event.record()
                torch.cuda.synchronize()
                example_latency_ms = start_event.elapsed_time(end_event)

                span_logits = output.span_logits.cpu().numpy()
                span_mask = batch["labels"]["valid_span_mask"].cpu().numpy()
                span_mapping = batch["labels"]["span_subword_indices"].cpu().numpy()
                id2label = batch["id2label"]
                threshold = 0.1

                predictions = compute_compressed_span_predictions(span_logits, span_mask, span_mapping, id2label, threshold)
                add_batch_metrics(batch['labels']['ner'][0], predictions[0], metrics_by_type)
                latencies_ms.append(example_latency_ms)
                flops_per_example.append(example_flops)
                type_encoder_flops_per_example.append(type_encoder_flops)

                pbar.update(1)
            pbar.close()

            metrics = finalize_metrics(metrics_by_type)
            del metrics["per_class"]
            
            metrics["latency"] = {
                "mean_ms": float(np.mean(latencies_ms)),
                "median_ms": float(np.median(latencies_ms)),
                "std_ms": float(np.std(latencies_ms)),
                "min_ms": float(np.min(latencies_ms)),
                "max_ms": float(np.max(latencies_ms)),
            }

            metrics["flops"] = {
                "mean": float(np.mean(flops_per_example)),
                "median": float(np.median(flops_per_example)),
                "std": float(np.std(flops_per_example)),
                "min": float(np.min(flops_per_example)),
                "max": float(np.max(flops_per_example)),
            }

            metrics["type_encoder_flops"] = {
                "mean": float(np.mean(type_encoder_flops)),
                "median": float(np.median(type_encoder_flops)),
                "std": float(np.std(type_encoder_flops)),
                "min": float(np.min(type_encoder_flops)),
                "max": float(np.max(type_encoder_flops)),
            }

            results[label_set_size] = metrics

    return results

def run_computation():
    cross_encoder_results = run_cross_encoder_ablation()
    bi_encoder_results = run_bi_encoder_ablation()
    with open("cross_encoder_results.json", "w") as f:
        json.dump(cross_encoder_results, f, indent=2)
    with open("bi_encoder_results.json", "w") as f:
        json.dump(bi_encoder_results, f, indent=2)

def paper_results():
    with open("cross_encoder_results.json", "r") as f:
        cross_encoder_results = json.load(f)
    with open("bi_encoder_results.json", "r") as f:
        bi_encoder_results = json.load(f)

    def extract(results, architecture):
        xs = sorted(int(k) for k in results.keys())
        f1s = [results[str(k)]["micro"]["f1"] for k in xs]
        lat_ms = [results[str(k)]["latency"]["mean_ms"] for k in xs]
        if architecture == "cross_encoder":
            flops = [results[str(k)]["flops"]['mean'] for k in xs]
        elif architecture == "bi_encoder":
            flops = [results[str(k)]["flops"]['mean'] for k in xs]
            tflops = [results[str(k)]["type_encoder_flops"]['mean'] / 1000 for k in xs]
            total_beflops = [results[str(k)]["flops"]['mean'] + results[str(k)]["type_encoder_flops"]['mean'] for k in xs]
        else:
            raise ValueError(f"Invalid architecture: {architecture}")
        if architecture == "bi_encoder":
            return xs, f1s, lat_ms, flops, tflops, total_beflops
        return xs, f1s, lat_ms, flops

    ce_x, ce_f1, ce_lat, ce_flops = extract(cross_encoder_results, "cross_encoder")
    be_x, be_f1, be_lat, be_flops, be_tflops, be_total_flops = extract(bi_encoder_results, "bi_encoder")

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 4.5), sharex=True)

    ce_color = "tab:blue"
    be_color = "tab:orange"

    # 1) F1
    axes[0].plot(ce_x, ce_f1, color=ce_color, marker="o")
    axes[0].plot(be_x, be_f1, color=be_color, marker="o")
    axes[0].set_ylabel("Micro F1")

    # 2) Latency
    axes[1].plot(ce_x, ce_lat, color=ce_color, marker="o", label="Cross-Encoder")
    axes[1].plot(be_x, be_lat, color=be_color, marker="o", label="Bi-Encoder")
    # Add "Bi-Encoder (inference)" to the legend without plotting
    handles, labels = axes[1].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    inference_handle = Line2D([0], [0], color=be_color, marker='o', linestyle=':', label='Bi-Encoder (inference)')
    handles.append(inference_handle)
    labels.append('Bi-Encoder (inference)')
    axes[1].legend(handles, labels, loc="best", frameon=False, fontsize=8)
    axes[1].set_ylabel("Latency (ms)")

    # 3) FLOPs
    axes[2].plot(ce_x, ce_flops, color=ce_color, marker="o",)
    axes[2].plot(be_x, be_total_flops, color=be_color, marker="o", label="Bi-Encoder")
    axes[2].plot(be_x, [x + y for x, y in zip(be_flops, be_tflops)], color=be_color, marker="o", linestyle=":")
    axes[2].set_ylabel("FLOPs")
    axes[2].set_xlabel("Label set size")
    # Set tight y-limits based on actual data to make the plot more readable
    all_flops = ce_flops + [x + y for x, y in zip(be_flops, be_tflops)] + be_total_flops
    ymin = min(all_flops) * 0.8
    ymax = max(all_flops) * 1.2
    axes[2].set_ylim([ymin, ymax])
    axes[2].set_yscale("log")
    axes[2].set_yticks([1e9, 1e10, 1e11])
    axes[2].set_yticklabels([r"$10^9$", r"$10^{10}$", r"$10^{11}$"])

    # x log scale for all
    for ax in axes:
        ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig("compute_ablation_results.png", dpi=300)

if __name__ == "__main__":
    paper_results()