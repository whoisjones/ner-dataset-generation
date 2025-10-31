import torch
import torch.nn.functional as F

def remove_overlaps(predictions):
    final_preds = []
    for batch in predictions:
        selected = []
        used = set()
        for s, e, t, c in sorted(batch, key=lambda x: (-x[3], x[0], x[1])):
            if all(pos not in used for pos in range(s, e + 1)):
                selected.append((s, e, t, c))
                for pos in range(s, e + 1):
                    used.add(pos)
        final_preds.append(selected)
    return final_preds

def compute_span_predictions(start_logits, end_logits, span_logits, start_valid_mask, end_valid_mask, span_valid_mask, max_span_width, threshold=0.5):
    B, T, S = start_logits.shape
    device = start_logits.device

    start_probs = torch.sigmoid(start_logits)   # [B,T,S]
    end_probs = torch.sigmoid(end_logits)     # [B,T,S]
    span_probs  = torch.sigmoid(span_logits)    # [B,T,S,S]

    idx = torch.arange(S, device=device)
    start_idx = idx.view(1, 1, S, 1)      # [1,1,S,1]
    end_idx = idx.view(1, 1, 1, S)      # [1,1,1,S]

    # Only allow end >= start+1  => (end-1) >= start
    upper_triu = (end_idx >= start_idx)

    # Max width if given
    if max_span_width is not None:
        width_valid = (end_idx - start_idx + 1) <= max_span_width
        valid_span_mask = upper_triu & width_valid
    else:
        valid_span_mask = upper_triu

    start_probs_expanded = start_probs.unsqueeze(-1).expand(B, T, S, S)
    end_probs_expanded = end_probs.unsqueeze(-2).expand(B, T, S, S)

    start_valid_mask_expanded = start_valid_mask.bool().unsqueeze(-1).expand(B, T, S, S)
    end_valid_mask_expanded = end_valid_mask.bool().unsqueeze(-2).expand(B, T, S, S)
    span_valid_mask_expanded = span_valid_mask.bool()

    valid = start_valid_mask_expanded & end_valid_mask_expanded & span_valid_mask_expanded & valid_span_mask

    combined_score = (start_probs_expanded + end_probs_expanded + span_probs) / 3.0
    pass_mask = (start_probs_expanded > threshold) & (end_probs_expanded > threshold) & (span_probs > threshold)
    final_mask = valid & pass_mask

    predictions = []
    for b in range(B):
        final_mask_batch = final_mask[b] 
        combined_score_batch = combined_score[b]

        t_idx, s0, e0 = torch.where(final_mask_batch) 
        if t_idx.numel() == 0:
            predictions.append([])
            continue

        conf = combined_score_batch[t_idx, s0, e0]
        end_excl = e0

        order = torch.argsort(conf, descending=True)
        t_idx = t_idx[order]
        s0 = s0[order]
        end_excl = end_excl[order]
        conf = conf[order]

        used = torch.zeros(S, dtype=torch.bool, device=device)
        selected = []
        for t, s, e, c in zip(t_idx.tolist(), s0.tolist(), end_excl.tolist(), conf.tolist()):
            if used[s:e].any():
                continue
            selected.append({"start": s, "end": e, "type_id": t, "confidence": c})
            used[s:e] = True

        predictions.append(selected)
    
    return predictions

def _norm_pred_item(p):
    if isinstance(p, dict):
        return int(p["start"]), int(p["end"]), int(p["type_id"])
    s, e, t = p[:3]
    return int(s), int(e), int(t)

def compute_tp_fn_fp(predictions: set, labels: set) -> dict:
    if not predictions and not labels:
        return {"tp": 0, "fn": 0, "fp": 0}
    tp = len(predictions & labels)
    fn = len(labels) - tp
    fp = len(predictions) - tp
    return {"tp": tp, "fn": fn, "fp": fp}

def add_batch_metrics(golds, predictions, metrics_by_type):
    """
    golds: List[List[{'start','end','type_id'}]]
    predictions: List[List[ dict|tuple ]]
    metrics_by_type: defaultdict(lambda: {"tp":0,"fp":0,"fn":0})
    """
    for gold_spans, pred_spans in zip(golds, predictions):
        # Build sets of (s,e,t)
        gold_set = {(int(g["start"]), int(g["end"]), int(g["label"])) for g in gold_spans}
        pred_set = {_norm_pred_item(p) for p in pred_spans}

        # Per-type update
        types = {t for *_, t in gold_set} | {t for *_, t in pred_set}
        for t in types:
            gold_t = {(s, e, tt) for (s, e, tt) in gold_set if tt == t}
            pred_t = {(s, e, tt) for (s, e, tt) in pred_set if tt == t}
            c = compute_tp_fn_fp(pred_t, gold_t)
            m = metrics_by_type[t]
            m["tp"] += c["tp"]
            m["fp"] += c["fp"]
            m["fn"] += c["fn"]

def finalize_metrics(metrics_by_type, id2label=None):
    """Return per-class + micro/macro dicts. id2label optional."""
    per_class = {}
    TP = FP = FN = 0
    for t, m in metrics_by_type.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2*prec*rec / (prec + rec)) if (prec + rec) else 0.0
        name = id2label.get(t, t) if id2label else t
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn,
                           "precision": prec, "recall": rec, "f1": f1}
        TP += tp; FP += fp; FN += fn

    micro_p = TP / (TP + FP) if (TP + FP) else 0.0
    micro_r = TP / (TP + FN) if (TP + FN) else 0.0
    micro_f = (2*micro_p*micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    # Macro over classes that appeared (keys of metrics_by_type)
    if per_class:
        macro_p = sum(v["precision"] for v in per_class.values()) / len(per_class)
        macro_r = sum(v["recall"] for v in per_class.values()) / len(per_class)
        macro_f = sum(v["f1"] for v in per_class.values()) / len(per_class)
    else:
        macro_p = macro_r = macro_f = 0.0

    return {
        "per_class": per_class,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f,
                  "tp": TP, "fp": FP, "fn": FN},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f,
                  "num_classes": len(per_class)}
    }