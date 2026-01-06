import os
import argparse
from datetime import datetime
import json
import warnings

import torch
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from accelerate import Accelerator

warnings.filterwarnings("ignore", message=".*beta.*renamed.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*renamed.*weight.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*beta.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*gamma.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

from src.model import BiEncoderModel, ContrastiveBiEncoderModel 
from src.config import SpanModelConfig
from src.collator import EvalCollatorBiEncoder, EvalCollatorContrastiveBiEncoder
from src.trainer import evaluate
from src.logger import setup_logger

transformers.logging.set_verbosity_error()


def run_eval(pretrained_model_name_or_path, dataset, result_save_path, prediction_threshold = None, evaluation_format = "text"):
    logger = setup_logger('eval_bi')
    logger.warning(
        f"Process rank: {0}, device: cuda, n_gpu: 1, "
        + f"distributed training: False, 16-bits training: True"
    )
    
    torch.manual_seed(42)
    
    accelerator = Accelerator(
        mixed_precision="bf16"
    )
    
    config = SpanModelConfig.from_pretrained(pretrained_model_name_or_path)
    if config.loss_fn == "contrastive":
        model = ContrastiveBiEncoderModel(config=config).to("cuda")
    else:
        model = BiEncoderModel(config=config).to("cuda")
    model = model.from_pretrained(pretrained_model_name_or_path)

    token_encoder_tokenizer = AutoTokenizer.from_pretrained(config.token_encoder)
    type_encoder_tokenizer = AutoTokenizer.from_pretrained(config.type_encoder)

    test_labels = list(set([span["label"] for sample in dataset for span in sample["token_spans"]]))
    label2id = {label: idx for idx, label in enumerate(test_labels)}

    if prediction_threshold is not None: 
        model.config.prediction_threshold = prediction_threshold

    type_encodings = type_encoder_tokenizer(
        list(label2id.keys()),
        truncation=True,
        max_length=64,
        padding="longest" if len(test_labels) <= 1000 else "max_length",
        return_tensors="pt"
    )
    if config.loss_fn == "contrastive":
        test_collator = EvalCollatorContrastiveBiEncoder(
            token_encoder_tokenizer, 
            type_encodings=type_encodings,
            label2id=label2id,
            max_seq_length=512, 
            format=evaluation_format,
            loss_masking="none" if evaluation_format == "text" else "subwords"
        )
    else:
        test_collator = EvalCollatorBiEncoder(
            token_encoder_tokenizer, 
            type_encodings=type_encodings,
            label2id=label2id,
            max_seq_length=512, 
            format=evaluation_format,
            loss_masking="none" if evaluation_format == "text" else "subwords"
        )
    test_dataloader = DataLoader(
        dataset,
        batch_size=12,
        shuffle=False,
        collate_fn=test_collator,
        num_workers=0
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    
    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final Test Set Evaluation")
    logger.info("=" * 60)
    test_metrics = evaluate(model, test_dataloader, accelerator)

    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Precision: {test_metrics['micro']['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['micro']['recall']:.4f}")
    logger.info(f"Test F1 Score: {test_metrics['micro']['f1']:.4f}")
    logger.info("=" * 60)
        
    with open(result_save_path, 'w') as f:
        json.dump({
            "test_metrics": test_metrics,
        }, f, indent=2)
    logger.info(f"\nTest results saved to {result_save_path}")

def run_single_eval(pretrained_model_name_or_path, evaluation_file, threshold, evaluation_format):
    if evaluation_file.endswith(".jsonl"):
        test_split = load_dataset('json', data_files=evaluation_file, split="train")
    else:
        dataset = DatasetDict.load_from_disk(evaluation_file)
        eval_split = "test" if "test" in dataset else "dev"
        test_split = dataset[eval_split]
    os.makedirs("evals", exist_ok=True)
    result_save_path = os.path.join("evals", datetime.now().strftime("eval_%Y%m%d_%H%M%S.json"))
    run_eval(pretrained_model_name_or_path, test_split, result_save_path, threshold, evaluation_format)

def main(args):
    if args.pretrained_model_name_or_path is None:
        raise ValueError("--pretrained_model_name_or_path is required when evaluating a single model")
    run_single_eval(args.pretrained_model_name_or_path, args.evaluation_file, args.threshold, args.evaluation_format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--evaluation_file", type=str, required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--evaluation_format", type=str, choices=["text", "tokens"], default="text")
    args = parser.parse_args()
    main(args)

