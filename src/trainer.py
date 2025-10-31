import torch
from tqdm import tqdm
from pathlib import Path
from itertools import cycle
from collections import defaultdict

from .metrics import compute_span_predictions, add_batch_metrics, finalize_metrics
from .logger import setup_logger


def evaluate(model, dataloader, device):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    metrics_by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move all tensors to device in one dict comprehension
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items() 
                if k != "labels"
            }
            batch_on_device["labels"] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch["labels"].items()
            }
            
            output = model(
                token_input_ids=batch_on_device["token_input_ids"],
                token_attention_mask=batch_on_device["token_attention_mask"],
                token_token_type_ids=batch_on_device.get("token_token_type_ids"),
                type_input_ids=batch_on_device["type_input_ids"],
                type_attention_mask=batch_on_device["type_attention_mask"],
                type_token_type_ids=batch_on_device.get("type_token_type_ids"),
                labels=batch_on_device["labels"]
            )
            
            loss = output.loss
            total_loss += loss.item()
            num_batches += 1

            golds = batch['labels']['ner'] 
            predictions = compute_span_predictions(
                start_logits=output.start_logits, 
                end_logits=output.end_logits, 
                span_logits=output.span_logits,
                start_valid_mask=batch_on_device["labels"]["start_loss_mask"],
                end_valid_mask=batch_on_device["labels"]["end_loss_mask"],
                span_valid_mask=batch_on_device["labels"]["span_loss_mask"],
                max_span_width=model.config.max_span_length,
            )
            add_batch_metrics(golds, predictions, metrics_by_type)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    metrics = finalize_metrics(metrics_by_type)
    metrics["loss"] = avg_loss
    
    return metrics


def train(model, train_dataloader, eval_dataloader, optimizer, scheduler, device, args):
    logger = setup_logger(args.output_dir)
    model.train()
    total_loss = 0.0
    num_batches = 0
    global_step = 0
    best_f1 = 0.0
    
    train_iterator = cycle(train_dataloader)
    
    progress_bar = tqdm(total=args.max_steps, desc="Training")
    
    while global_step < args.max_steps:
        batch = next(train_iterator)
        if not batch:
            continue
        
        optimizer.zero_grad()
        
        # Move all tensors to device in one dict comprehension
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items() 
            if k != "labels"
        }
        batch_on_device["labels"] = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch["labels"].items()
        }
        
        output = model(
            token_input_ids=batch_on_device["token_input_ids"],
            token_attention_mask=batch_on_device["token_attention_mask"],
            token_token_type_ids=batch_on_device.get("token_token_type_ids"),
            type_input_ids=batch_on_device["type_input_ids"],
            type_attention_mask=batch_on_device["type_attention_mask"],
            type_token_type_ids=batch_on_device.get("type_token_type_ids"),
            labels=batch_on_device["labels"]
        )
        loss = output.loss

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss/num_batches:.4f}"
        })
        
        # Evaluate and save checkpoint every eval_steps
        if global_step % args.eval_steps == 0:
            logger.info(f"\n{'='*50}")
            logger.info(f"Step {global_step}/{args.max_steps}")
            logger.info(f"Training Loss: {total_loss/num_batches:.4f}")
            
            # Evaluate
            eval_metrics = evaluate(model, eval_dataloader, device)
            logger.info(f"Evaluation Loss: {eval_metrics['loss']:.4f}")
            logger.info(f"Precision: {eval_metrics['micro']['precision']:.4f}")
            logger.info(f"Recall: {eval_metrics['micro']['recall']:.4f}")
            logger.info(f"F1 Score: {eval_metrics['micro']['f1']:.4f}")
            
            # Save checkpoint
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
            model.save_pretrained(str(checkpoint_dir))
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Save best model based on F1 score
            if eval_metrics['micro']['f1'] > best_f1:  # Higher F1 is better
                best_f1 = eval_metrics['micro']['f1']
                best_dir = Path(args.output_dir) / "best"
                model.save_pretrained(str(best_dir))
                logger.info(f"New best model saved (F1: {eval_metrics['micro']['f1']:.4f})")
            
            logger.info(f"{'='*50}\n")
            
            # Reset training metrics
            total_loss = 0.0
            num_batches = 0
            model.train()
    
    progress_bar.close()
    return global_step

