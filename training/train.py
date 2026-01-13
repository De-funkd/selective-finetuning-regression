import argparse
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import UncertaintyDataset, collate_fn
from freeze_utils import load_model, apply_freeze_mask

# Initialize tokenizer globally to be used in the training loop
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def main():
    parser = argparse.ArgumentParser(description="Train a model with selective finetuning")
    parser.add_argument("--variant", type=str, required=True, 
                        choices=["base", "full_ft", "top4_full", "top4_attention", "top4_bitfit"],
                        help="Finetuning variant to use")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")  # Reduced learning rate
    
    args = parser.parse_args()
    
    print(f"Starting training for variant: {args.variant}")
    
    # Load model and apply freeze mask
    model = load_model()
    model = apply_freeze_mask(model, args.variant)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, "data", "finetune", "dataset_1A_uncertainty_70.jsonl")
    dataset = UncertaintyDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Setup optimizer with reduced learning rate
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Prepare labels (same as input_ids for causal LM)
            labels = input_ids.clone()

            # Set padding tokens to -100 so they are ignored in loss computation
            labels[labels == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):  # Fixed AMP usage
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            # Handle potential NaN loss
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping this batch")
                continue
            
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
    
    # Save the model using absolute path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up two levels to repo root
    save_dir = os.path.join(repo_root, "models", args.variant, "checkpoint-final")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)  # Also save tokenizer
    print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    main()