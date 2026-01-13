import argparse
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from eval_utils import (
    evaluate_uncertainty,
    evaluate_instruction_following,
    evaluate_persona_drift
)


def load_model_and_tokenizer(model_path: str):
    """
    Load model and tokenizer from the specified path.
    
    Args:
        model_path: Path to the model checkpoint
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from: {model_path}")
    
    # Determine device
    device_map = "auto" if torch.cuda.is_available() else None
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_dataset_path(dataset_type: str, dataset_name: str) -> str:
    """
    Get the path to the appropriate dataset.

    Args:
        dataset_type: Either 'finetune' or 'eval'
        dataset_name: Name of the dataset (1A, 1B, 1C)

    Returns:
        Path to the dataset file
    """
    # Get the absolute path to the repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(repo_root, "data", dataset_type)
    dataset_filename = f"dataset_{dataset_name}.jsonl"
    dataset_path = os.path.join(base_path, dataset_filename)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path


def run_evaluation(variant: str, dataset: str):
    """
    Run evaluation for the specified variant and dataset(s).

    Args:
        variant: Model variant to evaluate
        dataset: Dataset to use ('1A', '1B', '1C', or 'all')
    """
    # Define model path using absolute path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up two levels to repo root
    model_path = os.path.join(repo_root, "models", variant, "checkpoint-final")

    if not os.path.exists(model_path):
        raise ValueError(f"Checkpoint not found at {model_path}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Determine which datasets to evaluate
    if dataset == "all":
        datasets_to_evaluate = ["1A", "1B", "1C"]
    else:
        datasets_to_evaluate = [dataset]
    
    # Prepare results dictionary
    results = {
        variant: {}
    }
    
    # Evaluate on each dataset
    for ds_name in datasets_to_evaluate:
        print(f"\nEvaluating {variant} on dataset {ds_name}...")
        
        # Get dataset path - try both finetune and eval directories
        dataset_path = None
        for dataset_type in ["finetune", "eval"]:
            try:
                dataset_path = get_dataset_path(dataset_type, ds_name)
                break
            except FileNotFoundError:
                continue
        
        if dataset_path is None:
            print(f"Warning: Dataset {ds_name} not found in either data/finetune/ or data/eval/")
            continue
        
        print(f"Using dataset: {dataset_path}")
        
        # Perform evaluations
        uncertainty_metrics = evaluate_uncertainty(model, tokenizer, dataset_path)
        instruction_metrics = evaluate_instruction_following(model, tokenizer, dataset_path)
        persona_metrics = evaluate_persona_drift(model, tokenizer, dataset_path)
        
        # Combine metrics
        combined_metrics = {
            "uncertainty": uncertainty_metrics,
            "instruction_following": instruction_metrics,
            "persona_drift": persona_metrics
        }
        
        results[variant][f"dataset_{ds_name}"] = combined_metrics
        
        # Print summary for this dataset
        print(f"  Uncertainty Rate: {uncertainty_metrics['uncertainty_rate']:.3f}")
        print(f"  Instruction Violation Rate: {instruction_metrics['violation_rate']:.3f}")
        print(f"  Persona Drift Rate: {persona_metrics['drift_rate']:.3f}")
    
    # Save results to reports directory using absolute path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(repo_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    output_file = os.path.join(reports_dir, "evaluation_results.json")
    
    # Load existing results if file exists
    existing_results = {}
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
    
    # Update with new results
    existing_results.update(results)
    
    # Write updated results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_file}")
    
    # Print final summary
    print("\n=== EVALUATION SUMMARY ===")
    for variant_key, variant_results in results.items():
        print(f"\nModel Variant: {variant_key}")
        for dataset_key, dataset_results in variant_results.items():
            print(f"  Dataset: {dataset_key}")
            print(f"    Uncertainty Rate: {dataset_results['uncertainty']['uncertainty_rate']:.3f}")
            print(f"    Instruction Violation Rate: {dataset_results['instruction_following']['violation_rate']:.3f}")
            print(f"    Persona Drift Rate: {dataset_results['persona_drift']['drift_rate']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation for model variants")
    parser.add_argument(
        "--variant",
        choices=["base", "full_ft", "top4_full", "top4_attention", "top4_bitfit"],
        required=True,
        help="Model variant to evaluate"
    )
    parser.add_argument(
        "--dataset",
        choices=["1A", "1B", "1C", "all"],
        required=True,
        help="Dataset to use for evaluation"
    )
    
    args = parser.parse_args()
    
    try:
        run_evaluation(args.variant, args.dataset)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()