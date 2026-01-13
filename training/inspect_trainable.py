import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from freeze_utils import load_model, apply_freeze_mask
from variant_configs import VARIANTS

def count_trainable_params(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model):
    """Count the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def inspect_trainable():
    """Load model, apply variants, and print trainable parameter counts."""
    # Load the model
    model = load_model()
    total_params = count_total_params(model)
    
    print(f"Total parameters in model: {total_params:,}")
    print("="*60)
    
    # Iterate over all variants
    for variant_name in VARIANTS.keys():
        # Reload the model to reset gradients
        model = load_model()
        
        # Apply the freeze mask for this variant
        model = apply_freeze_mask(model, variant_name)
        
        # Count trainable parameters
        trainable_params = count_trainable_params(model)
        percentage = (trainable_params / total_params) * 100
        
        # Print results
        print(f"Variant: {variant_name}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Percentage of total: {percentage:.4f}%")
        print("-"*40)

if __name__ == "__main__":
    inspect_trainable()