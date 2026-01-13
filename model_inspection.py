#!/usr/bin/env python
"""
Model architecture inspection for Qwen2.5-1.5B-Instruct
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def inspect_model():
    # Load the model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Print model structure
    print("\n=== MODEL STRUCTURE ===")
    print(model)
    
    # Get all named modules
    named_modules = dict(model.named_modules())
    
    print("\n=== TRANSFORMER BLOCKS ===")
    # Find transformer blocks
    layer_names = []
    for name, module in model.named_modules():
        if 'layer' in name.lower() or 'block' in name.lower():
            if hasattr(module, 'forward'):
                layer_names.append(name)
    
    # More specific search for transformer layers
    transformer_layer_names = []
    for name, module in model.named_modules():
        # Look for actual transformer layers (usually have attention and MLP components)
        if 'layers.' in name or 'h.' in name or 'blocks.' in name:
            if len(list(module.children())) > 0:  # Has submodules
                transformer_layer_names.append(name)
    
    # For Qwen models, typically the layers are in model.model.layers
    qwen_layers = []
    for name, module in model.named_modules():
        if name.startswith('model.layers') and '.' not in name.split('model.layers.')[-1]:
            qwen_layers.append(name)
    
    print(f"Found {len(qwen_layers)} transformer blocks:")
    for i, layer_name in enumerate(qwen_layers):
        print(f"  {i}: {layer_name}")
    
    print(f"\nTotal transformer blocks: {len(qwen_layers)}")
    
    # Inspect the first transformer block to understand its structure
    if qwen_layers:
        first_layer = dict(model.get_submodule(qwen_layers[0]).named_modules())
        print(f"\n=== STRUCTURE OF FIRST TRANSFORMER BLOCK ({qwen_layers[0]}) ===")
        for name, module in first_layer.items():
            print(f"  {name}: {type(module).__name__}")
    
    # Find attention projection layers
    print("\n=== ATTENTION PROJECTIONS ===")
    attention_projections = {}
    for name, param in model.named_parameters():
        if 'self_attn' in name and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name):
            layer_idx = None
            # Extract layer index
            for part in name.split('.'):
                if part.isdigit():
                    layer_idx = int(part)
                    break
            if layer_idx is not None:
                if layer_idx not in attention_projections:
                    attention_projections[layer_idx] = {'q_proj': [], 'k_proj': [], 'v_proj': [], 'o_proj': []}
                
                if 'q_proj' in name:
                    attention_projections[layer_idx]['q_proj'].append(name)
                elif 'k_proj' in name:
                    attention_projections[layer_idx]['k_proj'].append(name)
                elif 'v_proj' in name:
                    attention_projections[layer_idx]['v_proj'].append(name)
                elif 'o_proj' in name:
                    attention_projections[layer_idx]['o_proj'].append(name)
    
    # Print attention projections
    for layer_idx in sorted(attention_projections.keys()):
        print(f"Layer {layer_idx}:")
        for proj_type, params in attention_projections[layer_idx].items():
            if params:
                print(f"  {proj_type}: {params}")
    
    # Find MLP/feed-forward layers
    print("\n=== MLP/FEED-FORWARD LAYERS ===")
    mlp_layers = {}
    for name, param in model.named_parameters():
        if 'mlp' in name or 'feed_forward' in name or 'up_proj' in name or 'down_proj' in name or 'gate_proj' in name:
            layer_idx = None
            # Extract layer index
            for part in name.split('.'):
                if part.isdigit():
                    layer_idx = int(part)
                    break
            if layer_idx is not None:
                if layer_idx not in mlp_layers:
                    mlp_layers[layer_idx] = {'up_proj': [], 'down_proj': [], 'gate_proj': [], 'other': []}
                
                if 'up_proj' in name:
                    mlp_layers[layer_idx]['up_proj'].append(name)
                elif 'down_proj' in name:
                    mlp_layers[layer_idx]['down_proj'].append(name)
                elif 'gate_proj' in name:
                    mlp_layers[layer_idx]['gate_proj'].append(name)
                else:
                    mlp_layers[layer_idx]['other'].append(name)
    
    # Print MLP layers
    for layer_idx in sorted(mlp_layers.keys()):
        print(f"Layer {layer_idx}:")
        for mlp_type, params in mlp_layers[layer_idx].items():
            if params:
                print(f"  {mlp_type}: {params}")
    
    # Find LayerNorms
    print("\n=== LAYERNORMS ===")
    layernorms = []
    for name, module in model.named_modules():
        if 'norm' in name.lower() and ('layernorm' in str(type(module)).lower() or 'rmsnorm' in str(type(module)).lower()):
            layernorms.append((name, type(module).__name__))
    
    for name, norm_type in layernorms:
        print(f"  {name}: {norm_type}")
    
    # Find embedding layers
    print("\n=== EMBEDDING LAYERS ===")
    embeddings = []
    for name, module in model.named_modules():
        if 'embed' in name.lower() and 'embedding' in str(type(module)).lower():
            embeddings.append((name, type(module).__name__))
    
    for name, embed_type in embeddings:
        print(f"  {name}: {embed_type}")
    
    # Check for bias terms
    print("\n=== BIAS TERMS ===")
    bias_params = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            bias_params.append(name)
    
    print(f"Found {len(bias_params)} bias parameters")
    for name in bias_params[:10]:  # Show first 10
        print(f"  {name}")
    if len(bias_params) > 10:
        print(f"  ... and {len(bias_params)-10} more")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Model: {model_name}")
    print(f"Number of transformer blocks: {len(qwen_layers)}")
    print(f"Attention projections found: {sum(len(v) for v in attention_projections.values())}")
    print(f"MLP layers found: {sum(len(v) for v in mlp_layers.values())}")
    print(f"LayerNorm/RMSNorm found: {len(layernorms)}")
    print(f"Embedding layers found: {len(embeddings)}")
    print(f"Bias terms found: {len(bias_params)}")
    
    # Save architecture info to JSON
    arch_info = {
        'model_name': model_name,
        'num_layers': len(qwen_layers),
        'layer_names': qwen_layers,
        'attention_projections': attention_projections,
        'mlp_layers': mlp_layers,
        'layernorms': layernorms,
        'embeddings': embeddings,
        'bias_params': bias_params
    }
    
    with open('qwen25_architecture.json', 'w') as f:
        json.dump(arch_info, f, indent=2)
    
    print(f"\nArchitecture details saved to qwen25_architecture.json")

if __name__ == "__main__":
    inspect_model()