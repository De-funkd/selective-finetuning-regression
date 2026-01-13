# Qwen2.5-1.5B-Instruct Model Architecture Inspection Report

## Section B: Architecture Summary with Module Names

### High-Level Transformer Structure:
- **Number of transformer blocks**: 28
- **Naming convention for blocks**: `model.layers[i]` where i ∈ [0, 27]
- **Total parameters**: ~1.5B (estimated)

### Exact Module Names for Key Components:

#### Attention Projections (Q, K, V, Output):
- **Q projections**: `model.layers.[layer_idx].self_attn.q_proj.{weight,bias}`
- **K projections**: `model.layers.[layer_idx].self_attn.k_proj.{weight,bias}`
- **V projections**: `model.layers.[layer_idx].self_attn.v_proj.{weight,bias}`
- **O projections**: `model.layers.[layer_idx].self_attn.o_proj.{weight,bias}` (note: o_proj has no bias)

Example for layer 0:
- `model.layers.0.self_attn.q_proj.weight`
- `model.layers.0.self_attn.q_proj.bias`
- `model.layers.0.self_attn.k_proj.weight`
- `model.layers.0.self_attn.k_proj.bias`
- `model.layers.0.self_attn.v_proj.weight`
- `model.layers.0.self_attn.v_proj.bias`
- `model.layers.0.self_attn.o_proj.weight`

#### MLP / Feed-Forward Layers:
- **Gate projection**: `model.layers.[layer_idx].mlp.gate_proj.weight`
- **Up projection**: `model.layers.[layer_idx].mlp.up_proj.weight`
- **Down projection**: `model.layers.[layer_idx].mlp.down_proj.weight`

Example for layer 0:
- `model.layers.0.mlp.gate_proj.weight`
- `model.layers.0.mlp.up_proj.weight`
- `model.layers.0.mlp.down_proj.weight`

#### LayerNorms:
- **Input normalization**: `model.layers.[layer_idx].input_layernorm.weight`
- **Post-attention normalization**: `model.layers.[layer_idx].post_attention_layernorm.weight`
- **Final normalization**: `model.norm.weight`

Example for layer 0:
- `model.layers.0.input_layernorm.weight`
- `model.layers.0.post_attention_layernorm.weight`

#### Embedding Layers:
- **Token embeddings**: `model.embed_tokens.weight`

#### Bias Terms:
- **Attention biases**: `model.layers.[layer_idx].self_attn.[q/k/v]_proj.bias`
- Note: Output projection (`o_proj`) typically has no bias

## Section C: Freezing Logic Notes + Caveats

### Selectively Selecting Top 4 Transformer Blocks:
- For instruction-following models like Qwen2.5, empirical evidence suggests that the middle and later layers tend to be more important for task-specific adaptation.
- Common approach: Select top 4 layers by index (highest indices)
- **Top 4 transformer blocks**: layers 24, 25, 26, 27 (indices 24-27)
- Alternative approaches could use gradient-based importance or activation statistics

### Freezing Logic for Attention-Only Fine-tuning:
1. Freeze all parameters EXCEPT:
   - All attention projection weights and biases in selected layers (24-27)
   - Specific modules to unfreeze:
     * `model.layers.[24-27].self_attn.q_proj.{weight,bias}`
     * `model.layers.[24-27].self_attn.k_proj.{weight,bias}`
     * `model.layers.[24-27].self_attn.v_proj.{weight,bias}`
     * `model.layers.[24-27].self_attn.o_proj.weight` (no bias)

### Freezing Logic for BitFit-Style Fine-tuning:
1. Freeze all weight parameters
2. Unfreeze only bias terms in selected layers (24-27)
3. Specific modules to unfreeze:
   - `model.layers.[24-27].self_attn.q_proj.bias`
   - `model.layers.[24-27].self_attn.k_proj.bias`
   - `model.layers.[24-27].self_attn.v_proj.bias`
   - `model.layers.[24-27].mlp.gate_proj.bias` (if exists)
   - `model.layers.[24-27].mlp.up_proj.bias` (if exists)
   - `model.layers.[24-27].mlp.down_proj.bias` (if exists)
   - `model.layers.[24-27].input_layernorm.bias` (if exists)
   - `model.layers.[24-27].post_attention_layernorm.bias` (if exists)

### Sanity Checks:
✅ **Selective freezing/unfreezing is safe for Qwen2.5**:
- Standard transformer architecture with no unusual components
- No shared parameters between layers that would complicate selective updates
- Standard PyTorch parameter freezing mechanism works properly

✅ **Architectural quirks check**:
- Qwen2.5 uses grouped-query attention (K,V have smaller dimensions than Q,O) - this doesn't invalidate attention-only fine-tuning
- RMSNorm instead of LayerNorm - doesn't affect selective tuning strategies
- SwiGLU activation in MLP (gate_proj + up_proj) - doesn't affect attention-only
- No architectural features that would invalidate attention-only or bias-only fine-tuning

### Implementation Notes:

1. **To freeze all except attention projections in top 4 layers**:
```python
for name, param in model.named_parameters():
    if any(f'model.layers.{i}.self_attn' in name for i in [24, 25, 26, 27]):
        param.requires_grad = True  # Unfreeze attention
    else:
        param.requires_grad = False  # Freeze everything else
```

2. **To freeze all except bias terms in top 4 layers**:
```python
for name, param in model.named_parameters():
    if '.bias' in name and any(f'model.layers.{i}.' in name for i in [24, 25, 26, 27]):
        param.requires_grad = True  # Unfreeze bias in top 4 layers
    elif any(f'model.layers.{i}.' in name for i in [24, 25, 26, 27]):
        param.requires_grad = False  # Freeze weights in top 4 layers
    else:
        param.requires_grad = False  # Freeze everything else
```