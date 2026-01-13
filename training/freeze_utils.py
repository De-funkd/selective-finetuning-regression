import torch
from transformers import AutoModelForCausalLM

def load_model():
    """Load the Qwen2.5-1.5B-Instruct model."""
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return model

def apply_freeze_mask(model, variant_name):
    """
    Apply freezing/unfreezing based on the variant name.
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    if variant_name == "base":
        # All parameters remain frozen
        pass

    elif variant_name == "full_ft":
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

    elif variant_name == "top4_full":
        # Unfreeze all parameters in layers 24-27
        for i in range(24, 28):
            for param in model.model.layers[i].parameters():
                param.requires_grad = True

    elif variant_name == "top4_attention":
        # Unfreeze only attention projections in layers 24-27
        for i in [24, 25, 26, 27]:
            # Q projection
            for param in model.model.layers[i].self_attn.q_proj.parameters():
                param.requires_grad = True
            # K projection
            for param in model.model.layers[i].self_attn.k_proj.parameters():
                param.requires_grad = True
            # V projection
            for param in model.model.layers[i].self_attn.v_proj.parameters():
                param.requires_grad = True
            # O projection
            for param in model.model.layers[i].self_attn.o_proj.parameters():
                param.requires_grad = True

    elif variant_name == "top4_bitfit":
        # Unfreeze only bias parameters in q_proj, k_proj, v_proj for layers 24-27
        for i in [24, 25, 26, 27]:
            # Q projection bias
            if hasattr(model.model.layers[i].self_attn.q_proj, 'bias') and model.model.layers[i].self_attn.q_proj.bias is not None:
                model.model.layers[i].self_attn.q_proj.bias.requires_grad = True
            # K projection bias
            if hasattr(model.model.layers[i].self_attn.k_proj, 'bias') and model.model.layers[i].self_attn.k_proj.bias is not None:
                model.model.layers[i].self_attn.k_proj.bias.requires_grad = True
            # V projection bias
            if hasattr(model.model.layers[i].self_attn.v_proj, 'bias') and model.model.layers[i].self_attn.v_proj.bias is not None:
                model.model.layers[i].self_attn.v_proj.bias.requires_grad = True

    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    return model