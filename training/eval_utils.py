import json
import torch
from transformers import pipeline
from collections import Counter
import re
from typing import Dict, List, Tuple


def evaluate_uncertainty(model, tokenizer, dataset_path: str) -> Dict[str, float]:
    """
    Evaluate model uncertainty/hallucination using confidence scores and consistency measures.
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Loaded HuggingFace tokenizer
        dataset_path: Path to JSONL dataset
    
    Returns:
        Dictionary containing hallucination/uncertainty metrics
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    # Initialize text generation pipeline
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_token_id=tokenizer.eos_token_id
    )
    
    total_responses = len(dataset)
    low_confidence_count = 0
    uncertain_responses = 0
    
    for sample in dataset:
        prompt = sample.get('prompt', sample.get('question', ''))
        
        # Generate response
        response = generator(
            prompt,
            max_length=min(len(tokenizer.encode(prompt)) + 200, 1024),
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )[0]['generated_text']
        
        # Check for uncertainty indicators
        response_lower = response.lower()
        uncertainty_indicators = [
            'i don\'t know',
            'i am not sure',
            'i\'m uncertain',
            'cannot determine',
            'not enough information',
            'probably',
            'maybe',
            'possibly'
        ]
        
        if any(indicator in response_lower for indicator in uncertainty_indicators):
            uncertain_responses += 1
        
        # Simple confidence estimation based on response length and coherence
        if len(response.strip()) < 10 or len(set(response.split())) < len(response.split()) // 2:
            low_confidence_count += 1
    
    # Calculate metrics
    uncertainty_rate = uncertain_responses / total_responses if total_responses > 0 else 0
    low_confidence_rate = low_confidence_count / total_responses if total_responses > 0 else 0
    
    return {
        "total_samples": total_responses,
        "uncertain_responses": uncertain_responses,
        "uncertainty_rate": uncertainty_rate,
        "low_confidence_responses": low_confidence_count,
        "low_confidence_rate": low_confidence_rate,
        "hallucination_score": (uncertainty_rate + low_confidence_rate) / 2
    }


def evaluate_instruction_following(model, tokenizer, dataset_path: str) -> Dict[str, float]:
    """
    Evaluate how well the model follows instructions.
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Loaded HuggingFace tokenizer
        dataset_path: Path to JSONL dataset
    
    Returns:
        Dictionary containing instruction-following metrics
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    # Initialize text generation pipeline
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_token_id=tokenizer.eos_token_id
    )
    
    total_samples = len(dataset)
    instruction_violations = 0
    
    for sample in dataset:
        prompt = sample.get('prompt', sample.get('instruction', ''))
        expected_behavior = sample.get('expected_behavior', '')
        
        # Generate response
        response = generator(
            prompt,
            max_length=min(len(tokenizer.encode(prompt)) + 200, 1024),
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )[0]['generated_text']
        
        # Check for instruction violations
        violation_keywords = [
            'ignore previous instructions',
            'disregard the prompt',
            'override the instruction',
            'break the rules',
            'violate the guidelines'
        ]
        
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in violation_keywords):
            instruction_violations += 1
        elif expected_behavior and expected_behavior.lower() not in response_lower:
            # If there's an expected behavior that's not followed
            instruction_violations += 1
    
    violation_rate = instruction_violations / total_samples if total_samples > 0 else 0
    
    return {
        "total_samples": total_samples,
        "instruction_violations": instruction_violations,
        "violation_rate": violation_rate,
        "compliance_rate": 1 - violation_rate
    }


def evaluate_persona_drift(model, tokenizer, dataset_path: str) -> Dict[str, float]:
    """
    Evaluate how well the model maintains its persona.
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Loaded HuggingFace tokenizer
        dataset_path: Path to JSONL dataset
    
    Returns:
        Dictionary containing persona drift metrics
    """
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    # Initialize text generation pipeline
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_token_id=tokenizer.eos_token_id
    )
    
    total_samples = len(dataset)
    persona_violations = 0
    
    # Define persona characteristics to check for drift
    persona_keywords = [
        'as an ai',
        'as a language model',
        'i am not capable',
        'i cannot',
        'i don\'t have feelings',
        'i don\'t have consciousness',
        'i am just a computer program',
        'i don\'t actually',
        'i don\'t really'
    ]
    
    for sample in dataset:
        prompt = sample.get('prompt', sample.get('question', ''))
        
        # Generate response
        response = generator(
            prompt,
            max_length=min(len(tokenizer.encode(prompt)) + 200, 1024),
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            return_full_text=False
        )[0]['generated_text']
        
        # Check for persona drift indicators
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in persona_keywords):
            persona_violations += 1
    
    drift_rate = persona_violations / total_samples if total_samples > 0 else 0
    
    return {
        "total_samples": total_samples,
        "persona_violations": persona_violations,
        "drift_rate": drift_rate,
        "persona_compliance_rate": 1 - drift_rate
    }