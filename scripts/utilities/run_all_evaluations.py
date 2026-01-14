#!/usr/bin/env python3
"""
Script to run all model variants on all datasets and aggregate results.
"""
import subprocess
import sys
import os
from pathlib import Path
import json
import pandas as pd

def run_evaluation(variant, dataset):
    """Run a single evaluation and return the result."""
    cmd = [
        sys.executable, "-m", "training.run_eval",
        "--variant", variant,
        "--dataset", dataset
    ]
    
    print(f"Running: {variant} on {dataset}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"Error running {variant} on {dataset}: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        print(f"Exception running {variant} on {dataset}: {e}")
        return None

def run_all_evaluations():
    """Run all model variants on all datasets."""
    variants = ["base", "full_ft", "top4_full", "top4_attention", "top4_bitfit"]
    datasets = ["1A", "1B", "1C"]
    
    print("Starting evaluation of all variants on all datasets...")
    
    for variant in variants:
        for dataset in datasets:
            run_evaluation(variant, dataset)
    
    print("All evaluations completed!")

if __name__ == "__main__":
    run_all_evaluations()