#!/usr/bin/env python3
"""
Simple test script to verify that the evaluation modules can be imported correctly.
"""

import sys
import os
from pathlib import Path

# Add the repo root directory to the path
current_file_path = Path(__file__).resolve()
repo_root = current_file_path.parent.parent.parent  # Go up three levels to repo root from the script location
sys.path.insert(0, str(repo_root))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from training.eval_utils import (
            evaluate_uncertainty,
            evaluate_instruction_following,
            evaluate_persona_drift
        )
        print("✓ eval_utils module imported successfully")

        from training.run_eval import load_model_and_tokenizer, get_dataset_path, run_evaluation
        print("✓ run_eval module imported successfully")

        print("\nAll imports successful! The evaluation modules are ready to use.")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports()