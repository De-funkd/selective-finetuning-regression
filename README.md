# Selective Finetuning Regression Analysis

This repository contains code for analyzing regression in selective finetuning approaches.

## Project Structure

```
/home/sra/ansh/selective-finetuning-regression/
├───analyze_architecture.py
├───initial-qwen-report.md
├───model_inspection.py
├───qwen25_architecture_report.md
├───qwen25_architecture.json
├───README.md
├───requirements.txt
├───verify_freezing.py
├───.git/...
├───data/
│   ├───eval/
│   └───finetune/
├───models/
│   ├───README.md
│   ├───base/
│   ├───full_ft/
│   ├───top4_attention/
│   ├───top4_bitfit/
│   └───top4_full/
└───training/
    ├───eval_utils.py      # Evaluation utility functions
    └───run_eval.py        # CLI script for running evaluations
```

## Stage-3 Evaluation

Stage-3 evaluation includes three key metrics for assessing model quality:

### Evaluation Metrics

1. **Uncertainty/Hallucination Assessment**
   - Measures how often the model expresses uncertainty or generates low-confidence responses
   - Identifies potential hallucinations through confidence scoring

2. **Instruction Following**
   - Evaluates how well the model adheres to given instructions
   - Detects violations of guidelines or unexpected behavior

3. **Persona Drift**
   - Assesses whether the model maintains its intended persona
   - Identifies deviations from expected character traits

### Running Evaluations

Use the CLI script to run evaluations on different model variants:

```bash
# Evaluate a specific model variant on a specific dataset
python training/run_eval.py --variant base --dataset 1A

# Evaluate a model variant on all datasets
python training/run_eval.py --variant top4_full --dataset all

# Available variants: base, full_ft, top4_full, top4_attention, top4_bitfit
# Available datasets: 1A, 1B, 1C, all
```

Results are saved to `reports/evaluation_results.json` and printed to the console.

### Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure you have the following datasets in the appropriate directories:
- `data/finetune/dataset_1A.jsonl`
- `data/finetune/dataset_1B.jsonl`
- `data/finetune/dataset_1C.jsonl`
- Or in the `data/eval/` directory if using evaluation datasets

Model checkpoints should be located at:
- `models/<variant>/checkpoint-final/`

## Original Notes

When performing bias-only fine-tuning in Qwen2.5, it's important to note that:

"**Bias-only fine-tuning in Qwen2.5 affects only attention QKV biases, as RMSNorm and output projections are bias-free.**"

This is not a problem — it's an interesting constraint that stems from the model's architecture design.