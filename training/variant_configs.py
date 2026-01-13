VARIANTS = {
    "base": {
        "description": "Freeze all parameters - no training",
        "expected_behavioral_risk": "None - no changes to model behavior",
        "expected_regression_risk": "None - no regression possible"
    },
    "full_ft": {
        "description": "Full fine-tuning - all parameters trainable",
        "expected_behavioral_risk": "High - risk of catastrophic forgetting",
        "expected_regression_risk": "Medium - potential for overfitting"
    },
    "top4_full": {
        "description": "Unfreeze all parameters in top 4 transformer layers (24-27)",
        "expected_behavioral_risk": "Low - limited parameter changes",
        "expected_regression_risk": "Low - minimal parameter changes"
    },
    "top4_attention": {
        "description": "Unfreeze only attention projections in top 4 layers (24-27)",
        "expected_behavioral_risk": "Very Low - minimal parameter changes",
        "expected_regression_risk": "Very Low - minimal parameter changes"
    },
    "top4_bitfit": {
        "description": "Unfreeze only bias terms in attention QKV projections of top 4 layers (24-27)",
        "expected_behavioral_risk": "Very Low - minimal parameter changes",
        "expected_regression_risk": "Very Low - minimal parameter changes"
    }
}