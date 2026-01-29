## Overview
This repository contains scripts for training, evaluating, and analyzing a phishing detection model.
`data/` contains the data used for training and evaluation.
`figures/` contains the figures used in the thesis, produced by the various scripts in `src/`.
`src/` contains the scripts that analyze and evaluate the model.

## Requirements
- Python 3.10
- macOS / Linux recommended

## Setup
From the root of the repository

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run scripts
Run any script from `src/`:

```bash
cd src
python predictive_results.py
```

## Scripts
Short overview of scripts in `src/`:
- `src/ablation_study.py`: Runs ablation experiments and saves metrics.
- `src/check_external_dataset.py`: Checks the external dataset for duplicates/templates.
- `src/computation_performance.py`: Measures training and inference time.
- `src/error_analysis.py`: Saves FP/FN examples from the internal test set.
- `src/feature_influence_and_coefficients.py`: Plots top token and structural feature coefficients.
- `src/gridsearch.py`: Grid search for TF-IDF + LR hyperparameters.
- `src/helpers.py`: Shared helpers for data loading, splits, features, and models.
- `src/literature_review_plots.py`: Creates literature review plots.
- `src/loss_curves.py`: Plots train/val log-loss curves over iterations.
- `src/predictive_results.py`: Trains/calibrates and prints internal/external results.
- `src/produce_calibration_results.py`: Calibration plots and confidence summaries.
- `src/related_work_comparison.py`: Compares results to related work and plots.
- `src/roc_curves.py`: Plots ROC curves for internal/external data.
- `src/sampling_strategies_histogram.py`: Compares sampling strategies and plots.



## File tree
```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── Nazario.csv
│   ├── Phishing_validation_emails.csv
│   ├── SpamAssassin.csv
│   ├── legit.csv
│   ├── phishing.csv
│   └── gridsearch_results/
├── figures/
│   ├── ablation_outputs/
│   ├── calibration_results/
│   ├── error_analysis/
│   ├── feature_influence/
│   ├── literature_review_plots/
│   ├── loss_curve/
│   ├── related_work_comparison/
│   ├── roc_curves/
│   └── sampling_strategies_histograms_and_results/
└── src/
    ├── ablation_study.py
    ├── check_external_dataset.py
    ├── computation_performance.py
    ├── error_analysis.py
    ├── feature_influence_and_coefficients.py
    ├── gridsearch.py
    ├── helpers.py
    ├── literature_review_plots.py
    ├── loss_curves.py
    ├── predictive_results.py
    ├── produce_calibration_results.py
    ├── related_work_comparison.py
    ├── roc_curves.py
    └── sampling_strategies_histogram.py
```
