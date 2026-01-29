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

## Data
Expected data files are in `data/`
- `data/phishing.csv`
- `data/Phishing_validation_emails.csv`

## Run scripts
Run any script from `src/`:

```bash
cd src
python predictive_results.py
```