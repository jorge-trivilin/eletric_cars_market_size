# US EV Classification: Simple Cross-Validation Pipeline

## Project Overview

This project implements a simple cross-validation pipeline for predicting electric vehicle types (BEV vs PHEV) using XGBoost. The focus is on proper CV implementation with metrics to evaluate model performance and prevent data leakage.

## Cross-Validation Implementation

- **Stratified 5-Fold CV**: Ensures balanced class distribution in each fold
- **Preprocessing Inside CV**: Feature scaling happens within each fold to prevent data leakage
- **Multiple Metrics**: Tracks both accuracy and log loss across folds
- **Feature Importance Analysis**: Calculates and aggregates importance scores across all folds

## Key Pipeline Components

- `Pipeline` object combining preprocessing and classification steps
- `StratifiedKFold` for maintaining class distribution
- `cross_validate` for collecting multiple metrics simultaneously
- Per-fold feature importance extraction and aggregation

## Results

The pipeline demonstrates:
1. Performance comparison between models with/without Electric Range feature
2. Log loss evaluation for model calibration assessment
3. Consistent feature importance rankings across multiple folds

## Usage (Make sure to download the dataset before running the script)

Run the cross-validation pipeline:

```
python training.py
```

The script outputs:
- Mean accuracy and log loss metrics (with standard deviation)
- Per-fold performance breakdown
- Feature importance visualization