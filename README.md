# US EV Classification: Cross-Validation Pipeline

## Project Overview

This project implements a cross-validation pipeline for predicting electric vehicle types (BEV vs PHEV) using both XGBoost and Random Forest classifiers. The focus is on proper CV implementation with metrics to evaluate model performance and prevent data leakage.

## Cross-Validation Implementation

- **Stratified 5-Fold CV**: Ensures balanced class distribution in each fold
- **Preprocessing Inside CV**: Feature scaling happens within each fold to prevent data leakage
- **Multiple Metrics**: Tracks both accuracy and log loss across folds
- **Feature Importance Analysis**: Calculates and aggregates importance scores across all folds

## Pipeline Components

- `Pipeline` object combining preprocessing and classification steps
- `StratifiedKFold` for maintaining class distribution
- `cross_validate` for collecting multiple metrics simultaneously
- Per-fold feature importance extraction and aggregation

## Models Implemented

- **XGBoost**: Gradient boosting framework known for performance and speed
- **Random Forest**: Ensemble learning method using multiple decision trees

## Results

The pipelines demonstrate:
1. Performance comparison between models with/without Electric Range feature
2. Log loss evaluation for model calibration assessment
3. Consistent feature importance rankings across multiple folds
4. Comparative analysis between XGBoost and Random Forest approaches

## Usage (Make sure to download the dataset before running the script)

Run the cross-validation pipelines:

```
# Run XGBoost implementation
python training.py

# Run Random Forest implementation
python random_forest.py
```

The scripts output:
- Mean accuracy and log loss metrics (with standard deviation)
- Per-fold performance breakdown
- Feature importance visualizations
- Model comparison charts

## Model Comparison

The repository includes visualizations comparing:
- Feature importance rankings between models
- Accuracy and log loss metrics
- Training vs. testing performance to detect overfitting