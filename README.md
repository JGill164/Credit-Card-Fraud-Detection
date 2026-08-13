# Credit Card Fraud Detection

## Introduction

This project uses machine learning to detect credit card fraud. The dataset has many more normal transactions than fraud transactions, so different models are tested to improve fraud detection.

## What This Project Does

The program loads the dataset and splits it into **60% training, 20% validation, and 20% test data**. It scales the `Time` and `Amount` columns, trains Logistic Regression and Decision Tree models, uses class weights, tunes the Logistic Regression threshold, creates graphs, and compares the models using precision, recall, F1-score, and ROC-AUC.

## Requirements

Install the required libraries:

```bash
pip3 install numpy pandas kagglehub matplotlib scikit-learn
```

## How to Run

Go to the project folder:

```bash
cd Credit-Card-Fraud-Detection
```

Then run:

```bash
python3 data_pipeline.py
```

If you already have the dataset, you can run:

```bash
python3 data_pipeline.py --data creditcard.csv
```

## Models Used

The project uses Logistic Regression, class-weighted Logistic Regression, Decision Tree, and tuned Logistic Regression models.

## Output

The program shows the dataset split, class balance, model results, best thresholds, and final test results. It also creates Precision-Recall and ROC graphs.

## Files

The main file is `data_pipeline.py`. The program also creates `train.csv` for the 60% training data, `val.csv` for the 20% validation data, and `test.csv` for the 20% test data.

