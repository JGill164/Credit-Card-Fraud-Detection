# Credit Card Fraud Detection

## Introduction 

Credit card fraud is a major challenge in the financial industry, costing billions of dollars globally each year. Detecting fraudulent transactions automatically is critical, but difficult — fraud cases are rare, patterns are subtle, and models must be both accurate and fast enough to flag transactions in real time.

This project builds a machine learning system to classify credit card transactions as legitimate or fraudulent using supervised learning. We use a real-world dataset with a significant class imbalance (roughly 1 fraudulent transaction for every 579 legitimate ones), which reflects the actual difficulty of the problem. Working through that imbalance — rather than avoiding it with synthetic data — is a core part of what makes this project meaningful.

This file covers the first piece of the pipeline: loading the dataset and partitioning it into training, validation, and test sets.

---

## Setup

**Install dependencies:**
```bash
pip install pandas scikit-learn
```

**Download the dataset:**
1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it in a `data/` folder next to `pipeline.py`

```
project/
├── data/
│   └── creditcard.csv
└── pipeline.py
```

**Run:**
```bash
python pipeline.py
```

---

## What this does

Loads `creditcard.csv`, scales the `Time` and `Amount` columns, and splits the data into three stratified subsets:

| Split      | Size | Purpose |
|------------|------|---------|
| Train      | 60%  | Model learns from this |
| Validation | 20%  | Used to check performance during development |
| Test       | 20%  | Sealed until final evaluation |

Stratified splitting ensures the fraud ratio is preserved across all three subsets rather than concentrated in one by chance.
