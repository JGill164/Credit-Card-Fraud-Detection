import argparse
import os
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay

RANDOM_STATE = 42


#load
def download_data() -> str:
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print(f"[kaggle] Path to dataset files: {path}")
    return path


def load_data(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        csv_file = os.path.join(path, "creditcard.csv")
    else:
        csv_file = path

    df = pd.read_csv(csv_file)
    print(f"[load]   Shape      : {df.shape}")
    print(f"[load]   Fraud rate : {df['Class'].mean()*100:.4f}%  "
          f"({df['Class'].sum()} fraud / {len(df):,} total)")
    assert df.isnull().sum().sum() == 0, "Unexpected missing values found!"
    return df

def preprocess(X_train, X_val, X_test):
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    cols = ['Time', 'Amount']

    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_val[cols] = scaler.transform(X_val[cols])
    X_test[cols] = scaler.transform(X_test[cols])

    print("[prep]   Time & Amount scaled using training statistics")

    return X_train, X_val, X_test
# partition

def partition_data(
    df: pd.DataFrame,
    target: str = 'Class',
    val_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    X = df.drop(columns=[target])
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    val_ratio = val_size / (1 - test_size)  
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_state,
    )

    total = len(df)
    print("\n[split]  Partition summary (stratified, 60/20/20):")
    print(f"  {'Split':<8} {'n':>7}  {'%total':>7}  {'fraud':>6}  {'fraud%':>7}")
    print(f"  {'─'*42}")
    for name, X_s, y_s in [
        ('Train', X_train, y_train),
        ('Val',   X_val,   y_val),
        ('Test',  X_test,  y_test),
    ]:
        print(f"  {name:<8} {len(y_s):>7,}  {len(y_s)/total*100:>6.1f}%"
              f"  {y_s.sum():>6}  {y_s.mean()*100:>6.3f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


# Save

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test,
                out_dir: str = None):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    splits = {
        'train': (X_train, y_train),
        'val':   (X_val,   y_val),
        'test':  (X_test,  y_test),
    }

    print(f"\n[save]  Writing splits to: {out_dir}")
    for name, (X, y) in splits.items():
        df_out = X.copy()
        df_out['Class'] = y.values
        out_path = os.path.join(out_dir, f"{name}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"        {name}.csv  →  {df_out.shape}  ({out_path})")

    print("[save]  Done.")

# evaluateModel caluclates the true positive, false positive, and false negative
# returns precision, recall, and f1 score
def evaluateModel(y_pred, y_true):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    if precision + recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    rocAuc = roc_auc_score(y_true = y_true, y_score = y_pred)

    return precision, recall, f1, rocAuc, specificity

# Milestone 2 — point 1: analyze class distribution & class balancing

def analyze_class_distribution(y_train):
    """Report the train-split class imbalance and derive class weights.

    Uses sklearn's 'balanced' heuristic (n_samples / (n_classes * n_bin_count)),
    which is equivalent to inversely weighting each class by its frequency —
    i.e. fraud (the minority class) gets a much higher misclassification
    penalty than the legitimate class, per the balancing approach described
    in the project plan.
    """
    counts = y_train.value_counts().sort_index()
    n_legit, n_fraud = counts.get(0, 0), counts.get(1, 0)
    total = n_legit + n_fraud
    imbalance_ratio = n_legit / n_fraud if n_fraud else float('inf')

    # sklearn's 'balanced' formula: total / (n_classes * class_count)
    weight_legit = total / (2 * n_legit) if n_legit else 0
    weight_fraud = total / (2 * n_fraud) if n_fraud else 0

    print("\n[Milestone 2 · 1] Class distribution & balancing (train split)")
    print(f"        Legit (0) : {n_legit:>7,}  ({n_legit/total*100:.3f}%)")
    print(f"        Fraud (1) : {n_fraud:>7,}  ({n_fraud/total*100:.3f}%)")
    print(f"        Imbalance ratio (legit:fraud) ≈ {imbalance_ratio:,.1f} : 1")
    print(f"        Class weights (balanced)  → legit: {weight_legit:.4f}, "
          f"fraud: {weight_fraud:.4f}")

    return {0: weight_legit, 1: weight_fraud}


def train_and_evaluate(model, name, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    precision, recall, f1, rocAuc, specificity = evaluateModel(y_pred, y_val)

    print(f"\n[Evaluate]  {name}")
    print(f"        Precision : {precision:.4f}")
    print(f"        Recall    : {recall:.4f}")
    print(f"        F1-score  : {f1:.4f}")
    print(f"        ROC-AUC-score  : {rocAuc:.4f}")

    return model, {"precision": precision, "recall": recall, "f1": f1, "roc-auc": rocAuc, "specificity": specificity}

# create_graphs creates the precision vs recall graph and the ROC curve graph
def create_graphs(name ,y_true, X_val, model, metrics, threshold):
    y_prob = model[0].predict_proba(X_val)[:, 1]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    color = ("red", "orange")
    PrecisionRecallDisplay.from_predictions(y_true = y_true, y_score = y_prob, ax = axs[0])
    RocCurveDisplay.from_predictions(y_true = y_true, y_score = y_prob, ax = axs[1])
    axs[0].set_title(f"Precision and Recall curve for {name}")
    axs[1].set_title(f"ROC curve for {name}")
    for m, c, t in zip(metrics, color, threshold):
        axs[0].plot(m['recall'], m['precision'], marker = "*", markersize = 8, color = c, label=f"Threshold of {t}")
        axs[1].plot(1-m['specificity'], m['recall'], marker = "*", markersize = 8, color = c, label = f"Threshold of {t}")
    axs[0].legend()
    axs[1].legend()
  
# tune_model finds a the best threshold to maximize f1 score then applys it to the model and evaluates
def tune_model(model, name, X_train, y_train, X_val, y_val):
    newModel = TunedThresholdClassifierCV(estimator = model, scoring = "f1")
    newModel.fit(X_train, y_train)
    print(f"\n[Threshold evaluation]  {name}")
    print(f"        Best Threshold : {newModel.best_threshold_:.4f}")
    y_pred = newModel.predict(X_val)
    precision, recall, f1, rocAuc, specificity = evaluateModel(y_pred, y_val)

    print(f"\n[Evaluate]  {name} using threshold of {round(newModel.best_threshold_, 4)}")
    print(f"        Precision : {precision:.4f}")
    print(f"        Recall    : {recall:.4f}")
    print(f"        F1-score  : {f1:.4f}")
    print(f"        ROC-AUC-score  : {rocAuc:.4f}")
    
    return newModel, {"precision": precision, "recall": recall, "f1": f1, "roc-auc": rocAuc, "specificity": specificity}

# Main

def main():
    parser = argparse.ArgumentParser(description="Credit card fraud data pipeline")
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to creditcard.csv or directory. If omitted, downloads via kagglehub.'
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Credit Card Fraud — Data Pipeline (MVS)")
    print("=" * 50)

    data_path    = download_data() if args.data is None else args.data
    df_raw       = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = partition_data(df_raw)
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)

    print("\n[done]  Splits ready for modelling.")
    print(f"        X_train : {X_train.shape}")
    print(f"        X_val   : {X_val.shape}")
    print(f"        X_test  : {X_test.shape}")

    save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
   
    print("\n[MVS]  Baseline logistic regression (unweighted)")
    baseline_model, baseline_metrics = train_and_evaluate(
        linear_model.LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Logistic Regression — baseline (unweighted)",
        X_train, y_train, X_val, y_val,
    )

    # ---- Milestone 2 · point 1: analyze class distribution, apply class
    #      balancing technique (class weighting), and retrain the model ----
    class_weights = analyze_class_distribution(y_train)

    balanced_lr_model, balanced_lr_metrics = train_and_evaluate(
        linear_model.LogisticRegression(
            class_weight=class_weights, max_iter=1000, random_state=RANDOM_STATE
        ),
        "Logistic Regression — class-weighted",
        X_train, y_train, X_val, y_val,
    )

    # ---- Milestone 2 · point 2: decision tree model (also class-weighted,
    #      since the imbalance applies here too) for comparison ----
    tree_model, tree_metrics = train_and_evaluate(
        DecisionTreeClassifier(class_weight=class_weights, random_state=RANDOM_STATE),
        "Decision Tree — class-weighted",
        X_train, y_train, X_val, y_val,
    )

    # Milestone 2 Assess Feature Importance for Decision Tree Model
    print("\n[Evaluate]  Decision Tree Feature Importance")
    important_features = pd.Series(tree_model.feature_importances_, index = X_train.columns)
    top_features = important_features.sort_values(ascending=False).head(5)
    for col, i in top_features.items():
        print(f"        {col:<15} : {i:.4f}")

    # Milestone 2 class weighted logistic regression model determing a new threshold and applying it then evaluating
    baseline_tuned_model, baseline_tuned_metrics = tune_model(baseline_model, "Logistic Regression — baseline (unweighted)", X_train, y_train, X_val, y_val)
    balanced_tuned_model , balanced_tuned_metrics = tune_model(balanced_lr_model, "Logistic Regression — class-weighted", X_train, y_train, X_val, y_val)

    # Milestone 2 Visualization
    create_graphs("Logistic Regression — baseline (unweighted)" ,y_val, X_val, [baseline_model, baseline_tuned_model], [baseline_metrics, baseline_tuned_metrics], [0.5, round(baseline_tuned_model.best_threshold_, 4)])
    create_graphs("Logistic Regression — class-weighted" ,y_val, X_val, [balanced_lr_model, balanced_tuned_model], [balanced_lr_metrics, balanced_tuned_metrics], [0.5, round(balanced_tuned_model.best_threshold_, 4)])
    create_graphs("Decision Tree — class-weighted" ,y_val, X_val, [tree_model], [tree_metrics], [0.5])
    plt.show()

    print("\n[Summary]  Validation performance comparison")
    print(f"  {'Model':<38} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>8}")
    print(f"  {'-'*76}")
    for name, m in [
        ("Logistic Regression (baseline)", baseline_metrics),
        ("Logistic Regression (baseline - tuned)", baseline_tuned_metrics),
        ("Logistic Regression (balanced)", balanced_lr_metrics),
        ("Logistic Regression (balanced - tuned)", balanced_tuned_metrics),
        ("Decision Tree (balanced)", tree_metrics),
    ]:
        print(f"  {name:<38} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc-auc']:>8.4f}")

    print("\n[Final Evaluation]  Performance on Unseen Test Data")
    print(f"  {'Model':<38} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>8}")
    print(f"  {'-'*76}")
    for name, m in [
        ("Logistic Regression (baseline)", baseline_model),
        ("Logistic Regression (baseline - tuned)", baseline_tuned_model),
        ("Logistic Regression (balanced)", balanced_lr_model),
        ("Logistic Regression (balanced - tuned)", balanced_tuned_model),
        ("Decision Tree (balanced)", tree_model),
    ]:
        y_pred = m.predict(X_test)
        precision, recall, f1, rocAuc, specificity = evaluateModel(y_pred, y_test)
        print(f"  {name:<38} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f} {rocAuc:>8.4f}")    

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    main()
