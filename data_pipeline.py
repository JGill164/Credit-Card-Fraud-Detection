import argparse
import os
import numpy as np
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

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

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    print("[prep]   Time & Amount scaled (StandardScaler)")
    return df


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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall)/(precision + recall)
    return precision, recall, f1

#Main

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
    df_processed = preprocess(df_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = partition_data(df_processed)

    print("\n[done]  Splits ready for modelling.")
    print(f"        X_train : {X_train.shape}")
    print(f"        X_val   : {X_val.shape}")
    print(f"        X_test  : {X_test.shape}")

    save_splits(X_train, X_val, X_test, y_train, y_val, y_test)

    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    precision, recall, f1 = evaluateModel(y_pred,y_val)

    print("\n[Evaluate]  Model Evaluation.")
    print(f"        Precision : {precision}")
    print(f"        Recall   : {recall}")
    print(f"        F1-score  : {f1}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    main()
