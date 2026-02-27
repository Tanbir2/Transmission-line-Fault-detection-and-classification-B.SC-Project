#!/usr/bin/env python3
"""
Transmission Line Fault DETECTION (binary classification) using CatBoost.

Usage:
  python fault_detection_catboost.py --csv "Detect-Data-Without-Noise-Edited.csv" --plot

Notes:
- Requires: pandas, numpy, scikit-learn, matplotlib, seaborn, catboost
- Install (if needed): pip install pandas numpy scikit-learn matplotlib seaborn catboost
"""

import argparse
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from catboost import CatBoostClassifier
except Exception as e:
    raise ImportError(
        "catboost is required for this script. Install with: pip install catboost"
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fault detection with CatBoost (binary).")
    p.add_argument(
        "--csv",
        required=True,
        help="Path to detection CSV (must contain OUTPUT column).",
    )
    p.add_argument("--test-size", type=float, default=0.20, help="Test split fraction.")
    p.add_argument("--random-state", type=int, default=0, help="Random seed for split.")
    p.add_argument("--iterations", type=int, default=50, help="CatBoost iterations.")
    p.add_argument("--learning-rate", type=float, default=0.5, help="CatBoost learning rate.")
    p.add_argument("--seed", type=int, default=42, help="CatBoost random seed.")
    p.add_argument("--plot", action="store_true", help="Show confusion matrix plot.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.csv)

    if "OUTPUT" not in df.columns:
        raise ValueError("CSV must contain an 'OUTPUT' column for binary labels.")

    X = df.drop(["OUTPUT"], axis=1)
    y = df["OUTPUT"]

    # As in the appendix: treat all columns as 'cat_features' indices
    # (Note: in typical usage, only categorical columns should be listed.)
    cat_features = list(range(0, X.shape[1]))
    print("cat_features:", cat_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() == 2 else None
    )

    clf = CatBoostClassifier(
        iterations=args.iterations,
        random_seed=args.seed,
        learning_rate=args.learning_rate,
        custom_loss=["AUC", "Accuracy"],
        verbose=False,
    )

    clf.fit(X_train, y_train)

    print("CatBoost model is fitted:", clf.is_fitted())
    print("CatBoost model parameters:")
    print(clf.get_params())

    y_pred = clf.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="binary", zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred, average="binary", zero_division=0))
    print("F1-score :", f1_score(y_test, y_pred, average="binary", zero_division=0))

    if args.plot:
        cf_matrix = confusion_matrix(y_test, y_pred)
        ax = sns.heatmap(cf_matrix, annot=True, cmap="Greens", fmt="d")
        ax.set_title("Confusion Matrix\n\n")
        ax.set_xlabel("\nPredicted Values")
        ax.set_ylabel("Actual Values")
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
