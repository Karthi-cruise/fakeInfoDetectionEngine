import os
import argparse
import joblib

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_fake_news.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_processed_data(path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at {path}. Run preprocess.py first.")
    return pd.read_csv(path)


def build_vectorizer(max_features: int = 10000) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
    )


def train_models(df: pd.DataFrame, max_features: int = 10000) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)

    X = df["clean_text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = build_vectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    models = {
        "naive_bayes": MultinomialNB(),
        "log_reg": LogisticRegression(max_iter=1000, n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_val_vec)

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        }
        results[name] = metrics

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")

    # Save vectorizer separately
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)
    print(f"Saved TF-IDF vectorizer to {vec_path}")

    # Save metrics as a comparison table
    comparison_rows = []
    for name, m in results.items():
        comparison_rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
    print("\nModel comparison table saved to models/model_comparison.csv")
    print(comparison_df)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train fake news detection models.")
    parser.add_argument("--processed_path", type=str, default=PROCESSED_DATA_PATH, help="Path to processed CSV dataset.")
    parser.add_argument("--max_features", type=int, default=10000, help="Max number of TF-IDF features.")
    args = parser.parse_args()

    df = load_processed_data(args.processed_path)
    train_models(df, max_features=args.max_features)


if __name__ == "__main__":
    main()
