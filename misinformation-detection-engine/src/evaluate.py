import os
import argparse
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_fake_news.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_vectorizer_and_models() -> Tuple[Any, Dict[str, Any]]:
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    if not os.path.exists(vec_path):
        raise FileNotFoundError("TF-IDF vectorizer not found. Run train.py first.")

    vectorizer = joblib.load(vec_path)

    model_names = ["naive_bayes", "log_reg", "random_forest"]
    models = {}
    for name in model_names:
        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)

    if not models:
        raise FileNotFoundError("No trained models found in models/. Run train.py first.")

    return vectorizer, models


def load_eval_data(path: str = PROCESSED_DATA_PATH) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at {path}. Run preprocess.py first.")

    df = pd.read_csv(path)
    X = df["clean_text"].astype(str).values
    y = df["label"].astype(int).values
    return X, y


def evaluate_models() -> pd.DataFrame:
    vectorizer, models = load_vectorizer_and_models()
    X, y = load_eval_data()

    X_vec = vectorizer.transform(X)

    rows = []
    for name, model in models.items():
        print(f"\n=== Evaluating {name} ===")
        y_pred = model.predict(X_vec)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
        }
        print(classification_report(y, y_pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y, y_pred))

        rows.append(metrics)

        # Error analysis: false positives and false negatives
        fp_indices = np.where((y == 0) & (y_pred == 1))[0][:10]
        fn_indices = np.where((y == 1) & (y_pred == 0))[0][:10]

        print("\nSample False Positives (true=real, pred=fake):")
        for idx in fp_indices:
            print(f"- {X[idx][:200]}...")

        print("\nSample False Negatives (true=fake, pred=real):")
        for idx in fn_indices:
            print(f"- {X[idx][:200]}...")

    comparison_df = pd.DataFrame(rows)
    comparison_path = os.path.join(MODELS_DIR, "evaluation_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nEvaluation comparison table saved to {comparison_path}")
    print(comparison_df)

    return comparison_df


def explain_keywords(model, vectorizer, top_n: int = 15) -> pd.DataFrame:
    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        # Logistic Regression: coefficients for class 1
        coefs = model.coef_[0]
        top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
        top_neg_idx = np.argsort(coefs)[:top_n]

        df = pd.DataFrame(
            {
                "fake_keywords": feature_names[top_pos_idx],
                "fake_weights": coefs[top_pos_idx],
                "real_keywords": feature_names[top_neg_idx],
                "real_weights": coefs[top_neg_idx],
            }
        )
        return df

    if hasattr(model, "feature_log_prob_"):
        # Naive Bayes: class-conditional log probs
        log_probs = model.feature_log_prob_
        fake_scores = log_probs[1]
        real_scores = log_probs[0]

        top_fake_idx = np.argsort(fake_scores)[-top_n:][::-1]
        top_real_idx = np.argsort(real_scores)[-top_n:][::-1]

        df = pd.DataFrame(
            {
                "fake_keywords": feature_names[top_fake_idx],
                "fake_scores": fake_scores[top_fake_idx],
                "real_keywords": feature_names[top_real_idx],
                "real_scores": real_scores[top_real_idx],
            }
        )
        return df

    if hasattr(model, "feature_importances_"):
        # Random Forest: feature importances
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-top_n:][::-1]
        df = pd.DataFrame(
            {
                "keywords": feature_names[top_idx],
                "importance": importances[top_idx],
            }
        )
        return df

    raise ValueError("Model type not supported for keyword explanation.")


def predict_text(text: str, model_name: str = "log_reg") -> None:
    vectorizer, models = load_vectorizer_and_models()
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")

    model = models[model_name]
    clean = text  # Training already assumes cleaned text, but we keep raw for now
    X_vec = vectorizer.transform([clean])

    pred = model.predict(X_vec)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_vec)[0]
        confidence = float(np.max(proba))
    else:
        proba = None
        confidence = None

    label_str = "fake" if pred == 1 else "real"

    print(f"\nInput text prediction using {model_name}:")
    if confidence is not None:
        print(f"-> {label_str.upper()} with {confidence * 100:.2f}% confidence")
    else:
        print(f"-> {label_str.upper()} (no probability available)")

    # Basic explanation using top keywords
    try:
        explanation_df = explain_keywords(model, vectorizer, top_n=15)
        print("\nTop keywords driving predictions (global, not per-example):")
        print(explanation_df.head(10))
    except Exception as e:
        print(f"\nCould not compute keyword explanation: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models and/or run custom prediction.")
    parser.add_argument("--text", type=str, help="Custom input text to classify.")
    parser.add_argument("--model", type=str, default="log_reg", help="Model to use for custom prediction.")
    args = parser.parse_args()

    if args.text:
        predict_text(args.text, model_name=args.model)
    else:
        evaluate_models()


if __name__ == "__main__":
    main()
