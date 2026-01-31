import os
import re
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "fake_news.csv")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_fake_news.csv")


def clean_text(text: str) -> str:
    """
    Basic, explainable text cleaning:
    - Lowercasing
    - Remove URLs and HTML tags
    - Remove non-alphabetic characters
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw_dataset(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw dataset not found at {path}. Please place fake_news.csv there.")

    df = pd.read_csv(path)

    # Try to standardize expected columns
    possible_text_cols = ["text", "content", "article"]
    possible_title_cols = ["title", "headline"]
    possible_label_cols = ["label", "class", "target"]

    text_col = next((c for c in possible_text_cols if c in df.columns), None)
    title_col = next((c for c in possible_title_cols if c in df.columns), None)
    label_col = next((c for c in possible_label_cols if c in df.columns), None)

    if label_col is None:
        raise ValueError(f"Could not find a label column in dataset. Expected one of {possible_label_cols}.")

    # Build a unified text column (title + text if available)
    if text_col and title_col:
        df["full_text"] = (df[title_col].fillna("") + " " + df[text_col].fillna("")).str.strip()
    elif text_col:
        df["full_text"] = df[text_col].fillna("")
    elif title_col:
        df["full_text"] = df[title_col].fillna("")
    else:
        raise ValueError(f"Could not find any text column in dataset. Expected one of {possible_text_cols + possible_title_cols}.")

    df = df[[label_col, "full_text"]].rename(columns={label_col: "label", "full_text": "text"})

    # Normalize labels to {0, 1} if needed
    if df["label"].dtype == "object":
        df["label"] = df["label"].str.lower().map({"fake": 1, "false": 1, "real": 0, "true": 0})

    if not set(df["label"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Labels could not be normalized to {0, 1}. Please check dataset label values.")

    df = df.dropna(subset=["text", "label"])
    return df


def preprocess_and_save(raw_path: str = RAW_DATA_PATH, processed_path: str = PROCESSED_DATA_PATH) -> None:
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    df = load_raw_dataset(raw_path)
    df["clean_text"] = df["text"].apply(clean_text)

    # Drop empty cleaned texts
    df = df[df["clean_text"].str.len() > 0]

    # Save processed data
    df.to_csv(processed_path, index=False)

    # Also save a simple train/val split for convenience
    X_train, X_val, y_train, y_val = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    split_df = pd.DataFrame({"text": X_train, "label": y_train})
    split_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False)
    split_df = pd.DataFrame({"text": X_val, "label": y_val})
    split_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"), index=False)

    print(f"Processed data saved to: {processed_path}")
    print(f"Train/val splits saved in: {PROCESSED_DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess fake news dataset.")
    parser.add_argument("--raw_path", type=str, default=RAW_DATA_PATH, help="Path to raw CSV dataset.")
    parser.add_argument("--processed_path", type=str, default=PROCESSED_DATA_PATH, help="Output path for processed CSV.")
    args = parser.parse_args()

    preprocess_and_save(args.raw_path, args.processed_path)


if __name__ == "__main__":
    main()
