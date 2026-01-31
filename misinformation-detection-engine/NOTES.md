# Research Notes — Misinformation Detection Engine

Use this file for assumptions, model comparison notes, error analysis, and paper-ready bullets.

---

## Assumptions & Constraints

- **Language**: English-only text. No multilingual support in the current pipeline.
- **Labels**: Derived entirely from the dataset (no external fact-checking or human verification in production).
- **Scope**: This is a **misinformation detection classifier**, not a full fact-checking system. We predict dataset-style labels, not ground truth.
- **Domain**: News-style articles; performance on tweets, memes, or other short/noisy text is not guaranteed.
- **Train/val split**: Fixed random seed (42) for reproducibility.

---

## Dataset Notes

- **Source**: At least one well-known fake news dataset (e.g. Kaggle Fake News).
- **Labels**: `0` = real, `1` = fake.
- **Size**: Record here after EDA (e.g. N samples, balance).
- **Limitations**: English-only; dataset-based labels; possible label noise and temporal/domain bias.

---

## Model Comparison (fill after training)

| Model           | Accuracy | Precision | Recall | F1   | Pros                    | Cons                    |
|----------------|----------|-----------|--------|------|-------------------------|-------------------------|
| Naive Bayes    | —        | —         | —      | —    | Fast, interpretable     | Strong independence     |
| Logistic Reg   | —        | —         | —      | —    | Coefficients = features | Linear decision boundary|
| Random Forest  | —        | —         | —      | —    | Robust, non-linear      | Slower, less interpretable |

**Conclusion (example)**: *"Model A performed better because …"* — fill after you have metrics.

---

## Error Analysis Checklist

- [ ] False positives (real misclassified as fake): sample and note patterns.
- [ ] False negatives (fake misclassified as real): sample and note patterns.
- [ ] Common misclassified patterns: e.g. satire, borderline wording, short texts.
- [ ] Per-class precision/recall trade-offs.

---

## Confidence & Explainability

- **Confidence**: Use `predict_proba` for Naive Bayes, Logistic Regression, Random Forest. Output e.g. *"Fake with 76% confidence"*.
- **Explainability**: Top keywords (LR coefficients, NB log-probs, RF feature importance). No SHAP required for Week 2.

---

## Custom Input Testing

- Via notebook (`04_model_training.ipynb`) or CLI: `python src/evaluate.py --text "your article" --model log_reg`.
- Output: label, confidence, and (where supported) top influencing keywords.
