# Misinformation Detection Engine

An end-to-end NLP + machine learning pipeline for detecting fake news / misinformation in text. The system provides:

- Binary classification (`fake` vs. `real`)
- Confidence scores (e.g., *Fake with 76% confidence*)
- Basic explainability via influential keywords and feature importance
- Tools for error analysis and research-style model comparison

---

## Project Structure

```bash
misinformation-detection-engine/
├── data/
│   ├── README.md           # Data layout and usage
│   ├── raw/                # Place fake_news.csv here
│   └── processed/         # Output of preprocess.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_model_training.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── notes.md                # Research notes, model comparison template, assumptions
├── README.md
└── requirements.txt
```

This structure is designed to look and behave like a serious, research-oriented project.

- **Data**: See [data/README.md](data/README.md) for where to put raw data and what appears in `processed/`.
- **Research notes**: See [notes.md](notes.md) for assumptions, model comparison template, and error-analysis checklist.

---

## Dataset

**Recommended dataset**: Kaggle *Fake News* dataset (`news.csv` style), which typically includes:

- **Columns (example)**:
  - `title`: headline text
  - `text`: full article text
  - `label`: `0` = real, `1` = fake (or `FAKE` / `REAL` depending on the version)
- **Size**: ~20,000–40,000 news articles (depending on split/version)
- **Limitations**:
  - Mostly English-only news articles
  - Labels are dataset-based (from the original creators) and may have bias
  - Not updated in real time, so may not reflect the latest misinformation trends

### How to place the dataset

1. Download the fake news dataset (e.g., from Kaggle).
2. Save the main CSV file as:
   - `data/raw/fake_news.csv`
3. Adjust the file name / column names in `src/preprocess.py` if your dataset differs.

---

## NLP Pipeline

Implemented in `src/preprocess.py` and `src/train.py`:

- **Text cleaning**
  - Lowercasing
  - Removing URLs, HTML tags, and extra whitespace
  - Removing non-alphabetic characters / punctuation (configurable)
- **Tokenization**
  - Basic whitespace tokenization after cleaning
- **Stopword removal**
  - English stopwords using `nltk` or `scikit-learn`
- **Vectorization (TF-IDF)**
  - `sklearn.feature_extraction.text.TfidfVectorizer`
  - Optionally limit max features (e.g. 10,000) for efficiency

The processed dataset is saved to `data/processed/processed_fake_news.csv`.

---

## Models

Implemented in `src/train.py`:

- **Naive Bayes**
  - `sklearn.naive_bayes.MultinomialNB`
- **Logistic Regression**
  - `sklearn.linear_model.LogisticRegression`
- **(Optional bonus)** Random Forest
  - `sklearn.ensemble.RandomForestClassifier`

Models are trained on TF-IDF features and saved (with the vectorizer) for later evaluation and custom input testing.

---

## Evaluation

Implemented in `src/evaluate.py`:

For each model, we compute:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

We also:

- Build a **model comparison table** (metrics side-by-side)
- Perform basic **error analysis**:
  - Inspect false positives and false negatives
  - Look for common misclassified patterns

---

## Confidence Scoring

For models that support `predict_proba` (Naive Bayes, Logistic Regression, Random Forest), we:

- Use the predicted class probability as a **confidence score**
- Expose a helper function (and notebook cells) that output:
  - `Fake with 76% confidence` or
  - `Real with 88% confidence`

---

## Explainability (Basic)

We provide lightweight but real explainability:

- **Top keywords influencing prediction**:
  - For Logistic Regression: use the learned coefficients per feature.
  - For Naive Bayes: use class-conditional log probabilities.
- **Feature importance visualization**:
  - Simple bar plots of top positive/negative words for each class.

These are demonstrated in the notebooks, especially `03_feature_extraction.ipynb` and `04_model_training.ipynb`.

---

## Custom Input Testing

You can test arbitrary text via:

- Notebook cells in `04_model_training.ipynb`, or
- A CLI-style interface in `src/evaluate.py` (e.g., `python src/evaluate.py --text "your news article here"`).

The system returns:

- Predicted label (`fake` / `real`)
- Confidence score (probability)
- A short explanation (e.g., top contributing keywords)

---

## Assumptions & Constraints

- **Language**: English-only text.
- **Labels**: Derived entirely from the dataset (no external fact-checking).
- **Scope**: This is a **misinformation detection classifier**, not a full fact-checking system.
- **Domain**: News-style articles; performance on tweets, memes, or other formats is not guaranteed.

These assumptions should be clearly stated in your report / paper as well.

---

## How to Run

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download and place dataset**

   - Put your CSV file at `data/raw/fake_news.csv`.

3. **Run preprocessing**

   ```bash
   python src/preprocess.py
   ```

4. **Train models**

   ```bash
   python src/train.py
   ```

5. **Evaluate & analyze**

   ```bash
   python src/evaluate.py
   ```

6. **Use notebooks for research depth**

   - `01_data_exploration.ipynb`: EDA, label distribution, text length, etc.
   - `02_preprocessing.ipynb`: Cleaning, tokenization, stopwords examples.
   - `03_feature_extraction.ipynb`: TF-IDF exploration, top n-grams.
   - `04_model_training.ipynb`: Model training, comparison table, error analysis, custom input tests.

---

## Next Steps / Extensions

Once the Week 2 core system is stable, you can extend with:

- Web UI (Streamlit / Flask)
- Deep learning models (LSTM, BERT)
- Multiple datasets and domain adaptation
- Source credibility and user-level features
- Multilingual misinformation detection

This core engine is designed so that these extensions can plug into the same preprocessing and evaluation framework.
