# model.py
# ─────────────────────────────────────────────────────────────────────────────
# Machine Learning Module
# Handles: TF-IDF feature extraction, Logistic Regression training,
#          evaluation, prediction, and model persistence with pickle.
# ─────────────────────────────────────────────────────────────────────────────

import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
MODEL_PATH      = BASE_DIR / "trained_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
ENCODER_PATH    = BASE_DIR / "label_encoder.pkl"
DATASET_PATH    = BASE_DIR / "dataset.csv"


# ── TF-IDF Configuration ──────────────────────────────────────────────────────
TFIDF_CONFIG = {
    "max_features": 5000,
    "ngram_range":  (1, 2),      # unigrams + bigrams
    "sublinear_tf": True,        # apply log(1+tf) scaling
    "min_df":       1,
    "max_df":       0.95,
    "analyzer":     "word",
}

# Logistic Regression config
LR_CONFIG = {
    "C":             5.0,
    "max_iter":      1000,
    "solver":        "lbfgs",
    "class_weight":  "balanced",
    "random_state":  42,
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_dataset(path: str = str(DATASET_PATH)) -> pd.DataFrame:
    """Load and validate the training CSV."""
    df = pd.read_csv(path)
    required_cols = {"resume_text", "job_role"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")
    df = df.dropna(subset=["resume_text", "job_role"])
    df["resume_text"] = df["resume_text"].astype(str)
    df["job_role"]    = df["job_role"].astype(str).str.strip()
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    dataset_path: str = str(DATASET_PATH),
    test_size: float  = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Full training pipeline:
      1. Load dataset
      2. Encode labels
      3. TF-IDF vectorisation
      4. Train/test split
      5. Logistic Regression training
      6. Evaluation
      7. Persist model, vectoriser, and encoder

    Returns:
        Dict with accuracy, classification_report, cv_scores, classes.
    """
    print("📂  Loading dataset …")
    df = load_dataset(dataset_path)
    print(f"    {len(df)} samples | {df['job_role'].nunique()} classes")

    X = df["resume_text"].tolist()
    y = df["job_role"].tolist()

    # Label encoding
    encoder = LabelEncoder()
    y_enc   = encoder.fit_transform(y)

    # TF-IDF
    print("🔢  Fitting TF-IDF vectoriser …")
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_tfidf    = vectorizer.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y_enc, test_size=test_size,
        random_state=random_state, stratify=y_enc if len(set(y_enc)) > 1 else None,
    )

    # Model training
    print("🤖  Training Logistic Regression …")
    model = LogisticRegression(**LR_CONFIG)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(
        y_test, y_pred,
        target_names=encoder.classes_,
        zero_division=0,
    )

    # Cross-validation (on full data)
    cv_scores = cross_val_score(model, X_tfidf, y_enc, cv=min(5, len(df)), scoring="accuracy")

    print(f"\n✅  Accuracy : {accuracy:.4f}")
    print(f"    CV Mean  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Persist
    _save_artifacts(model, vectorizer, encoder)
    print("💾  Model saved.")

    return {
        "accuracy":              round(float(accuracy), 4),
        "cv_mean":               round(float(cv_scores.mean()), 4),
        "cv_std":                round(float(cv_scores.std()), 4),
        "classification_report": report,
        "classes":               list(encoder.classes_),
        "n_samples":             len(df),
        "n_features":            X_tfidf.shape[1],
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def _save_artifacts(model, vectorizer, encoder) -> None:
    with open(MODEL_PATH,      "wb") as f: pickle.dump(model,      f)
    with open(VECTORIZER_PATH, "wb") as f: pickle.dump(vectorizer, f)
    with open(ENCODER_PATH,    "wb") as f: pickle.dump(encoder,    f)


def load_artifacts() -> tuple:
    """Load (model, vectorizer, encoder) from disk."""
    if not all(p.exists() for p in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH]):
        raise FileNotFoundError(
            "Trained model not found. Please train the model first via the sidebar."
        )
    with open(MODEL_PATH,      "rb") as f: model      = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f: vectorizer = pickle.load(f)
    with open(ENCODER_PATH,    "rb") as f: encoder    = pickle.load(f)
    return model, vectorizer, encoder


def model_exists() -> bool:
    return all(p.exists() for p in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH])


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_job_role(preprocessed_text: str) -> dict:
    """
    Predict the most likely job role for a preprocessed resume text.

    Args:
        preprocessed_text: Output of resume_parser.preprocess_text()

    Returns:
        Dict with keys: predicted_role, confidence, top_roles (list of dicts)
    """
    model, vectorizer, encoder = load_artifacts()

    X = vectorizer.transform([preprocessed_text])
    proba     = model.predict_proba(X)[0]
    top_idx   = np.argsort(proba)[::-1][:5]   # top-5

    predicted_role = encoder.classes_[top_idx[0]]
    confidence     = float(proba[top_idx[0]])

    top_roles = [
        {
            "role":       encoder.classes_[i],
            "confidence": round(float(proba[i]) * 100, 2),
        }
        for i in top_idx
    ]

    return {
        "predicted_role": predicted_role,
        "confidence":     round(confidence * 100, 2),
        "top_roles":      top_roles,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = train_model()
    print("\n📊  Classification Report:")
    print(results["classification_report"])