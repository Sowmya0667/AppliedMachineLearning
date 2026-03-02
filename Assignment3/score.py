import sklearn
import sklearn.base
import joblib
import numpy as np
import re
from typing import Tuple
from scipy.sparse import hstack

# Path to the saved best model
DEFAULT_MODEL_PATH = "models/best_spam_model.joblib"


def extract_extra_features(text: str) -> np.ndarray:
    """
    Extract the 3 extra hand-crafted features that were used during training:
      - num_characters
      - num_words
      - num_sentences

    These MUST match exactly how they were computed in train.ipynb / prepare.ipynb.
    """
    num_characters = len(text)
    num_words = len(text.split())
    num_sentences = len(re.split(r'[.!?]+', text.strip()))
    return np.array([[num_characters, num_words, num_sentences]])


def score(text: str,
          model=None,
          threshold: float = 0.5) -> Tuple[bool, float]:
    """
    Score a text using a trained model and determine if it's spam.

    The saved model is expected to be a dict with keys:
        - 'tfidf'  : fitted TfidfVectorizer
        - 'clf'    : fitted classifier (LogisticRegression / RF / NB)

    OR a plain sklearn Pipeline that accepts raw text (text-only pipeline,
    no extra features).  The function auto-detects which format is loaded.

    Args:
        text      : Raw SMS/email string to classify.
        model     : Pre-loaded model dict or pipeline. If None, loaded from disk.
        threshold : Decision cutoff in [0, 1]. propensity >= threshold → spam.

    Returns:
        (prediction: bool, propensity: float)
    """
    # --- Input validation ---
    assert isinstance(text, str), \
        f"'text' must be str, got {type(text).__name__}"
    assert isinstance(threshold, (int, float)), \
        f"'threshold' must be numeric, got {type(threshold).__name__}"
    assert 0 <= threshold <= 1, \
        f"'threshold' must be in [0, 1], got {threshold}"

    # --- Load from disk if not provided ---
    if model is None:
        model = joblib.load(DEFAULT_MODEL_PATH)

    # --- Predict ---
    # Case 1: model is a dict  {'tfidf': ..., 'clf': ...}  (3003-feature setup)
    if isinstance(model, dict) and 'tfidf' in model and 'clf' in model:
        tfidf = model['tfidf']
        clf   = model['clf']
        X_text  = tfidf.transform([text])                  # (1, 3000)
        X_extra = extract_extra_features(text)             # (1, 3)
        X       = hstack([X_text, X_extra])                # (1, 3003)
        propensity = float(clf.predict_proba(X)[0][1])

    # Case 2: plain sklearn Pipeline (text-only, no extra features)
    else:
        propensity = float(model.predict_proba([text])[0][1])

    prediction: bool = propensity >= threshold
    return prediction, propensity