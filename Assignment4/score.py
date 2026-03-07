import os
import pandas as pd
import mlflow.pyfunc
from typing import Tuple

# ──────────────────────────────────────────────────────────
# Model path: prefer environment variable (set in Docker),
# fall back to the original local Windows path for dev use.
# ──────────────────────────────────────────────────────────
MLFLOW_MODEL_PATH = os.environ.get(
    "MLFLOW_MODEL_PATH",
    "file:///c:/Users/sowmy/Downloads/SEM4/AML/Assignment2/mlruns/"
    "723945164383203010/models/m-6360fe763adf42629ef6cdc810dd728c/artifacts",
)


def score(
    text: str,
    model=None,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    Score a text using the MLflow pyfunc model.

    Args:
        text:      Raw message string.
        model:     Optional preloaded model (loads from MLFLOW_MODEL_PATH if None).
        threshold: Decision threshold in [0, 1].

    Returns:
        (prediction: bool, propensity: float)
    """
    # ── Input validation ──────────────────────
    assert isinstance(text, str), \
        f"'text' must be str, got {type(text).__name__}"
    assert isinstance(threshold, (int, float)), \
        "'threshold' must be numeric"
    assert 0 <= threshold <= 1, \
        "'threshold' must be in [0, 1]"

    # ── Load model lazily ─────────────────────
    if model is None:
        model = mlflow.pyfunc.load_model(MLFLOW_MODEL_PATH)

    # ── Build input DataFrame ─────────────────
    # Column name must match the one used during training: "Message"
    input_df = pd.DataFrame({"Message": [text]})

    # ── Predict ───────────────────────────────
    prediction_output = model.predict(input_df)
    propensity = float(prediction_output[0])

    # ── Apply threshold (edge-safe) ───────────
    if threshold == 0:
        prediction_bool = True
    elif threshold == 1:
        prediction_bool = False
    else:
        prediction_bool = propensity >= threshold

    return prediction_bool, propensity
