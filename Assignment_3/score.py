import pandas as pd
import mlflow.pyfunc
from typing import Tuple

# Path to your best MLflow model (artifacts folder)
MLFLOW_MODEL_PATH = (
    "file:///c:/Users/sowmy/Downloads/SEM4/AML/Assignment2/mlruns/723945164383203010/models/m-6360fe763adf42629ef6cdc810dd728c/artifacts")


def score(text: str,
          model=None,
          threshold: float = 0.5) -> Tuple[bool, float]:
    """
    Score a text using the MLflow pyfunc model.

    Args:
        text: Raw message string
        model: Optional preloaded model (default loads from MLflow path)
        threshold: Decision threshold in [0,1]

    Returns:
        (prediction: bool, propensity: float)
    """

    # ----------------------------
    # Input validation
    # ----------------------------
    assert isinstance(text, str), \
        f"'text' must be str, got {type(text).__name__}"

    assert isinstance(threshold, (int, float)), \
        f"'threshold' must be numeric"

    assert 0 <= threshold <= 1, \
        f"'threshold' must be in [0,1]"

    # ----------------------------
    # Load model lazily
    # ----------------------------
    if model is None:
        model = mlflow.pyfunc.load_model(MLFLOW_MODEL_PATH)

    # ----------------------------
    # IMPORTANT: Column name must match training
    # Your training used column name "Message"
    # ----------------------------
    input_df = pd.DataFrame({"Message": [text]})

    # ----------------------------
    # Predict using MLflow model
    # ----------------------------
    prediction_output = model.predict(input_df)

    # Convert output to float
    propensity = float(prediction_output[0])

    # ----------------------------
    # Threshold logic (edge-safe)
    # ----------------------------
    if threshold == 0:
        prediction_bool = True
    elif threshold == 1:
        prediction_bool = False
    else:
        prediction_bool = propensity >= threshold

    return prediction_bool, propensity