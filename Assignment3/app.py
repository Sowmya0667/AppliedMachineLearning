from flask import Flask, request, jsonify
import joblib
import warnings
from score import score

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the trained model from the models/ folder
MODEL_PATH = "models/best_spam_model.joblib"
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    """Homepage with a simple HTML form."""
    return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Spam Classifier</title>
        </head>
        <body>
            <h1>Spam Classifier</h1>
            <form action="/score" method="post">
                <label for="text">Enter Text:</label><br>
                <input type="text" id="text" name="text" required><br><br>
                <input type="submit" value="Check Spam">
            </form>
        </body>
        </html>
    """


@app.route("/score", methods=["POST"])
def predict():
    """
    POST /score
    Accepts JSON  {"text": "..."}  or  form-data  text=...
    Returns JSON  {"prediction": "SPAM"/"HAM", "propensity": float}
    """
    try:
        # --- Parse input ---
        if request.is_json:
            data = request.get_json()
            text = data.get("text", "").strip() if data else ""
        else:
            text = request.form.get("text", "").strip()

        # --- Validate ---
        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # --- Score ---
        prediction, propensity = score(text, model, threshold=0.5)

        return jsonify({
            "prediction": "SPAM" if prediction else "HAM",
            "propensity": round(float(propensity), 6)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


if __name__ == "__main__":
    # use_reloader=False is critical: avoids double model-load and
    # makes the server easier to shut down cleanly from tests.
    app.run(debug=False, port=5000, use_reloader=False)