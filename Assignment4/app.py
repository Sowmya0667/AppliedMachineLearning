from flask import Flask, request, jsonify
import warnings
from score import score

warnings.filterwarnings("ignore")

app = Flask(__name__)


# -------------------------------------------------------------------
# Home Route
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
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


# -------------------------------------------------------------------
# Score Route
# -------------------------------------------------------------------
@app.route("/score", methods=["POST"])
def predict():
    try:
        # -------------------------
        # Get Input Text
        # -------------------------
        if request.is_json:
            data = request.get_json()
            text = data.get("text", "").strip() if data else ""
        else:
            text = request.form.get("text", "").strip()

        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # -------------------------
        # Get Prediction
        # -------------------------
        prediction, propensity = score(text, model=None, threshold=0.5)

        return jsonify(
            {
                "prediction": "SPAM" if prediction else "HAM",
                "propensity": round(float(propensity), 6),
            }
        ), 200

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


# -------------------------------------------------------------------
# Run App
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
