import pytest
import threading
import time
import warnings
import joblib
import requests

from score import score
from app import app as flask_app

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load model once for all unit tests
# ---------------------------------------------------------------------------
MODEL_PATH = "models/best_spam_model.joblib"
model = joblib.load(MODEL_PATH)


# ===========================================================================
# UNIT TESTS — score()

class TestScoreUnit:

    # 1. Smoke test
    def test_smoke(self):
        """score() must not raise and must return a 2-tuple."""
        try:
            result = score("Hello world", model, 0.5)
        except Exception as e:
            pytest.fail(f"score() raised an unexpected exception: {e}")

        assert isinstance(result, tuple), "score() should return a tuple"
        assert len(result) == 2, f"Expected 2 outputs, got {len(result)}"

    # 2. Format / type test
    def test_output_types(self):
        """prediction must be bool; propensity must be float-castable."""
        prediction, propensity = score("Sample text", model, 0.5)

        assert isinstance(prediction, bool), \
            f"prediction should be bool, got {type(prediction).__name__}"
        try:
            float(propensity)
        except (TypeError, ValueError):
            pytest.fail("propensity cannot be cast to float")

    # 3. prediction is 0 or 1
    def test_prediction_is_binary(self):
        """prediction must be 0 or 1."""
        prediction, _ = score("Win a free iPhone now!", model, 0.5)
        assert int(prediction) in (0, 1), \
            f"prediction should be 0 or 1, got {prediction}"

    # 4. propensity in [0, 1]
    def test_propensity_range(self):
        """propensity must be in [0, 1]."""
        _, propensity = score("See you at 5pm", model, 0.5)
        assert 0.0 <= propensity <= 1.0, \
            f"propensity out of range: {propensity}"

    # 5. threshold=0 -> always spam
    def test_threshold_zero_always_spam(self):
        """When threshold=0, every propensity >= 0 so prediction must be True."""
        for text in ["Be there tonight", "Claim your prize now"]:
            prediction, _ = score(text, model, threshold=0)
            assert int(prediction) == 1, \
                f"threshold=0 should always give prediction=1 for: '{text}'"

    # 6. threshold=1 -> always ham
    def test_threshold_one_always_ham(self):
        """When threshold=1, only propensity==1.0 would be spam, which never happens."""
        for text in ["Be there tonight", "Win a free vacation!"]:
            prediction, _ = score(text, model, threshold=1)
            assert int(prediction) == 0, \
                f"threshold=1 should always give prediction=0 for: '{text}'"

    # 7. Obvious spam -> 1
    def test_obvious_spam(self):
        """A clearly spammy message must be classified as spam."""
        text = (
            "Congratulations! You have been selected to win a FREE iPhone. "
            "Click the link below to claim your prize. Limited time offer!"
        )
        prediction, _ = score(text, model, threshold=0.5)
        assert int(prediction) == 1, \
            f"Expected spam prediction for obvious spam, got {prediction}"

    # 8. Obvious ham -> 0
    def test_obvious_ham(self):
        """A clearly non-spam message must be classified as ham."""
        text = "Don't forget about the team meeting tomorrow at 10am."
        prediction, _ = score(text, model, threshold=0.5)
        assert int(prediction) == 0, \
            f"Expected ham prediction for obvious ham, got {prediction}"

    # 9. Propensity consistent with threshold
    def test_propensity_consistent_with_threshold(self):
        """At threshold=0, prediction must always be True regardless of text."""
        text = "Call us now and win exciting prizes!"
        prediction_low, _ = score(text, model, threshold=0.0)
        assert int(prediction_low) == 1


# INTEGRATION TESTS — Flask app via background thread
# (Uses threading instead of subprocess to avoid WinError 10061 on Windows)


BASE_URL = "http://127.0.0.1:5001"   # port 5001 avoids clash with any running dev server


def _run_flask():
    """Run Flask in a daemon thread."""
    flask_app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)


@pytest.fixture(scope="module", autouse=True)
def start_flask_server():
    """Start Flask server once for the whole module; stop when done."""
    t = threading.Thread(target=_run_flask, daemon=True)
    t.start()
    # Wait until the server is actually accepting connections
    for _ in range(20):
        try:
            requests.get(f"{BASE_URL}/", timeout=1)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    yield
    # Daemon thread dies automatically when the test process exits


class TestFlaskIntegration:

    # 10. Server is up
    def test_flask_server_is_up(self):
        """GET / should return 200."""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200

    # 11. Homepage HTML
    def test_homepage_contains_form(self):
        """GET / should return the Spam Classifier HTML page."""
        response = requests.get(f"{BASE_URL}/")
        assert b"Spam Classifier" in response.content

    # 12. POST /score with JSON
    def test_score_endpoint_json(self):
        """POST /score with JSON must return prediction and propensity."""
        payload = {"text": "Congratulations! You have won a free prize. Call now!"}
        response = requests.post(f"{BASE_URL}/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "propensity" in data
        assert data["prediction"] in ("SPAM", "HAM")
        assert 0.0 <= data["propensity"] <= 1.0

    # 13. POST /score with form-data
    def test_score_endpoint_form_data(self):
        """POST /score with form-data must return prediction and propensity."""
        response = requests.post(
            f"{BASE_URL}/score",
            data={"text": "Are we still meeting at 3pm?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "propensity" in data

    # 14. Missing text → 400
    def test_missing_text_returns_400(self):
        """POST /score with no text must return 400."""
        response = requests.post(f"{BASE_URL}/score", json={})
        assert response.status_code == 400
        assert "error" in response.json()

    # 15. Empty text → 400
    def test_empty_text_returns_400(self):
        """POST /score with whitespace-only text must return 400."""
        response = requests.post(f"{BASE_URL}/score", json={"text": "  "})
        assert response.status_code == 400


# FLASK TEST CLIENT TESTS 

@pytest.fixture
def client():
    """Flask test client fixture."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def test_client_homepage(client):
    """GET / returns 200 with Spam Classifier HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Spam Classifier" in response.data


def test_client_score_json(client):
    """POST /score via test client with JSON."""
    response = client.post(
        "/score",
        json={"text": "You have won a free lottery ticket!"}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "propensity" in data


def test_client_score_form(client):
    """POST /score via test client with form-data."""
    response = client.post("/score", data={"text": "Let's catch up tomorrow"})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "propensity" in data


def test_client_missing_text(client):
    """POST /score with no text via test client → 400."""
    response = client.post("/score", json={})
    assert response.status_code == 400
    assert response.get_json() == {"error": "No input text provided"}

def test_score_loads_from_disk():
    """score() with model=None should load from disk and still work."""
    pred, prop = score("Free prize!", model=None, threshold=0.5)
    assert isinstance(pred, bool)
    assert 0.0 <= prop <= 1.0


def test_client_bad_model(client, monkeypatch):
    """If the model is None/broken, POST /score should return 500."""
    # Must patch 'score' inside app.py's namespace, not the module variable
    monkeypatch.setattr("app.score", lambda text, model, threshold: (_ for _ in ()).throw(RuntimeError("model broken")))
    response = client.post("/score", json={"text": "hello"})
    assert response.status_code == 500