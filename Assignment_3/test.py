import pytest
import threading
import time
import requests
import warnings

from score import score
from app import app as flask_app

warnings.filterwarnings("ignore")

# =====================================================================
# UNIT TESTS — score()
# =====================================================================

class TestScoreUnit:

    def test_smoke(self):
        result = score("Hello world", model=None, threshold=0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_types(self):
        prediction, propensity = score("Sample text", model=None, threshold=0.5)
        assert isinstance(prediction, bool)
        assert 0.0 <= float(propensity) <= 1.0

    def test_prediction_is_binary(self):
        prediction, _ = score("Win a free iPhone now!", model=None, threshold=0.5)
        assert int(prediction) in (0, 1)

    def test_propensity_range(self):
        _, propensity = score("See you at 5pm", model=None, threshold=0.5)
        assert 0.0 <= propensity <= 1.0

    def test_threshold_zero_always_spam(self):
        for text in ["Be there tonight", "Claim prize now"]:
            prediction, _ = score(text, model=None, threshold=0)
            assert int(prediction) == 1

    def test_threshold_one_always_ham(self):
        for text in ["Be there tonight", "Win a free vacation!"]:
            prediction, _ = score(text, model=None, threshold=1)
            assert int(prediction) == 0


# =====================================================================
# INTEGRATION TESTS — Flask (Background Thread)
# =====================================================================

BASE_URL = "http://127.0.0.1:5001"


def _run_flask():
    flask_app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)


@pytest.fixture(scope="module", autouse=True)
def start_flask_server():
    thread = threading.Thread(target=_run_flask, daemon=True)
    thread.start()

    # Wait for server to start
    for _ in range(20):
        try:
            requests.get(f"{BASE_URL}/", timeout=1)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)

    yield


class TestFlaskIntegration:

    def test_flask_server_is_up(self):
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200

    def test_homepage_contains_form(self):
        response = requests.get(f"{BASE_URL}/")
        assert b"Spam Classifier" in response.content

    def test_score_endpoint_json(self):
        payload = {"text": "You won a free prize!"}
        response = requests.post(f"{BASE_URL}/score", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "propensity" in data

    def test_missing_text_returns_400(self):
        response = requests.post(f"{BASE_URL}/score", json={})
        assert response.status_code == 400


# =====================================================================
# FLASK TEST CLIENT TESTS
# =====================================================================

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_client_homepage(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Spam Classifier" in response.data


def test_client_score_json(client):
    response = client.post("/score", json={"text": "Free lottery!"})
    assert response.status_code == 200


def test_client_missing_text(client):
    response = client.post("/score", json={})
    assert response.status_code == 400