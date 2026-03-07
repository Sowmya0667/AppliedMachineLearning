import os
import time
import subprocess
import threading
import pytest
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
        """Smoke test: function runs without crashing and returns a tuple."""
        result = score("Hello world", model=None, threshold=0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_types(self):
        """Format test: prediction is bool, propensity is float in [0,1]."""
        prediction, propensity = score("Sample text", model=None, threshold=0.5)
        assert isinstance(prediction, bool)
        assert isinstance(propensity, float)
        assert 0.0 <= propensity <= 1.0

    def test_prediction_is_binary(self):
        """Sanity check: prediction is 0 or 1."""
        prediction, _ = score("Win a free iPhone now!", model=None, threshold=0.5)
        assert int(prediction) in (0, 1)

    def test_propensity_range(self):
        """Sanity check: propensity score is between 0 and 1."""
        _, propensity = score("See you at 5pm", model=None, threshold=0.5)
        assert 0.0 <= propensity <= 1.0

    def test_threshold_zero_always_spam(self):
        """Edge case: threshold=0 → prediction always True (spam)."""
        for text in ["Be there tonight", "Claim prize now"]:
            prediction, _ = score(text, model=None, threshold=0)
            assert prediction is True, f"Expected True for '{text}' at threshold=0"

    def test_threshold_one_always_ham(self):
        """Edge case: threshold=1 → prediction always False (ham)."""
        for text in ["Be there tonight", "Win a free vacation!"]:
            prediction, _ = score(text, model=None, threshold=1)
            assert prediction is False, f"Expected False for '{text}' at threshold=1"

    def test_obvious_spam(self):
        """Typical input: obvious spam text → prediction is 1 (spam)."""
        prediction, _ = score(
            "Congratulations! You've won a $1000 gift card. Click here to claim now!",
            model=None,
            threshold=0.5,
        )
        assert prediction is True

    def test_obvious_ham(self):
        """Typical input: obvious ham text → prediction is 0 (not spam)."""
        prediction, _ = score(
            "Hey, are we still meeting for lunch tomorrow at noon?",
            model=None,
            threshold=0.5,
        )
        assert prediction is False


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
    # Wait for server to be ready
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


# =====================================================================
# DOCKER INTEGRATION TEST
# =====================================================================

DOCKER_IMAGE  = "spam-classifier"
DOCKER_CONTAINER = "spam-classifier-test"
DOCKER_HOST_PORT = 5002          # host port → maps to container's 5000
DOCKER_BASE_URL  = f"http://127.0.0.1:{DOCKER_HOST_PORT}"


def _docker_cmd(cmd: str) -> subprocess.CompletedProcess:
    """Run a docker CLI command and return the result."""
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
    )


def _wait_for_container(url: str, retries: int = 30, delay: float = 2.0) -> bool:
    """Poll until the container's /score endpoint is reachable."""
    for _ in range(retries):
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(delay)
    return False


def test_docker():
    """
    Docker integration test:
      1. Build the Docker image.
      2. Run the container with port binding.
      3. Send a POST /score request and validate the JSON response.
      4. Stop and remove the container (cleanup).
    """
    # ------------------------------------------------------------------
    # Step 0: Remove any leftover container from a previous test run
    # ------------------------------------------------------------------
    _docker_cmd(f"docker rm -f {DOCKER_CONTAINER}")

    # ------------------------------------------------------------------
    # Step 1: Build the Docker image
    # ------------------------------------------------------------------
    build_result = _docker_cmd(
        f"docker build -t {DOCKER_IMAGE} ."
    )
    assert build_result.returncode == 0, (
        f"Docker build failed:\n{build_result.stderr}"
    )

    # ------------------------------------------------------------------
    # Step 2: Run the container (detached, port-bound)
    # ------------------------------------------------------------------
    run_result = _docker_cmd(
        f"docker run -d "
        f"--name {DOCKER_CONTAINER} "
        f"-p {DOCKER_HOST_PORT}:5000 "
        f"{DOCKER_IMAGE}"
    )
    assert run_result.returncode == 0, (
        f"Docker run failed:\n{run_result.stderr}"
    )

    try:
        # --------------------------------------------------------------
        # Step 3: Wait for the Flask app inside the container to start
        # --------------------------------------------------------------
        is_up = _wait_for_container(f"{DOCKER_BASE_URL}/")
        assert is_up, (
            "Container did not become reachable within the timeout period."
        )

        # --------------------------------------------------------------
        # Step 4a: POST a spam-like message → expect SPAM prediction
        # --------------------------------------------------------------
        payload = {"text": "Congratulations! You've won a free prize. Claim now!"}
        response = requests.post(
            f"{DOCKER_BASE_URL}/score",
            json=payload,
            timeout=10,
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

        data = response.json()
        assert "prediction" in data, "Response JSON missing 'prediction' key"
        assert "propensity" in data, "Response JSON missing 'propensity' key"
        assert data["prediction"] in ("SPAM", "HAM"), (
            f"Unexpected prediction value: {data['prediction']}"
        )
        assert 0.0 <= float(data["propensity"]) <= 1.0, (
            f"Propensity out of range: {data['propensity']}"
        )

        # --------------------------------------------------------------
        # Step 4b: POST with missing text → expect 400 error
        # --------------------------------------------------------------
        err_response = requests.post(
            f"{DOCKER_BASE_URL}/score",
            json={},
            timeout=10,
        )
        assert err_response.status_code == 400, (
            f"Expected 400 for empty input, got {err_response.status_code}"
        )

    finally:
        # --------------------------------------------------------------
        # Step 5: Stop and remove the container (always runs)
        # --------------------------------------------------------------
        _docker_cmd(f"docker stop {DOCKER_CONTAINER}")
        _docker_cmd(f"docker rm {DOCKER_CONTAINER}")
