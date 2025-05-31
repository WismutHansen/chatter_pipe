import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock # For mocking
import torch # For dummy tensor and type hints
import sys # For stderr prints in fixture

# Assuming chatter_api.py is in the same directory or accessible via PYTHONPATH
from chatter_api import app, model as api_model, generate_speech_audio # Import app and the loaded model instance

client = TestClient(app)

# Fixture to ensure the model is available or mocked for tests
@pytest.fixture(autouse=True)
def ensure_model_is_loaded(monkeypatch):
    global api_model # Ensure we're referring to the global model from chatter_api

    if api_model is None:
        # Mock the global 'model' object in chatter_api if it failed to load
        mock_model_instance = MagicMock()
        dummy_tensor = torch.randn(1, 16000) # Example audio tensor
        mock_model_instance.generate.return_value = dummy_tensor
        mock_model_instance.sr = 16000 # Example sample rate

        # Patch the 'model' variable in the chatter_api module itself
        monkeypatch.setattr('chatter_api.model', mock_model_instance)
        # Also patch the 'model' that generate_speech_audio sees in its global scope
        # This is crucial because the function was defined with that global context.
        monkeypatch.setattr(generate_speech_audio.__globals__, 'model', mock_model_instance)
        api_model = mock_model_instance # Update the local reference for clarity
        print("API model (api_model) was None, fully mocked for testing.", file=sys.stderr)

    elif not hasattr(api_model, 'generate') or not hasattr(api_model, 'sr'):
        # This case implies that 'api_model' is not None but is not a valid model object
        mock_model_instance = MagicMock()
        dummy_tensor = torch.randn(1, 16000)
        mock_model_instance.generate.return_value = dummy_tensor
        mock_model_instance.sr = 16000

        monkeypatch.setattr('chatter_api.model', mock_model_instance)
        monkeypatch.setattr(generate_speech_audio.__globals__, 'model', mock_model_instance)
        api_model = mock_model_instance
        print("API model (api_model) seemed invalid, fully mocked for testing.", file=sys.stderr)
    else:
        # Model seems loaded and valid, patch its 'generate' method to control output
        # and avoid actual TTS, and to make it a MagicMock for introspection.
        original_generate_method = api_model.generate

        # Create a MagicMock for the generate method
        mock_generate_method = MagicMock()
        dummy_tensor = torch.randn(1, 16000) # Dummy audio tensor
        mock_generate_method.return_value = dummy_tensor

        # Patch the 'generate' method of the actual loaded model instance
        monkeypatch.setattr(api_model, 'generate', mock_generate_method)
        print(f"Patched 'generate' method of the loaded model instance (type: {type(api_model)}).", file=sys.stderr)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Chatterbox TTS API"}

def test_create_speech_success_wav():
    # This test will now use the mocked model.generate
    response = client.post("/speech", json={"text": "Hello world"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 0 # Should have some audio bytes

    # Example of asserting the mock was called (if api_model.generate is the mock)
    if hasattr(api_model, 'generate') and isinstance(api_model.generate, MagicMock):
        api_model.generate.assert_called_once_with(
            "Hello world",
            audio_prompt_path=None, # Default from SpeechRequest
            exaggeration=0.5,      # Default from SpeechRequest
            temperature=0.5        # Default from SpeechRequest
        )
        api_model.generate.reset_mock() # Reset for other tests


def test_create_speech_with_optional_params():
    response = client.post(
        "/speech",
        json={
            "text": "Hello with options",
            "voice": "dummy/path.wav", # This will be passed to the mocked generate
            "exaggeration": 0.8,
            "temperature": 0.2,
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 0

    if hasattr(api_model, 'generate') and isinstance(api_model.generate, MagicMock):
        api_model.generate.assert_called_once_with(
            "Hello with options",
            audio_prompt_path="dummy/path.wav",
            exaggeration=0.8,
            temperature=0.2
        )
        api_model.generate.reset_mock()


def test_create_speech_invalid_output_format():
    response = client.post("/speech", json={"text": "test format", "output_format": "mp3"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported output format: mp3. Only 'wav' is currently supported."}

def test_create_speech_missing_text():
    response = client.post("/speech", json={"output_format": "wav"}) # Missing 'text'
    assert response.status_code == 422 # Unprocessable Entity from Pydantic
    json_response = response.json()
    assert "detail" in json_response
    # Example check for Pydantic error detail
    found_error = False
    for err in json_response["detail"]:
        if err.get("type") == "missing" and "text" in err.get("loc", []):
            found_error = True
            break
    assert found_error, "Pydantic error detail for missing 'text' not found as expected."

def test_model_not_available_scenario(monkeypatch):
    # Temporarily make the model None as if it failed to load catastrophically
    # and the fixture didn't catch this specific way or was bypassed.
    # This tests the HTTPException(503) in the endpoint itself.

    # We need to ensure that for this test, the `model` object *within the endpoint's execution scope* is None.
    # The fixture `ensure_model_is_loaded` runs for every test. If it successfully mocks `api_model`,
    # then `api_model` (and `chatter_api.model`) won't be `None`.
    # So, we need to specifically re-patch `chatter_api.model` to `None` *after* the fixture has run.

    # Patch chatter_api.model to be None
    monkeypatch.setattr('chatter_api.model', None)
    # Critical: also patch the 'model' that the 'api_create_speech' endpoint sees in its global scope.
    # This is because the endpoint function closes over the 'model' global from its module.
    from chatter_api import api_create_speech
    monkeypatch.setattr(api_create_speech.__globals__, 'model', None)

    print(f"Testing 503: chatter_api.model is {app.model}, api_create_speech sees model {api_create_speech.__globals__['model']}", file=sys.stderr)


    response = client.post("/speech", json={"text": "Hello when model is None"})
    assert response.status_code == 503
    assert response.json() == {"detail": "TTS model is not available."}

    # It's good practice to restore, though monkeypatch handles it per test.
    # The fixture will run again for the next test and re-patch/mock.
    # No explicit restore needed here due to autouse fixture and monkeypatch's test scoping.
    # However, ensure this test doesn't interfere with api_model's state for other tests IF api_model was shared across tests
    # (which it is, but the fixture re-evaluates it).
    # Re-assigning global api_model for consistency if other tests were to import it and expect fixture's mock
    global api_model
    # api_model = ensure_model_is_loaded_fixture.api_model_mocked_instance # This is tricky, fixture scope.
    # Easiest is to rely on autouse=True to re-run fixture for next test.
    # For this test, after setting chatter_api.model to None, our local api_model might be stale.
    # This is fine as the test targets the endpoint's view of the model.

```
