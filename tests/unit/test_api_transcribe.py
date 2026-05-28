import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from step5_database.database import SayFitDB


@pytest.fixture
def test_db(tmp_path):
    return SayFitDB(db_path=tmp_path / "test_transcribe.db")


@pytest.fixture
def client(test_db):
    with patch("api.main.get_db", return_value=test_db):
        with patch("api.main.Path.exists", return_value=True):
            with TestClient(app) as c:
                yield c


@pytest.fixture
def mock_whisper_model():
    """Return a mock Whisper model whose transcribe() returns a known string."""
    model = MagicMock()
    model.transcribe.return_value = {"text": "  oatmeal with banana  "}
    return model


def test_transcribe_returns_text(client, mock_whisper_model):
    with patch("api.transcribe._get_whisper_model", return_value=mock_whisper_model):
        response = client.post(
            "/transcribe",
            files={"file": ("recording.webm", b"fake audio bytes", "audio/webm")},
        )
    assert response.status_code == 200
    assert response.json() == {"text": "oatmeal with banana"}


def test_transcribe_strips_whitespace(client, mock_whisper_model):
    """Whisper output often has leading/trailing spaces — these must be stripped."""
    mock_whisper_model.transcribe.return_value = {"text": "   two eggs and toast   "}
    with patch("api.transcribe._get_whisper_model", return_value=mock_whisper_model):
        response = client.post(
            "/transcribe",
            files={"file": ("audio.wav", b"fake wav bytes", "audio/wav")},
        )
    assert response.status_code == 200
    assert response.json()["text"] == "two eggs and toast"


def test_transcribe_missing_file(client):
    """No file field → FastAPI should return 422 automatically."""
    response = client.post("/transcribe")
    assert response.status_code == 422


def test_transcribe_empty_file(client, mock_whisper_model):
    """Empty bytes → endpoint raises 422 before calling whisper."""
    with patch("api.transcribe._get_whisper_model", return_value=mock_whisper_model):
        response = client.post(
            "/transcribe",
            files={"file": ("recording.webm", b"", "audio/webm")},
        )
    assert response.status_code == 422
    mock_whisper_model.transcribe.assert_not_called()


def test_transcribe_unsupported_content_type(client, mock_whisper_model):
    """Non-audio content type → 422, whisper never called."""
    with patch("api.transcribe._get_whisper_model", return_value=mock_whisper_model):
        response = client.post(
            "/transcribe",
            files={"file": ("document.pdf", b"pdf content", "application/pdf")},
        )
    assert response.status_code == 422
    mock_whisper_model.transcribe.assert_not_called()
