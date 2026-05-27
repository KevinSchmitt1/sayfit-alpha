import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.main import app
from step5_database.database import SayFitDB

@pytest.fixture
def test_db(tmp_path):
    # Create a SayFitDB pointing at a temp file
    return SayFitDB(db_path=tmp_path / "test_api.db")

@pytest.fixture
def client(test_db):
    with patch("api.main.get_db", return_value=test_db):
        with patch("api.main.Path.exists", return_value=True):  # bypass lifespan check
            with TestClient(app) as c:
                yield c

FAKE_PIPELINE_RESULT = {
    "results": [
        {
            "item_name": "banana",
            "matched_name": "BANANA",
            "amount_grams": 120,
            "nutrition": {
                "calories": 107.0,
                "protein": 1.3,
                "fat": 0.4,
                "carbs": 27.5,
            },
        }
    ]
}

FAKE_RETRIEVE_RESULT = {
    "items": [
        {
            "candidates": [
                {
                    "item_name": "APPLE",
                    "nutrition_per_100g": {
                        "calories": 52.0,
                        "protein": 0.3,
                        "fat": 0.2,
                        "carbs": 14.0,
                    },
                }
            ]
        }
    ]
}


def test_log_meal(client):
    with patch("api.main.run_pipeline", return_value=FAKE_PIPELINE_RESULT):
        response = client.post("/log", json={"uid": "user123", "text": "I ate a banana"})
        assert response.status_code == 200
        data = response.json()
        assert data["meal_id"] is not None
        assert data["uid"] == "user123"
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["item_name"] == "BANANA"
        assert item["calories"] == 107.0
        assert item["protein"] == 1.3
        assert item["fat"] == 0.4
        assert item["carbs"] == 27.5
        assert item["grams"] == 120


def test_get_today_empty(client):
    # Fresh DB has no meals — endpoint should return an empty list, not an error
    response = client.get("/meals/user123/today")
    assert response.status_code == 200
    assert response.json() == []


def test_get_history(client):
    # Fresh DB → all zeros, but the MealHistory shape must still be valid
    response = client.get("/meals/user123")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user123"
    assert "days" in data
    assert "daily_breakdown" in data
    assert "average_calories" in data
    assert "average_protein" in data


def test_patch_item(client):
    # Create a meal first, then update the item's grams using the real IDs from the response
    with patch("api.main.run_pipeline", return_value=FAKE_PIPELINE_RESULT):
        log_response = client.post("/log", json={"uid": "user123", "text": "I ate a banana"})
    meal_id = log_response.json()["meal_id"]
    item_id = log_response.json()["items"][0]["item_id"]

    response = client.patch(
        f"/meals/user123/items/{item_id}",
        json={"meal_id": meal_id, "grams": 200},
    )
    assert response.status_code == 204


def test_add_item(client):
    # Create a meal to attach the new item to, then POST the new item
    # retrieve() is called directly in this endpoint — patch it at api.main level
    with patch("api.main.run_pipeline", return_value=FAKE_PIPELINE_RESULT):
        log_response = client.post("/log", json={"uid": "user123", "text": "I ate a banana"})
    meal_id = log_response.json()["meal_id"]

    with patch("api.main.retrieve", return_value=FAKE_RETRIEVE_RESULT):
        response = client.post(
            "/meals/user123/items",
            json={"meal_id": meal_id, "item_name": "apple", "grams": 100},
        )
    assert response.status_code == 201
    data = response.json()
    assert "item_id" in data
    assert data["item_name"] == "APPLE"
    assert data["calories"] == 52.0
    assert data["grams"] == 100


def test_delete_item(client):
    # meal_id for delete_item is a query param, not in the path or body
    with patch("api.main.run_pipeline", return_value=FAKE_PIPELINE_RESULT):
        log_response = client.post("/log", json={"uid": "user123", "text": "I ate a banana"})
    meal_id = log_response.json()["meal_id"]
    item_id = log_response.json()["items"][0]["item_id"]

    response = client.delete(f"/meals/user123/items/{item_id}?meal_id={meal_id}")
    assert response.status_code == 204


def test_delete_meal(client):
    with patch("api.main.run_pipeline", return_value=FAKE_PIPELINE_RESULT):
        log_response = client.post("/log", json={"uid": "user123", "text": "I ate a banana"})
    meal_id = log_response.json()["meal_id"]

    response = client.delete(f"/meals/user123/{meal_id}")
    assert response.status_code == 204