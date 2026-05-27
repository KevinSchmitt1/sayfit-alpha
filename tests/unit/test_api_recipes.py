"""
Unit tests for the recipe API endpoints.

POST /recipes/{uid}/suggest       — returns ranked suggestions
POST /recipes/{uid}/log-suggestion — saves a chosen recipe to the meal log

Both endpoints use the same TestClient + mock pattern as test_api.py.
The `test_db` fixture is inherited from conftest.py (tests/unit/).
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from step5_database.database import SayFitDB

# ── Shared test data ──────────────────────────────────────────────────────────

REMAINING = {"calories": 600.0, "protein": 40.0, "fat": 25.0, "carbs": 80.0}
CONSUMED  = {"calories": 1400.0, "protein": 110.0, "fat": 45.0, "carbs": 170.0}

FAKE_RANKED = [
    {
        "title": "Chicken Stir Fry",
        "_fit_score": 82,
        "_portions": 1.0,
        "_source": "spoonacular",
        "ready_in_minutes": 30,
        "source_url": "https://example.com/chicken",
        "ingredients": ["chicken", "broccoli"],
        "ingredient_count": 2,
        "nutrition": {"calories": 500.0, "protein": 38.0, "fat": 15.0, "carbs": 40.0},
    }
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def test_db(tmp_path):
    return SayFitDB(db_path=tmp_path / "test_recipes.db")


@pytest.fixture
def client(test_db):
    with patch("api.main.get_db",    return_value=test_db):
        with patch("api.recipes.get_db", return_value=test_db):
            with patch("api.main.Path.exists", return_value=True):
                with TestClient(app) as c:
                    yield c


# ── suggest endpoint ──────────────────────────────────────────────────────────

def test_suggest_returns_ranked_suggestions(client, monkeypatch):
    monkeypatch.setenv("SPOONACULAR_API_KEY", "fake-key")
    with patch("api.recipes.compute_remaining", return_value=(REMAINING, {}, CONSUMED)):
        with patch("api.recipes.build_query_params", return_value={"number": 8}):
            with patch("api.recipes.fetch_recipes", return_value=[]):
                with patch("api.recipes.filter_and_rank", return_value=FAKE_RANKED):
                    response = client.post(
                        "/recipes/user1/suggest",
                        json={"source": "spoonacular", "preferences": {}},
                    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["suggestions"]) == 1
    s = data["suggestions"][0]
    assert s["title"] == "Chicken Stir Fry"
    assert s["fit_score"] == 82
    assert s["scale_factor"] == 1.0
    assert s["nutrition"]["protein"] == 38.0
    assert data["remaining"]["calories"] == 600.0
    assert data["message"] is None


def test_suggest_already_at_goal_returns_message(client):
    # Use kaggle source to skip the Spoonacular key check — the key is irrelevant here
    at_goal = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0}
    with patch("api.recipes.compute_remaining", return_value=(at_goal, {}, CONSUMED)):
        response = client.post("/recipes/user1/suggest", json={"source": "kaggle"})

    assert response.status_code == 200
    data = response.json()
    assert data["suggestions"] == []
    assert "goal" in data["message"].lower()


def test_suggest_low_calories_returns_message(client):
    almost_done = {"calories": 60.0, "protein": 5.0, "fat": 2.0, "carbs": 8.0}
    with patch("api.recipes.compute_remaining", return_value=(almost_done, {}, CONSUMED)):
        response = client.post("/recipes/user1/suggest", json={"source": "kaggle"})

    assert response.status_code == 200
    assert response.json()["suggestions"] == []


def test_suggest_under_200kcal_returns_quick_options_message(client):
    under_200 = {"calories": 150.0, "protein": 10.0, "fat": 5.0, "carbs": 20.0}
    with patch("api.recipes.compute_remaining", return_value=(under_200, {}, CONSUMED)):
        response = client.post("/recipes/user1/suggest", json={"source": "kaggle"})

    assert response.status_code == 200
    data = response.json()
    assert data["suggestions"] == []
    assert "200" in data["message"]


def test_suggest_missing_spoonacular_key_returns_400(client, monkeypatch):
    monkeypatch.delenv("SPOONACULAR_API_KEY", raising=False)
    with patch("api.recipes.compute_remaining", return_value=(REMAINING, {}, CONSUMED)):
        response = client.post(
            "/recipes/user1/suggest",
            json={"source": "spoonacular"},
        )

    assert response.status_code == 400
    assert "SPOONACULAR_API_KEY" in response.json()["detail"]


def test_suggest_kaggle_source_needs_no_api_key(client, monkeypatch):
    monkeypatch.delenv("SPOONACULAR_API_KEY", raising=False)
    with patch("api.recipes.compute_remaining", return_value=(REMAINING, {}, CONSUMED)):
        with patch("api.recipes.fetch_kaggle_recipes", return_value=[]):
            with patch("api.recipes.filter_and_rank", return_value=FAKE_RANKED):
                response = client.post(
                    "/recipes/user1/suggest",
                    json={"source": "kaggle"},
                )

    assert response.status_code == 200


def test_suggest_combo_without_key_falls_back_to_kaggle(client, monkeypatch):
    monkeypatch.delenv("SPOONACULAR_API_KEY", raising=False)
    with patch("api.recipes.compute_remaining", return_value=(REMAINING, {}, CONSUMED)):
        with patch("api.recipes.fetch_kaggle_recipes", return_value=[]) as mock_kaggle:
            with patch("api.recipes.filter_and_rank", return_value=[]):
                response = client.post(
                    "/recipes/user1/suggest",
                    json={"source": "combo"},
                )

    assert response.status_code == 200
    mock_kaggle.assert_called_once()   # fell back to kaggle-only


def test_suggest_no_recipes_found_returns_empty_list(client, monkeypatch):
    monkeypatch.setenv("SPOONACULAR_API_KEY", "fake-key")
    with patch("api.recipes.compute_remaining", return_value=(REMAINING, {}, CONSUMED)):
        with patch("api.recipes.build_query_params", return_value={"number": 8}):
            with patch("api.recipes.fetch_recipes", return_value=[]):
                with patch("api.recipes.filter_and_rank", return_value=[]):
                    response = client.post(
                        "/recipes/user1/suggest",
                        json={"source": "spoonacular"},
                    )

    assert response.status_code == 200
    assert response.json()["suggestions"] == []


def test_suggest_preferences_are_forwarded_to_query_builder(client, monkeypatch):
    monkeypatch.setenv("SPOONACULAR_API_KEY", "fake-key")
    prefs = {
        "target_calories": 400,
        "taste": "savory",
        "max_time_minutes": 30,
        "ingredients": ["chicken"],
        "few_ingredients": False,
    }

    with patch("api.recipes.compute_remaining", return_value=(REMAINING, {}, CONSUMED)):
        with patch("api.recipes.build_query_params", return_value={"number": 8}) as mock_bq:
            with patch("api.recipes.fetch_recipes", return_value=[]):
                with patch("api.recipes.filter_and_rank", return_value=[]):
                    client.post(
                        "/recipes/user1/suggest",
                        json={"source": "spoonacular", "preferences": prefs},
                    )

    called_prefs = mock_bq.call_args[0][1]
    assert called_prefs["taste"] == "savory"
    assert called_prefs["max_time_minutes"] == 30
    assert "chicken" in called_prefs["ingredients"]


# ── log-suggestion endpoint ───────────────────────────────────────────────────

def test_log_suggestion_saves_to_db_and_returns_meal(client):
    response = client.post(
        "/recipes/user1/log-suggestion",
        json={
            "title": "Chicken Stir Fry",
            "portions": 1.0,
            "nutrition": {"calories": 500.0, "protein": 38.0, "fat": 15.0, "carbs": 40.0},
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["uid"] == "user1"
    assert data["meal_id"] is not None
    assert len(data["items"]) == 1
    item = data["items"][0]
    assert item["item_name"] == "Chicken Stir Fry"
    assert item["calories"] == 500.0
    assert item["protein"]  == 38.0
    assert item["grams"]    == 0     # recipes are logged as servings, not grams


def test_log_suggestion_scales_macros_by_portions(client):
    response = client.post(
        "/recipes/user1/log-suggestion",
        json={
            "title": "Protein Shake",
            "portions": 2.0,
            "nutrition": {"calories": 150.0, "protein": 25.0, "fat": 3.0, "carbs": 10.0},
        },
    )

    assert response.status_code == 201
    item = response.json()["items"][0]
    assert item["calories"] == 300.0   # 150 × 2
    assert item["protein"]  == 50.0    # 25 × 2


def test_log_suggestion_appears_in_todays_meals(client):
    # Log a recipe, then verify it shows up in GET /meals/{uid}/today
    client.post(
        "/recipes/user1/log-suggestion",
        json={
            "title": "Oatmeal",
            "portions": 1.0,
            "nutrition": {"calories": 300.0, "protein": 10.0, "fat": 5.0, "carbs": 55.0},
        },
    )

    today_response = client.get("/meals/user1/today")
    assert today_response.status_code == 200
    meals = today_response.json()
    titles = [item["item_name"] for m in meals for item in m["items"]]
    assert "Oatmeal" in titles
