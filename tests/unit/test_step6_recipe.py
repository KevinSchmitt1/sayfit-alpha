"""
Unit tests for step6_recipe functions.

Coverage:
  filter_and_rank  — pure scoring/filtering logic, no mocks needed
  compute_remaining — DB query, get_db() mocked
  fetch_recipes    — Spoonacular HTTP client, urlopen mocked
  build_query_params — LLM call, llm_client mocked
"""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from step6_recipe.recipe_fetcher import fetch_recipes
from step6_recipe.recipe_filter import filter_and_rank
from step6_recipe.recipe_query_builder import build_query_params
from step6_recipe.recipe_suggester import compute_remaining


# ── Shared fixtures ───────────────────────────────────────────────────────────

REMAINING = {"calories": 600.0, "protein": 40.0, "fat": 25.0, "carbs": 80.0}

def _make_recipe(**overrides) -> dict:
    base = {
        "title": "Test Recipe",
        "ready_in_minutes": 30,
        "source_url": "https://example.com",
        "ingredients": ["chicken", "rice"],
        "ingredient_count": 2,
        "nutrition": {"calories": 500.0, "protein": 35.0, "fat": 15.0, "carbs": 60.0},
    }
    base.update(overrides)
    return base


# ── filter_and_rank ───────────────────────────────────────────────────────────

def test_filter_and_rank_good_recipe_is_returned():
    # Recipe that comfortably covers remaining macros at 1 portion
    ranked = filter_and_rank([_make_recipe()], REMAINING)

    assert len(ranked) == 1
    assert ranked[0]["title"] == "Test Recipe"
    assert ranked[0]["_fit_score"] > 0
    assert ranked[0]["_portions"] == 1.0


def test_filter_and_rank_hard_discard_low_protein():
    # < 5g protein while remaining protein > 30g → always discarded
    low_protein = _make_recipe(nutrition={"calories": 300.0, "protein": 2.0, "fat": 5.0, "carbs": 60.0})
    ranked = filter_and_rank([low_protein], REMAINING)

    assert ranked == []


def test_filter_and_rank_scales_up_small_recipe():
    # 200 kcal / 20g protein recipe needs ~2.4x to cover 80% kcal and 60% protein
    small = _make_recipe(nutrition={"calories": 200.0, "protein": 20.0, "fat": 5.0, "carbs": 20.0})
    ranked = filter_and_rank([small], REMAINING)

    assert len(ranked) == 1
    assert ranked[0]["_portions"] > 1.0


def test_filter_and_rank_discards_over_budget_after_scaling():
    # Low-protein recipe needs large scale that pushes calories > 120% of remaining
    # remaining = {cal: 300, prot: 40} → need scale ~2.4x for protein → 200 * 2.5 = 500 > 300 * 1.2 = 360
    tight_remaining = {"calories": 300.0, "protein": 40.0, "fat": 10.0, "carbs": 50.0}
    recipe = _make_recipe(nutrition={"calories": 200.0, "protein": 10.0, "fat": 5.0, "carbs": 20.0})
    ranked = filter_and_rank([recipe], tight_remaining)

    assert ranked == []


def test_filter_and_rank_empty_input():
    assert filter_and_rank([], REMAINING) == []


def test_filter_and_rank_respects_top_n():
    recipes = [_make_recipe(title=f"Recipe {i}") for i in range(5)]
    ranked = filter_and_rank(recipes, REMAINING, top_n=2)

    assert len(ranked) <= 2


def test_filter_and_rank_best_score_first():
    # High-protein recipe should score higher than a calorie-only recipe
    high_protein = _make_recipe(nutrition={"calories": 550.0, "protein": 38.0, "fat": 15.0, "carbs": 60.0})
    low_protein  = _make_recipe(title="Low Prot", nutrition={"calories": 580.0, "protein": 8.0, "fat": 20.0, "carbs": 70.0})
    ranked = filter_and_rank([low_protein, high_protein], REMAINING)

    # high_protein should rank first (protein weight is 45%)
    first_titles = [r["title"] for r in ranked]
    assert first_titles[0] == "Test Recipe"   # high_protein has default title


# ── compute_remaining ─────────────────────────────────────────────────────────

def test_compute_remaining_with_profile():
    mock_db = MagicMock()
    mock_db.get_daily_totals.return_value = {
        "calories": 800.0, "protein": 50.0, "fat": 30.0, "carbs": 100.0,
    }
    mock_db.get_user_profile.return_value = {
        "kcal_daily": 2000.0, "protein_daily": 150.0,
        "fat_daily": 70.0,    "carbs_daily": 250.0,
    }

    with patch("step6_recipe.recipe_suggester.get_db", return_value=mock_db):
        remaining, profile, consumed = compute_remaining("user1")

    assert remaining["calories"] == 1200.0
    assert remaining["protein"]  == 100.0
    assert remaining["fat"]      == 40.0
    assert remaining["carbs"]    == 150.0
    assert profile is not None
    assert consumed["calories"]  == 800.0


def test_compute_remaining_no_profile_returns_defaults():
    mock_db = MagicMock()
    mock_db.get_daily_totals.return_value = {
        "calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0,
    }
    mock_db.get_user_profile.return_value = None

    with patch("step6_recipe.recipe_suggester.get_db", return_value=mock_db):
        remaining, profile, consumed = compute_remaining("new_user")

    # No profile → conservative defaults
    assert remaining["calories"] == 500.0
    assert remaining["protein"]  == 30.0
    assert profile is None


def test_compute_remaining_clamps_to_zero_when_over_target():
    mock_db = MagicMock()
    mock_db.get_daily_totals.return_value = {
        "calories": 2500.0, "protein": 200.0, "fat": 100.0, "carbs": 300.0,
    }
    mock_db.get_user_profile.return_value = {
        "kcal_daily": 2000.0, "protein_daily": 150.0,
        "fat_daily": 70.0,    "carbs_daily": 250.0,
    }

    with patch("step6_recipe.recipe_suggester.get_db", return_value=mock_db):
        remaining, _, _ = compute_remaining("user1")

    # All macros exceeded → remaining is 0, never negative
    assert remaining["calories"] == 0.0
    assert remaining["protein"]  == 0.0


# ── fetch_recipes ─────────────────────────────────────────────────────────────

def _make_urlopen_mock(data: dict):
    """Return a context-manager mock that yields a response with `data` as JSON."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_fetch_recipes_success():
    fake_data = {
        "results": [{
            "id": 1,
            "title": "Chicken Stir Fry",
            "readyInMinutes": 25,
            "sourceUrl": "https://example.com/recipe",
            "nutrition": {"nutrients": [
                {"name": "Calories",       "amount": 480.0},
                {"name": "Protein",        "amount": 36.0},
                {"name": "Fat",            "amount": 14.0},
                {"name": "Carbohydrates",  "amount": 42.0},
            ]},
            "extendedIngredients": [{"name": "chicken"}, {"name": "broccoli"}],
        }],
    }

    with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(fake_data)):
        result = fetch_recipes({"number": 8}, "fake-key")

    assert len(result) == 1
    assert result[0]["title"] == "Chicken Stir Fry"
    assert result[0]["nutrition"]["protein"] == 36.0
    assert result[0]["ingredient_count"] == 2
    assert "chicken" in result[0]["ingredients"]


def test_fetch_recipes_quota_exceeded_returns_empty():
    http_err = urllib.error.HTTPError(url="", code=402, msg="Payment Required", hdrs=None, fp=None)
    http_err.read = lambda: b"quota exceeded"

    with patch("urllib.request.urlopen", side_effect=http_err):
        result = fetch_recipes({"number": 8}, "fake-key")

    assert result == []


def test_fetch_recipes_network_error_returns_empty():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        result = fetch_recipes({"number": 8}, "fake-key")

    assert result == []


def test_fetch_recipes_empty_results():
    with patch("urllib.request.urlopen", return_value=_make_urlopen_mock({"results": []})):
        result = fetch_recipes({"number": 8}, "fake-key")

    assert result == []


# ── build_query_params ────────────────────────────────────────────────────────

@pytest.mark.llm_path
def test_build_query_params_parses_llm_response():
    fake_json = json.dumps({
        "number": 8,
        "maxCalories": 660,
        "minProtein": 12,
        "addRecipeNutrition": True,
        "addRecipeInformation": True,
        "instructionsRequired": True,
    })

    mock_response = MagicMock()
    mock_response.choices[0].message.content = fake_json

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("llm_client.get_client", return_value=mock_client):
        with patch("llm_client.reasoning_model", return_value="test-model"):
            result = build_query_params(
                {"calories": 600.0, "protein": 40.0, "fat": 20.0, "carbs": 60.0},
                {"taste": "any", "max_time_minutes": None, "ingredients": [], "few_ingredients": False},
            )

    assert result["number"] == 8
    assert result["maxCalories"] == 660
    assert result["addRecipeNutrition"] is True


@pytest.mark.llm_path
def test_build_query_params_falls_back_on_llm_error():
    # get_client() is outside the try/except in build_query_params, so the
    # exception that triggers fallback must come from client.chat.completions.create()
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API call failed")

    with patch("llm_client.get_client", return_value=mock_client):
        with patch("llm_client.reasoning_model", return_value="test-model"):
            result = build_query_params(
                {"calories": 500.0, "protein": 35.0, "fat": 10.0, "carbs": 40.0},
                {"taste": "savory", "max_time_minutes": 30, "ingredients": ["chicken"], "few_ingredients": False},
            )

    assert result["number"] == 8
    assert result["addRecipeNutrition"] is True
    assert result.get("type") == "main course"
    assert result.get("maxReadyTime") == 30
    assert "chicken" in result.get("includeIngredients", "")
