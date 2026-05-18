import pytest

from step5_database.database import SayFitDB

SAMPLE_ITEMS = [
    {
        "item_name": "banana",
        "matched_name": "banana raw",
        "amount_grams": 120,
        "unit": "g",
        "calories": 106.8,
        "protein": 1.3,
        "fat": 0.4,
        "carbs": 27.6,
        "confidence": "high",
        "confidence_note": "",
    }
]

TWO_ITEMS = [
    *SAMPLE_ITEMS,
    {
        "item_name": "egg",
        "matched_name": "egg boiled",
        "amount_grams": 60,
        "unit": "g",
        "calories": 78.0,
        "protein": 6.3,
        "fat": 5.3,
        "carbs": 0.6,
        "confidence": "high",
        "confidence_note": "",
    },
]

PIPELINE_RERANKED = {
    "results": [
        {
            "item_name": "apple",
            "matched_name": "apple raw",
            "amount_grams": 180,
            "unit": "g",
            "confidence": "high",
            "confidence_note": "",
            "nutrition": {"calories": 94.0, "protein": 0.5, "fat": 0.3, "carbs": 25.1},
        }
    ]
}


@pytest.fixture
def db(tmp_path):
    return SayFitDB(db_path=tmp_path / "test.db")


class TestSaveMeal:
    def test_returns_meal_id_string(self, db):
        meal_id = db.save_meal("user1", SAMPLE_ITEMS, "i ate a banana", meal_date="2026-01-01")
        assert isinstance(meal_id, str)
        assert len(meal_id) > 0

    def test_custom_meal_id_used(self, db):
        meal_id = db.save_meal(
            "user1", SAMPLE_ITEMS, "i ate a banana",
            meal_date="2026-01-01", meal_id="my-fixed-id"
        )
        assert meal_id == "my-fixed-id"

    def test_auto_creates_user(self, db):
        db.save_meal("new_user", SAMPLE_ITEMS, "banana", meal_date="2026-01-01")
        totals = db.get_daily_totals("new_user", "2026-01-01")
        assert totals["meal_count"] == 1

    def test_meal_calories_stored(self, db):
        db.save_meal("user1", SAMPLE_ITEMS, "banana", meal_date="2026-01-01")
        totals = db.get_daily_totals("user1", "2026-01-01")
        assert totals["calories"] == pytest.approx(106.8, abs=0.1)


class TestGetDailyTotals:
    def test_empty_day_returns_zeros(self, db):
        totals = db.get_daily_totals("user1", "2026-01-01")
        assert totals["calories"] == 0
        assert totals["meal_count"] == 0

    def test_single_meal_aggregated(self, db):
        db.save_meal("user1", TWO_ITEMS, "banana and egg", meal_date="2026-01-01")
        totals = db.get_daily_totals("user1", "2026-01-01")
        assert totals["calories"] == pytest.approx(106.8 + 78.0, abs=0.1)

    def test_all_macros_aggregated_correctly(self, db):
        db.save_meal("user1", TWO_ITEMS, "banana and egg", meal_date="2026-01-01")
        totals = db.get_daily_totals("user1", "2026-01-01")
        assert totals["protein"] == pytest.approx(1.3 + 6.3, abs=0.1)
        assert totals["fat"] == pytest.approx(0.4 + 5.3, abs=0.1)
        assert totals["carbs"] == pytest.approx(27.6 + 0.6, abs=0.1)

    def test_two_meals_summed(self, db):
        db.save_meal("user1", SAMPLE_ITEMS, "breakfast", meal_date="2026-01-01")
        db.save_meal("user1", SAMPLE_ITEMS, "lunch", meal_date="2026-01-01")
        totals = db.get_daily_totals("user1", "2026-01-01")
        assert totals["meal_count"] == 2
        assert totals["calories"] == pytest.approx(106.8 * 2, abs=0.1)

    def test_different_dates_not_mixed(self, db):
        db.save_meal("user1", SAMPLE_ITEMS, "breakfast", meal_date="2026-01-01")
        totals = db.get_daily_totals("user1", "2026-01-02")
        assert totals["calories"] == 0


class TestCalibration:
    def test_get_calibration_returns_none_when_not_set(self, db):
        result = db.get_calibration("user1", "banana")
        assert result is None

    def test_add_then_get_returns_saved_value(self, db):
        db.add_calibration("user1", "banana", 150.0)
        result = db.get_calibration("user1", "banana")
        assert result is not None
        assert result["preferred_grams"] == pytest.approx(150.0)

    def test_second_add_increments_corrections_count(self, db):
        db.add_calibration("user1", "banana", 120.0)
        db.add_calibration("user1", "banana", 150.0)
        result = db.get_calibration("user1", "banana")
        assert result["corrections_count"] == 2

    def test_second_add_updates_grams(self, db):
        db.add_calibration("user1", "banana", 120.0)
        db.add_calibration("user1", "banana", 200.0)
        result = db.get_calibration("user1", "banana")
        assert result["preferred_grams"] == pytest.approx(200.0)

    def test_different_users_isolated(self, db):
        db.add_calibration("user1", "banana", 120.0)
        result = db.get_calibration("user2", "banana")
        assert result is None


class TestDeleteMeal:
    def test_soft_delete_removes_from_daily_totals(self, db):
        meal_id = db.save_meal("user1", SAMPLE_ITEMS, "banana", meal_date="2026-01-01")
        db.delete_meal(meal_id)
        totals = db.get_daily_totals("user1", "2026-01-01")
        assert totals["meal_count"] == 0
        assert totals["calories"] == 0

    def test_other_meals_unaffected_after_delete(self, db):
        meal_id = db.save_meal("user1", SAMPLE_ITEMS, "breakfast", meal_date="2026-01-01")
        db.save_meal("user1", SAMPLE_ITEMS, "lunch", meal_date="2026-01-01")
        db.delete_meal(meal_id)
        totals = db.get_daily_totals("user1", "2026-01-01")
        assert totals["meal_count"] == 1


class TestGetMealsForDay:
    def test_returns_empty_list_for_unknown_date(self, db):
        result = db.get_meals_for_day("user1", "2026-01-01")
        assert result == []

    def test_saved_meal_appears_in_day(self, db):
        db.save_meal("user1", SAMPLE_ITEMS, "banana", meal_date="2026-01-01")
        meals = db.get_meals_for_day("user1", "2026-01-01")
        assert len(meals) == 1
        assert meals[0]["input_text"] == "banana"

    def test_meal_items_nested_in_result(self, db):
        db.save_meal("user1", SAMPLE_ITEMS, "banana", meal_date="2026-01-01")
        meals = db.get_meals_for_day("user1", "2026-01-01")
        assert "items" in meals[0]
        assert len(meals[0]["items"]) == 1
        assert meals[0]["items"][0]["item_name"] == "banana"

    def test_deleted_meal_not_returned(self, db):
        meal_id = db.save_meal("user1", SAMPLE_ITEMS, "banana", meal_date="2026-01-01")
        db.delete_meal(meal_id)
        meals = db.get_meals_for_day("user1", "2026-01-01")
        assert meals == []


class TestDeleteMealItem:
    def test_soft_deletes_single_item(self, db):
        meal_id = db.save_meal("user1", TWO_ITEMS, "banana and egg", meal_date="2026-01-01")
        meals = db.get_meals_for_day("user1", "2026-01-01")
        item_id = meals[0]["items"][0]["item_id"]
        db.delete_meal_item(item_id, meal_id)
        meals = db.get_meals_for_day("user1", "2026-01-01")
        assert len(meals[0]["items"]) == 1

    def test_meal_totals_recalculated_after_item_delete(self, db):
        meal_id = db.save_meal("user1", TWO_ITEMS, "banana and egg", meal_date="2026-01-01")
        meals = db.get_meals_for_day("user1", "2026-01-01")
        banana_item = next(i for i in meals[0]["items"] if i["item_name"] == "banana")
        db.delete_meal_item(banana_item["item_id"], meal_id)
        totals = db.get_daily_totals("user1", "2026-01-01")
        assert totals["calories"] == pytest.approx(78.0, abs=0.1)

    def test_deleting_last_item_soft_deletes_meal(self, db):
        meal_id = db.save_meal("user1", SAMPLE_ITEMS, "banana", meal_date="2026-01-01")
        meals = db.get_meals_for_day("user1", "2026-01-01")
        item_id = meals[0]["items"][0]["item_id"]
        db.delete_meal_item(item_id, meal_id)
        assert db.get_meals_for_day("user1", "2026-01-01") == []


class TestUpdateMealItemGrams:
    def test_scales_calories_proportionally(self, db):
        # banana: 120g, 106.8 kcal → scale to 240g → 213.6 kcal
        meal_id = db.save_meal("user1", SAMPLE_ITEMS, "banana", meal_date="2026-01-01")
        meals = db.get_meals_for_day("user1", "2026-01-01")
        item_id = meals[0]["items"][0]["item_id"]
        db.update_meal_item_grams(item_id, meal_id, 240.0)
        meals = db.get_meals_for_day("user1", "2026-01-01")
        assert meals[0]["items"][0]["calories"] == pytest.approx(213.6, abs=0.5)

    def test_meal_totals_updated_after_gram_change(self, db):
        meal_id = db.save_meal("user1", SAMPLE_ITEMS, "banana", meal_date="2026-01-01")
        meals = db.get_meals_for_day("user1", "2026-01-01")
        item_id = meals[0]["items"][0]["item_id"]
        db.update_meal_item_grams(item_id, meal_id, 240.0)
        totals = db.get_daily_totals("user1", "2026-01-01")
        assert totals["calories"] == pytest.approx(213.6, abs=0.5)


class TestSavePipelineResult:
    def test_returns_meal_id(self, db):
        meal_id = db.save_pipeline_result(
            PIPELINE_RERANKED, uid="user1", input_text="apple", meal_date="2026-01-01"
        )
        assert isinstance(meal_id, str)
        assert len(meal_id) > 0

    def test_nutrition_persisted(self, db):
        db.save_pipeline_result(
            PIPELINE_RERANKED, uid="user1", input_text="apple", meal_date="2026-01-01"
        )
        totals = db.get_daily_totals("user1", "2026-01-01")
        assert totals["calories"] == pytest.approx(94.0, abs=0.1)
