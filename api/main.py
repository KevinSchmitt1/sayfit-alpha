import datetime

import fastapi

from api.schemas import FoodItem, Meal, MealCreate
from main import run_pipeline
from step5_database.database import get_db

app = fastapi.FastAPI()


@app.post("/log", response_model=Meal)
def log_meal(meal: MealCreate):
    """Run pipeline on text input, save to DB, return structured meal."""
    result = run_pipeline(
        text=meal.text,
        date_time=datetime.datetime.now().isoformat(),
        uid=meal.uid,
    )

    meal_id = get_db().save_meal(
        user_id=meal.uid,
        items=result["results"],
        input_text=meal.text,
    )

    items = [
        FoodItem(
            item_id=item.get("item_id", ""),
            item_name=item.get("matched_name", item.get("item_name", "")),
            calories=item["nutrition"]["calories"],
            protein=item["nutrition"]["protein"],
            fat=item["nutrition"]["fat"],
            carbs=item["nutrition"]["carbs"],
            grams=item["amount_grams"],
        )
        for item in result["results"]
    ]

    return Meal(
        meal_id=meal_id,
        date_time=datetime.datetime.now().isoformat(),
        uid=meal.uid,
        items=items,
    )
