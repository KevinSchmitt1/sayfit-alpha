import datetime
from contextlib import asynccontextmanager
from pathlib import Path

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from api.recipes import router as recipes_router
from api.transcribe import router as transcribe_router
from api.schemas import FoodItem, ItemCreate, ItemPatch, Meal, MealCreate, MealHistory
from main import run_pipeline
from step2_retrieval.retriever import retrieve
from step5_database.database import get_db


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Validate required files and initialise the database before serving."""
    print("Server is starting — checking mandatory files...")

    # Initialise SQLite DB (creates file + tables if they don't exist yet)
    db_path = Path("data/sayfit_meals.db")
    db_existed = db_path.exists()
    get_db()  # sqlite3.connect creates the file; _ensure_db() creates tables
    print("Database file found." if db_existed else "Database file not found — created fresh at 'data/sayfit_meals.db'.")

    # FAISS index must be present — shipped by sayfit-data-repo, not built here
    index_file = Path("data/faiss_index/food.index")
    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{index_file}'. "
            "Run sayfit-data-repo to build and place the index, then restart."
        )
    print("FAISS index found.")

    instrumentator.expose(app)
    yield

app = fastapi.FastAPI(lifespan=lifespan)
app.include_router(recipes_router)
app.include_router(transcribe_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
instrumentator = Instrumentator().instrument(app)

PIPELINE_DURATION = Histogram(
    "pipeline_duration_seconds",
    "End-to-end duration of run_pipeline()",
    buckets=[1, 2, 5, 10, 15, 20, 30, 60],
)

PIPELINE_ERRORS = Counter(
    "pipeline_errors_total",
    "Pipeline failures by error type",
    ["error_type"],
)

@app.post("/log", response_model=Meal)
def log_meal(meal: MealCreate):
    """Run pipeline on text input, save to DB, return structured meal."""
    with PIPELINE_DURATION.time():
        try:
            result = run_pipeline(
                text=meal.text,
                date_time=datetime.datetime.now().isoformat(),
                uid=meal.uid,
            )
        except Exception as e:
            PIPELINE_ERRORS.labels(error_type=type(e).__name__).inc()
            raise

    saved = get_db().save_pipeline_result(
        reranked=result,
        uid=meal.uid,
        input_text=meal.text,
    )
    meal_id = saved["meal_id"]
    item_ids = saved["item_ids"]

    items = [
        FoodItem(
            item_id=item_ids[i],
            item_name=item.get("matched_name", item.get("item_name", "")),
            calories=item["nutrition"]["calories"],
            protein=item["nutrition"]["protein"],
            fat=item["nutrition"]["fat"],
            carbs=item["nutrition"]["carbs"],
            grams=item["amount_grams"],
        )
        for i, item in enumerate(result["results"])
    ]

    return Meal(
        meal_id=meal_id,
        date_time=datetime.datetime.now().isoformat(),
        uid=meal.uid,
        items=items,
    )


@app.get("/meals/{uid}/today", response_model=list[Meal])
def get_today(uid: str):
    today = datetime.date.today().isoformat()
    meals = get_db().get_meals_for_day(uid, today)
    return [
        Meal(
            meal_id=m["meal_id"],
            date_time=m["logged_at"],
            uid=uid,
            items=[
                FoodItem(
                    item_id=item["item_id"],
                    item_name=item["matched_name"] or item["item_name"],
                    calories=item["calories"],
                    protein=item["protein"],
                    fat=item["fat"],
                    carbs=item["carbs"],
                    grams=item["amount_grams"],
                )
                for item in m["items"]
            ],
        )
        for m in meals
    ]


@app.get("/meals/{uid}", response_model=MealHistory)
def get_history(uid: str, days: int = 30):
    return get_db().get_stats(uid, days=days)


@app.patch("/meals/{uid}/items/{item_id}", status_code=204)
def patch_item(uid: str, item_id: str, body: ItemPatch):
    get_db().update_meal_item_grams(item_id, body.meal_id, body.grams)


@app.delete("/meals/{uid}/items/{item_id}", status_code=204)
def delete_item(uid: str, item_id: str, meal_id: str):
    get_db().delete_meal_item(item_id, meal_id)


@app.delete("/meals/{uid}/{meal_id}", status_code=204)
def delete_meal(uid: str, meal_id: str):
    get_db().delete_meal(meal_id)


@app.post("/meals/{uid}/items", response_model=FoodItem, status_code=201)
def add_item(uid: str, body: ItemCreate):
    retrieval = retrieve([body.item_name])
    candidates = retrieval["items"][0]["candidates"] if retrieval.get("items") else []

    if not candidates:
        raise fastapi.HTTPException(
            status_code=404,
            detail=f"No nutrition data found for '{body.item_name}'",
        )

    top = candidates[0]
    nutr = top.get("nutrition_per_100g", {})
    scale = body.grams / 100.0

    calories = round((nutr.get("calories") or 0) * scale, 1)
    protein = round((nutr.get("protein") or 0) * scale, 1)
    fat = round((nutr.get("fat") or 0) * scale, 1)
    carbs = round((nutr.get("carbs") or 0) * scale, 1)
    matched_name = top.get("item_name", body.item_name)

    item_id = get_db().add_meal_item(
        meal_id=body.meal_id,
        item_name=body.item_name,
        matched_name=matched_name,
        amount_grams=body.grams,
        calories=calories,
        protein=protein,
        fat=fat,
        carbs=carbs,
    )

    return FoodItem(
        item_id=item_id,
        item_name=matched_name,
        calories=calories,
        protein=protein,
        fat=fat,
        carbs=carbs,
        grams=body.grams,
    )