from typing import Optional

from pydantic import BaseModel, Field


class MealCreate(BaseModel):
    uid: str = Field(..., description="User identifier")
    text: str = Field(..., description="Raw transcription text, e.g. 'i ate a pepperoni pizza and 3 eggs'")


class FoodItem(BaseModel):
    item_id: str = Field(..., description="Unique identifier for the food item")
    item_name: str = Field(..., description="Name of the food item")
    calories: float = Field(..., description="Calories in kcal")
    protein: float = Field(..., description="Protein in grams")
    fat: float = Field(..., description="Fat in grams")
    carbs: float = Field(..., description="Carbohydrates in grams")
    grams: float = Field(..., description="Portion size in grams")


class Meal(BaseModel):
    meal_id: str = Field(..., description="Unique identifier for the meal")
    date_time: str = Field(..., description="Logged timestamp in ISO format")
    items: list[FoodItem] = Field(..., description="Food items in the meal")
    uid: str = Field(..., description="User identifier")


class ItemPatch(BaseModel):
    meal_id: str = Field(..., description="ID of the meal containing this item")
    grams: float = Field(..., gt=0, description="New portion size in grams")


class ItemCreate(BaseModel):
    meal_id: str = Field(..., description="ID of the meal to add the item to")
    item_name: str = Field(..., description="Name of the food item to add")
    grams: float = Field(..., gt=0, description="Portion size in grams")


class DailyMacros(BaseModel):
    meal_date: str
    calories: float
    protein: float
    fat: float
    carbs: float
    meal_count: int


class MealHistory(BaseModel):
    user_id: str
    days: int
    daily_breakdown: list[DailyMacros]
    average_calories: float
    average_protein: float


class TranscribeResponse(BaseModel):
    text: str


# ── Recipe endpoints ──────────────────────────────────────────────────────────

class RecipePreferences(BaseModel):
    target_calories: Optional[int] = Field(None, description="Target kcal for this meal; None = use full remaining budget")
    taste: str = Field("any", description="'savory', 'sweet', or 'any'")
    max_time_minutes: Optional[int] = Field(None, description="Max prep time in minutes; None = no limit")
    ingredients: list[str] = Field(default_factory=list, description="Ingredients already on hand")
    few_ingredients: bool = Field(False, description="Limit results to max 5 ingredients")


class RecipeSuggestRequest(BaseModel):
    source: str = Field("spoonacular", description="'spoonacular', 'kaggle', or 'combo'")
    preferences: RecipePreferences = Field(default_factory=RecipePreferences)


class RecipeNutrition(BaseModel):
    calories: float
    protein: float
    fat: float
    carbs: float


class RecipeSuggestion(BaseModel):
    title: str
    fit_score: int = Field(..., description="Macro fit score 0–100")
    scale_factor: float = Field(..., description="How many servings of the original recipe this represents (> 1 when scaled up). The nutrition values already reflect this — do NOT pass this as portions to log-suggestion.")
    ready_in_minutes: int
    source_url: str
    ingredients: list[str]
    ingredient_count: int
    source: str = Field(..., description="'spoonacular' or 'kaggle'")
    nutrition: RecipeNutrition


class RecipeMacros(BaseModel):
    calories: float
    protein: float
    fat: float
    carbs: float


class RecipeSuggestResponse(BaseModel):
    remaining: RecipeMacros
    suggestions: list[RecipeSuggestion]
    message: Optional[str] = Field(None, description="Explanatory message when suggestions is empty")


class RecipeLogRequest(BaseModel):
    title: str = Field(..., description="Recipe title — used as the meal name")
    portions: float = Field(1.0, gt=0, description="Number of portions to log")
    nutrition: RecipeNutrition
