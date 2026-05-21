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
