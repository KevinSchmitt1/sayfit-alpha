from pydantic import BaseModel, Field

class MealCreate(BaseModel):
    uid: str = Field(..., description="User identifier")
    text: str = Field(..., description="Raw transcription text, e.g. 'i ate a pepperoni pizza and 3 eggs'")

class FoodItem(BaseModel):
    item_id: str = Field(..., description="Unique identifier for the food item")
    item_name: str = Field(..., description="Name of the food item")
    calories: float = Field(..., description="Calories in the food item")
    protein: float = Field(..., description="Protein content in grams")
    fat: float = Field(..., description="Fat content in grams")
    carbs: float = Field(..., description="Carbohydrate content in grams")
    grams: float = Field(..., description="Weight of the food item in grams")

class Meal(BaseModel):
    meal_id: str = Field(..., description="Unique identifier for the meal")
    date_time: str = Field(..., description="Date and time of the meal in ISO format")
    items: list[FoodItem] = Field(..., description="List of food items in the meal")
    uid: str = Field(..., description="User identifier")
