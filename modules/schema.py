from pydantic import BaseModel,Field,ConfigDict
from typing import List
from enum import Enum

class ServingUnit(str,Enum):
    g="g"
    ml="ml"

class NutritionUnit(str,Enum):
    g="g"
    mg="mg"
    kcal="kcal"

class ServingSize(BaseModel):
    value:float=Field(default=0.0)
    unit:ServingUnit=Field(default=ServingUnit.g)

class NutritionValue(BaseModel):
    value:float=Field(default=0.0)
    unit:NutritionUnit=Field(default=NutritionUnit.g)

class NutritionSchema(BaseModel):
    totalServingSize:ServingSize=Field(default_factory=ServingSize)
    servingSize:ServingSize=Field(default_factory=ServingSize)
    calories:NutritionValue=Field(default_factory=lambda:NutritionValue(unit=NutritionUnit.kcal))
    sodium:NutritionValue=Field(default_factory=lambda:NutritionValue(unit=NutritionUnit.mg))
    carbohydrate:NutritionValue=Field(default_factory=NutritionValue)
    sugar:NutritionValue=Field(default_factory=NutritionValue)
    fat:NutritionValue=Field(default_factory=NutritionValue)
    transFat:NutritionValue=Field(default_factory=NutritionValue)
    saturatedFat:NutritionValue=Field(default_factory=NutritionValue)
    cholesterol:NutritionValue=Field(default_factory=lambda:NutritionValue(unit=NutritionUnit.mg))
    protein:NutritionValue=Field(default_factory=NutritionValue)
    model_config=ConfigDict(extra="forbid")

class IngredientSchema(BaseModel):
    ingredients:str=""
    allergens:List[str]=Field(default_factory=list)
    model_config=ConfigDict(extra="forbid")