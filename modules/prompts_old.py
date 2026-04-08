from .schema import NutritionSchema,IngredientSchema
from langchain_core.output_parsers import PydanticOutputParser

def build_free_ocr_user_prompt():
    return f"""
Extract all visible text from the image.
"""

def build_nutrition_user_prompt():
    parser=PydanticOutputParser(pydantic_object=NutritionSchema)
    format_instructions=parser.get_format_instructions()

    return f"""
Extract nutrition facts.

{format_instructions}
"""

def build_ingredient_user_prompt():
    parser=PydanticOutputParser(pydantic_object=IngredientSchema)
    format_instructions=parser.get_format_instructions()

    return f"""
Extract Ingredients and allergens.

{format_instructions}
"""