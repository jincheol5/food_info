from .schema import NutritionSchema
from langchain_core.output_parsers import PydanticOutputParser

# def build_nutrition_user_prompt(raw_text:str):
#     parser=PydanticOutputParser(pydantic_object=NutritionSchema)
#     format_instructions=parser.get_format_instructions()

#     return f"""
# Extract nutrition facts from OCR text.

# {format_instructions}

# OCR text:
# {raw_text}
# """

def build_nutrition_user_prompt():
    parser=PydanticOutputParser(pydantic_object=NutritionSchema)
    format_instructions=parser.get_format_instructions()

    return f"""
Extract nutrition facts.

{format_instructions}
"""
