from .schema import NutritionSchema,IngredientSchema
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

def build_ingredient_user_prompt():
    parser=PydanticOutputParser(pydantic_object=IngredientSchema)
    format_instructions=parser.get_format_instructions()

    return f"""
원재료 정보와 알레르기 유발 물질을 추출해.

Instructions:
1. ingredients: 식품 라벨의 "원재료" 항목을 그대로 복사한다.
2. allergens: "알레르기 유발물질" 항목에 표시된 재료들을 리스트로 추출한다.
3. 알레르기는 추론하지 말고 명시된 것만 추출한다.

{format_instructions}
"""