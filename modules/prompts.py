import textwrap
from .schema import NutritionSchema,IngredientSchema
from langchain_core.output_parsers import PydanticOutputParser

class Prompts:
    FOOD_IMG_CLASSIFIER_SYSTEM_PROMPT=textwrap.dedent(f"""
        You are a strict image text classifier.

        Your task is to classify the image based on the presence of specific types of text.

        Definitions:
        - Ingredient text: "원재료", "원재료명", "원재료 및 함량", "Ingredients", "Ingredient"

        - Nutrition text: "영양정보", "영양성분", "Nutrition Facts", "칼로리", "탄수화물", "단백질", "지방"

        Classification rules:
        1. If ingredient-related text appears anywhere in the image, then output 0
        2. Else if nutrition-related text appears, then output 1
        3. Else, then output 2

        Important:
        - If both ingredient text and nutrition text appear, output 1

        Constraints:
        - Output only one number: 0, 1, or 2
        - No explanation
        - No reasoning
        - No additional text
        - Do not describe the image
        - Do not guess unreadable text
        """).strip()
    
    @staticmethod
    def build_img_classifier_system_retry_prompt(invalid_case=None):
        if invalid_case is None:
            feedback=textwrap.dedent(f"""
            """)
        else:
            feedback=textwrap.dedent(f"""
                Previous attempt feedback:
                - {invalid_case}
                - Valid outputs are only: 0, 1, 2.
                - Return exactly one digit.
            """)
        return textwrap.dedent(f"""
        You are a strict image text classifier.

        Your task is to classify the image based on the presence of specific types of text.

        Definitions:
        - Ingredient text: "원재료", "원재료명", "원재료 및 함량", "Ingredients", "Ingredient"

        - Nutrition text: "영양정보", "영양성분", "Nutrition Facts", "칼로리", "탄수화물", "단백질", "지방"

        Classification rules:
        1. If ingredient-related text appears anywhere in the image, then output 0
        2. Else if nutrition-related text appears, then output 1
        3. Else, then output 2

        Important:
        - If both ingredient text and nutrition text appear, output 1

        Constraints:
        - Output only one number: 0, 1, or 2
        - No explanation
        - No reasoning
        - No additional text
        - Do not describe the image
        - Do not guess unreadable text
    
        {feedback}
        """).strip()

    FOOD_IMG_CLASSIFIER_HUMAN_PROMPT=f"""Classify the image."""

    NUTRITION_SYSTEM_PROMPT=textwrap.dedent("""
        You are a nutrition facts extraction system.

        Your task is to extract nutrition information from the given input accurately and consistently.

        General Rules:
        - Always return the result as a JSON object.
        - Do not include any explanations, comments, markdown, or extra text.
        - Do not output code fences (e.g., ```).
        - Only output the final JSON.

        Extraction Rules:
        - Extract only explicitly available information. Do not guess or infer missing values.
        - If a value is missing, unreadable, or uncertain, use the default value.
        - Use numeric values (float) for all numbers. Do not use strings for numbers.
        - Use only valid units provided in the schema.
        - Do not invent units or fields.

        Behavior:
        - Be precise and conservative.
        - If unsure, return default values rather than guessing.
        - Ensure the output is always valid and parseable JSON.
        """).strip()

    @staticmethod
    def build_nutrition_user_prompt():
        parser=PydanticOutputParser(pydantic_object=NutritionSchema)
        format_instructions=parser.get_format_instructions()
        return textwrap.dedent(f"""
            Extract nutrition facts.

            {format_instructions}
            """).strip()