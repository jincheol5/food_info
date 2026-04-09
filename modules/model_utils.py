from pydantic import ValidationError
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from .prompts import Prompts
from .schema import NutritionSchema

class ModelUtils:
    @staticmethod
    def get_classifier_message(image_path:str):
        system_msg=SystemMessage(content=Prompts.FOOD_IMG_CLASSIFIER_SYSTEM_PROMPT)
        human_msg=HumanMessage(
            content=[
                {
                    "type":"text",
                    "text":Prompts.FOOD_IMG_CLASSIFIER_HUMAN_PROMPT
                },
                {
                    "type":"image_url",
                    "image_url":image_path
                }
            ]
        )
        classifier_message=[
            system_msg,
            human_msg
        ]
        return classifier_message

    @staticmethod
    def get_nutrition_message(image_path:str):
        system_msg=SystemMessage(content=Prompts.NUTRITION_SYSTEM_PROMPT)
        human_msg=HumanMessage(
            content=[
                {
                    "type":"text",
                    "text":Prompts.build_nutrition_human_prompt()
                },
                {
                    "type":"image_url",
                    "image_url":image_path
                }
            ]
        )
        nutrition_message=[
            system_msg,
            human_msg
        ]
        return nutrition_message

    @staticmethod
    def check_classifier_output(response:AIMessage):
        output=str(response.content).strip()
        if output not in {"0","1","2"}:
            raise ValueError(f"Invalid output: {output}")
        return output

    @staticmethod
    def check_nutrition_etl_output(response:AIMessage):
        output=str(response.content).strip()
        try:
            NutritionSchema.model_validate_json(output)
        except ValidationError:
            raise ValueError(f"Invalid output")
        return output