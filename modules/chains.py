from pydantic import ValidationError
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from .model_utils import ModelUtils
from .schema import NutritionSchema

class FoodInfoChain:
    """
    """
    @staticmethod
    def get_food_img_classify_chain(model_name:str=f"qwen3.5:35b",port:int=11434):
        prompt=RunnableLambda(ModelUtils.get_nutrition_message)
        model=ChatOllama(
            model=model_name,
            base_url=f"http://localhost:{port}"
        )
        check_output=RunnableLambda(ModelUtils.check_classifier_output)
        food_img_classify_chain=(prompt|model|check_output).with_retry(
            stop_after_attempt=3,
            retry_if_exception_type=(ValueError,)
        )
        return food_img_classify_chain

    @staticmethod
    def get_nutrition_extract_chain(model_name:str=f"qwen3.5:35b",port:int=11434):
        prompt=RunnableLambda(ModelUtils.get_nutrition_message)
        model=ChatOllama(
            model=model_name,
            base_url=f"http://localhost:{port}"
        )
        model=model.with_structured_output(NutritionSchema) # 성공 시 결과값 NutritionSchema 객체, 실패 시 ValidationError
        nutrition_extract_chain=(prompt|model).with_retry(
            stop_after_attempt=3,
            retry_if_exception_type=(ValidationError,)
        )
        return nutrition_extract_chain