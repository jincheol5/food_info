from typing_extensions import Literal
from langchain_core.messages import SystemMessage,HumanMessage
from .prompts import get_system_prompt_for_classifier,get_human_prompt_for_classifier

class ModelUtils:
    @staticmethod
    def get_classifier_message(image_path:str):
        system_msg=SystemMessage(content=get_system_prompt_for_classifier())
        human_msg=HumanMessage(
            content=[
                {
                    "type":"text",
                    "text":get_human_prompt_for_classifier()
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
    def get_batch_classifier_messages(image_paths:list):
        batch_messages=[]
        for image_path in image_paths:
            batch_messages.append(ModelUtils.get_classifier_message(image_path=image_path))
        return batch_messages
