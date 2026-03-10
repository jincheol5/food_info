from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from modules import build_ingredient_user_prompt
from pathlib import Path

image_path=str(Path(__file__).resolve().parent.parent / "food" / "ingredient" / "2087686023125.png")
# 2087686023125
# 2087686040757

vlm=ChatOllama(
    model="qwen3.5:9b",
    base_url="http://localhost:11435"
)

msg=HumanMessage(
    content=[
        {"type":"text","text":build_ingredient_user_prompt()},
        {
            "type":"image_url",
            "image_url":{
                "url":image_path
            }
        }
    ]
)
response=vlm.invoke([msg])
print(response.content)