from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from modules import build_nutrition_user_prompt
from pathlib import Path

image_path=str(Path(__file__).resolve().parent.parent / "food" / "nutrition" / "8801045500614.png")
# 8800279679073
# 8800336394796
# 8801045500614


vlm=ChatOllama(
    model="qwen3-vl:30b",
    base_url="http://localhost:11435"
)

msg=HumanMessage(
    content=[
        {"type":"text","text":build_nutrition_user_prompt()},
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