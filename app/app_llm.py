from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from modules import build_free_ocr_user_prompt,build_extract_nutrition_user_prompt
from pathlib import Path

image_path=str(Path(__file__).resolve().parent.parent / "food" / "nutrition" / "ex_1.png")
# image_path=str(Path(__file__).resolve().parent.parent / "food" / "ingredient" / "ex_1.png")
# 2087686023125
# 2087686040757
# 2700038864145

vlm=ChatOllama(
    model="qwen3.5:0.8b",
    temperature=0,
    base_url="http://localhost:11434"
)

msg=HumanMessage(
    content=[
        {"type":"text","text":build_free_ocr_user_prompt()},
        {
            "type":"image_url",
            "image_url":{
                "url":image_path
            }
        }
    ]
)
response=vlm.invoke([msg])
raw_text=response.content
print(raw_text)

llm=ChatOllama(
    model="qwen3.5:2b",
    temperature=0,
    base_url="http://localhost:11434"
)

msg=HumanMessage(
    content=[
        {"type":"text","text":build_extract_nutrition_user_prompt(raw_text)},
    ]
)
response=llm.invoke([msg])
print(response.content)