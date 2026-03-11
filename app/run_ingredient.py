import json
import argparse
from pathlib import Path
from modules import IngredientSchema,State,get_ingredient_graph
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser

def main(app_config:dict):
    """
    """
    vlm=ChatOllama(
        model=app_config['model_name'],
        base_url=f"http://localhost:{app_config['port']}"
    )
    parser=PydanticOutputParser(pydantic_object=IngredientSchema)
    image_path=str(Path(__file__).resolve().parent.parent / "food" / "ingredient" / f"{app_config['image_name']}.png")

    state:State={
        "vlm":vlm,
        "parser":parser,
        "image_path":image_path,
        "vlm_output":"",
        "parserd_json":{},
        "valid":False
    }
    graph=get_ingredient_graph() # CompiledGraph
    result=graph.invoke(state)
    print(
        json.dumps(
            result["parserd_json"].model_dump(),
            indent=2,
            ensure_ascii=False
            )
    )

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="qwen3.5:0.8b")
    parser.add_argument("--port",type=int,default=11434)
    parser.add_argument("--image_name",type=str,default="ex_1")
    args=parser.parse_args()
    app_config={
        # app 관련
        'model_name':args.model_name,
        'port':args.port,
        'image_name':args.image_name
    }
    main(app_config=app_config)