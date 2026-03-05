import torch
import argparse
from modules import OCR_VLM_Runnable,Nutrition_LLM_Runnable,NutritionSchema
from pathlib import Path
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

def main(image_path):
    ocr_vlm_config={
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda:0",
        "trust_remote_code": True,
        "use_safetensors": True,
        "output_path": Path(__file__).resolve().parent.parent / "ocr_results"
    }
    nutri_llm_config={
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda:0",
        "trust_remote_code": True,
        "use_safetensors": True
    }
    
    ocr_vlm_runnable=OCR_VLM_Runnable(model_config=ocr_vlm_config)
    nutri_llm_runnable=Nutrition_LLM_Runnable(model_config=nutri_llm_config)
    parser=PydanticOutputParser(pydantic_object=NutritionSchema)
    
    chain=ocr_vlm_runnable | nutri_llm_runnable | parser
    result=None
    try:
        result=chain.invoke(image_path) # pydantic 객체
        result=result.model_dump(mode="json")
    except OutputParserException as e:
        print("Parsing failed:", e)
    
    print(f"<<Final Result>>",end="\n\n")
    print(result)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_name",type=str,default=f"nutri_1")
    args=parser.parse_args()
    image_path=Path(__file__).resolve().parent.parent / "food" / f"{args.image_name}.png"
    main(image_path=image_path)