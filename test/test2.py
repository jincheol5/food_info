import torch
from modules import ModelUtils
from pathlib import Path
from langchain_core.runnables import RunnableLambda

def run_deepseek_ocr(inputs:dict):
    """
    inputs
        model
        tokenizer
        image_path
    """
    model=inputs['model']
    tokenizer=inputs['tokenizer']
    image_path=inputs['image_path']
    prompt="<image>\nFree OCR."
    output_path=Path(__file__).resolve().parent.parent / "ocr_results"

    with torch.no_grad():
        model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_path,
            base_size=640,   
            image_size=640,
            crop_mode=False,
            save_results=True
        )

    print(f"<<OCR result>>",end="\n\n")
    result_file=output_path / "result.mmd"
    with open(result_file,"r",encoding="utf-8") as f:
        raw_text=f.read()
    return raw_text

def main(config:dict):
    model=ModelUtils.load_local_llm(
        model_name="DeepSeek-OCR",
        model_type="base",
        config=config
    )
    model=model.eval().cuda()

    tokenizer=ModelUtils.load_local_tokenizer(
        model_name="DeepSeek-OCR",
        config=config
    )

    image_path=Path(__file__).resolve().parent.parent / "food" / "nutrition_1.png"
    inputs={
        "model":model,
        "tokenizer":tokenizer,
        "image_path":image_path
    }

    ocr_runnable=RunnableLambda(run_deepseek_ocr)
    result=ocr_runnable.invoke(inputs)
    print(result)

if __name__=="__main__":
    config={
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "use_safetensors": True
    }
    main(config=config)