import torch
import argparse
from pathlib import Path
from modules import ModelUtils

def main(image_name:str):
    """
    """
    model_config={
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda:0",
        "trust_remote_code": True,
        "use_safetensors": True,
        "output_path": Path(__file__).resolve().parent.parent / "ocr_results"
    }

    model=ModelUtils.load_local_llm(
        model_name="DeepSeek-OCR",
        model_type="base",
        model_config=model_config
    )
    model=model.to(model_config['device_map'])
    tokenizer=ModelUtils.load_local_tokenizer(
        model_name="DeepSeek-OCR",
        model_config=model_config
    )
    prompt="<image>\nFree OCR."
    with torch.no_grad():
        model.infer(
            tokenizer,
            prompt=prompt,
            image_file=Path(__file__).resolve().parent.parent / "food" / f"{image_name}.png",
            output_path=model_config['output_path'],
            base_size=640,   
            image_size=640,
            crop_mode=False,
            save_results=True
        )

    result_file=model_config['output_path'] / "result.mmd"
    with open(result_file,"r",encoding="utf-8") as f:
        raw_text=f.read()
    print(f"<<OCR result>>")
    print(raw_text)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image_name",type=str,default=f"nutri_1")
    args=parser.parse_args()
    main(image_name=args.image_name)