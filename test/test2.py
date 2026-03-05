import torch
from modules import DeepSeek_OCR_Runnable
from pathlib import Path

def main(config:dict):
    vlm_runnable=DeepSeek_OCR_Runnable(config=config)
    image_path=Path(__file__).resolve().parent.parent / "food" / "nutrition_1.png"
    raw_text=vlm_runnable.invoke(image_path=image_path)
    print(raw_text)

if __name__=="__main__":
    config={
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "use_safetensors": True,
        "output_path": Path(__file__).resolve().parent.parent / "ocr_results"
    }
    main(config=config)