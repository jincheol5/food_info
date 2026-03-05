import os
import torch
from langchain_core.runnables import Runnable
from .utils import ModelUtils

class DeepSeek_OCR_Runnable(Runnable):
    def __init__(self,config:dict):
        """
        """
        self.model=ModelUtils.load_local_llm(
            model_name="DeepSeek-OCR",
            model_type="base",
            config=config
        )
        self.tokenizer=ModelUtils.load_local_tokenizer(
            model_name="DeepSeek-OCR",
            config=config
        )
        self.output_path=config['output']
        os.makedirs(self.output_path,exist_ok=True) 

    def set_output_path(self,output_path):
        self.output_path=output_path

    def invoke(self,image_path):
        prompt="<image>\nFree OCR."
        with torch.no_grad():
            self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=self.output_path,
                base_size=640,   
                image_size=640,
                crop_mode=False,
                save_results=True
            )

        print(f"<<OCR result>>",end="\n\n")
        result_file=self.output_path / "result.mmd"
        with open(result_file,"r",encoding="utf-8") as f:
            raw_text=f.read()
        return raw_text

