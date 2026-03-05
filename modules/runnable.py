import os
import torch
from langchain_core.runnables import Runnable
from .utils import ModelUtils
from .prompts import build_nutrition_user_prompt

class OCR_VLM_Runnable(Runnable):
    def __init__(self,model_config:dict):
        """
        VLM: DeepSeek-OCR (small)
        """
        self.model=ModelUtils.load_local_llm(
            model_name="DeepSeek-OCR",
            model_type="base",
            model_config=model_config
        )
        self.model=self.model.to(model_config['device_map'])
        self.tokenizer=ModelUtils.load_local_tokenizer(
            model_name="DeepSeek-OCR",
            model_config=model_config
        )
        self.output_path=model_config['output_path']
        os.makedirs(self.output_path,exist_ok=True) 

    def set_output_path(self,output_path):
        self.output_path=output_path

    def invoke(self,image_path,config=None):
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

        result_file=self.output_path / "result.mmd"
        with open(result_file,"r",encoding="utf-8") as f:
            raw_text=f.read()
        return raw_text

class Nutrition_LLM_Runnable(Runnable):
    def __init__(self,model_config:dict):
        """
        LLM: Qwen2.5-1.5B-Instruct
        """
        self.model=ModelUtils.load_local_llm(
            model_name="Qwen2.5-1.5B-Instruct",
            model_type="causal",
            model_config=model_config
        )
        self.model=self.model.to(model_config['device_map'])
        self.tokenizer=ModelUtils.load_local_tokenizer(
            model_name="Qwen2.5-1.5B-Instruct",
            model_config=model_config
        )

    def invoke(self,raw_text:str,config=None):
        """
        """
        messages=[
            {"role":"system","content":"You are an information extraction system."},
            {"role":"user","content":build_nutrition_user_prompt(raw_text=raw_text)}
        ]
        inputs=self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        outputs=self.model.generate(**inputs,max_new_tokens=1000)
        result=self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True)
        return result