import os
from typing_extensions import Literal
from transformers import AutoModelForCausalLM,AutoModel,AutoTokenizer

class ModelUtils:
    dir_path=os.path.join('..','data','llm')
    @staticmethod
    def save_pretrained_llm_from_HF(HF_path:str,model_name:str,model_type:Literal['base','causal'],config:dict):
        """
        config
            dtype: "auto"
            device_map: "auto"
            trust_remote_code: bool
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        os.makedirs(model_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성
        match model_type:
            case 'base':
                model=AutoModel.from_pretrained(
                    pretrained_model_name_or_path=HF_path,
                    dtype=config['dtype'],
                    device_map=config['device_map'],
                    trust_remote_code=config['trust_remote_code']
                )
            case 'causal':
                model=AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=HF_path,
                    dtype=config['dtype'],
                    device_map=config['device_map'],
                    trust_remote_code=config['trust_remote_code']
                )
        model.save_pretrained(model_path)
        print(f"Save pretrained causal llm: {model_name}!")

    @staticmethod
    def save_tokenizer_from_HF(HF_path:str,model_name:str,config:dict):
        """
        """
        tokenizer_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        os.makedirs(tokenizer_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=HF_path,
            trust_remote_code=config['trust_remote_code']
        )
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Save tokenizer for pretrained causal llm: {model_name}!")
    
    @staticmethod
    def load_local_llm(model_name:str,model_type:Literal['base','causal'],config:dict):
        """
        config
            dtype: torch.bfloat16 (GPU)
            device_map: "auto"
            trust_remote_code: bool
            use_safetensors: bool
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        match model_type:
            case 'base':
                model=AutoModel.from_pretrained(
                    model_path, 
                    dtype=config['dtype'],
                    device_map=config['device_map'],
                    trust_remote_code=config['trust_remote_code'],
                    use_safetensors=config['use_safetensors']
            case 'causal':
                model=AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    dtype=config['dtype'],
                    device_map=config['device_map'],
                    trust_remote_code=config['trust_remote_code'],
                    use_safetensors=config['use_safetensors']
                )
        return model

    @staticmethod
    def load_local_tokenizer(model_name:str,config:dict):
        """
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            trust_remote_code=config['trust_remote_code']
        )
        return tokenizer