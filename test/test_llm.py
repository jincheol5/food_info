import torch
import argparse
from pathlib import Path
from modules import ModelUtils

def main(question:str):
    """
    """
    model_config={
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda:0",
        "trust_remote_code": True,
        "use_safetensors": True
    }

    model=ModelUtils.load_local_llm(
        model_name="Qwen2.5-1.5B-Instruct",
        model_type="causal",
        model_config=model_config
    )
    model=model.to(model_config['device_map'])
    tokenizer=ModelUtils.load_local_tokenizer(
        model_name="Qwen2.5-1.5B-Instruct",
        model_config=model_config
    )

    messages=[
        {"role":"system","content":"You are an AI assistant. Answer my questions kindly and accurately."},
        {"role":"user","content":question}
    ]
    inputs=tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs=model.generate(**inputs,max_new_tokens=1000)
    result=tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True)
    print(f"<<LLM result>>")
    print(result)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--question",type=str,default=f"AI에 대해 설명해줘.")
    args=parser.parse_args()
    main(question=args.question)