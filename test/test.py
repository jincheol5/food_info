import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

model=AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    trust_remote_code=True,
    dtype=torch.float16, 
    device_map="auto"  
)
tokenizer=AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    trust_remote_code=True
)

pipe=pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7
)

hf_llm=HuggingFacePipeline(pipeline=pipe)
chat_model=ChatHuggingFace(llm=hf_llm)

prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant."),
    ("user","{question}")
])

chain=prompt | chat_model

response=chain.invoke({
    "question": "Explain artificial intelligence in simple terms."
})

print("=== Model Response ===")
print(response.content)