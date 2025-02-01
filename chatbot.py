from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import HuggingFacePipeline
import torch
import os
offload_dir = "./model_offload"
os.makedirs(offload_dir, exist_ok=True)

app = FastAPI(title="Qwen Inference API")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

try:
    peft_model_id = "Rakesh7n/Qwen2.5-0.5_alpaca-finance_finetuned"
    
    model = AutoModelForCausalLM.from_pretrained(peft_model_id, torch_dtype=torch.float16, device_map='auto',offload_folder=offload_dir)
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}") from e

generation_config = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_config
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Adam, a helpful financial assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

memory = ConversationBufferMemory(return_messages=True)

chain = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

@app.get("/health")
def health_check():
    """Endpoint for service health verification"""
    return {"status": "healthy"}

@app.post("/chat")
def chat_completion(request: PromptRequest):
    """Chat Completion Endpoint"""
    try:
        response = chain.invoke({"input": request.prompt})
        return {"response": response['response'].split('Assistant:')[-1].strip()}
    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")
