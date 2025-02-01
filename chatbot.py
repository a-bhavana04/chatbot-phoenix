from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import HuggingFacePipeline
import torch
import os

# Set up the offload directory for model files
offload_dir = "./model_offload"
os.makedirs(offload_dir, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Model loading and setup
try:
    peft_model_id = "Rakesh7n/Qwen2.5-0.5_alpaca-finance_finetuned"
    
    model = AutoModelForCausalLM.from_pretrained(
        peft_model_id, 
        torch_dtype=torch.float16, 
        device_map='auto',
        offload_folder=offload_dir
    )
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}") from e

# Configuration for text generation
generation_config = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_config
)

# Set up the LLM, prompt, and memory
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

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint for service health verification"""
    return jsonify({"status": "healthy"}), 200

@app.route("/chat", methods=["POST"])
def chat_completion():
    """Chat Completion Endpoint"""
    try:
        data = request.json
        prompt_text = data.get("prompt", "")
        if not prompt_text:
            return jsonify({"error": "No prompt provided"}), 400
        
        response = chain.invoke({"input": prompt_text})
        response_text = response['response'].split('Assistant:')[-1].strip()

        return jsonify({"response": response_text}), 200
    except Exception as e:
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
