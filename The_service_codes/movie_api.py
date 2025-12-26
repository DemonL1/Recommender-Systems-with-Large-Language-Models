from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from fastapi.middleware.cors import CORSMiddleware  # Added: CORS dependency

# Initialize FastAPI
app = FastAPI(title="Movie Recommendation API")

# Added: Allow all CORS requests (for local testing, no security risk)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from all frontend domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Model path (merged model)
model_path = "/root/autodl-tmp/qwen3-movie-merged"

# Load model and Tokenizer (loaded once at startup)
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
print("Model loaded successfully!")

# Define request body format
class RecommendRequest(BaseModel):
    user_input: str  # User preferences passed from frontend
    max_length: int = 500
    temperature: float = 0.6

# Define API endpoint
@app.post("/recommend")
def recommend_movies(request: RecommendRequest):
    try:
        inputs = tokenizer(request.user_input, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            top_p=0.95,
            do_sample=True
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"status": "success", "recommendation": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}