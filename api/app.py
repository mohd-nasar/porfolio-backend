import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 150
    temperature: float = 0.7
    top_p: float = 0.9

# Load model from Hugging Face Hub
def load_model():
    MODEL_NAME = "nasar/pModel"  # Your HF model repo
    
    try:
        # Check if this is a PEFT model
        config = PeftConfig.from_pretrained(MODEL_NAME)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, MODEL_NAME)
    except:
        # Fallback to regular model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer

# Load model (cached between invocations)
model, tokenizer = load_model()

# Generation function with safeguards
def generate_text(params: GenerationRequest):
    try:
        inputs = tokenizer(params.prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=min(params.max_length, 200),
                temperature=params.temperature,
                top_p=params.top_p,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )

        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(500, "GPU memory exceeded - try shorter prompt")
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")

# API endpoint
@app.post("/generate")
async def generate_response(request: GenerationRequest):
    if not request.prompt or len(request.prompt) > 1000:
        raise HTTPException(400, "Prompt must be 1-1000 characters")
    
    try:
        response_text = generate_text(request)
        return {"output": response_text[:5000]}  # Ensure response <4.5MB
    except Exception as e:
        raise HTTPException(500, str(e))

# Health check
@app.get("/")
async def health_check():
    return {"status": "ready", "model": "nasar/pModel from Hugging Face"}

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
