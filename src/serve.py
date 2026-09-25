"""FastAPI wrapper around the fine-tuned model. Run with: uv run uvicorn src.serve:app --reload"""

import json

import torch
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train import ADAPTER_OUT, BASE_MODEL, CATEGORY, build_messages, get_attributes, load_split

device = "cuda" if torch.cuda.is_available() else "cpu"

train_rows = [r for r in load_split("train") if r["category"] == CATEGORY]
ATTRIBUTES = get_attributes(train_rows)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto").to(device)
model = PeftModel.from_pretrained(base_model, ADAPTER_OUT).to(device)

app = FastAPI()


class ExtractRequest(BaseModel):
    title: str
    description: str


def generate_json(title: str, description: str, max_new_tokens: int = 300) -> str:
    messages = build_messages(title, description, ATTRIBUTES)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


@app.post("/extract")
def extract(request: ExtractRequest) -> dict:
    raw = generate_json(request.title, request.description)
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start : end + 1])
