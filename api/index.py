import os
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


app = FastAPI(title="Harassment Detection API")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = str(PROJECT_ROOT)
TOKENIZER_NAME = MODEL_DIR
MODEL_NAME = MODEL_DIR

try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
except Exception as exc:
    tokenizer = None
    model = None
    model_init_error = exc
else:
    model_init_error = None


class PredictRequest(BaseModel):
    comment: str


class PredictResponse(BaseModel):
    comment: str
    label: str
    confidence: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if tokenizer is None or model is None:
        raise RuntimeError(f"Model could not be initialized: {model_init_error}")

    inputs = tokenizer(payload.comment, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    confidence, pred_id = torch.max(probs, dim=1)
    pred_id = int(pred_id.item())
    confidence_value = float(confidence.item())

    label = model.config.id2label.get(pred_id, str(pred_id))

    return PredictResponse(
        comment=payload.comment,
        label=label,
        confidence=confidence_value,
    )
