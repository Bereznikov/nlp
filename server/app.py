from __future__ import annotations

import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from faster_whisper import WhisperModel

app = FastAPI(title="TG Audio Sentiment Server")

MODEL_DIR = os.getenv("MODEL_DIR", "models/bert")
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")

# Load classifier
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
clf = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
clf.eval()

asr = WhisperModel(WHISPER_SIZE, device=WHISPER_DEVICE)


def classify_text(text: str) -> dict:
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = clf(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return {
        "label_int": pred,
        "label": "positive" if pred == 1 else "negative",
        "prob_pos": float(probs[1]),
        "prob_neg": float(probs[0]),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "voice.ogg")[1] or ".ogg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments, info = asr.transcribe(tmp_path, language="ru")
        text = " ".join([s.text.strip() for s in segments]).strip()
        if not text:
            return JSONResponse(status_code=500, content={"error": "ASR_failed"})
        result = classify_text(text)
        result["text"] = text
        return result
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
