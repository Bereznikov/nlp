"""
Вывод текста с использованием сохраненной модели HuggingFace
Предполагается, что модель сохранена с помощью `save_pretrained` в папку models/bert/

Использование:
python -m src.predict_text --model-dir models/bert --text "Это отличный товар"
"""

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    enc = tok(args.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    label = "positive" if pred == 1 else "negative"

    print({"label": label, "prob_pos": float(probs[1]), "prob_neg": float(probs[0])})


if __name__ == "__main__":
    main()
