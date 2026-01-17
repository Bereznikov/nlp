from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


@dataclass
class BertBundle:
    model_name: str
    tokenizer: any
    model: any


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int = 128):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def finetune_bert(
    x_train: List[str],
    y_train: np.ndarray,
    x_val: List[str],
    y_val: np.ndarray,
    model_name: str = "DeepPavlov/rubert-base-cased",
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5,
    out_dir: str = "./models/bert",
) -> BertBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = HFDataset(x_train, y_train, tokenizer)
    val_ds = HFDataset(x_val, y_val, tokenizer)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        dataloader_pin_memory=False
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    return BertBundle(model_name=model_name, tokenizer=tokenizer, model=model)


@torch.no_grad()
def predict_bert(bundle, texts: List[str], max_len: int = 128, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает:
      y_pred: (n,)
      y_prob: (n,2) вероятности классов
    Исправление: переносим входные тензоры на device модели (cpu/mps/cuda).
    """
    model = bundle.model
    tokenizer = bundle.tokenizer

    model.eval()

    # Определяем device, на котором сейчас модель (после Trainer она часто на mps)
    device = next(model.parameters()).device

    all_probs = []
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )

        # ВАЖНО: переносим входы на устройство модели
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1)

        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    return y_pred, y_prob
