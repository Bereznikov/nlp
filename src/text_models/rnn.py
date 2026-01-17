from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё]+", text.lower())


@dataclass
class Vocab:
    stoi: Dict[str, int]

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(tok, 0) for tok in tokenize(text)]


def build_vocab(texts: List[str], max_words: int = 30000) -> Vocab:
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    most = c.most_common(max_words)
    stoi = {w: i + 1 for i, (w, _) in enumerate(most)}  # 0 = PAD/UNK
    return Vocab(stoi=stoi)


class SentDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, vocab: Vocab, max_len: int = 80):
        self.labels = labels.astype(np.int64)
        self.vocab = vocab
        self.max_len = max_len
        self.seqs = [self._pad(vocab.encode(t)) for t in texts]

    def _pad(self, seq: List[int]) -> List[int]:
        if len(seq) >= self.max_len:
            return seq[: self.max_len]
        return seq + [0] * (self.max_len - len(seq))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 128, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


@dataclass
class RNNBundle:
    vocab: Vocab
    model: BiLSTM


def train_rnn(
    x_train: List[str],
    y_train: np.ndarray,
    x_val: List[str],
    y_val: np.ndarray,
    max_words: int = 30000,
    max_len: int = 80,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> RNNBundle:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_vocab(x_train, max_words=max_words)

    train_ds = SentDataset(x_train, y_train, vocab, max_len=max_len)
    val_ds = SentDataset(x_val, y_val, vocab, max_len=max_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BiLSTM(vocab_size=len(vocab.stoi)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        model.eval()
        correct = 0
        n = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                n += len(yb)
        print(f"Epoch {ep}: train_loss={total/len(train_dl):.4f}, val_acc={correct/max(n,1):.4f}")

    return RNNBundle(vocab=vocab, model=model)


def predict_rnn(bundle: RNNBundle, texts: List[str], max_len: int = 80, device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle.model.eval()
    bundle.model.to(device)

    ds = SentDataset(texts, np.zeros(len(texts)), bundle.vocab, max_len=max_len)
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    preds = []
    prob_pos = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            logits = bundle.model(xb)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            preds.extend(pred.cpu().numpy().tolist())
            prob_pos.extend(probs[:, 1].cpu().numpy().tolist())

    return np.array(preds), np.array(prob_pos)
