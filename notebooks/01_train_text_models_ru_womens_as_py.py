import os
import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split

from src.text_models.bert import finetune_bert, predict_bert
from src.text_models.rnn import train_rnn, predict_rnn
from src.text_models.classic import train_classic_models, predict_classic
from src.text_models.metrics import evaluate_binary



RAW_PATH = "../data/raw/Womens Clothing E-Commerce Reviews.csv"
PROC_EN = "../data/processed/dataset_en.csv"
PROC_RU = "../data/processed/dataset_ru.csv"

df_raw = pd.read_csv(RAW_PATH)
out = df_raw[["Review Text", "Rating"]].copy().dropna(subset=["Review Text"]) 
out["Rating"] = pd.to_numeric(out["Rating"], errors="coerce")
out = out.dropna(subset=["Rating"]) 
out = out[out["Rating"].between(1,5)]
out["text"] = out["Review Text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
out["label"] = (out["Rating"] >= 4).astype(int)
df_en = out[["text","label"]]

df_en.to_csv(PROC_EN, index=False)
print("Saved", PROC_EN, "rows=", len(df_en))
print(df_en["label"].value_counts())
df_en.head()

translator = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru")

def translate_batch(texts, batch_size=16):
    ru = []
    for i in range(0, len(texts), batch_size):
        res = translator(texts[i:i+batch_size])
        ru.extend([r["translation_text"] for r in res])
        if (i//batch_size) % 50 == 0:
            print(f"Translated {len(ru)}/{len(texts)}")
    return ru

max_rows = 5000  # 0 для полного датасета
work = df_en.copy()
if max_rows:
    work = work.head(max_rows)

work["text_ru"] = translate_batch(work["text"].tolist(), batch_size=16)
work.to_csv(PROC_RU, index=False)
print("Saved", PROC_RU)
work.head()
df = pd.read_csv(PROC_RU)

X = df["text_ru"].astype(str).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# далее ещё сделаем val из train для RNN/BERT
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print("Train:", len(X_tr), "Val:", len(X_val), "Test:", len(X_test))
bundle = train_classic_models(X_tr, y_tr)

results = {}
for name in bundle.models.keys():
    y_pred, y_prob = predict_classic(bundle, X_test, name)
    m = evaluate_binary(y_test, y_pred, y_prob)
    results[f"classic/{name}"] = m
    print(name, m.as_dict())

rnn_bundle = train_rnn(
    x_train=list(X_tr), y_train=y_tr,
    x_val=list(X_val), y_val=y_val,
    epochs=3
)

y_pred, y_prob = predict_rnn(rnn_bundle, list(X_test))
m = evaluate_binary(y_test, y_pred, y_prob)
results["rnn/bilstm"] = m
print(m.as_dict())

bert_bundle = finetune_bert(
    x_train=list(X_tr), y_train=y_tr,
    x_val=list(X_val), y_val=y_val,
    model_name="DeepPavlov/rubert-base-cased",
    epochs=1,
    out_dir="../models/bert_tmp"
)

y_pred, y_prob = predict_bert(bert_bundle, list(X_test))
m = evaluate_binary(y_test, y_pred, y_prob)
results["bert/rubert"] = m
print(m.as_dict())

rows = []
for name, m in results.items():
    d = m.as_dict()
    rows.append({
        "model": name,
        "accuracy": d["accuracy"],
        "macro_f1": d["macro_f1"],
        "weighted_f1": d["weighted_f1"],
        "roc_auc": d["roc_auc"],
        "pr_auc": d["pr_auc"],
        "cm": d["confusion"].tolist(),
    })

df_res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
df_res
# Обычно для деплоя выбираем RuBERT. Сохраняем в ../models/bert/

out_dir = "../models/bert"
os.makedirs(out_dir, exist_ok=True)

# сохраняем tokenizer+model в формате HuggingFace
bert_bundle.tokenizer.save_pretrained(out_dir)
bert_bundle.model.save_pretrained(out_dir)
print("Saved HF model to", out_dir)
# Для анализа ошибок используем предсказания BERT

y_pred = np.array(y_pred)

fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

print("False Positive:", len(fp_idx))
print("False Negative:", len(fn_idx))

print("Примеры FP:")
for i in fp_idx[:5]:
    print("-", X_test[i])

print("Примеры FN:")
for i in fn_idx[:5]:
    print("-", X_test[i])
