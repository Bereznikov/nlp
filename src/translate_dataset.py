"""
Перевод dataset с EN на RU используя Helsinki-NLP opus-mt-en-ru.

Input CSV columns: text,label
Output CSV columns: text,label,text_ru

Использование:
python -m src.translate_dataset --input data/processed/dataset_en.csv --output data/processed/dataset_ru.csv --batch-size 16
"""

import argparse
import math
from typing import List

import pandas as pd
from transformers import pipeline


def chunks(xs: List[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Input must have columns: text,label")

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    translator = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru")

    texts = df["text"].astype(str).tolist()
    out_ru = []

    total = len(texts)
    num_batches = math.ceil(total / args.batch_size)

    for bi, batch in enumerate(chunks(texts, args.batch_size), start=1):
        res = translator(batch)
        out_ru.extend([r["translation_text"] for r in res])
        if bi % 10 == 0 or bi == num_batches:
            print(f"Translated {len(out_ru)}/{total}")

    df_out = df.copy()
    df_out["text_ru"] = out_ru
    df_out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
