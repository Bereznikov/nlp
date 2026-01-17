"""
Подготовка Women's Clothing E-Commerce Reviews dataset.

Outputs: CSV с 2 колонками:
- text: review text
- label: 0 негативный (оценка 1-3), 1 позитивный (оценка 4-5)

Использование:
python -m src.preprocess_womens_dataset --input <raw.csv> --output <dataset_en.csv>
"""

import argparse
import pandas as pd

from src.utils.csv_parser import load_womens_reviews_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = load_womens_reviews_csv(args.input)

    # Обязательные колонки: 'Review Text', 'Rating'
    if "Review Text" not in df.columns or "Rating" not in df.columns:
        raise ValueError(f"Expected columns 'Review Text' and 'Rating'. Found: {list(df.columns)}")

    out = df[["Review Text", "Rating"]].copy()

    out = out.dropna(subset=["Review Text"])  # remove NaN text

    out["Rating"] = pd.to_numeric(out["Rating"], errors="coerce")
    out = out.dropna(subset=["Rating"])  # remove NaN rating
    out = out[out["Rating"].between(1, 5)]

    out["text"] = (
        out["Review Text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )

    # label mapping
    out["label"] = (out["Rating"] >= 4).astype(int)

    out = out[["text", "label"]]

    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print("Rows:", len(out))
    print("Label counts:\n", out["label"].value_counts())


if __name__ == "__main__":
    main()
