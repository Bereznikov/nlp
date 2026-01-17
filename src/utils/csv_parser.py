import pandas as pd


def load_womens_reviews_csv(path: str) -> pd.DataFrame:
    """Загрузка Women's Clothing E-Commerce Reviews CSV"""
    return pd.read_csv(path)
