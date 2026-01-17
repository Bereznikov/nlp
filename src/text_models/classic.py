from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


@dataclass
class ClassicBundle:
    vectorizer: TfidfVectorizer
    models: Dict[str, Any]


def train_classic_models(x_train: np.ndarray, y_train: np.ndarray) -> ClassicBundle:
    """
    ROC/PR-AUC считаем по decision_function (score) для SVM.
    """
    vec = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float64,
    )
    X = vec.fit_transform(x_train)

    models: Dict[str, Any] = {}

    lr = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
    )
    lr.fit(X, y_train)
    models["logreg"] = lr

    # Без калибровки: будем использовать decision_function как score
    svm = LinearSVC(class_weight="balanced", C=1.0)
    svm.fit(X, y_train)
    models["svm_linear"] = svm

    # Для TF-IDF и дисбаланса ComplementNB
    nb = ComplementNB(alpha=1.0)
    nb.fit(X, y_train)
    models["naive_bayes"] = nb

    return ClassicBundle(vectorizer=vec, models=models)


def predict_classic(
    bundle: ClassicBundle,
    X_text: Union[np.ndarray, list],
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает:
      y_pred: (n,)
      y_score_or_prob:
        - если есть predict_proba -> (n,2)
        - иначе если есть decision_function -> (n,) score
        - иначе -> (n,) из y_pred (как fallback)
    """
    model = bundle.models[model_name]
    X = bundle.vectorizer.transform(X_text)

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)
        return y_pred, y_prob

    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        score = np.asarray(score, dtype=np.float64).ravel()
        return y_pred, score

    return y_pred, y_pred.astype(np.float64)
