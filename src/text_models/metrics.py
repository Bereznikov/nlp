from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,   # PR-AUC (Average Precision)
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion: np.ndarray

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "macro_f1": float(self.macro_f1),
            "weighted_f1": float(self.weighted_f1),
            "roc_auc": None if self.roc_auc is None else float(self.roc_auc),
            "pr_auc": None if self.pr_auc is None else float(self.pr_auc),
            "confusion": self.confusion.tolist(),
        }


def _extract_score(y_score_or_prob) -> Optional[np.ndarray]:
    """
    Принимает:
      - (n,2) вероятности -> вернём prob_pos = [:,1]
      - (n,) score -> вернём как есть
    """
    if y_score_or_prob is None:
        return None

    arr = np.asarray(y_score_or_prob)

    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            return arr[:, 1].astype(np.float64)
        return arr.ravel().astype(np.float64)

    return arr.astype(np.float64)


def evaluate_binary(y_true, y_pred, y_score_or_prob=None) -> BinaryMetrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    score = _extract_score(y_score_or_prob)

    roc = pr = None
    if score is not None and len(np.unique(y_true)) == 2:
        # подстрахуемся от NaN/Inf
        score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        roc = roc_auc_score(y_true, score)
        pr = average_precision_score(y_true, score)

    return BinaryMetrics(
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        roc_auc=roc,
        pr_auc=pr,
        confusion=cm,
    )
