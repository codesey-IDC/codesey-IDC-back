"""Ridge 선형 회귀 IT 부하 예측 모델 정의 (학습·비교·서빙 공용).

피처는 LightGBM 과 같다 (features.feature_names). 시각·요일은 범주형(one-hot), 나머지는 표준화한다.
90% 예측 구간은 split conformal: 예측 ± Val 절대 오차의 (1-α) 분위수 (유한 표본 보정).

저장 파일 (output/forecast/models/)
  ridge_24h.joblib, ridge_168h.joblib   학습된 Pipeline
  ridge_config.json                      모델 파일명, 피처 순서, 구간 반폭(kW), α
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ['hour', 'dayofweek']
RIDGE_ALPHA = 1.0
CONFIG_NAME = 'ridge_config.json'


def model_file(name: str) -> str:
    return f'ridge_{name}.joblib'


def build_pipeline(feats: list[str], alpha: float = RIDGE_ALPHA) -> Pipeline:
    """피처 목록 → (one-hot + 표준화) → Ridge 파이프라인."""
    numeric = [f for f in feats if f not in CATEGORICAL]
    pre = ColumnTransformer(
        [
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
            ('num', StandardScaler(), numeric),
        ]
    )
    return Pipeline([('pre', pre), ('ridge', Ridge(alpha=alpha))])


def conformal_halfwidth(y_val: np.ndarray, pred_val: np.ndarray, alpha: float = 0.1) -> float:
    """split conformal 구간 반폭: Val 절대 오차의 ⌈(n+1)(1-α)⌉/n 분위수."""
    scores = np.abs(np.asarray(y_val, dtype=float) - np.asarray(pred_val, dtype=float))
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method='higher'))
