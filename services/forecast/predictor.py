"""학습된 LightGBM 분위수 모델 로딩·예측 (서빙).

output/forecast/models/ 의 model_config.json 과 모델 파일을 읽어,
최근 이력(최소 168시간)을 받아 앞으로 horizon_hours 시간의
중앙값(q50) + 90% 예측구간(q05-offset, q95+offset)을 돌려준다.

안전장치
  - 이력 168시간 미만 또는 최근 168시간에 빈 시간이 있으면 ValueError
  - 피처 순서가 모델이 기억하는 순서와 다르면 AssertionError
  - q05 <= q50 <= q95 가 깨지면 정렬
  - 입력/예측값이 학습 범위를 벗어나면 warnings 에 기록 (MPC 가 Fallback 판단)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from services.forecast.features import (
    TARGET, HORIZONS, MIN_HISTORY_HOURS,
    build_forecast_features, model_key_for, to_hourly,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / 'output' / 'forecast' / 'models'
CONFIG_NAME = 'model_config.json'
QUANTILE_KEYS = ('q05', 'q50', 'q95')


def read_booster(path: Path) -> lgb.Booster:
    """모델 텍스트 파일을 읽는다. Windows git 이 CRLF 로 바꿔 놓아도 읽히게 LF 로 정규화."""
    text = path.read_text(encoding='utf-8').replace('\r\n', '\n')
    return lgb.Booster(model_str=text)


@dataclass
class ForecastResult:
    model: str                      # '24h' | '168h'
    origin_time: pd.Timestamp       # 예측 시점 t (이력의 마지막 시각)
    timestamps: list[pd.Timestamp]  # t+1 ... t+horizon
    predictions: list[float]        # q50
    pi_lower: list[float]           # q05 - offset
    pi_upper: list[float]           # q95 + offset
    warnings: list[str] = field(default_factory=list)
    in_training_range: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    """model_config.json + 분위수 모델 6개를 들고 있는 예측기."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        with open(self.model_dir / CONFIG_NAME, encoding='utf-8') as f:
            self.config = json.load(f)
        self.boosters: dict[str, dict[str, lgb.Booster]] = {}
        for key, spec in self.config['models'].items():
            self.boosters[key] = {
                q: read_booster(self.model_dir / spec['files'][q]) for q in QUANTILE_KEYS
            }
            for q, booster in self.boosters[key].items():
                assert booster.feature_name() == spec['features'], (
                    f'{key} {q} 피처 순서 불일치: {booster.feature_name()} != {spec["features"]}')

    @classmethod
    def load(cls, model_dir: Path | str = DEFAULT_MODEL_DIR) -> 'Predictor | None':
        """모델/설정이 없으면 None (서비스가 503 을 내도록)."""
        model_dir = Path(model_dir)
        if not (model_dir / CONFIG_NAME).exists():
            return None
        try:
            return cls(model_dir)
        except (OSError, AssertionError, KeyError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------
    def prepare_history(self, history: pd.DataFrame) -> pd.Series:
        """이력(5분 또는 1시간, timestamp + it_power_kw)을 1시간 Series 로 바꾸고 검증."""
        if TARGET not in history.columns:
            raise ValueError(f'이력에 {TARGET} 컬럼이 필요합니다')
        hourly = to_hourly(history, [TARGET])[TARGET]
        if len(hourly) < MIN_HISTORY_HOURS:
            raise ValueError(
                f'이력이 최소 {MIN_HISTORY_HOURS}시간 필요합니다 (받은 이력: {len(hourly)}시간)')
        recent = hourly.iloc[-MIN_HISTORY_HOURS:]
        if recent.isna().any():
            missing = recent[recent.isna()].index
            raise ValueError(
                f'최근 {MIN_HISTORY_HOURS}시간 안에 데이터가 없는 시각이 {len(missing)}개 있습니다 '
                f'(예: {missing[0]})')
        return hourly

    def forecast(self, history: pd.DataFrame, horizon_hours: int) -> ForecastResult:
        key = model_key_for(horizon_hours)
        spec = self.config['models'][key]
        hourly = self.prepare_history(history)

        X = build_forecast_features(hourly, horizon_hours)
        assert list(X.columns) == spec['features'], '피처 순서가 model_config 과 다름'

        offset = float(spec['cqr_offset_kw'])
        b = self.boosters[key]
        lo = b['q05'].predict(X) - offset
        mid = b['q50'].predict(X)
        hi = b['q95'].predict(X) + offset

        warnings: list[str] = []
        stacked = np.vstack([lo, mid, hi])
        if (np.diff(stacked, axis=0) < 0).any():
            n_bad = int((np.diff(stacked, axis=0) < 0).any(axis=0).sum())
            warnings.append(f'quantile_crossing: {n_bad}개 시점에서 q05/q50/q95 순서가 어긋나 정렬함')
            stacked = np.sort(stacked, axis=0)
        lo, mid, hi = stacked

        rng = self.config['train_target_range_kw']
        in_range = True
        recent = hourly.iloc[-MIN_HISTORY_HOURS:]
        if recent.min() < rng['min'] or recent.max() > rng['max']:
            in_range = False
            warnings.append(
                f'input_out_of_range: 최근 {MIN_HISTORY_HOURS}시간 입력 {recent.min():.0f}~{recent.max():.0f}kW 가 '
                f'학습 범위 {rng["min"]:.0f}~{rng["max"]:.0f}kW 를 벗어남. 트리 모델은 범위 밖을 '
                f'따라가지 못하므로 Fallback 을 검토할 것')
        if mid.min() < rng['min'] or mid.max() > rng['max']:
            in_range = False
            warnings.append(
                f'prediction_out_of_range: 예측 중앙값 {mid.min():.0f}~{mid.max():.0f}kW 가 '
                f'학습 범위를 벗어남')

        return ForecastResult(
            model=key,
            origin_time=hourly.index[-1],
            timestamps=list(X.index),
            predictions=[round(float(v), 1) for v in mid],
            pi_lower=[round(float(v), 1) for v in lo],
            pi_upper=[round(float(v), 1) for v in hi],
            warnings=warnings,
            in_training_range=in_range,
        )


def max_horizon() -> int:
    return HORIZONS['168h']
