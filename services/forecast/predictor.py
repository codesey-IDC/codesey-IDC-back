"""학습된 IT 부하 예측 모델 로딩·예측 (서빙). 두 종류를 지원한다.

  kind='lgbm'   LightGBM 분위수 모델 (scripts/train_forecast). 구간 = [q05 - offset, q95 + offset] (CQR)
  kind='ridge'  Ridge 선형 회귀 (scripts/train_ridge.py). 구간 = 예측 ± 반폭 (split conformal)

output/forecast/models/ 의 model_config.json(공통: 학습 범위 등)과 모델 파일을 읽어,
최근 이력(최소 168시간)을 받아 앞으로 horizon_hours 시간의 중앙값 + 90% 예측구간을 돌려준다.

안전장치 (두 모델 공통)
  - 이력 168시간 미만 또는 최근 168시간에 빈 시간이 있으면 ValueError
  - 피처 순서가 모델이 기억하는 순서와 다르면 AssertionError
  - 하한 <= 중앙값 <= 상한 이 깨지면 정렬
  - 입력/예측값이 학습 범위를 벗어나면 warnings 에 기록 (MPC 가 Fallback 판단)
  - 학습 범위 안이라도 입력 이력이 고착·급변·평균 이동이면 warnings + input_quality_ok=False
    (check_input_quality). 이런 입력은 범위 검사를 통과하지만 예측을 조용히 틀리게 만든다
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from services.common.settings import repo_path, section
from services.forecast import ridge
from services.forecast.features import (
    TARGET,
    MIN_HISTORY_HOURS,
    build_forecast_features,
    model_key_for,
    to_hourly,
)

DEFAULT_MODEL_DIR = repo_path('forecast_models')  # config/settings.yaml 의 paths.forecast_models
CONFIG_NAME = 'model_config.json'
QUANTILE_KEYS = ('q05', 'q50', 'q95')
KINDS = ('lgbm', 'ridge')
_QUALITY: dict = section('forecast')['input_quality']  # 입력 이력 품질 검사 한계값


def read_booster(path: Path) -> lgb.Booster:
    """모델 텍스트 파일을 읽는다. Windows git 이 CRLF 로 바꿔 놓아도 읽히게 LF 로 정규화."""
    text = path.read_text(encoding='utf-8').replace('\r\n', '\n')
    return lgb.Booster(model_str=text)


@dataclass
class InputCheck:
    """입력 이력 품질 검사 한 항목."""

    rule: str  # 'stuck' | 'jump' | 'level_shift'
    label: str
    ok: bool | None  # None = 이력이 짧아 검사하지 못함
    measured: str  # 측정값 (사람이 읽는 글)
    limit: str  # 기준


def check_input_quality(hourly: pd.Series) -> list[InputCheck]:
    """학습 범위 안에서 생기는 입력 이상을 찾는다 (범위 검사는 forecast() 가 따로 한다).

    hourly: 1시간 간격 IT 부하. 마지막 MIN_HISTORY_HOURS 시간을 본다.
      stuck        값이 멈춤 (센서 고착): 연속 N시간 표준편차가 거의 0
      jump         1시간 사이 급변 (단발 스파이크·계단의 첫 순간)
      level_shift  최근 평균이 1주 전 같은 시간대보다 크게 다름 (계단·점진 드리프트).
                   1주 전 값이 필요해 이력이 MIN_HISTORY_HOURS + shift_window_hours 이상일 때만 검사한다.
    한계값은 config/settings.yaml 의 forecast.input_quality.
    """
    q = _QUALITY
    recent = hourly.iloc[-MIN_HISTORY_HOURS:]
    window, std_limit = int(q['stuck_window_hours']), float(q['stuck_std_kw'])
    flat = float(recent.rolling(window).std().min())
    jump = float(recent.diff().abs().max())
    checks = [
        InputCheck(
            'stuck',
            '값 고착 (센서 멈춤)',
            flat >= std_limit,
            f'{window}시간 표준편차 최소 {flat:,.1f}kW',
            f'{std_limit:,.1f}kW 이상',
        ),
        InputCheck(
            'jump',
            '1시간 사이 급변',
            jump <= float(q['jump_limit_kw']),
            f'1시간 변화 최대 {jump:,.0f}kW',
            f'{float(q["jump_limit_kw"]):,.0f}kW 이하',
        ),
    ]
    span, shift_limit = int(q['shift_window_hours']), float(q['shift_limit_kw'])
    diff = (hourly - hourly.shift(MIN_HISTORY_HOURS)).iloc[-span:]
    limit_text = f'±{shift_limit:,.0f}kW 이내'
    if len(hourly) < MIN_HISTORY_HOURS + span or diff.isna().any():
        checks.append(
            InputCheck(
                'level_shift',
                '평균 이동·드리프트 (1주 전 대비)',
                None,
                f'이력이 {MIN_HISTORY_HOURS + span}시간보다 짧아 검사하지 못함',
                limit_text,
            )
        )
    else:
        shift = float(diff.mean())
        checks.append(
            InputCheck(
                'level_shift',
                '평균 이동·드리프트 (1주 전 대비)',
                abs(shift) <= shift_limit,
                f'최근 {span}시간 평균이 1주 전보다 {shift:+,.0f}kW',
                limit_text,
            )
        )
    return checks


@dataclass
class ForecastResult:
    model: str  # '24h' | '168h'
    origin_time: pd.Timestamp  # 예측 시점 t (이력의 마지막 시각)
    timestamps: list[pd.Timestamp]  # t+1 ... t+horizon
    predictions: list[float]  # q50
    pi_lower: list[float]  # q05 - offset
    pi_upper: list[float]  # q95 + offset
    warnings: list[str] = field(default_factory=list)
    in_training_range: bool = True
    input_quality_ok: bool = True  # 입력 이력에 고착·급변·평균 이동이 없다
    input_checks: list[InputCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Predictor:
    """model_config.json + 모델 파일을 들고 있는 예측기 (kind='lgbm' 분위수 6개 | 'ridge' 2개)."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR, kind: str = 'lgbm') -> None:
        if kind not in KINDS:
            raise ValueError(f'kind 는 {KINDS} 중 하나여야 합니다: {kind}')
        self.kind = kind
        self.model_dir = Path(model_dir)
        with open(self.model_dir / CONFIG_NAME, encoding='utf-8') as f:
            self.config = json.load(f)
        self.boosters: dict[str, dict[str, lgb.Booster]] = {}
        self.ridge: dict[str, dict] = {}
        if kind == 'lgbm':
            for key, spec in self.config['models'].items():
                self.boosters[key] = {
                    q: read_booster(self.model_dir / spec['files'][q]) for q in QUANTILE_KEYS
                }
                for q, booster in self.boosters[key].items():
                    assert (
                        booster.feature_name() == spec['features']
                    ), f'{key} {q} 피처 순서 불일치: {booster.feature_name()} != {spec["features"]}'
        else:
            with open(self.model_dir / ridge.CONFIG_NAME, encoding='utf-8') as f:
                ridge_config = json.load(f)
            for key, spec in ridge_config['models'].items():
                pipeline = joblib.load(self.model_dir / spec['file'])
                assert (
                    list(pipeline.feature_names_in_) == spec['features']
                ), f'{key} ridge 피처 순서 불일치'
                self.ridge[key] = {
                    'pipeline': pipeline,
                    'features': spec['features'],
                    'halfwidth_kw': float(spec['conformal_halfwidth_kw']),
                }

    @classmethod
    def load(
        cls, model_dir: Path | str = DEFAULT_MODEL_DIR, kind: str = 'lgbm'
    ) -> 'Predictor | None':
        """모델/설정이 없으면 None (서비스가 503 을 내도록)."""
        model_dir = Path(model_dir)
        if not (model_dir / CONFIG_NAME).exists():
            return None
        try:
            return cls(model_dir, kind)
        except (OSError, AssertionError, KeyError, ValueError, json.JSONDecodeError):
            return None

    def _interval(self, key: str, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(하한, 중앙값, 상한) 예측."""
        if self.kind == 'ridge':
            spec = self.ridge[key]
            assert list(X.columns) == spec['features'], '피처 순서가 ridge_config 와 다름'
            mid = spec['pipeline'].predict(X)
            return mid - spec['halfwidth_kw'], mid, mid + spec['halfwidth_kw']
        spec = self.config['models'][key]
        assert list(X.columns) == spec['features'], '피처 순서가 model_config 과 다름'
        offset = float(spec['cqr_offset_kw'])
        b = self.boosters[key]
        return b['q05'].predict(X) - offset, b['q50'].predict(X), b['q95'].predict(X) + offset

    # ------------------------------------------------------------------
    def prepare_history(self, history: pd.DataFrame) -> pd.Series:
        """이력(5분 또는 1시간, timestamp + it_power_kw)을 1시간 Series 로 바꾸고 검증."""
        if TARGET not in history.columns:
            raise ValueError(f'이력에 {TARGET} 컬럼이 필요합니다')
        hourly = to_hourly(history, [TARGET])[TARGET]
        if len(hourly) < MIN_HISTORY_HOURS:
            raise ValueError(
                f'이력이 최소 {MIN_HISTORY_HOURS}시간 필요합니다 (받은 이력: {len(hourly)}시간)'
            )
        recent = hourly.iloc[-MIN_HISTORY_HOURS:]
        if recent.isna().any():
            missing = recent[recent.isna()].index
            raise ValueError(
                f'최근 {MIN_HISTORY_HOURS}시간 안에 데이터가 없는 시각이 {len(missing)}개 있습니다 '
                f'(예: {missing[0]})'
            )
        return hourly

    def forecast(self, history: pd.DataFrame, horizon_hours: int) -> ForecastResult:
        key = model_key_for(horizon_hours)
        hourly = self.prepare_history(history)

        X = build_forecast_features(hourly, horizon_hours)
        lo, mid, hi = self._interval(key, X)

        warnings: list[str] = []
        stacked = np.vstack([lo, mid, hi])
        if (np.diff(stacked, axis=0) < 0).any():
            n_bad = int((np.diff(stacked, axis=0) < 0).any(axis=0).sum())
            warnings.append(
                f'quantile_crossing: {n_bad}개 시점에서 q05/q50/q95 순서가 어긋나 정렬함'
            )
            stacked = np.sort(stacked, axis=0)
        lo, mid, hi = stacked

        rng = self.config['train_target_range_kw']
        in_range = True
        recent = hourly.iloc[-MIN_HISTORY_HOURS:]
        if recent.min() < rng['min'] or recent.max() > rng['max']:
            in_range = False
            warnings.append(
                f'input_out_of_range: 최근 {MIN_HISTORY_HOURS}시간 입력 '
                f'{recent.min():.0f}~{recent.max():.0f}kW 가 '
                f'학습 범위 {rng["min"]:.0f}~{rng["max"]:.0f}kW 를 벗어남. 학습 범위 밖 예측은 '
                f'믿기 어려우므로 Fallback 을 검토할 것'
            )
        if mid.min() < rng['min'] or mid.max() > rng['max']:
            in_range = False
            warnings.append(
                f'prediction_out_of_range: 예측 중앙값 {mid.min():.0f}~{mid.max():.0f}kW 가 '
                f'학습 범위를 벗어남'
            )

        checks = check_input_quality(hourly)
        failed = [c for c in checks if c.ok is False]
        warnings += [f'input_{c.rule}: {c.label} — {c.measured} (기준 {c.limit})' for c in failed]

        return ForecastResult(
            model=key,
            origin_time=hourly.index[-1],
            timestamps=list(X.index),
            predictions=[round(float(v), 1) for v in mid],
            pi_lower=[round(float(v), 1) for v in lo],
            pi_upper=[round(float(v), 1) for v in hi],
            warnings=warnings,
            in_training_range=in_range,
            input_quality_ok=not failed,
            input_checks=checks,
        )
