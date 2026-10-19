"""IT 부하 예측용 피처 계산 (학습·서빙 공용).

train_forecast.py(학습)와 services/forecast/predictor.py(서빙)가 모두 이 파일의
함수로 피처를 만든다. 한쪽에만 따로 피처 코드를 두면 나중에 한쪽만 바뀌어도
에러 없이 틀린 예측이 나오므로, 피처 로직은 반드시 여기서만 수정한다.

피처 (모두 예측 시점 t 이전에 알 수 있는 값만 사용, 모델 입력 순서대로)
  h                 몇 시간 뒤를 예측하는지 (1 ~ max_h)
  last_value        t 시점의 IT 부하
  rolling_mean_24h  t 까지 최근 24시간 평균 (t 포함)
  lag_168           예측 대상 시각(t+h)의 168시간 전 값
  lag_24            예측 대상 시각의 24시간 전 값  ※ 24h 모델에만 있음
  hour              예측 대상 시각의 시 (0~23)
  dayofweek         예측 대상 시각의 요일 (0=월 ~ 6=일)
  is_weekend        예측 대상 시각이 주말이면 1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = 'it_power_kw'
WEATHER_COLS = ['outdoor_temp_c', 'outdoor_humidity', 'wind_speed']
RESAMPLE_RULE = 'h'           # 5분 데이터 → 1시간 평균
HORIZONS = {'24h': 24, '168h': 168}
MIN_HISTORY_HOURS = 168       # lag_168 때문에 최소 168시간 이력이 필요
ROLLING_HOURS = 24


def feature_names(max_h: int) -> list[str]:
    """모델 입력 피처 이름을 학습 순서 그대로 돌려준다."""
    names = ['h', 'last_value', 'rolling_mean_24h', 'lag_168']
    if max_h <= 24:
        names.append('lag_24')
    names += ['hour', 'dayofweek', 'is_weekend']
    return names


def model_key_for(horizon_hours: int) -> str:
    """요청 horizon에 맞는 모델 키 ('24h' 또는 '168h')."""
    if not 1 <= horizon_hours <= HORIZONS['168h']:
        raise ValueError(f'horizon_hours는 1~{HORIZONS["168h"]} 사이여야 합니다: {horizon_hours}')
    return '24h' if horizon_hours <= HORIZONS['24h'] else '168h'


def to_hourly(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """timestamp 컬럼(또는 DatetimeIndex)을 가진 데이터를 1시간 평균으로 집계한다."""
    out = df
    if 'timestamp' in out.columns:
        out = out.copy()
        out['timestamp'] = pd.to_datetime(out['timestamp'])
        out = out.set_index('timestamp')
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError('timestamp 컬럼 또는 DatetimeIndex 가 필요합니다')
    out = out.sort_index()
    return out[value_cols].resample(RESAMPLE_RULE).mean()


def load_hourly(path: str) -> pd.DataFrame:
    """5분 parquet 을 1시간 평균으로 집계하고 Day 번호(1~)를 붙인다. (학습용)"""
    raw = pd.read_parquet(path)
    hourly = to_hourly(raw, [TARGET] + WEATHER_COLS)
    start = hourly.index[0].normalize()
    hourly['day'] = (hourly.index.normalize() - start).days + 1
    return hourly


def _feature_block(y: np.ndarray, idx: pd.DatetimeIndex, roll24: np.ndarray,
                   origin: np.ndarray, h: int, use_lag24: bool
                   ) -> tuple[dict, pd.DatetimeIndex]:
    """예측 시점 위치 origin(배열)에서 h시간 뒤를 맞히는 피처 블록 (공용 핵심).

    lag 위치가 origin 을 넘으면(=미래 참조) assert 로 멈춘다.
    """
    target_time = idx[origin] + pd.Timedelta(hours=h)
    lag168_pos = origin + h - 168
    assert (lag168_pos >= 0).all(), 'lag_168 위치가 이력 시작 이전'
    assert (lag168_pos <= origin).all(), 'lag_168이 예측 시점 이후 값을 참조'

    cols: dict = {
        'h': np.full(len(origin), h, dtype=int),
        'last_value': y[origin],
        'rolling_mean_24h': roll24[origin],
        'lag_168': y[lag168_pos],
    }
    if use_lag24:
        lag24_pos = origin + h - 24
        assert (lag24_pos <= origin).all(), 'lag_24가 예측 시점 이후 값을 참조'
        cols['lag_24'] = y[lag24_pos]
    dow = np.asarray(target_time.dayofweek)
    cols['hour'] = np.asarray(target_time.hour)
    cols['dayofweek'] = dow
    cols['is_weekend'] = (dow >= 5).astype(int)
    return cols, target_time


def build_direct_dataset(hourly: pd.DataFrame, max_h: int,
                         exog_cols: list[str] | None = None) -> pd.DataFrame:
    """(학습용) 예측 시점 t 에서 h시간 뒤(t+h)를 맞히는 학습 행을 모두 만든다.

    반환 컬럼: origin_time, target_time, origin_day, target_day, 피처들, [exog], y
    """
    y = hourly[TARGET].to_numpy()
    n = len(y)
    idx = hourly.index
    days = hourly['day'].to_numpy()
    roll24 = hourly[TARGET].rolling(ROLLING_HOURS).mean().to_numpy()
    use_lag24 = max_h <= 24

    frames = []
    for h in range(1, max_h + 1):
        origin = np.arange(0, n - h)
        ok = (origin + h - 168 >= 0) & ~np.isnan(roll24[origin])  # 초기 168시간 제외
        origin = origin[ok]
        target = origin + h

        feats, target_time = _feature_block(y, idx, roll24, origin, h, use_lag24)
        assert (target_time == idx[target]).all(), '시간 인덱스가 1시간 간격이 아님'

        cols = {
            'origin_time': idx[origin],
            'target_time': target_time,
            'origin_day': days[origin],
            'target_day': days[target],
            **feats,
        }
        for c in exog_cols or []:
            cols[c] = hourly[c].to_numpy()[target]
        cols['y'] = y[target]
        frames.append(pd.DataFrame(cols))

    return pd.concat(frames, ignore_index=True)


def build_forecast_features(hourly_y: pd.Series, horizon_hours: int) -> pd.DataFrame:
    """(서빙용) 이력의 마지막 시점을 예측 시점 t 로 두고 h=1..horizon 피처를 만든다.

    hourly_y: 1시간 간격 DatetimeIndex 를 가진 IT 부하 Series (최소 168시간).
    반환: target_time 인덱스, 피처 컬럼은 feature_names(max_h) 순서.
    """
    if len(hourly_y) < MIN_HISTORY_HOURS:
        raise ValueError(f'이력이 {MIN_HISTORY_HOURS}시간 이상 필요합니다: {len(hourly_y)}시간')
    key = model_key_for(horizon_hours)
    max_h = HORIZONS[key]

    y = hourly_y.to_numpy(dtype=float)
    idx = hourly_y.index
    roll24 = hourly_y.rolling(ROLLING_HOURS).mean().to_numpy()
    origin = np.array([len(y) - 1])
    use_lag24 = max_h <= 24

    rows, times = [], []
    for h in range(1, horizon_hours + 1):
        feats, target_time = _feature_block(y, idx, roll24, origin, h, use_lag24)
        rows.append({k: v[0] for k, v in feats.items()})
        times.append(target_time[0])

    out = pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='target_time'))
    return out[feature_names(max_h)]
