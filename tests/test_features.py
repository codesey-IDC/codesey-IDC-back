"""피처 계산(학습·서빙 공용) 테스트: 누수 없음, 순서 고정, 서빙 피처가 학습 피처와 같은 값."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.forecast import features as F


def _hourly(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range('2026-03-15', periods=n, freq='1h')
    df = pd.DataFrame({F.TARGET: np.arange(n, dtype=float) + 1000.0}, index=idx)
    df['day'] = (df.index.normalize() - df.index[0].normalize()).days + 1
    return df


def test_feature_names_order() -> None:
    assert F.feature_names(24) == [
        'h', 'last_value', 'rolling_mean_24h', 'lag_168', 'lag_24', 'hour', 'dayofweek', 'is_weekend']
    assert F.feature_names(168) == [
        'h', 'last_value', 'rolling_mean_24h', 'lag_168', 'hour', 'dayofweek', 'is_weekend']


def test_model_key_for() -> None:
    assert F.model_key_for(1) == '24h'
    assert F.model_key_for(24) == '24h'
    assert F.model_key_for(25) == '168h'
    assert F.model_key_for(168) == '168h'
    with pytest.raises(ValueError):
        F.model_key_for(0)
    with pytest.raises(ValueError):
        F.model_key_for(169)


def test_to_hourly_from_5min() -> None:
    ts = pd.date_range('2026-03-15', periods=24 * 12, freq='5min')
    df = pd.DataFrame({'timestamp': ts, F.TARGET: np.repeat(np.arange(24, dtype=float), 12)})
    hourly = F.to_hourly(df, [F.TARGET])
    assert len(hourly) == 24
    assert hourly[F.TARGET].tolist() == list(range(24))


def test_direct_dataset_no_leakage() -> None:
    """y 가 등차수열이면 lag_168 == y(target) - 168, lag_24 == y(target) - 24 이어야 한다."""
    data = F.build_direct_dataset(_hourly(), 24)
    assert list(data.columns[4:-1]) == F.feature_names(24)
    assert (data['lag_168'] == data['y'] - 168).all()
    assert (data['lag_24'] == data['y'] - 24).all()
    assert (data['last_value'] == data['y'] - data['h']).all()
    assert (data['origin_time'] + pd.to_timedelta(data['h'], unit='h') == data['target_time']).all()


def test_direct_dataset_168_has_no_lag24() -> None:
    data = F.build_direct_dataset(_hourly(), 168)
    assert 'lag_24' not in data.columns
    assert data['h'].max() == 168


def test_forecast_features_match_training_rows() -> None:
    """서빙 피처는 '마지막 시점을 origin 으로 한 학습 행'과 값이 완전히 같아야 한다."""
    hourly = _hourly(300)
    y = hourly[F.TARGET]
    train = F.build_direct_dataset(hourly, 24)
    origin = hourly.index[-25]            # 학습 행이 존재하려면 24시간 뒤가 데이터 안에 있어야 함
    serve = F.build_forecast_features(y.loc[:origin], 24)
    ref = train[train['origin_time'] == origin].sort_values('h').set_index('target_time')
    pd.testing.assert_frame_equal(serve, ref[serve.columns], check_dtype=False, check_names=False)


def test_forecast_features_requires_168h() -> None:
    y = _hourly(100)[F.TARGET]
    with pytest.raises(ValueError):
        F.build_forecast_features(y, 24)
