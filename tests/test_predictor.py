"""Predictor 테스트 (실제 저장된 모델 사용). 모델이 없으면 skip."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.forecast.predictor import DEFAULT_MODEL_DIR, CONFIG_NAME, Predictor, REPO_ROOT

DATA_PATH = REPO_ROOT / 'data' / 'synthetic_idc_90d.parquet'

pytestmark = pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / CONFIG_NAME).exists() or not DATA_PATH.exists(),
    reason='학습된 모델 또는 데이터 없음')


@pytest.fixture(scope='module')
def predictor() -> Predictor:
    return Predictor()


@pytest.fixture(scope='module')
def history() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)[['timestamp', 'it_power_kw']]


@pytest.mark.parametrize('horizon, key', [(1, '24h'), (24, '24h'), (25, '168h'), (168, '168h')])
def test_forecast_shape_and_order(predictor, history, horizon, key) -> None:
    r = predictor.forecast(history, horizon)
    assert r.model == key
    assert len(r.predictions) == len(r.pi_lower) == len(r.pi_upper) == len(r.timestamps) == horizon
    assert r.timestamps[0] == r.origin_time + pd.Timedelta(hours=1)
    assert r.timestamps[-1] == r.origin_time + pd.Timedelta(hours=horizon)
    lo, mid, hi = map(np.array, (r.pi_lower, r.predictions, r.pi_upper))
    assert (lo <= mid).all() and (mid <= hi).all()
    assert r.in_training_range
    assert not [w for w in r.warnings if 'out_of_range' in w]


def test_reproduces_saved_test_predictions(predictor, history) -> None:
    """학습 때 저장한 test_pred_24h.parquet 의 한 origin 을 서빙 경로로 재현."""
    tp = pd.read_parquet(REPO_ROOT / 'output' / 'forecast' / 'test_pred_24h.parquet')
    origin = tp['origin_time'].min()
    ref = tp[tp['origin_time'] == origin].sort_values('h')
    hist = history[history['timestamp'] < origin + pd.Timedelta(hours=1)]
    r = predictor.forecast(hist, 24)
    assert r.origin_time == origin
    np.testing.assert_allclose(r.predictions, ref['q50'], atol=0.06)
    np.testing.assert_allclose(r.pi_lower, ref['q05'], atol=0.06)
    np.testing.assert_allclose(r.pi_upper, ref['q95'], atol=0.06)


def test_out_of_range_input_warns(predictor, history) -> None:
    surge = history.copy()
    surge['it_power_kw'] *= 1.3          # server_surge 시나리오
    r = predictor.forecast(surge, 24)
    assert not r.in_training_range
    assert any(w.startswith('input_out_of_range') for w in r.warnings)


def test_short_history_rejected(predictor, history) -> None:
    with pytest.raises(ValueError, match='168'):
        predictor.forecast(history.tail(12 * 100), 24)


def test_gap_in_recent_history_rejected(predictor, history) -> None:
    hist = history.copy()
    cut = hist['timestamp'].max() - pd.Timedelta(hours=10)
    hist = hist[(hist['timestamp'] < cut) | (hist['timestamp'] >= cut + pd.Timedelta(hours=2))]
    with pytest.raises(ValueError, match='데이터가 없는 시각'):
        predictor.forecast(hist, 24)


def test_bad_horizon(predictor, history) -> None:
    with pytest.raises(ValueError):
        predictor.forecast(history, 0)
    with pytest.raises(ValueError):
        predictor.forecast(history, 169)
