"""냉각 수요 예측(target='cooling_power')과 예측 모델 비교 테스트."""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import scripts.compare_forecast_models as CMP
from services.common.settings import repo_path
from services.forecast import main as forecast_svc
from services.twin.main import app as twin_app

needs_model = pytest.mark.skipif(forecast_svc._predictor is None, reason='예측 모델 없음')
NOW = pd.Timestamp('2026-06-04 06:00')


@pytest.fixture(scope='module')
def hourly() -> pd.DataFrame:
    raw = pd.read_parquet(repo_path('data')).set_index('timestamp')
    return raw.resample('h').agg(
        {
            'it_power_kw': 'mean',
            'cooling_power_kw': 'mean',
            'cooling_mode': lambda s: s.mode().iloc[0],
        }
    )


@pytest.fixture()
def client() -> TestClient:
    forecast_svc.set_twin_client(TestClient(twin_app))
    return TestClient(forecast_svc.app)


def _history(hourly: pd.DataFrame) -> list[dict]:
    hist = hourly.it_power_kw.loc[NOW - pd.Timedelta(hours=192) : NOW - pd.Timedelta(hours=1)]
    return [{'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()]


# ---------------------------------------------------------------- 냉각 수요 예측 API


@needs_model
def test_cooling_power_forecast(client: TestClient, hourly: pd.DataFrame) -> None:
    r = client.post(
        '/api/v1/forecast',
        json={'horizon_hours': 24, 'target': 'cooling_power', 'history': _history(hourly)},
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert b['model'] == f'{forecast_svc.DEFAULT_MODEL}-24h+twin' and b['outdoor_source'] == 'data'
    for key in (
        'predictions',
        'pi_lower',
        'pi_upper',
        'it_predictions',
        'outdoor_temp_c',
        'cooling_mode',
        'pue',
    ):
        assert len(b[key]) == 24, key
    assert all(lo <= p <= hi for lo, p, hi in zip(b['pi_lower'], b['predictions'], b['pi_upper']))
    actual = hourly.cooling_power_kw.loc[NOW : NOW + pd.Timedelta(hours=23)].to_numpy()
    nmae = np.abs(np.array(b['predictions']) - actual).sum() / actual.sum() * 100
    assert nmae < 10.0  # 과제 기준


@needs_model
def test_free_cooling_switch_below_15c(client: TestClient, hourly: pd.DataFrame) -> None:
    """외기 15℃ 이하면 외기 냉각(자연공조), 18℃ 이상이면 칠러."""
    outdoor = [10.0] * 12 + [25.0] * 12
    b = client.post(
        '/api/v1/forecast',
        json={
            'horizon_hours': 24,
            'target': 'cooling_power',
            'history': _history(hourly),
            'outdoor_forecast_c': outdoor,
        },
    ).json()
    assert b['outdoor_source'] == 'request'
    assert set(b['cooling_mode'][:12]) == {'free_cooling'} and set(b['cooling_mode'][12:]) == {
        'chiller'
    }
    assert np.mean(b['predictions'][:12]) < np.mean(b['predictions'][12:])


@needs_model
def test_it_load_target_unchanged(client: TestClient, hourly: pd.DataFrame) -> None:
    b = client.post(
        '/api/v1/forecast', json={'horizon_hours': 24, 'history': _history(hourly)}
    ).json()
    assert b['target'] == 'it_load' and b['cooling_mode'] is None
    assert b['model'] == f'{forecast_svc.DEFAULT_MODEL}-24h'


def test_outdoor_length_422(client: TestClient) -> None:
    r = client.post(
        '/api/v1/forecast',
        json={'horizon_hours': 24, 'target': 'cooling_power', 'outdoor_forecast_c': [20.0] * 3},
    )
    assert r.status_code == 422


@needs_model
def test_twin_down_503(hourly: pd.DataFrame) -> None:
    forecast_svc.set_twin_client(
        httpx.Client(
            base_url='http://twin', transport=httpx.MockTransport(lambda req: httpx.Response(503))
        )
    )
    r = TestClient(forecast_svc.app).post(
        '/api/v1/forecast',
        json={'horizon_hours': 24, 'target': 'cooling_power', 'history': _history(hourly)},
    )
    assert r.status_code == 503


# ---------------------------------------------------------------- 예측 모델 비교


def test_conformal_halfwidth() -> None:
    """유한 표본 보정: n=100, α=0.1 → ⌈101 × 0.9⌉ / 100 = 91% 분위수 (절대 오차 0..99 중 91)."""
    y = np.zeros(100)
    pred = np.arange(100, dtype=float)
    assert CMP.conformal_halfwidth(y, pred, alpha=0.1) == 91.0


@needs_model
def test_model_comparison_meets_criteria() -> None:
    """3개 모델 모두 같은 분할로 평가, 선정은 Val 기준, 선정 모델은 과제 기준 통과."""
    report = CMP.compare()
    models = report['models']
    assert set(models) == {'lightgbm', 'ridge', 'seasonal_naive'}
    val_score = {m: r['24h']['val_mape'] + r['168h']['val_mape'] for m, r in models.items()}
    assert report['selected'] == min(val_score, key=val_score.get)
    best = models[report['selected']]
    assert best['24h']['mape'] < 5.0 and best['168h']['mape'] < 8.0
    assert best['24h']['coverage_pct'] >= 85.0 and best['168h']['coverage_pct'] >= 85.0
    assert models['lightgbm']['24h']['mape'] < models['seasonal_naive']['24h']['mape']
