"""forecast-service API 테스트 (FastAPI TestClient)."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.forecast import main as svc

client = TestClient(svc.app)
needs_model = pytest.mark.skipif(svc._predictor is None, reason='학습된 모델 없음')


def test_health() -> None:
    r = client.get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok' and body['service'] == 'forecast'


@needs_model
def test_forecast_default_history() -> None:
    r = client.post('/api/v1/forecast', json={'horizon_hours': 24})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['model'] == 'lgbm-24h'
    assert body['unit'] == 'kW'
    assert len(body['predictions']) == len(body['pi_lower']) == len(body['pi_upper']) == 24
    assert len(body['timestamps']) == 24
    assert all(lo <= p <= hi for lo, p, hi in zip(body['pi_lower'], body['predictions'], body['pi_upper']))


@needs_model
def test_forecast_168h_cooling_alias() -> None:
    r = client.post('/api/v1/forecast', json={'horizon_hours': 168, 'target': 'cooling_load'})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['model'] == 'lgbm-168h' and body['target'] == 'cooling_load'
    assert len(body['predictions']) == 168


@needs_model
def test_forecast_with_history_payload() -> None:
    hist = pd.read_parquet(svc.DATA_PATH)[['timestamp', 'it_power_kw']].tail(12 * 200)
    payload = {
        'horizon_hours': 6,
        'history': [{'timestamp': t.isoformat(), 'it_power_kw': float(v)}
                    for t, v in zip(hist['timestamp'], hist['it_power_kw'])],
    }
    r = client.post('/api/v1/forecast', json=payload)
    assert r.status_code == 200, r.text
    assert len(r.json()['predictions']) == 6


@needs_model
def test_forecast_short_history_422() -> None:
    hist = pd.read_parquet(svc.DATA_PATH)[['timestamp', 'it_power_kw']].tail(12 * 50)
    payload = {
        'horizon_hours': 6,
        'history': [{'timestamp': t.isoformat(), 'it_power_kw': float(v)}
                    for t, v in zip(hist['timestamp'], hist['it_power_kw'])],
    }
    r = client.post('/api/v1/forecast', json=payload)
    assert r.status_code == 422
    assert '168' in r.json()['detail']


def test_invalid_horizon_422() -> None:
    assert client.post('/api/v1/forecast', json={'horizon_hours': 0}).status_code == 422
    assert client.post('/api/v1/forecast', json={'horizon_hours': 169}).status_code == 422
