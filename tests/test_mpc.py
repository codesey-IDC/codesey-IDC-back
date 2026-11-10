"""MPC 테스트: 정상 계획, 예측 불신 시 naive, 트윈 장애 시 fallback, API. 다른 모듈은 HTTP 로만 호출."""
from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.forecast import main as forecast_svc
from services.mpc import main as mpc_svc
from services.mpc.controller import MPCConfig, MPCController
from services.twin.main import app as twin_app

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'synthetic_idc_90d.parquet'
pytestmark = pytest.mark.skipif(not DATA_PATH.exists() or forecast_svc._predictor is None,
                                reason='데이터 또는 예측 모델 없음')

FAST = MPCConfig(cem_iterations=3, cem_samples=32)
NOW = pd.Timestamp('2026-06-03 06:00')        # Test 구간의 더운 날 아침


@pytest.fixture(scope='module')
def hourly() -> pd.DataFrame:
    raw = pd.read_parquet(DATA_PATH).set_index('timestamp')
    return raw[['it_power_kw', 'outdoor_temp_c']].resample('h').mean()


def _inputs(hourly: pd.DataFrame, scale: float = 1.0) -> tuple[list[dict], list[float]]:
    hist = hourly.it_power_kw.loc[NOW - pd.Timedelta(hours=192):NOW - pd.Timedelta(hours=1)] * scale
    history = [{'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()]
    outdoor = hourly.outdoor_temp_c.loc[NOW:NOW + pd.Timedelta(hours=23)].tolist()
    return history, outdoor


def _controller(twin=None) -> MPCController:
    return MPCController(twin or TestClient(twin_app), TestClient(forecast_svc.app), FAST)


def _down_client() -> httpx.Client:
    return httpx.Client(base_url='http://twin',
                        transport=httpx.MockTransport(lambda req: httpx.Response(503, text='down')))


def test_plan_beats_reference_on_hot_day(hourly) -> None:
    history, outdoor = _inputs(hourly)
    r = _controller().plan(20.0, history, outdoor)
    assert r.mode == 'mpc'
    assert len(r.plan_c) == 24 and 16.0 <= r.setpoint_c <= 27.0
    assert r.expected_facility_energy_kwh < r.baseline_facility_energy_kwh
    assert max(r.predicted_room_temp_c) <= 27.0 + 1e-6
    assert r.timestamps[0].startswith('2026-06-03T06')


def test_out_of_range_load_uses_naive_forecast(hourly) -> None:
    """학습 범위 밖 부하면 예측 대신 최근 최대 부하 유지 가정으로 계속 최적화."""
    history, outdoor = _inputs(hourly, scale=1.3)
    r = _controller().plan(20.0, history, outdoor)
    assert r.mode == 'mpc_naive'
    assert len(set(r.forecast_it_kw)) == 1
    assert r.forecast_it_kw[0] == pytest.approx(max(h['it_power_kw'] for h in history[-3:]))
    assert any(w.startswith('forecast_out_of_range') for w in r.warnings)


def test_twin_down_falls_back_to_reference(hourly) -> None:
    history, outdoor = _inputs(hourly)
    r = _controller(twin=_down_client()).plan(22.0, history, outdoor)
    assert r.mode == 'fallback' and r.setpoint_c == 20.0
    assert any(w.startswith('twin_unavailable') for w in r.warnings)


def test_bad_outdoor_length(hourly) -> None:
    history, outdoor = _inputs(hourly)
    with pytest.raises(ValueError):
        _controller().plan(20.0, history, outdoor[:10])


def test_config_blocks_must_sum_to_horizon() -> None:
    with pytest.raises(ValueError):
        MPCConfig(blocks=(1, 1, 1))


def test_api_plan(hourly) -> None:
    mpc_svc.set_controller(_controller())
    client = TestClient(mpc_svc.app)
    history, outdoor = _inputs(hourly)
    r = client.post('/api/v1/plan', json={'room_temp_c': 20.0, 'history': history, 'outdoor_forecast_c': outdoor})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['mode'] == 'mpc' and len(body['plan_c']) == 24
    assert body['expected_saving_pct'] > 0


def test_api_plan_short_history_422(hourly) -> None:
    client = TestClient(mpc_svc.app)
    history, outdoor = _inputs(hourly)
    r = client.post('/api/v1/plan', json={'room_temp_c': 20.0, 'history': history[:100], 'outdoor_forecast_c': outdoor})
    assert r.status_code == 422
