"""최적 냉각 제어 API(/api/v1/control/optimize, /control/mpc)와 MPC 최적화(scipy)·변화율 제약 테스트."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.forecast import main as forecast_svc
from services.mpc import controller as C
from services.mpc import main as mpc_svc
from services.mpc.controller import MPCConfig, MPCController
from services.twin.main import app as twin_app

needs_model = pytest.mark.skipif(forecast_svc._predictor is None, reason='예측 모델 없음')
FAST = MPCConfig(de_maxiter=3, de_popsize=2)
NOW = '2026-06-04T06:00:00'


@pytest.fixture()
def client() -> TestClient:
    mpc_svc.set_controller(MPCController(TestClient(twin_app), TestClient(forecast_svc.app), FAST))
    return TestClient(mpc_svc.app)


# ---------------------------------------------------------------- 최적화 방식


def test_uses_scipy_differential_evolution() -> None:
    """과제 제약: MPC 최적화는 CasADi / CVXPY / scipy.optimize 중 하나."""
    assert C.differential_evolution.__module__.startswith('scipy.optimize')


def test_rate_limit_repairs_plans() -> None:
    ctl = MPCController(TestClient(twin_app), TestClient(forecast_svc.app), FAST)
    ctl._last_setpoint = 20.0
    plans = np.array([[27.0, 16.0, 27.0, 27.0], [20.5, 21.0, 21.0, 21.0]])
    out = ctl._limit_rate(plans)
    np.testing.assert_allclose(out[0], [22.0, 20.0, 22.0, 24.0])
    np.testing.assert_allclose(out[1], plans[1])  # 이미 지키는 계획은 그대로
    assert np.abs(np.diff(np.hstack([[[20.0]] * 2, out]), axis=1)).max() <= 2.0 + 1e-9


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        MPCConfig(max_setpoint_rate_c_per_h=0)
    with pytest.raises(ValueError):
        MPCConfig(de_maxiter=0)


# ---------------------------------------------------------------- /api/v1/control/optimize


@needs_model
def test_control_optimize_from_state_only(client: TestClient) -> None:
    """현재 상태만 보내면 이력·외기는 트윈 계측값에서 받아 24시간 최적 설정값과 예상 PUE 를 돌려준다."""
    r = client.post(
        '/api/v1/control/optimize',
        json={
            'timestamp': NOW,
            'room_temp_c': 20.0,
            'current_setpoint_c': 20.0,
            'tes_soc_kwh': 6000.0,
        },
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert b['mode'] == 'mpc' and b['inputs_from'] == 'twin_telemetry'
    assert b['timestamps'][0].startswith('2026-06-04T06')
    for key in (
        'supply_temp_c',
        'tes_kw',
        'cooling_mode',
        'expected_pue',
        'expected_room_temp_c',
        'price_krw_per_kwh',
    ):
        assert len(b[key]) == 24, key
    assert min(b['expected_pue']) > 1.0 and b['expected_average_pue'] > 1.0
    steps = np.abs(np.diff([20.0] + b['supply_temp_c']))
    assert steps.max() <= 2.0 + 1e-6  # 변화율 제약 (시간당 ±2℃)
    assert max(b['expected_room_temp_c']) <= 27.0 + 1e-6
    assert 0 < b['computation_time_s'] < 300  # 과제 기준: 5분 이내
    assert b['expected_energy_cost_krw'] < b['baseline_energy_cost_krw']


@needs_model
def test_control_mpc_full_result(client: TestClient) -> None:
    r = client.post(
        '/api/v1/control/mpc',
        json={'timestamp': NOW, 'room_temp_c': 22.0, 'current_setpoint_c': 22.0},
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert (
        len(b['plan_c']) == 24 and len(b['forecast_it_kw']) == 24 and len(b['tes_plan_kw']) == 24
    )
    assert abs(b['setpoint_c'] - 22.0) <= 2.0 + 1e-6
    assert b['computation_time_s'] > 0


@needs_model
def test_control_with_explicit_inputs(client: TestClient) -> None:
    tele = (
        TestClient(twin_app)
        .get('/api/v1/telemetry', params={'start': '2026-05-27T06:00:00', 'hours': 192})
        .json()
    )
    history = [
        {'timestamp': t, 'it_power_kw': v}
        for t, v in zip(tele['timestamps'][:168], tele['it_power_kw'][:168])
    ]
    r = client.post(
        '/api/v1/control/optimize',
        json={'room_temp_c': 20.0, 'history': history, 'outdoor_forecast_c': [25.0] * 24},
    )
    assert r.status_code == 200, r.text
    assert r.json()['inputs_from'] == 'request'


def test_control_needs_timestamp_without_inputs(client: TestClient) -> None:
    assert client.post('/api/v1/control/optimize', json={'room_temp_c': 20.0}).status_code == 422


@needs_model
def test_control_outside_data_422(client: TestClient) -> None:
    r = client.post(
        '/api/v1/control/optimize', json={'timestamp': '2026-03-16T00:00:00', 'room_temp_c': 20.0}
    )
    assert r.status_code == 422
