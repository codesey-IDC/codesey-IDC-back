"""대시보드용 API 테스트: 트윈 계측값·ESG, MPC 폐루프 시뮬레이션·시나리오·사전 계산 결과."""
from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.forecast import main as forecast_svc
from services.mpc import main as mpc_svc
from services.mpc import simulation
from services.mpc.controller import MPCConfig, MPCController
from services.twin import esg
from services.twin.main import app as twin_app

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'synthetic_idc_90d.parquet'
needs_data = pytest.mark.skipif(not DATA_PATH.exists(), reason='데이터 없음')
needs_model = pytest.mark.skipif(not DATA_PATH.exists() or forecast_svc._predictor is None,
                                 reason='데이터 또는 예측 모델 없음')

FAST = MPCConfig(cem_iterations=2, cem_samples=24)
START = '2026-06-04T00:00:00'
twin = TestClient(twin_app)


@pytest.fixture()
def mpc_client() -> TestClient:
    mpc_svc.set_controller(MPCController(TestClient(twin_app), TestClient(forecast_svc.app), FAST))
    return TestClient(mpc_svc.app)


# ---------------------------------------------------------------- 트윈: 계측값

@needs_data
def test_telemetry_matches_hourly_mean_of_data() -> None:
    r = twin.get('/api/v1/telemetry', params={'start': START, 'hours': 6})
    assert r.status_code == 200, r.text
    body = r.json()
    raw = pd.read_parquet(DATA_PATH).set_index('timestamp')
    expected = raw[['it_power_kw', 'outdoor_temp_c']].resample('h').mean().loc[START:'2026-06-04 05:00']
    assert body['timestamps'][0] == START and len(body['timestamps']) == 6
    assert body['it_power_kw'] == pytest.approx(expected.it_power_kw.tolist())
    assert body['outdoor_temp_c'] == pytest.approx(expected.outdoor_temp_c.tolist())


@needs_data
def test_telemetry_default_is_end_of_data() -> None:
    body = twin.get('/api/v1/telemetry', params={'hours': 3}).json()
    assert len(body['timestamps']) == 3 and body['timestamps'][-1] == body['data_end']


@needs_data
def test_telemetry_outside_data_404() -> None:
    assert twin.get('/api/v1/telemetry', params={'start': '2030-01-01T00:00:00'}).status_code == 404


# ---------------------------------------------------------------- 트윈: ESG

def test_esg_known_values() -> None:
    s = esg.summarize([3000, 3000], [300, 800], ['free_cooling', 'chiller'])
    assert s['it_energy_mwh'] == pytest.approx(6.0)
    assert s['total_energy_mwh'] == pytest.approx(6.0 * 1.05 + 1.1)
    assert s['carbon_tco2'] == pytest.approx(s['total_energy_mwh'] * 0.459)
    assert s['water_by_mode_l'] == pytest.approx({'chiller': 5400.0, 'hybrid': 0.0, 'free_cooling': 900.0})
    assert s['wue_l_per_kwh'] == pytest.approx(6300.0 / 6000.0)


def test_esg_chiller_uses_more_water_than_free_cooling() -> None:
    free = esg.summarize([3000] * 4, [250] * 4, ['free_cooling'] * 4)
    chiller = esg.summarize([3000] * 4, [250] * 4, ['chiller'] * 4)
    assert chiller['water_usage_l'] > free['water_usage_l']


def test_api_esg() -> None:
    r = twin.post('/api/v1/esg', json={'it_power_kw': [3000, 3000], 'cooling_power_kw': [300, 800],
                                       'cooling_mode': ['hybrid', 'chiller'], 'dt_hours': 0.5})
    assert r.status_code == 200, r.text
    assert r.json()['hours_by_mode'] == {'chiller': 0.5, 'hybrid': 0.5, 'free_cooling': 0.0}


def test_api_esg_bad_input_422() -> None:
    bad_mode = {'it_power_kw': [3000], 'cooling_power_kw': [300], 'cooling_mode': ['unknown']}
    bad_length = {'it_power_kw': [3000, 3000], 'cooling_power_kw': [300], 'cooling_mode': ['chiller']}
    assert twin.post('/api/v1/esg', json=bad_mode).status_code == 422
    assert twin.post('/api/v1/esg', json=bad_length).status_code == 422


# ---------------------------------------------------------------- MPC: 폐루프 시뮬레이션

@needs_model
def test_simulate_fixed_matches_reference_operation(mpc_client) -> None:
    """20℃ 고정은 서버실 온도가 20℃ 에 머물고 위반이 없다."""
    r = mpc_client.post('/api/v1/simulate', json={'start': START, 'hours': 6, 'controller': 'fixed_20'})
    assert r.status_code == 200, r.text
    body = r.json()
    s = body['summary']
    assert s['hours'] == 6 and len(body['timeseries']['timestamp']) == 6
    assert s['max_room_temp_c'] == pytest.approx(20.0) and s['violation_hours'] == 0
    assert s['temp_maintenance_pct'] == 100.0 and not s['diverged']
    assert s['facility_energy_kwh'] == pytest.approx(sum(body['timeseries']['facility_power_kw']))


@needs_model
def test_simulate_mpc_saves_energy_and_keeps_temperature(mpc_client) -> None:
    base = mpc_client.post('/api/v1/simulate', json={'start': START, 'hours': 6, 'controller': 'fixed_20'}).json()
    mpc = mpc_client.post('/api/v1/simulate', json={'start': START, 'hours': 6, 'controller': 'mpc'}).json()
    assert mpc['summary']['facility_energy_kwh'] < base['summary']['facility_energy_kwh']
    assert mpc['summary']['temp_ok'] and set(mpc['summary']['mode_hours']) == {'mpc'}


@needs_model
@pytest.mark.parametrize('scenario', ['server_surge', 'chiller_failure', 'heatwave'])
def test_simulate_crisis_keeps_room_under_limit(mpc_client, scenario) -> None:
    """위기 시나리오 3종 모두 서버실 온도 27℃ 이하."""
    r = mpc_client.post('/api/v1/simulate', json={'start': START, 'hours': 6, 'controller': 'mpc',
                                                  'scenario': scenario})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['summary']['temp_ok'] and not body['summary']['diverged']
    assert body['scenario_label'] and body['strategy']


@needs_model
def test_simulate_surge_scales_load_and_distrusts_forecast(mpc_client) -> None:
    normal = mpc_client.post('/api/v1/simulate', json={'start': START, 'hours': 4, 'controller': 'mpc'}).json()
    surge = mpc_client.post('/api/v1/simulate', json={'start': START, 'hours': 4, 'controller': 'mpc',
                                                      'scenario': 'server_surge'}).json()
    ratio = [s / n for s, n in zip(surge['timeseries']['it_base_kw'], normal['timeseries']['it_base_kw'])]
    assert ratio == pytest.approx([1.3] * 4)
    assert 'mpc_naive' in surge['summary']['mode_hours']        # 학습 범위 밖 부하 → 예측 불신


@needs_model
def test_simulate_heatwave_raises_outdoor(mpc_client) -> None:
    r = mpc_client.post('/api/v1/simulate', json={'start': START, 'hours': 3, 'controller': 'fixed_20',
                                                  'scenario': 'heatwave'}).json()
    assert min(r['timeseries']['outdoor_temp_c']) >= 38.0
    assert set(r['timeseries']['cooling_mode']) == {'chiller'}


@needs_model
def test_simulate_outside_data_422(mpc_client) -> None:
    too_early = {'start': '2026-03-16T00:00:00', 'hours': 6}     # 이전 192시간 이력 없음
    too_late = {'start': '2026-06-12T20:00:00', 'hours': 6}
    assert mpc_client.post('/api/v1/simulate', json=too_early).status_code == 422
    assert mpc_client.post('/api/v1/simulate', json=too_late).status_code == 422
    assert mpc_client.post('/api/v1/simulate', json={'start': START, 'scenario': 'meteor'}).status_code == 422


@needs_model
def test_plan_falls_back_when_every_plan_diverges() -> None:
    """냉각 용량이 발열의 절반이면 어떤 설정 온도로도 발산한다 → 예외 대신 fallback."""
    twin_client = TestClient(twin_app)
    tele = simulation.fetch_telemetry(twin_client, pd.Timestamp(START), 24)
    hist = tele.it_power_kw.loc[:pd.Timestamp(START) - pd.Timedelta(hours=1)]
    history = [{'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()]
    outdoor = tele.outdoor_temp_c.loc[START:].iloc[:24].tolist()
    r = MPCController(twin_client, TestClient(forecast_svc.app), FAST).plan(20.0, history, outdoor,
                                                                             capacity_factor=0.5)
    assert r.mode == 'fallback' and r.setpoint_c == 20.0
    assert any(w.startswith('twin_unavailable') for w in r.warnings)


def test_simulate_twin_down_503() -> None:
    down = httpx.Client(base_url='http://twin', transport=httpx.MockTransport(
        lambda req: (_ for _ in ()).throw(httpx.ConnectError('down'))))
    mpc_svc.set_controller(MPCController(down, down, FAST))
    r = TestClient(mpc_svc.app).post('/api/v1/simulate', json={'start': START, 'hours': 3})
    assert r.status_code == 503


# ---------------------------------------------------------------- MPC: 시나리오 목록, 사전 계산 결과

def test_scenarios_list(mpc_client) -> None:
    body = mpc_client.get('/api/v1/scenarios').json()
    assert [s['name'] for s in body] == ['normal', 'server_surge', 'chiller_failure', 'heatwave']
    by_name = {s['name']: s for s in body}
    assert by_name['server_surge']['it_scale'] == 1.3
    assert by_name['heatwave']['outdoor_min_c'] == 38.0
    assert by_name['chiller_failure']['capacity_factor'] < 1.0


@pytest.mark.skipif(not (ROOT / 'output' / 'mpc' / 'metrics_base.json').exists(), reason='사전 계산 결과 없음')
def test_results_base(mpc_client) -> None:
    r = mpc_client.get('/api/v1/results', params={'scenario': 'base'})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body['controllers']) == set(body['timeseries'])
    assert len(body['timeseries']['mpc']['timestamp']) == body['hours']
    assert 'controller' not in body['timeseries']['mpc']


def test_results_unknown_404(mpc_client) -> None:
    assert mpc_client.get('/api/v1/results', params={'scenario': 'nope'}).status_code == 404
    assert mpc_client.get('/api/v1/results', params={'scenario': '../x'}).status_code == 422
