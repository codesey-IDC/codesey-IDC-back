"""디지털 트윈 테스트: 90일 데이터 재현, 기본 설정 = 정적 계산, 물리 상식, API."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.twin import physics as P
from services.twin.main import app

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'synthetic_idc_90d.parquet'
needs_data = pytest.mark.skipif(not DATA_PATH.exists(), reason='데이터 없음')
client = TestClient(app)


@pytest.fixture(scope='module')
def data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


# ---------------------------------------------------------------- 1단계: 데이터 재현

@needs_data
def test_static_reproduces_90d_data(data) -> None:
    """원본 데이터 25,920행 전부를 복원 공식으로 재현."""
    it = P.it_power_from_cpu(data['cpu_utilization'])
    np.testing.assert_allclose(it, data['it_power_kw'], atol=1e-6)

    r = P.simulate_static(data['it_power_kw'], data['outdoor_temp_c'])
    np.testing.assert_allclose(r['cooling_power_kw'], data['cooling_power_kw'], atol=1e-6)
    np.testing.assert_allclose(r['free_cooling_fraction'], data['free_cooling_fraction'], atol=1e-9)
    np.testing.assert_allclose(r['pue'], data['pue'], atol=1e-9)
    np.testing.assert_allclose(r['server_heat_kw'], data['server_heat_kw'], atol=1e-6)
    assert (r['cooling_mode'] == data['cooling_mode'].to_numpy()).all()


@needs_data
def test_rollout_at_reference_setpoint_matches_data(data) -> None:
    """기준 설정 온도 + 서버실 = 설정 온도로 시작하면 5분 단위 rollout 이 원본과 같다."""
    day = data.head(12 * 48)
    n = len(day)
    r = P.rollout(day['it_power_kw'], day['outdoor_temp_c'],
                  [P.DEFAULT_PARAMS.supply_temp_ref_c] * n, dt_hours=5 / 60)
    np.testing.assert_allclose(r.cooling_power_kw, day['cooling_power_kw'], atol=1e-6)
    np.testing.assert_allclose(r.pue, day['pue'], atol=1e-9)
    np.testing.assert_allclose(r.room_temp_c, P.DEFAULT_PARAMS.supply_temp_ref_c, atol=1e-9)
    assert r.violation_steps == 0 and not r.warnings


def test_static_known_points() -> None:
    """설명용 예시 값: 3,000kW 에서 외기 25℃ / 10℃."""
    hot = P.simulate_static([3000.0], [25.0])
    assert hot['cooling_mode'][0] == 'chiller'
    assert hot['cooling_power_kw'][0] == pytest.approx(3000 / 2.7)
    assert hot['pue'][0] == pytest.approx(1 + 1 / 2.7 + 0.05)
    cold = P.simulate_static([3000.0], [10.0])
    assert cold['cooling_mode'][0] == 'free_cooling'
    assert cold['cooling_power_kw'][0] == pytest.approx(3000 / 13)


# ---------------------------------------------------------------- 2단계: 물리 상식

def _rollout(supply, it=3000.0, t_out=20.0, hours=24, **kw):
    return P.rollout([it] * hours, [t_out] * hours, [supply] * hours, **kw)


def test_higher_setpoint_uses_less_cooling_energy() -> None:
    energies = [_rollout(s).total_cooling_energy_kwh for s in (18.0, 20.0, 22.0, 24.0)]
    assert energies == sorted(energies, reverse=True)
    assert energies[0] > energies[-1]


def test_higher_setpoint_widens_free_cooling() -> None:
    fc_ref, _ = P.cooling_efficiency(17.0, 20.0)
    fc_hi, _ = P.cooling_efficiency(17.0, 23.0)
    assert fc_ref < 1.0 and fc_hi == 1.0


def test_cooling_off_heats_room() -> None:
    r = _rollout(20.0, hours=2, capacity_factor=0.0)
    # 발열 3,000kW / 열용량 200kWh/℃ → 시간당 15℃ 이상 상승 (24℃ 넘으면 팬 발열이 더해짐)
    assert r.room_temp_c[0] > 35.0
    assert r.server_fan_extra_kw[0] > 0
    assert r.cooling_power_kw[0] == 0.0
    assert r.violation_steps == 2
    assert any(w.startswith('room_temp_violation') for w in r.warnings)


def test_room_converges_to_new_setpoint() -> None:
    r = _rollout(24.0, hours=6, initial_room_temp_c=20.0)
    assert r.room_temp_c[0] > 23.9
    assert r.room_temp_c[-1] == pytest.approx(24.0, abs=1e-6)


def test_energy_balance() -> None:
    """열용량 × 온도 변화 = Σ(발열(팬 포함) - 제거열) × 시간."""
    rng = np.random.default_rng(0)
    it = rng.uniform(2500, 3400, 24)
    supply = rng.uniform(18, 26, 24)
    r = P.rollout(it, rng.uniform(5, 30, 24), supply, initial_room_temp_c=21.0)
    stored = P.DEFAULT_PARAMS.thermal_capacity_kwh_per_c * (r.room_temp_c[-1] - 21.0)
    heat = np.array(r.it_power_kw)
    assert stored == pytest.approx(np.sum(heat - np.array(r.heat_removed_kw)) * 1.0, abs=1e-6)


# ---------------------------------------------------------------- 설정 온도의 단점

def test_no_downside_at_reference() -> None:
    """기준 20℃ 에서는 팬 증가 0, 고장률 1.0 (원본 데이터와 같은 조건)."""
    r = P.simulate_static([3000.0], [25.0], [20.0])
    assert r['server_fan_extra_kw'][0] == 0.0
    assert r['xfactor'][0] == pytest.approx(1.0)


def test_fan_power_rises_above_threshold() -> None:
    ratios = P.fan_extra_ratio([20.0, 24.0, 25.0, 27.0])
    assert ratios[0] == ratios[1] == 0.0
    assert 0 < ratios[2] < ratios[3]
    assert ratios[3] == pytest.approx(0.03)


def test_fan_heat_is_cooled_too() -> None:
    """팬 증가분도 열이 되어 냉각 전력이 그만큼 늘어난다."""
    r = P.simulate_static([3000.0], [25.0], [27.0])
    assert r['server_heat_kw'][0] == pytest.approx(3000 * 1.03)
    fc, cop = P.cooling_efficiency(25.0, 27.0)
    assert r['cooling_power_kw'][0] == pytest.approx(3090 * P.cooling_power_per_kw(fc, cop))


def test_facility_energy_has_interior_optimum() -> None:
    """설정 온도를 무작정 올리면 손해: 전체 전력이 최소가 되는 지점이 범위 안쪽에 있다."""
    temps = np.arange(20.0, 27.01, 0.5)
    energy = [_rollout(s, t_out=25.0).total_facility_energy_kwh for s in temps]
    best = temps[int(np.argmin(energy))]
    assert 20.0 < best < 27.0
    assert energy[-1] > min(energy)


def test_pue_can_improve_while_total_energy_worsens() -> None:
    """팬 전력은 IT 로 잡혀서, 너무 올리면 PUE 는 좋아져도 전체 전력은 늘어난다."""
    a, b = _rollout(25.0, t_out=20.0), _rollout(27.0, t_out=20.0)   # 외기 20℃ 최적 ≈ 25℃
    assert b.average_pue < a.average_pue
    assert b.total_facility_energy_kwh > a.total_facility_energy_kwh


def test_ride_through_limits_high_load() -> None:
    """부하가 크면 같은 온도에서도 비상 여유 시간이 짧아져 위반이 된다."""
    normal = _rollout(27.0, it=3000.0, hours=2)
    surge = _rollout(27.0, it=4500.0, hours=2)
    assert normal.min_ride_through_min >= P.DEFAULT_PARAMS.min_ride_through_min
    assert surge.min_ride_through_min < P.DEFAULT_PARAMS.min_ride_through_min
    assert any(w.startswith('ride_through_violation') for w in surge.warnings)


def test_xfactor_rises_with_temperature() -> None:
    assert _rollout(18.0).average_xfactor < _rollout(20.0).average_xfactor < _rollout(26.0).average_xfactor


def test_capacity_limit_warns() -> None:
    r = _rollout(20.0, it=4500.0, hours=3, capacity_factor=0.5)   # 2,500kW 한계 < 발열 4,500kW
    assert r.capacity_limited_steps == 3
    assert max(r.heat_removed_kw) == pytest.approx(2500.0)
    assert any(w.startswith('cooling_capacity_limited') for w in r.warnings)


def test_setpoint_is_clipped() -> None:
    r = _rollout(40.0, hours=1)
    assert any(w.startswith('supply_temp_clipped') for w in r.warnings)


def test_batch_matches_single_rollout() -> None:
    """배치 평가 결과가 계획 하나씩 rollout 한 합계와 같다."""
    rng = np.random.default_rng(1)
    it, t_out = rng.uniform(2500, 4000, 24), rng.uniform(5, 32, 24)
    plans = rng.uniform(16, 27, (5, 24))
    batch = P.rollout_batch(it, t_out, plans, initial_room_temp_c=22.0)
    for j, plan in enumerate(plans):
        single = P.rollout(it, t_out, plan, initial_room_temp_c=22.0)
        for key in ('total_facility_energy_kwh', 'total_cooling_energy_kwh', 'max_room_temp_c',
                    'min_ride_through_min', 'average_xfactor', 'violation_steps'):
            assert batch[key][j] == pytest.approx(getattr(single, key)), key


def test_rollout_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        P.rollout([3000.0, 3000.0], [20.0], [20.0, 20.0])
    with pytest.raises(ValueError):
        P.rollout([0.0], [20.0], [20.0])


# ---------------------------------------------------------------- API

def test_health() -> None:
    r = client.get('/health')
    assert r.status_code == 200 and r.json()['service'] == 'twin'


def test_params() -> None:
    body = client.get('/api/v1/params').json()
    assert body['free_cooling_cop'] == 13.0 and 'thermal_capacity_kwh_per_c' in body


def test_api_simulate() -> None:
    r = client.post('/api/v1/simulate', json={'points': [
        {'it_power_kw': 3000, 'outdoor_temp_c': 25},
        {'cpu_utilization': 0.25, 'outdoor_temp_c': 10},
    ]})
    assert r.status_code == 200, r.text
    a, b = r.json()['results']
    assert a['cooling_mode'] == 'chiller' and a['cooling_power_kw'] == pytest.approx(3000 / 2.7)
    assert b['it_power_kw'] == pytest.approx(2150 + 3600 * 0.25)


def test_api_simulate_needs_load() -> None:
    r = client.post('/api/v1/simulate', json={'points': [{'outdoor_temp_c': 20}]})
    assert r.status_code == 422


def test_api_rollout() -> None:
    r = client.post('/api/v1/rollout', json={
        'it_power_kw': [3000] * 24, 'outdoor_temp_c': [22] * 24, 'supply_temp_c': [22] * 24})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body['room_temp_c']) == 24
    assert body['average_pue'] > 1.0 and body['violation_steps'] == 0


def test_api_rollout_batch() -> None:
    r = client.post('/api/v1/rollout/batch', json={
        'it_power_kw': [3000] * 24, 'outdoor_temp_c': [22] * 24,
        'plans': [[20] * 24, [26] * 24], 'initial_room_temp_c': 20})
    assert r.status_code == 200, r.text
    e = r.json()['total_facility_energy_kwh']
    assert len(e) == 2 and e[1] < e[0]


def test_api_rollout_batch_length_mismatch_422() -> None:
    r = client.post('/api/v1/rollout/batch', json={
        'it_power_kw': [3000] * 24, 'outdoor_temp_c': [22] * 24,
        'plans': [[20] * 23], 'initial_room_temp_c': 20})
    assert r.status_code == 422


def test_api_rollout_length_mismatch_422() -> None:
    r = client.post('/api/v1/rollout', json={
        'it_power_kw': [3000] * 24, 'outdoor_temp_c': [22] * 23, 'supply_temp_c': [22] * 24})
    assert r.status_code == 422
