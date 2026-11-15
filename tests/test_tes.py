"""냉수 축열조(TES) 테스트: 물리 제약, 요금 절감, API, MPC 계획, Rule-based 시간표, 시뮬레이션."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.forecast import main as forecast_svc
from services.mpc import simulation
from services.mpc.controller import MPCConfig, MPCController
from services.mpc.rule_based import tes_schedule_kw
from services.twin import physics as P
from services.twin.main import app as twin_app

p = P.DEFAULT_PARAMS
twin = TestClient(twin_app)
needs_model = pytest.mark.skipif(forecast_svc._predictor is None, reason='예측 모델 없음')


def _run(
    tes: list[float] | None,
    hours: int = 6,
    t_out: float = 28.0,
    soc0: float = 6000.0,
    **kw: Any,
) -> P.RolloutResult:
    return P.rollout(
        [3000.0] * hours, [t_out] * hours, [22.0] * hours, tes_kw=tes, initial_tes_kwh=soc0, **kw
    )


# ---------------------------------------------------------------- 물리


def test_zero_tes_equals_no_tes() -> None:
    """축열조를 쓰지 않으면 (출력 0) 기존 계산과 같다 → 90일 데이터 재현 유지."""
    a = P.rollout([3000.0] * 6, [28.0] * 6, [22.0] * 6)
    b = _run([0.0] * 6)
    np.testing.assert_allclose(a.cooling_power_kw, b.cooling_power_kw)
    np.testing.assert_allclose(a.room_temp_c, b.room_temp_c)


def test_charge_stores_energy_and_costs_power() -> None:
    idle, charge = _run([0.0] * 2, hours=2), _run([2000.0] * 2, hours=2)
    stored = charge.tes_soc_kwh[-1] - idle.tes_soc_kwh[-1]
    assert stored == pytest.approx(4000.0, rel=0.02)
    assert np.all(np.array(charge.cooling_power_kw) > np.array(idle.cooling_power_kw))
    np.testing.assert_allclose(charge.tes_kw, 2000.0)


def test_discharge_replaces_chiller_power() -> None:
    """칠러만 쓰는 외기(28℃)에서 방전하면 그만큼 칠러 전력이 준다."""
    idle, dis = _run([0.0] * 2, hours=2), _run([-1500.0] * 2, hours=2)
    _, cop = P.cooling_efficiency(28.0, 22.0)
    saved = np.array(idle.cooling_power_kw) - np.array(dis.cooling_power_kw)
    np.testing.assert_allclose(saved, 1500.0 / cop, rtol=1e-6)
    np.testing.assert_allclose(dis.tes_kw, -1500.0)


def test_soc_stays_within_capacity() -> None:
    full = _run([5000.0] * 10, hours=10, t_out=10.0, soc0=11000.0)
    assert max(full.tes_soc_kwh) <= p.tes_capacity_kwh + 1e-6
    empty = _run([-5000.0] * 6, hours=6, soc0=1000.0)
    assert min(empty.tes_soc_kwh) >= 0.0
    assert sum(-x for x in empty.tes_kw) == pytest.approx(1000.0, rel=0.02)  # 가진 만큼만 방전


def test_commands_clipped_to_rate_limits() -> None:
    r = _run([10000.0, -10000.0], hours=2, soc0=6000.0)
    assert r.tes_kw[0] == pytest.approx(p.tes_max_charge_kw)
    assert r.tes_kw[1] == pytest.approx(-p.tes_max_discharge_kw)


def test_no_discharge_during_full_free_cooling() -> None:
    """외기 냉각 100% 이면 칠러가 맡을 부분이 없어 방전하지 않는다."""
    r = _run([-2000.0] * 2, hours=2, t_out=5.0)
    assert np.allclose(r.tes_kw, 0.0, atol=1e-9)


def test_charge_limited_by_chiller_headroom() -> None:
    """칠러 용량이 거의 다 쓰이면 충전할 여유가 없다 (용량 3,200kW, 부하 3,000kW)."""
    r = _run([2000.0], hours=1, capacity_factor=3200.0 / p.cooling_capacity_kw)
    assert r.tes_kw[0] < 300.0


def test_discharge_adds_cooling_capacity() -> None:
    """칠러 용량이 부족해도 축열조 방전으로 서버실 온도를 지킨다."""
    no_tes = _run([0.0] * 4, hours=4, capacity_factor=0.5)
    with_tes = _run([-3000.0] * 4, hours=4, capacity_factor=0.5, soc0=12000.0)
    assert no_tes.max_room_temp_c > 27.0
    assert with_tes.max_room_temp_c < no_tes.max_room_temp_c


def test_tes_shifts_cost_to_cheap_hours() -> None:
    """같은 하루, 경부하 충전·최대부하 방전 → 전기요금 감소 (전력량은 조금 늘 수 있다)."""
    hours = 24
    price = np.array([86.4] * 8 + [139.6] * 7 + [209.9] * 6 + [139.6] + [86.4] * 2)
    sched = np.where(price < 100, 2000.0, np.where(price > 200, -2500.0, 0.0))
    kw = dict(price_krw_per_kwh=price.tolist(), initial_tes_kwh=0.0)
    base = P.rollout([3000.0] * hours, [26.0] * hours, [22.0] * hours, **kw)
    shifted = P.rollout(
        [3000.0] * hours, [26.0] * hours, [22.0] * hours, tes_kw=sched.tolist(), **kw
    )
    assert shifted.total_energy_cost_krw < base.total_energy_cost_krw


def test_batch_matches_single_with_tes() -> None:
    rng = np.random.default_rng(3)
    plans = rng.uniform(18, 27, (4, 8))
    tes = rng.uniform(-3000, 2000, (4, 8))
    batch = P.rollout_batch(
        [3000.0] * 8, [28.0] * 8, plans, 22.0, tes_plans=tes, initial_tes_kwh=5000.0
    )
    for j in range(4):
        single = P.rollout(
            [3000.0] * 8,
            [28.0] * 8,
            plans[j],
            initial_room_temp_c=22.0,
            tes_kw=tes[j],
            initial_tes_kwh=5000.0,
        )
        assert batch['total_facility_energy_kwh'][j] == pytest.approx(
            single.total_facility_energy_kwh
        )
        assert batch['final_tes_soc_kwh'][j] == pytest.approx(single.final_tes_soc_kwh)


def test_bad_tes_input() -> None:
    with pytest.raises(ValueError):
        _run([0.0], hours=2)
    with pytest.raises(ValueError):
        _run([0.0] * 2, hours=2, soc0=99999.0)
    with pytest.raises(ValueError):
        P.rollout_batch([3000.0] * 2, [25.0] * 2, [[20.0] * 2], 20.0, tes_plans=[[0.0] * 2] * 2)


# ---------------------------------------------------------------- API


def test_api_rollout_with_tes() -> None:
    r = twin.post(
        '/api/v1/rollout',
        json={
            'it_power_kw': [3000] * 2,
            'outdoor_temp_c': [28] * 2,
            'supply_temp_c': [22] * 2,
            'tes_kw': [2000, -1500],
            'initial_tes_kwh': 6000,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['tes_kw'][0] == pytest.approx(2000.0) and body['tes_kw'][1] == pytest.approx(
        -1500.0
    )
    assert body['final_tes_soc_kwh'] == body['tes_soc_kwh'][-1]


def test_api_batch_tes_validation() -> None:
    ok = twin.post(
        '/api/v1/rollout/batch',
        json={
            'it_power_kw': [3000] * 2,
            'outdoor_temp_c': [28] * 2,
            'plans': [[22] * 2],
            'tes_plans': [[0, 0]],
            'initial_room_temp_c': 22,
        },
    )
    assert ok.status_code == 200 and len(ok.json()['final_tes_soc_kwh']) == 1
    bad = twin.post(
        '/api/v1/rollout/batch',
        json={
            'it_power_kw': [3000] * 2,
            'outdoor_temp_c': [28] * 2,
            'plans': [[22] * 2],
            'tes_plans': [[0, 0], [0, 0]],
            'initial_room_temp_c': 22,
        },
    )
    assert bad.status_code == 422


def test_params_expose_tes() -> None:
    body = twin.get('/api/v1/params').json()
    assert (
        body['tes_capacity_kwh'] == p.tes_capacity_kwh
        and body['tes_max_charge_kw'] == p.tes_max_charge_kw
    )


# ---------------------------------------------------------------- Rule-based 시간표


def test_tes_schedule_rule() -> None:
    assert tes_schedule_kw('off_peak') == p.tes_max_charge_kw
    assert tes_schedule_kw('peak') == -p.tes_max_discharge_kw
    assert tes_schedule_kw('mid') == 0.0


# ---------------------------------------------------------------- MPC


def _inputs(now: pd.Timestamp) -> tuple[list[dict], list[float]]:
    tele = simulation.fetch_telemetry(twin, now, 24)
    hist = tele.it_power_kw.loc[: now - pd.Timedelta(hours=1)]
    return (
        [{'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()],
        tele.outdoor_temp_c.loc[now:].iloc[:24].tolist(),
    )


@needs_model
def test_mpc_plans_charge_cheap_discharge_expensive() -> None:
    """여름 평일 06시 계획: 최대부하(15~21시)에는 평균적으로 방전, 경부하(22~08시)보다 출력이 낮다."""
    now = pd.Timestamp('2026-06-04 06:00')
    history, outdoor = _inputs(now)
    ctl = MPCController(
        TestClient(twin_app), TestClient(forecast_svc.app), MPCConfig(de_maxiter=4, de_popsize=4)
    )
    r = ctl.plan(20.0, history, outdoor, tes_soc_kwh=6000.0)
    assert r.mode == 'mpc' and len(r.tes_plan_kw) == 24 and len(r.predicted_tes_soc_kwh) == 24
    tes, price = np.array(r.tes_plan_kw), np.array(r.price_krw_per_kwh)
    assert tes[price == price.max()].mean() < 0
    assert tes[price == price.max()].mean() < tes[price == price.min()].mean()
    assert r.expected_cost_saving_pct > 0


@needs_model
def test_mpc_without_tes_config() -> None:
    now = pd.Timestamp('2026-06-04 06:00')
    history, outdoor = _inputs(now)
    ctl = MPCController(
        TestClient(twin_app),
        TestClient(forecast_svc.app),
        MPCConfig(use_tes=False, de_maxiter=2, de_popsize=2),
    )
    r = ctl.plan(20.0, history, outdoor, tes_soc_kwh=6000.0)
    assert r.tes_kw == 0.0 and r.tes_plan_kw == []


@needs_model
def test_simulation_rule_tes_follows_schedule() -> None:
    r = simulation.run_closed_loop(
        twin, TestClient(forecast_svc.app), pd.Timestamp('2026-06-04 13:00'), 4, 'rule_tes'
    )
    ts = r['timeseries']
    assert ts['tariff_period'] == ['mid', 'mid', 'peak', 'peak']
    assert ts['tes_kw'][:2] == [0.0, 0.0] and ts['tes_kw'][2] < 0
    assert ts['tes_soc_kwh'][-1] < ts['tes_soc_kwh'][1]
