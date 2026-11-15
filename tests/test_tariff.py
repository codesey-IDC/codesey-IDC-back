"""전기요금 테스트: 요금표(계절·시간대·주말·공휴일), 트윈 요금 계산, API, MPC 비용 목적, ESG 비용."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.common.settings import repo_path, section
from services.forecast import main as forecast_svc
from services.mpc import simulation
from services.mpc.controller import MPCConfig, MPCController
from services.twin import esg
from services.twin import physics as P
from services.twin.main import app as twin_app
from services.twin.tariff import DEFAULT_TARIFF as T, Tariff

twin = TestClient(twin_app)
needs_model = pytest.mark.skipif(forecast_svc._predictor is None, reason='예측 모델 없음')


# ---------------------------------------------------------------- 요금표


@pytest.mark.parametrize(
    'when, season, period, price',
    [
        ('2026-06-04 16:00', 'summer', 'peak', 209.9),  # 목요일 여름 최대부하 15~21시
        ('2026-06-04 08:00', 'summer', 'mid', 139.6),
        ('2026-06-04 21:00', 'summer', 'mid', 139.6),
        ('2026-06-04 22:00', 'summer', 'off_peak', 86.4),
        ('2026-06-04 07:00', 'summer', 'off_peak', 86.4),
        ('2026-05-29 16:00', 'spring_fall', 'peak', 132.2),  # 5월 = 봄
        ('2026-01-14 10:00', 'winter', 'peak', 186.7),  # 겨울 최대부하 9~12시
        ('2026-01-14 13:00', 'winter', 'mid', 139.8),
        ('2026-01-14 17:00', 'winter', 'peak', 186.7),
    ],
)
def test_weekday_rates(when: str, season: str, period: str, price: float) -> None:
    t = pd.Timestamp(when)
    assert (T.season(t), T.period(t), T.price(t)) == (season, period, price)


def test_weekend_and_holiday_rules() -> None:
    assert T.period(pd.Timestamp('2026-06-13 16:00')) == 'mid'  # 토요일 최대부하 → 중간부하
    assert T.period(pd.Timestamp('2026-06-14 16:00')) == 'off_peak'  # 일요일 → 경부하
    assert T.period(pd.Timestamp('2026-06-03 16:00')) == 'off_peak'  # 지방선거일 (수)
    assert T.period(pd.Timestamp('2026-06-06 16:00')) == 'off_peak'  # 현충일 (토, 공휴일이 우선)


def test_config_values_match_pdf_table() -> None:
    """설정 파일 값 = 요금표 PDF 일반용전력(을) 고압A 선택Ⅲ."""
    cfg = section('tariff')
    assert cfg['basic_charge_krw_per_kw'] == 9810.0
    assert cfg['rates_krw_per_kwh']['summer'] == {'off_peak': 86.4, 'mid': 139.6, 'peak': 209.9}
    assert cfg['rates_krw_per_kwh']['spring_fall'] == {
        'off_peak': 86.4,
        'mid': 108.5,
        'peak': 132.2,
    }
    assert cfg['rates_krw_per_kwh']['winter'] == {'off_peak': 93.7, 'mid': 139.8, 'peak': 186.7}


def test_every_hour_has_one_period() -> None:
    for season in ('summer', 'spring_fall', 'winter'):
        assert set(T.period_hours[season]) == set(range(24))
    sched = T.schedule(pd.date_range('2026-06-01', periods=24 * 7, freq='h'))
    assert set(sched.period) == {'off_peak', 'mid', 'peak'}


def test_adders_are_added() -> None:
    cfg = dict(section('tariff')) | {
        'climate_env_krw_per_kwh': 9.0,
        'fuel_adjust_krw_per_kwh': 5.0,
    }
    assert Tariff.from_config(cfg).price(pd.Timestamp('2026-06-04 16:00')) == pytest.approx(
        209.9 + 14.0
    )


def test_bad_seasons_rejected() -> None:
    cfg = dict(section('tariff')) | {'seasons': {'summer': [6, 7, 8]}}
    with pytest.raises(ValueError, match='1~12월'):
        Tariff.from_config(cfg)


# ---------------------------------------------------------------- 트윈 요금 계산


def test_rollout_cost_equals_power_times_price() -> None:
    prices = [86.4, 139.6, 209.9]
    r = P.rollout([3000.0] * 3, [25.0] * 3, [22.0] * 3, price_krw_per_kwh=prices)
    expected = np.array(r.facility_power_kw) * np.array(prices)
    np.testing.assert_allclose(r.energy_cost_krw, expected)
    assert r.total_energy_cost_krw == pytest.approx(expected.sum())


def test_rollout_without_prices_has_no_cost() -> None:
    r = P.rollout([3000.0], [25.0], [22.0])
    assert r.energy_cost_krw is None and r.total_energy_cost_krw is None


def test_batch_cost_matches_single() -> None:
    prices = list(np.linspace(80, 210, 6))
    plans = np.array([[20.0] * 6, [25.0] * 6])
    batch = P.rollout_batch([3000.0] * 6, [28.0] * 6, plans, 20.0, price_krw_per_kwh=prices)
    for j, plan in enumerate(plans):
        single = P.rollout(
            [3000.0] * 6, [28.0] * 6, plan, initial_room_temp_c=20.0, price_krw_per_kwh=prices
        )
        assert batch['total_energy_cost_krw'][j] == pytest.approx(single.total_energy_cost_krw)


def test_bad_prices_rejected() -> None:
    with pytest.raises(ValueError):
        P.rollout([3000.0] * 2, [25.0] * 2, [22.0] * 2, price_krw_per_kwh=[100.0])
    with pytest.raises(ValueError):
        P.rollout([3000.0], [25.0], [22.0], price_krw_per_kwh=[-1.0])


def test_esg_cost() -> None:
    s = esg.summarize(
        [3000, 3000], [300, 800], ['free_cooling', 'chiller'], price_krw_per_kwh=[100.0, 200.0]
    )
    facility = np.array([3000 * 1.05 + 300, 3000 * 1.05 + 800])
    assert s['energy_cost_krw'] == pytest.approx((facility * [100.0, 200.0]).sum())
    assert s['cost_krw_per_mwh'] == pytest.approx(s['energy_cost_krw'] / s['total_energy_mwh'])
    assert esg.summarize([3000], [300], ['chiller'])['energy_cost_krw'] is None


# ---------------------------------------------------------------- API


def test_api_tariff() -> None:
    r = twin.get('/api/v1/tariff', params={'start': '2026-06-04T14:00:00', 'hours': 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['period'] == ['mid', 'peak', 'peak']
    assert body['price_krw_per_kwh'] == [139.6, 209.9, 209.9]
    assert body['name'].startswith('일반용전력(을)')


def test_api_rollout_with_prices() -> None:
    r = twin.post(
        '/api/v1/rollout',
        json={
            'it_power_kw': [3000] * 2,
            'outdoor_temp_c': [25] * 2,
            'supply_temp_c': [22] * 2,
            'price_krw_per_kwh': [100, 200],
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()['energy_cost_krw']) == 2
    bad = twin.post(
        '/api/v1/rollout',
        json={
            'it_power_kw': [3000] * 2,
            'outdoor_temp_c': [25] * 2,
            'supply_temp_c': [22] * 2,
            'price_krw_per_kwh': [100],
        },
    )
    assert bad.status_code == 422


def test_api_esg_with_prices() -> None:
    r = twin.post(
        '/api/v1/esg',
        json={
            'it_power_kw': [3000],
            'cooling_power_kw': [300],
            'cooling_mode': ['chiller'],
            'price_krw_per_kwh': [150],
        },
    )
    assert r.status_code == 200 and r.json()['energy_cost_krw'] == pytest.approx(
        (3000 * 1.05 + 300) * 150
    )


# ---------------------------------------------------------------- MPC 비용 목적


@needs_model
def test_mpc_plan_reports_cost_and_pue() -> None:
    raw = pd.read_parquet(repo_path('data'))
    hourly = raw.set_index('timestamp')[['it_power_kw', 'outdoor_temp_c']].resample('h').mean()
    now = pd.Timestamp('2026-06-04 06:00')
    hist = hourly.it_power_kw.loc[now - pd.Timedelta(hours=192) : now - pd.Timedelta(hours=1)]
    history = [{'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()]
    outdoor = hourly.outdoor_temp_c.loc[now : now + pd.Timedelta(hours=23)].tolist()
    ctl = MPCController(
        TestClient(twin_app), TestClient(forecast_svc.app), MPCConfig(de_maxiter=3, de_popsize=3)
    )
    r = ctl.plan(20.0, history, outdoor)
    assert r.mode == 'mpc' and r.objective == 'cost'
    assert r.price_krw_per_kwh[9] == 209.9  # 06시 + 9 = 15시 최대부하
    assert len(r.predicted_pue) == 24 and min(r.predicted_pue) > 1.0
    assert r.expected_energy_cost_krw < r.baseline_energy_cost_krw
    assert r.expected_cost_saving_pct > 0


def test_mpc_objective_validation() -> None:
    with pytest.raises(ValueError, match='objective'):
        MPCConfig(objective='pue')


@needs_model
def test_simulation_records_cost() -> None:
    r = simulation.run_closed_loop(
        TestClient(twin_app),
        TestClient(forecast_svc.app),
        pd.Timestamp('2026-06-04 14:00'),
        3,
        'fixed_20',
    )
    ts, s = r['timeseries'], r['summary']
    assert ts['tariff_period'] == ['mid', 'peak', 'peak']
    assert ts['energy_cost_krw'] == pytest.approx(
        [p * f for p, f in zip(ts['price_krw_per_kwh'], ts['facility_power_kw'])]
    )
    assert s['energy_cost_krw'] == pytest.approx(sum(ts['energy_cost_krw']))
