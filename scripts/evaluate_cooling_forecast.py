"""냉각 수요(냉각 전력) 예측 평가 — forecast-service target='cooling_power' 를 REST 로 호출.

Test 구간(2026-05-30 ~ 06-12)의 매시각에서 앞으로 24시간 냉각 전력을 예측하고 실제값(1시간 평균)과 비교한다.

  nMAE          Σ|예측 - 실제| / Σ실제 × 100 (과제 기준 10% 이내)
  구간 적중률    실제값이 90% 예측 구간 안에 든 비율
  모드 적중률    예상 냉각 모드 = 실제 냉각 모드(그 시간 5분값의 최빈값) 비율
  구조 오차      IT 부하를 실제값으로 넣었을 때의 nMAE (1시간 평균 처리 등 모델 구조에서 오는 오차)
  외기 민감도    외기 예보에 평균 0, 표준편차 σ℃ 오차를 더했을 때 nMAE·모드 적중률 (σ = 0, 1, 2, 3)
  냉각 방식별    Test 구간 실제 운전의 냉각 방식별 시간·냉각 전력량·PUE, 대표 외기에서의 PUE·하루 전력량

실행:  python scripts/evaluate_cooling_forecast.py
결과:  output/forecast/cooling_metrics.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트 (services 패키지)

import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=DeprecationWarning)
from fastapi.testclient import TestClient  # noqa: E402

from services.common.settings import repo_path, section  # noqa: E402
from services.forecast import main as forecast_svc  # noqa: E402
from services.twin.main import app as twin_app  # noqa: E402

RESULT_PATH = repo_path('forecast_models').parent / 'cooling_metrics.json'
TEST_START = pd.Timestamp(section('simulation')['test_start'])
TEST_END = pd.Timestamp('2026-06-12 23:00')
HORIZON = 24
HISTORY_HOURS = section('simulation')['history_hours']
NOISE_SIGMAS_C = (0.0, 1.0, 2.0, 3.0)
SEED = 0


def load_hourly() -> pd.DataFrame:
    raw = pd.read_parquet(repo_path('data')).set_index('timestamp')
    return raw.resample('h').agg(
        {
            'it_power_kw': 'mean',
            'outdoor_temp_c': 'mean',
            'cooling_power_kw': 'mean',
            'cooling_mode': lambda s: s.mode().iloc[0],
        }
    )


def nmae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.abs(pred - actual).sum() / np.abs(actual).sum() * 100)


def forecast_rows(
    client: TestClient, hourly: pd.DataFrame, sigma: float, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    origins = pd.date_range(TEST_START, TEST_END - pd.Timedelta(hours=HORIZON - 1), freq='h')
    for now in origins:
        hist = hourly.it_power_kw.loc[
            now - pd.Timedelta(hours=HISTORY_HOURS) : now - pd.Timedelta(hours=1)
        ]
        future = hourly.loc[now : now + pd.Timedelta(hours=HORIZON - 1)]
        outdoor = (
            (future.outdoor_temp_c.to_numpy() + rng.normal(0, sigma, HORIZON)) if sigma else None
        )
        r = client.post(
            '/api/v1/forecast',
            json={
                'horizon_hours': HORIZON,
                'target': 'cooling_power',
                'history': [
                    {'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()
                ],
                'outdoor_forecast_c': None if outdoor is None else outdoor.tolist(),
            },
        )
        r.raise_for_status()
        b = r.json()
        rows.append(
            pd.DataFrame(
                {
                    'origin': now,
                    'target': future.index,
                    'h': np.arange(1, HORIZON + 1),
                    'pred': b['predictions'],
                    'lo': b['pi_lower'],
                    'hi': b['pi_upper'],
                    'mode_pred': b['cooling_mode'],
                    'actual': future.cooling_power_kw.to_numpy(),
                    'mode_actual': future.cooling_mode.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame) -> dict:
    buckets = {'h1-6': (1, 6), 'h7-12': (7, 12), 'h13-24': (13, 24)}
    return {
        'rows': len(df),
        'nmae_pct': nmae(df.pred.to_numpy(), df.actual.to_numpy()),
        'nmae_by_horizon_pct': {
            k: nmae(g.pred.to_numpy(), g.actual.to_numpy())
            for k, (a, b) in buckets.items()
            for g in [df[df.h.between(a, b)]]
        },
        'coverage_pct': float(((df.actual >= df.lo) & (df.actual <= df.hi)).mean() * 100),
        'mode_accuracy_pct': float((df.mode_pred == df.mode_actual).mean() * 100),
    }


def structure_error(twin: TestClient, hourly: pd.DataFrame) -> float:
    """IT 부하를 실제값으로 넣었을 때 (IT 예측 오차 0) 트윈 냉각 전력의 nMAE."""
    test = hourly.loc[TEST_START:TEST_END]
    r = twin.post(
        '/api/v1/simulate',
        json={
            'points': [
                {'it_power_kw': float(i), 'outdoor_temp_c': float(t)}
                for i, t in zip(test.it_power_kw, test.outdoor_temp_c)
            ]
        },
    )
    pred = np.array([x['cooling_power_kw'] for x in r.json()['results']])
    return nmae(pred, test.cooling_power_kw.to_numpy())


def cooling_mode_comparison(twin: TestClient, hourly: pd.DataFrame) -> dict:
    """냉각 방식별 에너지: Test 구간 실제 운전 집계 + 대표 외기(IT 3,000kW, 기준 설정)에서의 비교."""
    raw = (
        pd.read_parquet(repo_path('data'))
        .set_index('timestamp')
        .loc[TEST_START : TEST_END + pd.Timedelta(minutes=55)]
    )
    actual = {}
    for mode, g in raw.groupby('cooling_mode'):
        hours = len(g) / 12
        actual[mode] = {
            'hours': hours,
            'cooling_energy_mwh': float(g.cooling_power_kw.sum() / 12 / 1000),
            'average_pue': float(g.pue.mean()),
            'cooling_kw_per_it_kw': float(g.cooling_power_kw.sum() / g.it_power_kw.sum()),
        }
    typical = {}
    for label, t in (
        ('free_cooling (외기 5℃)', 5.0),
        ('hybrid (외기 16.5℃)', 16.5),
        ('chiller (외기 25℃)', 25.0),
    ):
        x = twin.post(
            '/api/v1/simulate', json={'points': [{'it_power_kw': 3000.0, 'outdoor_temp_c': t}]}
        ).json()
        x = x['results'][0]
        typical[label] = {
            'cooling_mode': x['cooling_mode'],
            'pue': x['pue'],
            'cooling_power_kw': x['cooling_power_kw'],
            'daily_facility_mwh': x['facility_power_kw'] * 24 / 1000,
        }
    return {'test_period_actual': actual, 'typical_it_3000kw': typical}


def main() -> None:
    hourly = load_hourly()
    twin = TestClient(twin_app)
    forecast_svc.set_twin_client(twin)
    client = TestClient(forecast_svc.app)
    rng = np.random.default_rng(SEED)

    sensitivity = {}
    for sigma in NOISE_SIGMAS_C:
        sensitivity[f'sigma_{sigma:g}c'] = summarize(forecast_rows(client, hourly, sigma, rng))
        s = sensitivity[f'sigma_{sigma:g}c']
        print(
            f'외기 오차 σ={sigma:g}℃: nMAE {s["nmae_pct"]:5.2f}%  '
            f'모드 적중 {s["mode_accuracy_pct"]:5.1f}%  '
            f'구간 적중 {s["coverage_pct"]:5.1f}%',
            flush=True,
        )

    base = sensitivity['sigma_0c']
    report = {
        'period': {'start': str(TEST_START), 'end': str(TEST_END), 'horizon_hours': HORIZON},
        'criteria': {'nmae_max_pct': 10.0},
        'cooling_power_forecast': base,
        'structure_error_nmae_pct': structure_error(twin, hourly),
        'outdoor_error_sensitivity': sensitivity,
        'cooling_mode_comparison': cooling_mode_comparison(twin, hourly),
        'assumption': '외기 예보 = 실측 (σ=0). 냉각 전력은 기준 설정 온도(20℃) 운전 기준',
    }
    print('\n=== 냉각 수요 예측 (Test, 24시간 앞) ===')
    print(
        f'nMAE {base["nmae_pct"]:.2f}% (기준 10% 이내)  구간별 '
        + ', '.join(f'{k} {v:.2f}%' for k, v in base['nmae_by_horizon_pct'].items())
    )
    print(
        f'90% 구간 적중률 {base["coverage_pct"]:.1f}%, 냉각 모드 적중률 {base["mode_accuracy_pct"]:.1f}%'
    )
    print(f'구조 오차 (IT 실제값 사용) nMAE {report["structure_error_nmae_pct"]:.2f}%')
    print('\n=== 냉각 방식별 (IT 3,000kW, 기준 설정) ===')
    for label, x in report['cooling_mode_comparison']['typical_it_3000kw'].items():
        print(
            f'{label:22s} PUE {x["pue"]:.3f}  냉각 {x["cooling_power_kw"]:7.0f}kW  '
            f'하루 {x["daily_facility_mwh"]:6.1f}MWh'
        )
    print('Test 구간 실제 운전:')
    for mode, x in report['cooling_mode_comparison']['test_period_actual'].items():
        print(
            f'  {mode:13s} {x["hours"]:5.0f}시간  냉각 {x["cooling_energy_mwh"]:6.1f}MWh  '
            f'평균 PUE {x["average_pue"]:.3f}  '
            f'IT 1kW당 냉각 {x["cooling_kw_per_it_kw"]:.3f}kW'
        )
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f'\n저장: {RESULT_PATH}')


if __name__ == '__main__':
    main()
