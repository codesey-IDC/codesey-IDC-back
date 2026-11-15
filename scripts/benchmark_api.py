"""API 응답 시간 측정 (과제 기준: 95th percentile 10초 이내).

세 서비스를 띄운 상태에서 실행한다 (start.bat 또는 docker-compose):
  python scripts/benchmark_api.py              # 엔드포인트마다 반복 호출해 평균·p95·최대
  python scripts/benchmark_api.py --repeat 10
결과: output/benchmark/api_latency.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트 (services 패키지)

import argparse
import json
import os
import time
from collections.abc import Callable

import httpx
import numpy as np

from services.common.settings import repo_path, service_url

LIMIT_S = 10.0
START = '2026-06-04T06:00:00'


def cases(c: httpx.Client) -> list[tuple[str, Callable[[], httpx.Response]]]:
    twin, fc, mpc = service_url('twin'), service_url('forecast'), service_url('mpc')
    tele = c.get(
        f'{twin}/api/v1/telemetry', params={'start': '2026-05-27T06:00:00', 'hours': 192}
    ).json()
    history = [
        {'timestamp': t, 'it_power_kw': v} for t, v in zip(tele['timestamps'], tele['it_power_kw'])
    ]
    plan = {'room_temp_c': 22.0, 'history': history, 'outdoor_forecast_c': [25.0] * 24}
    state = {
        'timestamp': START,
        'room_temp_c': 22.0,
        'current_setpoint_c': 22.0,
        'tes_soc_kwh': 6000.0,
    }
    rollout = {
        'it_power_kw': [3000.0] * 24,
        'outdoor_temp_c': [25.0] * 24,
        'supply_temp_c': [24.0] * 24,
    }
    return [
        ('GET  twin /health', lambda: c.get(f'{twin}/health')),
        ('GET  twin /api/v1/params', lambda: c.get(f'{twin}/api/v1/params')),
        (
            'POST twin /api/v1/simulate',
            lambda: c.post(
                f'{twin}/api/v1/simulate',
                json={'points': [{'it_power_kw': 3000.0, 'outdoor_temp_c': 25.0}] * 24},
            ),
        ),
        ('POST twin /api/v1/rollout', lambda: c.post(f'{twin}/api/v1/rollout', json=rollout)),
        (
            'POST twin /api/v1/rollout/batch (108개)',
            lambda: c.post(
                f'{twin}/api/v1/rollout/batch',
                json={
                    'it_power_kw': [3000.0] * 24,
                    'outdoor_temp_c': [25.0] * 24,
                    'plans': [[24.0] * 24] * 108,
                    'tes_plans': [[0.0] * 24] * 108,
                    'initial_room_temp_c': 22.0,
                },
            ),
        ),
        (
            'GET  twin /api/v1/telemetry',
            lambda: c.get(f'{twin}/api/v1/telemetry', params={'hours': 216}),
        ),
        (
            'GET  twin /api/v1/tariff',
            lambda: c.get(f'{twin}/api/v1/tariff', params={'start': START, 'hours': 24}),
        ),
        (
            'POST twin /api/v1/esg',
            lambda: c.post(
                f'{twin}/api/v1/esg',
                json={
                    'it_power_kw': [3000.0] * 24,
                    'cooling_power_kw': [500.0] * 24,
                    'cooling_mode': ['chiller'] * 24,
                },
            ),
        ),
        ('GET  forecast /health', lambda: c.get(f'{fc}/health')),
        (
            'POST forecast /api/v1/forecast (24h)',
            lambda: c.post(
                f'{fc}/api/v1/forecast', json={'horizon_hours': 24, 'history': history}
            ),
        ),
        (
            'POST forecast /api/v1/forecast (168h)',
            lambda: c.post(
                f'{fc}/api/v1/forecast', json={'horizon_hours': 168, 'history': history}
            ),
        ),
        (
            'POST forecast /api/v1/forecast (cooling_power)',
            lambda: c.post(
                f'{fc}/api/v1/forecast',
                json={
                    'horizon_hours': 24,
                    'target': 'cooling_power',
                    'history': history,
                    'outdoor_forecast_c': [25.0] * 24,
                },
            ),
        ),
        ('GET  mpc /health', lambda: c.get(f'{mpc}/health')),
        (
            'POST mpc /api/v1/control/optimize',
            lambda: c.post(f'{mpc}/api/v1/control/optimize', json=state),
        ),
        ('POST mpc /api/v1/control/mpc', lambda: c.post(f'{mpc}/api/v1/control/mpc', json=state)),
        ('POST mpc /api/v1/plan', lambda: c.post(f'{mpc}/api/v1/plan', json=plan)),
        ('GET  mpc /api/v1/scenarios', lambda: c.get(f'{mpc}/api/v1/scenarios')),
        (
            'GET  mpc /api/v1/results',
            lambda: c.get(f'{mpc}/api/v1/results', params={'scenario': 'base'}),
        ),
        (
            'POST mpc /api/v1/simulate (rule_pid 24h)',
            lambda: c.post(
                f'{mpc}/api/v1/simulate',
                json={'start': START, 'hours': 24, 'controller': 'rule_pid'},
            ),
        ),
        (
            'POST mpc /api/v1/simulate/jobs (mpc 72h 등록)',
            lambda: c.post(f'{mpc}/api/v1/simulate/jobs', json=JOB),
        ),
        (
            'GET  mpc /api/v1/simulate/jobs/{id} (상태)',
            lambda: c.get(f'{mpc}/api/v1/simulate/jobs/{_job_id(c)}'),
        ),
    ]


JOB = {'start': START, 'hours': 72, 'controller': 'mpc', 'scenario': 'normal'}


def _job_id(c: httpx.Client) -> str:
    return c.post(f'{service_url("mpc")}/api/v1/simulate/jobs', json=JOB).json()['job_id']


def job_completion_s(c: httpx.Client) -> float:
    """참고: 72시간 MPC 시뮬레이션 작업이 끝날 때까지 걸린 시간 (API 응답 시간이 아님)."""
    t = time.perf_counter()
    job = c.post(f'{service_url("mpc")}/api/v1/simulate/jobs', json=JOB | {'hours': 71}).json()
    while job['status'] in ('queued', 'running'):
        time.sleep(0.5)
        job = c.get(f'{service_url("mpc")}/api/v1/simulate/jobs/{job["job_id"]}').json()
    return time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser(description='API 응답 시간 측정')
    ap.add_argument('--repeat', type=int, default=20)
    args = ap.parse_args()
    results = {}
    with httpx.Client(timeout=600.0) as c:
        completion = job_completion_s(c)
        print(
            f'참고: MPC 71시간 시뮬레이션 작업 완료까지 {completion:.1f}초 (등록·상태 조회 API 는 바로 응답)',
            flush=True,
        )
        for name, call in cases(c):
            n = 3 if 'simulate (' in name else args.repeat
            times = []
            for _ in range(n):
                t = time.perf_counter()
                r = call()
                times.append(time.perf_counter() - t)
                if r.status_code >= 400:
                    raise SystemExit(f'{name}: HTTP {r.status_code} {r.text[:200]}')
            t = np.array(times)
            results[name] = {
                'n': n,
                'mean_s': float(t.mean()),
                'p95_s': float(np.percentile(t, 95)),
                'max_s': float(t.max()),
                'ok': bool(np.percentile(t, 95) <= LIMIT_S),
            }
            s = results[name]
            print(
                f'{name:48s} 평균 {s["mean_s"]:6.2f}s  p95 {s["p95_s"]:6.2f}s  '
                f'최대 {s["max_s"]:6.2f}s  '
                f'{"OK" if s["ok"] else "초과"}',
                flush=True,
            )
    out = repo_path('mpc_results').parent / 'benchmark'
    os.makedirs(out, exist_ok=True)
    with open(out / 'api_latency.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'limit_s': LIMIT_S,
                'endpoints': results,
                'mpc_simulation_job_completion_s': {'hours': 71, 'seconds': completion},
            },
            f,
            ensure_ascii=False,
            indent=4,
        )
    print(
        f'저장: {out / "api_latency.json"}  (기준 p95 {LIMIT_S:.0f}초 이내: '
        f'{sum(r["ok"] for r in results.values())}/{len(results)})'
    )


if __name__ == '__main__':
    main()
