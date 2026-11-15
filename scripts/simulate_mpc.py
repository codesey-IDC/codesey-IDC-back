"""MPC 폐루프 시뮬레이션: Test 구간 동안 매시간 제어기가 설정 온도를 정하고 트윈을 1시간 진행.

비교 제어기
  fixed_20   기준 20℃ 고정 (원본 데이터의 운영 방식)
  rule_pid   Rule-based: 서버실 온도를 목표 20℃ 에 맞추는 PID (과제의 MPC 비교 기준)
  rule_tes   Rule-based PID + 축열조 요금 시간표 운전 (경부하 충전, 최대부하 방전)
  fixed_26   26℃ 고정 (허용 한계 바로 아래)
  reactive   반응형: 직전 1시간 부하 + 현재 외기만 보고 그 순간 정상 상태 최적값 선택 (예측 없음)
  mpc        예측 서비스 24시간 예측 + 트윈으로 24시간 계획 최적화, 첫 1시간만 적용

설비(plant) 역할도 트윈이 한다: 실제 시간별 IT 부하·외기로 /api/v1/rollout 을 1시간씩 진행.
모든 모듈 호출은 REST(FastAPI TestClient) 로 한다.

가정
  - 외기온도 예보는 실제값과 같다 (perfect forecast). 오차의 영향은 scripts/sensitivity_analysis.py 로 본다.
  - IT 부하는 MPC 가 forecast-service 예측값(q50)을 쓴다 (실제값을 모름).
  - 1시간 평균값으로 진행한다 (5분 변동 무시).

실행:
  python scripts/simulate_mpc.py                     # Test 14일 (2026-05-30 ~ 06-12)
  python scripts/simulate_mpc.py --days 3
  python scripts/simulate_mpc.py --surge             # 6/5 10~15시 IT 부하 1.3배 (서버 급증) 시나리오
결과: output/mpc/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트 (services 패키지)

import argparse
import json
import os
import time
import warnings
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=DeprecationWarning)
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from services.common.settings import repo_path, section  # noqa: E402
from services.forecast.main import app as forecast_app  # noqa: E402
from services.mpc.controller import MPCConfig, MPCController  # noqa: E402
from services.mpc.rule_based import PIDController, tes_schedule_kw  # noqa: E402
from services.mpc.simulation import _reactive_setpoint  # noqa: E402
from services.twin.main import app as twin_app  # noqa: E402

DATA_PATH = repo_path('data')
OUT_DIR = str(repo_path('mpc_results'))
TEST_START = pd.Timestamp(section('simulation')['test_start'])
HISTORY_HOURS: int = section('simulation')['history_hours']


@contextmanager
def open_clients() -> Iterator[tuple[TestClient, TestClient]]:
    """트윈·예측 서비스 TestClient 를 연결 하나로 열어 둔다.

    with 없이 TestClient 를 쓰면 요청마다 이벤트 루프를 새로 만든다. Windows 에서는 그때마다
    내부 소켓 쌍을 만드는데, 수만 번 반복하면(MPC 14일) 소켓 생성이 응답 없이 멈추는 일이 있었다.
    """
    with TestClient(twin_app) as twin, TestClient(forecast_app) as forecast:
        yield twin, forecast


def load_hourly() -> pd.DataFrame:
    raw = pd.read_parquet(DATA_PATH).set_index('timestamp')
    return raw[['it_power_kw', 'outdoor_temp_c']].resample('h').mean()


class NoisyForecast:
    """예측 서비스 응답에 일부러 오차를 섞는 대리 클라이언트 (민감도 분석용).

    예측값 × (1 + bias + noise), noise ~ N(0, noise_pct%) 를 시간마다 따로 뽑는다.
    오차 0 이면 예측 서비스 응답을 그대로 돌려준다.
    """

    def __init__(
        self,
        inner: TestClient,
        noise_pct: float = 0.0,
        bias_pct: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.inner, self.noise, self.bias = inner, noise_pct / 100, bias_pct / 100
        self.rng = rng or np.random.default_rng(0)

    def post(self, url: str, json: dict) -> httpx.Response:
        r = self.inner.post(url, json=json)
        if r.status_code != 200 or not (self.noise or self.bias):
            return r
        body = r.json()
        pred = np.asarray(body['predictions'], dtype=float)
        scale = (
            1 + self.bias + self.rng.normal(0, self.noise, len(pred))
            if self.noise
            else 1 + self.bias
        )
        body['predictions'] = (pred * scale).tolist()
        return httpx.Response(200, json=body, request=r.request)


def run(
    hourly: pd.DataFrame,
    start: pd.Timestamp,
    hours: int,
    controller: str,
    twin: TestClient,
    forecast: TestClient | NoisyForecast,
    latency_h: int = 0,
    outdoor_sigma_c: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """controller 로 [start, start + hours) 를 1시간씩 진행한다.

    latency_h: 제어 지연. k 시에 내린 결정이 k + latency_h 시에 적용된다 (그 전에는 직전 값 유지).
    outdoor_sigma_c: MPC 에 주는 외기 예보에 N(0, σ) 오차를 시간마다 더한다 (실제 설비에는 실제값).
    """
    rng = np.random.default_rng(seed)
    params = twin.get('/api/v1/params').json()
    ref = params['supply_temp_ref_c']
    mpc = MPCController(twin, forecast, MPCConfig()) if controller == 'mpc' else None
    pid = PIDController() if controller in ('rule_pid', 'rule_tes') else None
    tariff = twin.get('/api/v1/tariff', params={'start': start.isoformat(), 'hours': hours}).json()
    stamps = pd.to_datetime(tariff['timestamps'])
    prices = pd.Series(tariff['price_krw_per_kwh'], index=stamps)
    periods = pd.Series(tariff['period'], index=stamps)

    room, rows, fallbacks, naive = float(ref), [], 0, 0
    soc = float(params['tes_capacity_kwh'] * params['tes_initial_soc'])
    last_sp = float(ref)  # 직전에 적용한 설정 온도 (MPC 변화율 제약 기준)
    pending: deque[tuple[float, float]] = deque()  # 아직 적용되지 않은 (설정 온도, 축열조) 결정
    for k in range(hours):
        now = start + pd.Timedelta(hours=k)
        actual = hourly.loc[now]
        tes_cmd = 0.0
        mode = controller
        if controller == 'fixed_20':
            sp = ref
        elif controller in ('rule_pid', 'rule_tes'):
            sp = pid.update(room)
            if controller == 'rule_tes':
                tes_cmd = tes_schedule_kw(
                    periods[now], params['tes_max_charge_kw'], params['tes_max_discharge_kw']
                )
        elif controller == 'fixed_26':
            sp = 26.0
        elif controller == 'reactive':
            prev = hourly.loc[now - pd.Timedelta(hours=1)]
            sp = _reactive_setpoint(
                twin,
                params,
                MPCConfig().reliability_weight_kwh_per_h,
                float(prev.it_power_kw),
                float(actual.outdoor_temp_c),
            )
        else:
            hist = hourly.it_power_kw.loc[
                now - pd.Timedelta(hours=HISTORY_HOURS) : now - pd.Timedelta(hours=1)
            ]
            history = [
                {'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()
            ]
            outdoor = hourly.outdoor_temp_c.loc[now : now + pd.Timedelta(hours=23)].tolist()
            outdoor += [outdoor[-1]] * (24 - len(outdoor))  # 데이터 끝: 마지막 값 유지
            if outdoor_sigma_c:
                outdoor = (np.asarray(outdoor) + rng.normal(0, outdoor_sigma_c, 24)).tolist()
            plan = mpc.plan(room, history, outdoor, tes_soc_kwh=soc, current_setpoint_c=last_sp)
            sp, mode, tes_cmd = plan.setpoint_c, plan.mode, plan.tes_kw
            fallbacks += int(plan.mode == 'fallback')
            naive += int(plan.mode == 'mpc_naive')

        # 제어 지연: 지금 결정은 latency_h 시간 뒤에 적용, 지금은 그만큼 전에 내린 결정을 적용
        pending.append((sp, tes_cmd))
        sp, tes_cmd = pending.popleft() if len(pending) > latency_h else (last_sp, 0.0)

        step = twin.post(
            '/api/v1/rollout',
            json={
                'it_power_kw': [float(actual.it_power_kw)],
                'outdoor_temp_c': [float(actual.outdoor_temp_c)],
                'supply_temp_c': [sp],
                'initial_room_temp_c': room,
                'price_krw_per_kwh': [float(prices[now])],
                'tes_kw': [tes_cmd],
                'initial_tes_kwh': soc,
            },
        ).json()
        room, soc, last_sp = step['room_temp_c'][0], step['tes_soc_kwh'][0], sp
        rows.append(
            {
                'timestamp': now,
                'controller': controller,
                'mode': mode,
                'setpoint_c': sp,
                'it_base_kw': float(actual.it_power_kw),
                'outdoor_temp_c': float(actual.outdoor_temp_c),
                'room_temp_c': room,
                'it_power_kw': step['it_power_kw'][0],
                'fan_extra_kw': step['server_fan_extra_kw'][0],
                'cooling_power_kw': step['cooling_power_kw'][0],
                'facility_power_kw': step['facility_power_kw'][0],
                'pue': step['pue'][0],
                'ride_through_min': step['ride_through_min'][0],
                'xfactor_avg': step['average_xfactor'],
                'violation': step['violation_steps'] > 0,
                'price_krw_per_kwh': float(prices[now]),
                'energy_cost_krw': step['energy_cost_krw'][0],
                'tes_kw': step['tes_kw'][0],
                'tes_soc_kwh': soc,
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        'hours': hours,
        'energy_cost_krw': float(df.energy_cost_krw.sum()),
        'facility_energy_kwh': float(df.facility_power_kw.sum()),
        'cooling_energy_kwh': float(df.cooling_power_kw.sum()),
        'fan_extra_energy_kwh': float(df.fan_extra_kw.sum()),
        'it_energy_kwh': float(df.it_power_kw.sum()),
        'average_pue': float(df.facility_power_kw.sum() / df.it_power_kw.sum()),
        'average_xfactor': float(df.xfactor_avg.mean()),
        'max_room_temp_c': float(df.room_temp_c.max()),
        'min_ride_through_min': float(df.ride_through_min.min()),
        'violation_hours': int(df.violation.sum()),
        'setpoint_change_c_total': float(np.abs(np.diff(df.setpoint_c, prepend=ref)).sum()),
        'fallback_hours': fallbacks,
        'naive_forecast_hours': naive,
        # 시작 저장량보다 적으면 그만큼 공짜로 쓴 셈 (비교 시 확인)
        'final_tes_soc_kwh': float(soc),
    }
    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser(description='MPC 폐루프 시뮬레이션')
    ap.add_argument('--days', type=int, default=14)
    ap.add_argument('--start', default=str(TEST_START))
    ap.add_argument('--surge', action='store_true', help='6/5 10~15시 IT 부하 1.3배')
    ap.add_argument('--controllers', default='fixed_20,rule_pid,rule_tes,fixed_26,reactive,mpc')
    args = ap.parse_args()

    hourly = load_hourly()
    tag = 'surge' if args.surge else 'base'
    if args.surge:
        surge = (hourly.index >= '2026-06-05 10:00') & (hourly.index < '2026-06-05 16:00')
        hourly.loc[surge, 'it_power_kw'] *= 1.3
    start, hours = pd.Timestamp(args.start), args.days * 24

    os.makedirs(OUT_DIR, exist_ok=True)
    frames, summaries = [], {}
    with open_clients() as (twin, forecast):
        for name in args.controllers.split(','):
            t0 = time.time()
            df, s = run(hourly, start, hours, name, twin, forecast)
            frames.append(df)
            summaries[name] = s
            print(f'{name:9s} 완료 ({time.time() - t0:.0f}초)')

    first = next(iter(summaries.values()))
    base, rule = summaries.get('fixed_20', first), summaries.get(
        'rule_pid', summaries.get('fixed_20', first)
    )
    for s in summaries.values():
        s['saving_vs_fixed_20_pct'] = (
            1 - s['facility_energy_kwh'] / base['facility_energy_kwh']
        ) * 100
        s['saving_vs_rule_pid_pct'] = (
            1 - s['facility_energy_kwh'] / rule['facility_energy_kwh']
        ) * 100
        s['cost_saving_vs_rule_pid_pct'] = (
            1 - s['energy_cost_krw'] / rule['energy_cost_krw']
        ) * 100

    print(f'\n[{tag}] {start} 부터 {hours}시간   (Rule-based(PID) 대비 절감률: 요금% / 전력량%)')
    print(
        f'{"":10s}{"요금 백만원":>11s}{"요금%":>7s}{"전체 kWh":>11s}{"전력%":>7s}{"냉각 kWh":>10s}{"팬 kWh":>8s}'
        f'{"PUE":>7s}{"고장률":>7s}{"최고℃":>7s}{"여유분":>7s}{"위반h":>6s}{"변경℃":>7s}'
        f'{"FB":>4s}{"naive":>6s}'
    )
    for name, s in summaries.items():
        print(
            f'{name:10s}{s["energy_cost_krw"] / 1e6:11.2f}{s["cost_saving_vs_rule_pid_pct"]:7.2f}'
            f'{s["facility_energy_kwh"]:11.0f}{s["saving_vs_rule_pid_pct"]:7.2f}'
            f'{s["cooling_energy_kwh"]:10.0f}{s["fan_extra_energy_kwh"]:8.0f}'
            f'{s["average_pue"]:7.3f}'
            f'{s["average_xfactor"]:7.2f}{s["max_room_temp_c"]:7.1f}'
            f'{s["min_ride_through_min"]:7.1f}'
            f'{s["violation_hours"]:6d}{s["setpoint_change_c_total"]:7.0f}'
            f'{s["fallback_hours"]:4d}'
            f'{s["naive_forecast_hours"]:6d}'
        )

    pd.concat(frames).to_csv(os.path.join(OUT_DIR, f'timeseries_{tag}.csv'), index=False)
    with open(os.path.join(OUT_DIR, f'metrics_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {'start': str(start), 'hours': hours, 'scenario': tag, 'controllers': summaries},
            f,
            ensure_ascii=False,
            indent=4,
        )
    print(f'\n저장: {OUT_DIR}/metrics_{tag}.json, timeseries_{tag}.csv')


if __name__ == '__main__':
    main()
