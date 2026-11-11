"""MPC 폐루프 시뮬레이션: Test 구간 동안 매시간 제어기가 설정 온도를 정하고 트윈을 1시간 진행.

비교 제어기
  fixed_20   기준 20℃ 고정 (원본 데이터의 운영 방식)
  fixed_26   26℃ 고정 (허용 한계 바로 아래)
  reactive   반응형: 직전 1시간 부하 + 현재 외기만 보고 그 순간 정상 상태 최적값 선택 (예측 없음)
  mpc        예측 서비스 24시간 예측 + 트윈으로 24시간 계획 최적화, 첫 1시간만 적용

설비(plant) 역할도 트윈이 한다: 실제 시간별 IT 부하·외기로 /api/v1/rollout 을 1시간씩 진행.
모든 모듈 호출은 REST(FastAPI TestClient) 로 한다.

가정
  - 외기온도 예보는 실제값과 같다 (perfect forecast). 예보 오차는 반영하지 않음.
  - IT 부하는 MPC 가 forecast-service 예측값(q50)을 쓴다 (실제값을 모름).
  - 1시간 평균값으로 진행한다 (5분 변동 무시).

실행:
  python simulate_mpc.py                     # Test 14일 (2026-05-30 ~ 06-12)
  python simulate_mpc.py --days 3
  python simulate_mpc.py --surge             # 6/5 10~15시 IT 부하 1.3배 (서버 급증) 시나리오
결과: output/mpc/
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=DeprecationWarning)
from fastapi.testclient import TestClient  # noqa: E402

from services.forecast.main import app as forecast_app  # noqa: E402
from services.mpc.controller import MPCConfig, MPCController  # noqa: E402
from services.twin.main import app as twin_app  # noqa: E402

DATA_PATH = 'data/synthetic_idc_90d.parquet'
OUT_DIR = 'output/mpc'
TEST_START = pd.Timestamp('2026-05-30 00:00')
HISTORY_HOURS = 192


def load_hourly() -> pd.DataFrame:
    raw = pd.read_parquet(DATA_PATH).set_index('timestamp')
    return raw[['it_power_kw', 'outdoor_temp_c']].resample('h').mean()


class Reactive:
    """예측 없이 지금 조건만 보고 정상 상태 최적값을 고르는 일반 제어."""

    def __init__(self, twin: TestClient, reliability_weight: float, params: dict) -> None:
        self.twin, self.w, self.min_ride = twin, reliability_weight, params['min_ride_through_min']
        # 정상 상태에서 서버실 = 설정 온도이므로 후보는 서버실 허용 범위 안에서만
        self.grid = np.arange(params['room_temp_min_c'], params['room_temp_max_c'] + 1e-9, 0.5)

    def decide(self, it_kw: float, outdoor_c: float) -> float:
        r = self.twin.post('/api/v1/simulate', json={'points': [
            {'it_power_kw': it_kw, 'outdoor_temp_c': outdoor_c, 'supply_temp_c': float(s)} for s in self.grid]})
        res = r.json()['results']
        cost = [x['facility_power_kw'] + self.w * (x['xfactor'] - 1.0)
                + (1e6 if x['ride_through_min'] < self.min_ride else 0.0) for x in res]
        return float(self.grid[int(np.argmin(cost))])


def run(hourly: pd.DataFrame, start: pd.Timestamp, hours: int, controller: str,
        twin: TestClient, forecast: TestClient) -> tuple[pd.DataFrame, dict]:
    params = twin.get('/api/v1/params').json()
    ref = params['supply_temp_ref_c']
    mpc = MPCController(twin, forecast, MPCConfig()) if controller == 'mpc' else None
    reactive = Reactive(twin, MPCConfig().reliability_weight_kwh_per_h, params) \
        if controller == 'reactive' else None

    room, rows, fallbacks, naive = float(ref), [], 0, 0
    for k in range(hours):
        now = start + pd.Timedelta(hours=k)
        actual = hourly.loc[now]
        mode = controller
        if controller == 'fixed_20':
            sp = ref
        elif controller == 'fixed_26':
            sp = 26.0
        elif controller == 'reactive':
            prev = hourly.loc[now - pd.Timedelta(hours=1)]
            sp = reactive.decide(float(prev.it_power_kw), float(actual.outdoor_temp_c))
        else:
            hist = hourly.it_power_kw.loc[now - pd.Timedelta(hours=HISTORY_HOURS):now - pd.Timedelta(hours=1)]
            history = [{'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()]
            outdoor = hourly.outdoor_temp_c.loc[now:now + pd.Timedelta(hours=23)].tolist()
            outdoor += [outdoor[-1]] * (24 - len(outdoor))     # 데이터 끝: 마지막 값 유지
            plan = mpc.plan(room, history, outdoor)
            sp, mode = plan.setpoint_c, plan.mode
            fallbacks += int(plan.mode == 'fallback')
            naive += int(plan.mode == 'mpc_naive')

        step = twin.post('/api/v1/rollout', json={
            'it_power_kw': [float(actual.it_power_kw)], 'outdoor_temp_c': [float(actual.outdoor_temp_c)],
            'supply_temp_c': [sp], 'initial_room_temp_c': room}).json()
        room = step['room_temp_c'][0]
        rows.append({
            'timestamp': now, 'controller': controller, 'mode': mode, 'setpoint_c': sp,
            'it_base_kw': float(actual.it_power_kw), 'outdoor_temp_c': float(actual.outdoor_temp_c),
            'room_temp_c': room, 'it_power_kw': step['it_power_kw'][0],
            'fan_extra_kw': step['server_fan_extra_kw'][0], 'cooling_power_kw': step['cooling_power_kw'][0],
            'facility_power_kw': step['facility_power_kw'][0], 'pue': step['pue'][0],
            'ride_through_min': step['ride_through_min'][0], 'xfactor_avg': step['average_xfactor'],
            'violation': step['violation_steps'] > 0,
        })

    df = pd.DataFrame(rows)
    summary = {
        'hours': hours,
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
    }
    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser(description='MPC 폐루프 시뮬레이션')
    ap.add_argument('--days', type=int, default=14)
    ap.add_argument('--start', default=str(TEST_START))
    ap.add_argument('--surge', action='store_true', help='6/5 10~15시 IT 부하 1.3배')
    ap.add_argument('--controllers', default='fixed_20,fixed_26,reactive,mpc')
    args = ap.parse_args()

    hourly = load_hourly()
    tag = 'surge' if args.surge else 'base'
    if args.surge:
        surge = (hourly.index >= '2026-06-05 10:00') & (hourly.index < '2026-06-05 16:00')
        hourly.loc[surge, 'it_power_kw'] *= 1.3
    start, hours = pd.Timestamp(args.start), args.days * 24

    twin, forecast = TestClient(twin_app), TestClient(forecast_app)
    os.makedirs(OUT_DIR, exist_ok=True)
    frames, summaries = [], {}
    for name in args.controllers.split(','):
        t0 = time.time()
        df, s = run(hourly, start, hours, name, twin, forecast)
        frames.append(df)
        summaries[name] = s
        print(f'{name:9s} 완료 ({time.time() - t0:.0f}초)')

    base_e = summaries.get('fixed_20', next(iter(summaries.values())))['facility_energy_kwh']
    for s in summaries.values():
        s['saving_vs_fixed_20_pct'] = (1 - s['facility_energy_kwh'] / base_e) * 100

    print(f'\n[{tag}] {start} 부터 {hours}시간')
    print(f'{"":10s}{"전체 kWh":>11s}{"절감%":>8s}{"냉각 kWh":>10s}{"팬 kWh":>8s}{"PUE":>7s}'
          f'{"고장률":>7s}{"최고℃":>7s}{"여유분":>7s}{"위반h":>6s}{"변경℃":>7s}{"FB":>4s}{"naive":>6s}')
    for name, s in summaries.items():
        print(f'{name:10s}{s["facility_energy_kwh"]:11.0f}{s["saving_vs_fixed_20_pct"]:8.2f}'
              f'{s["cooling_energy_kwh"]:10.0f}{s["fan_extra_energy_kwh"]:8.0f}{s["average_pue"]:7.3f}'
              f'{s["average_xfactor"]:7.2f}{s["max_room_temp_c"]:7.1f}{s["min_ride_through_min"]:7.1f}'
              f'{s["violation_hours"]:6d}{s["setpoint_change_c_total"]:7.0f}{s["fallback_hours"]:4d}'
              f'{s["naive_forecast_hours"]:6d}')

    pd.concat(frames).to_csv(os.path.join(OUT_DIR, f'timeseries_{tag}.csv'), index=False)
    with open(os.path.join(OUT_DIR, f'metrics_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump({'start': str(start), 'hours': hours, 'scenario': tag, 'controllers': summaries},
                  f, ensure_ascii=False, indent=4)
    print(f'\n저장: {OUT_DIR}/metrics_{tag}.json, timeseries_{tag}.csv')


if __name__ == '__main__':
    main()
