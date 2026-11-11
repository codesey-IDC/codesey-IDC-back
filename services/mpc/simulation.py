"""폐루프 시뮬레이션: 매시간 제어기가 설정 온도를 정하고 트윈을 1시간 진행한다.

simulate_mpc.py(14일 일괄 실험 스크립트)와 같은 절차를 서비스 안에서 돌려,
대시보드가 구간·제어기·위기 시나리오를 골라 API 한 번으로 결과를 받게 한다.
계측값(IT 부하·외기)도 파일이 아니라 twin-service /api/v1/telemetry 로 받는다.

제어기
  fixed_20   기준 20℃ 고정 (원본 데이터의 운영 방식)
  reactive   예측 없이 직전 1시간 부하 + 현재 외기로 그 순간 최적값 선택
  mpc        24시간 예측 + 계획 최적화, 첫 1시간만 적용

위기 시나리오 (start 시각부터 구간 끝까지 적용)
  normal           변경 없음
  server_surge     IT 부하 1.3배
  chiller_failure  냉각 용량 75% (칠러 4대 중 1대 정지)
  heatwave         외기온도 최저 38℃

가정: 외기 예보는 실제값과 같다. 1시간 평균값으로 진행한다 (5분 변동 무시).
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd

from services.mpc.controller import MPCConfig, MPCController

HISTORY_HOURS = 192                     # 예측 모델 입력(최소 168시간)에 여유를 둔 이력 길이
CONTROLLERS = ('fixed_20', 'reactive', 'mpc')


@dataclass(frozen=True)
class Scenario:
    label: str
    strategy: str                       # 대응 전략 설명 (대시보드 표시용)
    it_scale: float = 1.0
    outdoor_min_c: float | None = None
    capacity_factor: float = 1.0


SCENARIOS: dict[str, Scenario] = {
    'normal': Scenario('정상 운영', '예측 기반 설정 온도 최적화'),
    'server_surge': Scenario('서버 급증 (IT 부하 +30%)',
                             '학습 범위 밖 부하 → 예측 대신 최근 최대 부하 유지 가정(mpc_naive)으로 계획', it_scale=1.3),
    'chiller_failure': Scenario('칠러 고장 (냉각 용량 75%)',
                                '남은 냉각 용량 안에서 운전, 용량 한계에 걸리지 않게 설정 온도 조정', capacity_factor=0.75),
    'heatwave': Scenario('폭염 (외기 38℃ 이상)',
                         '외기 냉각 불가 → 칠러 효율이 좋아지는 선까지만 설정 온도 상향', outdoor_min_c=38.0),
}


class SimulationError(ValueError):
    """요청 구간·제어기·시나리오가 잘못됐을 때 (API 에서 422)."""


def fetch_telemetry(twin: httpx.Client, start: pd.Timestamp, hours: int) -> pd.DataFrame:
    """시뮬레이션에 필요한 계측값: 이력(start 이전 192시간) + 구간 + 외기 예보용 23시간."""
    first = start - pd.Timedelta(hours=HISTORY_HOURS)
    r = twin.get('/api/v1/telemetry', params={'start': first.isoformat(), 'hours': HISTORY_HOURS + hours + 23})
    if r.status_code != 200:
        raise SimulationError(f'계측값을 받을 수 없습니다 (twin telemetry {r.status_code}): {r.text[:200]}')
    body = r.json()
    df = pd.DataFrame({'it_power_kw': body['it_power_kw'], 'outdoor_temp_c': body['outdoor_temp_c']},
                      index=pd.to_datetime(body['timestamps']))
    last = start + pd.Timedelta(hours=hours - 1)
    if df.index[0] > first or df.index[-1] < last:
        raise SimulationError(
            f'구간이 데이터 범위를 벗어납니다. 시작 시각은 {pd.Timestamp(body["data_start"]) + pd.Timedelta(hours=HISTORY_HOURS)} '
            f'이후, 끝 시각은 {body["data_end"]} 이전이어야 합니다 (요청: {start} ~ {last})')
    return df


def apply_scenario(df: pd.DataFrame, start: pd.Timestamp, scenario: Scenario) -> pd.DataFrame:
    out = df.copy()
    active = out.index >= start
    out.loc[active, 'it_power_kw'] *= scenario.it_scale
    if scenario.outdoor_min_c is not None:
        out.loc[active, 'outdoor_temp_c'] = out.loc[active, 'outdoor_temp_c'].clip(lower=scenario.outdoor_min_c)
    return out


def _reactive_setpoint(twin: httpx.Client, params: dict, weight: float, it_kw: float, outdoor_c: float) -> float:
    """지금 조건의 정상 상태에서 전체 전력이 가장 적은 설정 온도 (서버실 허용 범위 안)."""
    grid = np.arange(params['room_temp_min_c'], params['room_temp_max_c'] + 1e-9, 0.5)
    r = twin.post('/api/v1/simulate', json={'points': [
        {'it_power_kw': it_kw, 'outdoor_temp_c': outdoor_c, 'supply_temp_c': float(s)} for s in grid]})
    r.raise_for_status()
    cost = [x['facility_power_kw'] + weight * (x['xfactor'] - 1.0)
            + (1e6 if x['ride_through_min'] < params['min_ride_through_min'] else 0.0) for x in r.json()['results']]
    return float(grid[int(np.argmin(cost))])


def run_closed_loop(twin: httpx.Client, forecast: httpx.Client, start: pd.Timestamp, hours: int,
                    controller: str = 'mpc', scenario: str = 'normal',
                    config: MPCConfig = MPCConfig()) -> dict:
    """[start, start + hours) 를 1시간씩 진행한 시계열과 요약을 돌려준다."""
    if controller not in CONTROLLERS:
        raise SimulationError(f'controller 는 {list(CONTROLLERS)} 중 하나여야 합니다: {controller}')
    if scenario not in SCENARIOS:
        raise SimulationError(f'scenario 는 {list(SCENARIOS)} 중 하나여야 합니다: {scenario}')
    sc = SCENARIOS[scenario]
    start = pd.Timestamp(start).tz_localize(None).floor('h')
    data = apply_scenario(fetch_telemetry(twin, start, hours), start, sc)

    params = twin.get('/api/v1/params').json()
    ref = float(params['supply_temp_ref_c'])
    t_min, t_max = params['room_temp_min_c'], params['room_temp_max_c']
    mpc = MPCController(twin, forecast, config) if controller == 'mpc' else None

    room, rows, warnings, diverged = ref, [], [], False
    for k in range(hours):
        now = start + pd.Timedelta(hours=k)
        actual = data.loc[now]
        mode = controller
        if controller == 'fixed_20':
            setpoint = ref
        elif controller == 'reactive':
            prev = data.loc[now - pd.Timedelta(hours=1)]
            setpoint = _reactive_setpoint(twin, params, config.reliability_weight_kwh_per_h,
                                          float(prev.it_power_kw), float(actual.outdoor_temp_c))
        else:
            hist = data.it_power_kw.loc[now - pd.Timedelta(hours=HISTORY_HOURS):now - pd.Timedelta(hours=1)]
            history = [{'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()]
            outdoor = data.outdoor_temp_c.loc[now:now + pd.Timedelta(hours=config.horizon_hours - 1)].tolist()
            outdoor += [outdoor[-1]] * (config.horizon_hours - len(outdoor))    # 데이터 끝: 마지막 값 유지
            plan = mpc.plan(room, history, outdoor, sc.capacity_factor)
            setpoint, mode = plan.setpoint_c, plan.mode

        r = twin.post('/api/v1/rollout', json={
            'it_power_kw': [float(actual.it_power_kw)], 'outdoor_temp_c': [float(actual.outdoor_temp_c)],
            'supply_temp_c': [setpoint], 'initial_room_temp_c': room, 'capacity_factor': sc.capacity_factor})
        step = r.json() if r.status_code == 200 else None
        if step is None or step['room_temp_c'][0] is None:
            diverged = True
            warnings.append(f'diverged: {now} 에 서버실 온도({room:.1f}℃)가 트윈 계산 범위를 벗어나 중단')
            break
        room = step['room_temp_c'][0]
        rows.append({
            'timestamp': now.isoformat(), 'mode': mode, 'setpoint_c': setpoint,
            'it_base_kw': float(actual.it_power_kw), 'outdoor_temp_c': float(actual.outdoor_temp_c),
            'room_temp_c': room, 'it_power_kw': step['it_power_kw'][0],
            'fan_extra_kw': step['server_fan_extra_kw'][0], 'heat_removed_kw': step['heat_removed_kw'][0],
            'cooling_power_kw': step['cooling_power_kw'][0], 'facility_power_kw': step['facility_power_kw'][0],
            'pue': step['pue'][0], 'free_cooling_fraction': step['free_cooling_fraction'][0],
            'cooling_mode': step['cooling_mode'][0], 'ride_through_min': step['ride_through_min'][0],
            'xfactor': step['average_xfactor'], 'violation': step['violation_steps'] > 0,
            'capacity_limited': step['capacity_limited_steps'] > 0,
        })

    if not rows:
        raise SimulationError(warnings[0] if warnings else '시뮬레이션 결과가 없습니다')
    df = pd.DataFrame(rows)
    in_range = (df.room_temp_c >= t_min - 1e-6) & (df.room_temp_c <= t_max + 1e-6)
    summary = {
        'hours': len(df),
        'facility_energy_kwh': float(df.facility_power_kw.sum()),
        'cooling_energy_kwh': float(df.cooling_power_kw.sum()),
        'fan_extra_energy_kwh': float(df.fan_extra_kw.sum()),
        'it_energy_kwh': float(df.it_power_kw.sum()),
        'average_pue': float(df.facility_power_kw.sum() / df.it_power_kw.sum()),
        'average_xfactor': float(df.xfactor.mean()),
        'average_setpoint_c': float(df.setpoint_c.mean()),
        'max_room_temp_c': float(df.room_temp_c.max()),
        'min_room_temp_c': float(df.room_temp_c.min()),
        'temp_maintenance_pct': float(in_range.mean() * 100),      # 서버실 18~27℃ 유지율
        'temp_ok': bool(df.room_temp_c.max() <= t_max + 1e-6),
        'min_ride_through_min': float(df.ride_through_min.min()),
        'violation_hours': int(df.violation.sum()),                 # 온도 범위 또는 비상 여유 시간 위반
        'capacity_limited_hours': int(df.capacity_limited.sum()),
        'mode_hours': {str(k): int(v) for k, v in df['mode'].value_counts().items()},
        'diverged': diverged,
    }
    return {
        'start': start.isoformat(), 'hours': hours, 'controller': controller,
        'scenario': scenario, 'scenario_label': sc.label, 'strategy': sc.strategy,
        'summary': summary, 'timeseries': {c: df[c].tolist() for c in df.columns}, 'warnings': warnings,
    }
