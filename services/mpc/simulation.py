"""폐루프 시뮬레이션: 매시간 제어기가 설정 온도를 정하고 트윈을 1시간 진행한다.

scripts/simulate_mpc.py(14일 일괄 실험 스크립트)와 같은 절차를 서비스 안에서 돌려,
대시보드가 구간·제어기·위기 시나리오를 골라 API 한 번으로 결과를 받게 한다.
계측값(IT 부하·외기)도 파일이 아니라 twin-service /api/v1/telemetry 로 받는다.

제어기
  fixed_20   기준 20℃ 고정 (원본 데이터의 운영 방식)
  rule_pid   Rule-based: 서버실 온도를 목표(20℃)에 맞추는 PID (rule_based.py). MPC 비교 기준
  rule_tes   Rule-based PID + 축열조 요금 시간표 운전 (경부하 충전, 최대부하 방전)
  reactive   예측 없이 직전 1시간 부하 + 현재 외기로 그 순간 최적값 선택
  mpc        24시간 예측 + 설정 온도·축열조 계획 최적화, 첫 1시간만 적용

축열조 저장량은 시간마다 이어받는다 (시작: 용량 × tes_initial_soc). 축열조를 쓰지 않는 제어기는 출력 0.

위기 시나리오 (start 시각부터 구간 끝까지 적용, config/settings.yaml 의 scenarios)
  normal           변경 없음
  server_surge     IT 부하 1.3배
  chiller_failure  칠러 4대 중 1대 탈락 (냉각 용량 75%)
  heatwave         외기온도 최저 38℃
  복합 위기         heatwave_chiller, surge_heatwave, triple_crisis(세 위기 동시),
                   chiller_double_failure(칠러 2대 탈락, 냉각 용량 50%)

서버실 온도가 위험 온도(critical_temp_c, 32℃)에 닿으면 그 시각에서 멈추고 summary.hours_to_critical 에
시작부터 걸린 시간을 적는다. 그 위는 서버가 보호 정지하는 영역이라 트윈 모델이 맞지 않기 때문이다.

가정: 외기 예보는 실제값과 같다. 1시간 평균값으로 진행한다 (5분 변동 무시).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd

from services.common.settings import section
from services.mpc.controller import MPCConfig, MPCController
from services.mpc.rule_based import PIDController, tes_schedule_kw

HISTORY_HOURS: int = section('simulation')[
    'history_hours'
]  # 예측 모델 입력(최소 168시간)에 여유를 둔 이력 길이
CONTROLLERS = ('fixed_20', 'rule_pid', 'rule_tes', 'reactive', 'mpc')


@dataclass(frozen=True)
class Scenario:
    label: str
    strategy: str  # 대응 전략 설명 (대시보드 표시용)
    it_scale: float = 1.0
    outdoor_min_c: float | None = None
    capacity_factor: float = 1.0


def load_scenarios() -> dict[str, Scenario]:
    """config/settings.yaml 의 scenarios 항목. 칠러 고장은 탈락 대수 / 전체 대수로 냉각 용량 배율을 정한다."""
    units = section('datacenter')['chiller_units']
    out = {}
    for name, s in section('scenarios').items():
        failed = s.get('failed_chillers', 0)
        if not 0 <= failed < units:
            raise ValueError(
                f'시나리오 {name}: failed_chillers({failed})는 0 이상 칠러 대수({units}) 미만이어야 합니다'
            )
        out[name] = Scenario(
            label=s['label'],
            strategy=s['strategy'],
            it_scale=float(s.get('it_scale', 1.0)),
            outdoor_min_c=s.get('outdoor_min_c'),
            capacity_factor=(units - failed) / units,
        )
    return out


SCENARIOS: dict[str, Scenario] = load_scenarios()


class SimulationError(ValueError):
    """요청 구간·제어기·시나리오가 잘못됐을 때 (API 에서 422)."""


def fetch_telemetry(twin: httpx.Client, start: pd.Timestamp, hours: int) -> pd.DataFrame:
    """시뮬레이션에 필요한 계측값: 이력(start 이전 192시간) + 구간 + 외기 예보용 23시간."""
    first = start - pd.Timedelta(hours=HISTORY_HOURS)
    r = twin.get(
        '/api/v1/telemetry',
        params={'start': first.isoformat(), 'hours': HISTORY_HOURS + hours + 23},
    )
    if r.status_code != 200:
        raise SimulationError(
            f'계측값을 받을 수 없습니다 (twin telemetry {r.status_code}): {r.text[:200]}'
        )
    body = r.json()
    df = pd.DataFrame(
        {'it_power_kw': body['it_power_kw'], 'outdoor_temp_c': body['outdoor_temp_c']},
        index=pd.to_datetime(body['timestamps']),
    )
    last = start + pd.Timedelta(hours=hours - 1)
    if df.index[0] > first or df.index[-1] < last:
        earliest = pd.Timestamp(body['data_start']) + pd.Timedelta(hours=HISTORY_HOURS)
        raise SimulationError(
            f'구간이 데이터 범위를 벗어납니다. 시작 시각은 {earliest} '
            f'이후, 끝 시각은 {body["data_end"]} 이전이어야 합니다 (요청: {start} ~ {last})'
        )
    return df


def fetch_prices(twin: httpx.Client, start: pd.Timestamp, hours: int) -> pd.DataFrame:
    """[start, start + hours) 의 시간별 전기 단가 (twin-service /api/v1/tariff). index = 시각."""
    r = twin.get('/api/v1/tariff', params={'start': start.isoformat(), 'hours': hours})
    if r.status_code != 200:
        raise SimulationError(
            f'전기 단가를 받을 수 없습니다 (twin tariff {r.status_code}): {r.text[:200]}'
        )
    body = r.json()
    return pd.DataFrame(
        {'period': body['period'], 'price_krw_per_kwh': body['price_krw_per_kwh']},
        index=pd.to_datetime(body['timestamps']),
    )


def apply_scenario(df: pd.DataFrame, start: pd.Timestamp, scenario: Scenario) -> pd.DataFrame:
    out = df.copy()
    active = out.index >= start
    out.loc[active, 'it_power_kw'] *= scenario.it_scale
    if scenario.outdoor_min_c is not None:
        out.loc[active, 'outdoor_temp_c'] = out.loc[active, 'outdoor_temp_c'].clip(
            lower=scenario.outdoor_min_c
        )
    return out


def _reactive_setpoint(
    twin: httpx.Client, params: dict, weight: float, it_kw: float, outdoor_c: float
) -> float:
    """지금 조건의 정상 상태에서 전체 전력이 가장 적은 설정 온도 (서버실 허용 범위 안)."""
    grid = np.arange(params['room_temp_min_c'], params['room_temp_max_c'] + 1e-9, 0.5)
    r = twin.post(
        '/api/v1/simulate',
        json={
            'points': [
                {'it_power_kw': it_kw, 'outdoor_temp_c': outdoor_c, 'supply_temp_c': float(s)}
                for s in grid
            ]
        },
    )
    r.raise_for_status()
    cost = [
        x['facility_power_kw']
        + weight * (x['xfactor'] - 1.0)
        + (1e6 if x['ride_through_min'] < params['min_ride_through_min'] else 0.0)
        for x in r.json()['results']
    ]
    return float(grid[int(np.argmin(cost))])


def validate_request(controller: str, scenario: str) -> None:
    """제어기·시나리오 이름 검증 (작업 등록 전에 바로 422 를 내기 위해 따로 둔다)."""
    if controller not in CONTROLLERS:
        raise SimulationError(
            f'controller 는 {list(CONTROLLERS)} 중 하나여야 합니다: {controller}'
        )
    if scenario not in SCENARIOS:
        raise SimulationError(f'scenario 는 {list(SCENARIOS)} 중 하나여야 합니다: {scenario}')


def run_closed_loop(
    twin: httpx.Client,
    forecast: httpx.Client,
    start: pd.Timestamp,
    hours: int,
    controller: str = 'mpc',
    scenario: str = 'normal',
    config: MPCConfig = MPCConfig(),
    initial_room_temp_c: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """[start, start + hours) 를 1시간씩 진행한 시계열과 요약을 돌려준다.

    initial_room_temp_c: 시작 서버실 온도. 생략하면 기준 설정 온도(20℃)에서 시작한다.
    progress: 1시간 진행할 때마다 progress(끝낸 시간 수, 전체 시간 수) 로 부른다 (작업 진행률).
    """
    validate_request(controller, scenario)
    sc = SCENARIOS[scenario]
    start = pd.Timestamp(start).tz_localize(None).floor('h')
    data = apply_scenario(fetch_telemetry(twin, start, hours), start, sc)

    params = twin.get('/api/v1/params').json()
    ref = float(params['supply_temp_ref_c'])
    t_min, t_max = params['room_temp_min_c'], params['room_temp_max_c']
    t_critical = float(params['critical_temp_c'])
    prices = fetch_prices(twin, start, hours)
    mpc = MPCController(twin, forecast, config) if controller == 'mpc' else None
    pid = PIDController() if controller in ('rule_pid', 'rule_tes') else None

    room = ref if initial_room_temp_c is None else float(initial_room_temp_c)
    soc = float(params['tes_capacity_kwh'] * params['tes_initial_soc'])
    last_setpoint = ref  # 직전에 적용한 설정 온도 (MPC 변화율 제약 기준)
    rows, warnings, diverged = [], [], False
    hours_to_critical: int | None = None  # 위험 온도에 닿아 중단한 경우, 시작부터 걸린 시간
    for k in range(hours):
        now = start + pd.Timedelta(hours=k)
        actual = data.loc[now]
        mode, tes_cmd = controller, 0.0
        if controller == 'fixed_20':
            setpoint = ref
        elif controller in ('rule_pid', 'rule_tes'):
            setpoint = pid.update(room)
            if controller == 'rule_tes':
                tes_cmd = tes_schedule_kw(
                    prices.loc[now, 'period'],
                    params['tes_max_charge_kw'],
                    params['tes_max_discharge_kw'],
                )
        elif controller == 'reactive':
            prev = data.loc[now - pd.Timedelta(hours=1)]
            setpoint = _reactive_setpoint(
                twin,
                params,
                config.reliability_weight_kwh_per_h,
                float(prev.it_power_kw),
                float(actual.outdoor_temp_c),
            )
        else:
            hist = data.it_power_kw.loc[
                now - pd.Timedelta(hours=HISTORY_HOURS) : now - pd.Timedelta(hours=1)
            ]
            history = [
                {'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()
            ]
            outdoor = data.outdoor_temp_c.loc[
                now : now + pd.Timedelta(hours=config.horizon_hours - 1)
            ].tolist()
            outdoor += [outdoor[-1]] * (
                config.horizon_hours - len(outdoor)
            )  # 데이터 끝: 마지막 값 유지
            plan = mpc.plan(
                room, history, outdoor, sc.capacity_factor, soc, current_setpoint_c=last_setpoint
            )
            setpoint, mode, tes_cmd = plan.setpoint_c, plan.mode, plan.tes_kw

        r = twin.post(
            '/api/v1/rollout',
            json={
                'it_power_kw': [float(actual.it_power_kw)],
                'outdoor_temp_c': [float(actual.outdoor_temp_c)],
                'supply_temp_c': [setpoint],
                'initial_room_temp_c': room,
                'capacity_factor': sc.capacity_factor,
                'price_krw_per_kwh': [prices.loc[now, 'price_krw_per_kwh']],
                'tes_kw': [tes_cmd],
                'initial_tes_kwh': soc,
            },
        )
        step = r.json() if r.status_code == 200 else None
        if step is None or step['room_temp_c'][0] is None:
            diverged = True
            warnings.append(
                f'diverged: {now} 에 서버실 온도({room:.1f}℃)가 트윈 계산 범위를 벗어나 중단'
            )
            break
        room, soc, last_setpoint = step['room_temp_c'][0], step['tes_soc_kwh'][0], setpoint
        rows.append(
            {
                'timestamp': now.isoformat(),
                'mode': mode,
                'setpoint_c': setpoint,
                'it_base_kw': float(actual.it_power_kw),
                'outdoor_temp_c': float(actual.outdoor_temp_c),
                'room_temp_c': room,
                'it_power_kw': step['it_power_kw'][0],
                'fan_extra_kw': step['server_fan_extra_kw'][0],
                'heat_removed_kw': step['heat_removed_kw'][0],
                'cooling_power_kw': step['cooling_power_kw'][0],
                'facility_power_kw': step['facility_power_kw'][0],
                'pue': step['pue'][0],
                'free_cooling_fraction': step['free_cooling_fraction'][0],
                'cooling_mode': step['cooling_mode'][0],
                'ride_through_min': step['ride_through_min'][0],
                'xfactor': step['average_xfactor'],
                'violation': step['violation_steps'] > 0,
                'capacity_limited': step['capacity_limited_steps'] > 0,
                'tariff_period': prices.loc[now, 'period'],
                'price_krw_per_kwh': prices.loc[now, 'price_krw_per_kwh'],
                'energy_cost_krw': step['energy_cost_krw'][0],
                'tes_kw': step['tes_kw'][0],
                'tes_soc_kwh': soc,
            }
        )
        if progress is not None:
            progress(k + 1, hours)
        if room >= t_critical:
            # 위험 온도 이후는 서버 보호 정지 영역이라 트윈 모델이 맞지 않는다 → 여기서 멈춘다
            hours_to_critical = k + 1
            capacity_kw = params['cooling_capacity_kw'] * sc.capacity_factor
            warnings.append(
                f'critical_temp_reached: 시작 {hours_to_critical}시간 만에 서버실 온도가 위험 온도 '
                f'{t_critical:.0f}℃ 에 도달({room:.1f}℃)해 중단. 냉각 용량 {capacity_kw:,.0f}kW 가 '
                f'발열 {step["it_power_kw"][0]:,.0f}kW 보다 작아 설정 온도 조정으로는 막을 수 없음'
            )
            break

    if not rows:
        raise SimulationError(warnings[0] if warnings else '시뮬레이션 결과가 없습니다')
    df = pd.DataFrame(rows)
    in_range = (df.room_temp_c >= t_min - 1e-6) & (df.room_temp_c <= t_max + 1e-6)
    summary = {
        'hours': len(df),
        'energy_cost_krw': float(df.energy_cost_krw.sum()),  # 전력량요금 (기본요금·부가세 제외)
        'facility_energy_kwh': float(df.facility_power_kw.sum()),
        'cooling_energy_kwh': float(df.cooling_power_kw.sum()),
        'fan_extra_energy_kwh': float(df.fan_extra_kw.sum()),
        'it_energy_kwh': float(df.it_power_kw.sum()),
        'average_pue': float(df.facility_power_kw.sum() / df.it_power_kw.sum()),
        'average_xfactor': float(df.xfactor.mean()),
        'average_setpoint_c': float(df.setpoint_c.mean()),
        'max_room_temp_c': float(df.room_temp_c.max()),
        'min_room_temp_c': float(df.room_temp_c.min()),
        'temp_maintenance_pct': float(in_range.mean() * 100),  # 서버실 18~27℃ 유지율
        'temp_ok': bool(df.room_temp_c.max() <= t_max + 1e-6),
        'min_ride_through_min': float(df.ride_through_min.min()),
        'violation_hours': int(df.violation.sum()),  # 온도 범위 또는 비상 여유 시간 위반
        'capacity_limited_hours': int(df.capacity_limited.sum()),
        'mode_hours': {str(k): int(v) for k, v in df['mode'].value_counts().items()},
        'final_tes_soc_kwh': float(df.tes_soc_kwh.iloc[-1]),
        'diverged': diverged,
        'hours_to_critical': hours_to_critical,
    }
    return {
        'start': start.isoformat(),
        'hours': hours,
        'controller': controller,
        'scenario': scenario,
        'scenario_label': sc.label,
        'strategy': sc.strategy,
        'summary': summary,
        'timeseries': {c: df[c].tolist() for c in df.columns},
        'warnings': warnings,
    }
