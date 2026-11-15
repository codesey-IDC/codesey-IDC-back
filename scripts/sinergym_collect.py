"""Sinergym(EnergyPlus) 데이터센터 환경을 1년 돌려 비교용 시계열을 모은다.

이 스크립트만 Sinergym 이 필요하다 (설치가 무거워 Docker 이미지 sailugr/sinergym 안에서 실행).
우리 저장소 코드는 import 하지 않는다. 결과 CSV 는 scripts/validate_sinergym.py 가 읽는다.

  docker compose --profile validation run --rm sinergym-validate      # 수집 + 비교 한 번에

환경: Eplus-datacenter_cw-mixed-continuous-stochastic-v1
  - 과제가 지정한 'Eplus-datacenter-mixed-continuous-stochastic-v1' 은 Sinergym 3.x 에서
    datacenter_cw(냉수·칠러) / datacenter_dx(직접팽창) 로 나뉘었다. 우리 모델이 칠러 + 외기 냉각이라 cw 를 쓴다.
  - 건물: 2ZoneDataCenterHVAC_wEconomizer_CW (동·서 2개 전산실, 외기 이코노마이저 + 냉수 칠러)
  - 날씨: 뉴욕 JFK TMY3 1년 (mixed 기후) + 외기온도 잡음 (stochastic). seed 를 고정해 재현 가능.
  - 제어: 구간 내내 같은 값 (전산실 냉방 설정 온도, 냉수 공급 온도). 10분 간격 52,560 스텝.

EnergyPlus 가 기본으로 내주는 값에는 IT 전력이 없어서 (CPU 부하율만 있다) 출력 변수를 추가해 직접 읽는다.
부하율 × 설계 전력으로 어림하면 실제 IT 전력과 크게 다르다 (부하율 0.5 에서 49kW vs 실제 81kW).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ENV_NAME = 'Eplus-datacenter_cw-mixed-continuous-stochastic-v1'
BASELINE_SETPOINT_C = 25.0  # 행동 공간(20~30℃)의 가운데. 우리 모델의 기준 운전에 대응시킨다
CHW_SUPPLY_C = 7.5  # 냉수 공급 온도: 행동 공간(5~10℃)의 가운데로 고정
DEFAULT_SETPOINTS = (22.0, 25.0, 28.0)
ZONES = (('East Zone', 'east'), ('West Zone', 'west'))
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / 'output' / 'validation' / 'sinergym_samples.csv'
)


def make_env():
    """IT 전력 출력 변수를 추가한 Sinergym 환경."""
    import gymnasium as gym
    import sinergym  # noqa: F401  (환경 등록)

    variables = dict(gym.spec(ENV_NAME).kwargs['variables'])
    for zone, z in ZONES:
        variables[f'{z}_ite_cpu_electricity_rate'] = ('Zone ITE CPU Electricity Rate', zone)
        variables[f'{z}_ite_fan_electricity_rate'] = ('Zone ITE Fan Electricity Rate', zone)
        variables[f'{z}_ite_ups_electricity_rate'] = ('Zone ITE UPS Electricity Rate', zone)
    return gym.make(ENV_NAME, variables=variables)


def run_year(env, cooling_setpoint_c: float, seed: int, max_steps: int | None) -> pd.DataFrame:
    """설정 온도를 고정하고 한 에피소드(1년)를 돌려 1시간 평균 시계열로 돌려준다."""
    names = env.get_wrapper_attr('observation_variables')
    action = np.array([cooling_setpoint_c, CHW_SUPPLY_C], dtype=np.float32)
    env.reset(seed=seed)
    rows = []
    while True:
        obs, _, terminated, truncated, _ = env.step(action)
        rows.append(obs)
        if terminated or truncated or (max_steps and len(rows) >= max_steps):
            break
    d = pd.DataFrame(np.asarray(rows, dtype=float), columns=names)

    def kw(*cols: str) -> pd.Series:
        return sum(d[c] for c in cols) / 1000.0

    out = pd.DataFrame(
        {
            'month': d['month'].astype(int),
            'day': d['day_of_month'].astype(int),
            'hour': d['hour'].astype(int),
            'outdoor_temp_c': d['outdoor_temperature'],
            'cpu_loading': d['cpu_loading_fraction'],
            # IT 전력 = 서버 CPU + 서버 팬 (UPS 손실은 IT 가 아니라 기타 손실)
            'it_power_kw': kw(
                *[f'{z}_ite_{c}_electricity_rate' for _, z in ZONES for c in ('cpu', 'fan')]
            ),
            'ups_loss_kw': kw(*[f'{z}_ite_ups_electricity_rate' for _, z in ZONES]),
            'hvac_power_kw': kw('HVAC_electricity_demand_rate'),  # 냉각 설비 전체 전력
            'chiller_kw': kw('chiller_electricity_rate'),
            'supply_fan_kw': kw('east_fan_electricity_rate', 'west_fan_electricity_rate'),
            'pump_kw': kw('cw_pump_electricity_rate', 'condenser_pump_electricity_rate'),
            'tower_kw': kw(
                'cooling_tower_fan_electricity_rate', 'cooling_tower_basin_heater_electricity_rate'
            ),
            'evap_cooler_kw': kw(
                'east_dec_electricity_rate',
                'east_iec_electricity_rate',
                'west_dec_electricity_rate',
                'west_iec_electricity_rate',
            ),
            'zone_temp_c': (d['east_zone_air_temperature'] + d['west_zone_air_temperature']) / 2,
            'supply_air_temp_c': (d['east_supply_temperature'] + d['west_supply_temperature']) / 2,
        }
    )
    hourly = out.groupby(['month', 'day', 'hour'], sort=False).mean().reset_index()
    hourly.insert(0, 'cooling_setpoint_c', cooling_setpoint_c)
    return hourly


def main() -> int:
    ap = argparse.ArgumentParser(description='Sinergym 데이터센터 환경 실행 → 비교용 CSV')
    ap.add_argument(
        '--setpoints',
        default=','.join(map(str, DEFAULT_SETPOINTS)),
        help=f'전산실 냉방 설정 온도 목록 (℃). {BASELINE_SETPOINT_C} 가 기준 운전',
    )
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument(
        '--max-steps', type=int, default=None, help='빠른 확인용: 스텝 수 제한 (10분 단위)'
    )
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Sinergym 은 import 한 폴더에 에피소드 출력을 잔뜩 만든다 → 저장소가 아니라 임시 폴더에서 돌린다
    os.chdir(tempfile.mkdtemp(prefix='sinergym_'))
    try:
        import sinergym
    except ImportError:
        print(
            '[보류] Sinergym 이 설치되어 있지 않습니다. Docker 로 실행하세요:\n'
            '  docker compose --profile validation run --rm sinergym-validate',
            file=sys.stderr,
        )
        return 2

    setpoints = [float(s) for s in args.setpoints.split(',')]
    if BASELINE_SETPOINT_C not in setpoints:
        setpoints.append(BASELINE_SETPOINT_C)

    env = make_env()
    frames = []
    try:
        for sp in sorted(setpoints):
            t0 = time.time()
            df = run_year(env, sp, args.seed, args.max_steps)
            frames.append(df)
            print(
                f'[수집] 설정 {sp:.1f}℃: {len(df)}시간, 외기 {df.outdoor_temp_c.min():.1f}~'
                f'{df.outdoor_temp_c.max():.1f}℃, IT 평균 {df.it_power_kw.mean():.1f}kW, '
                f'냉각 평균 {df.hvac_power_kw.mean():.1f}kW ({time.time() - t0:.0f}초)',
                flush=True,
            )
    finally:
        env.close()

    result = pd.concat(frames, ignore_index=True)
    result.insert(0, 'sinergym_version', sinergym.__version__)
    result.insert(1, 'env', ENV_NAME)
    result.insert(2, 'seed', args.seed)
    result.to_csv(out_path, index=False, float_format='%.5f')
    print(f'[수집] 저장: {out_path} ({len(result)}행)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
