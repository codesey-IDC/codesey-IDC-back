"""지역별 비교: 같은 가상 데이터센터(같은 IT 부하)를 서울·춘천·제주의 외기에 놓았을 때.

지역마다 다른 것은 외기(기상청 ASOS 실측)뿐이다 (scripts/build_dataset.py). 예측 모델은 IT 부하만 쓰므로
지역과 무관하게 같은 모델을 쓴다. 이 스크립트는 두 가지를 본다.

  1. 전체 기간 (기본 2026-03-23 ~ 10-31): 기준 운전(20℃ 고정)과 Rule-based(PID) 의 PUE·전기요금·냉각 방식별 시간
  2. 계절별 1주일 4구간: MPC 가 Rule-based(PID) 대비 전기요금을 얼마나 줄이는지 (MPC 는 1시간에 약 2.7초라 전체 기간 대신 구간만)

폐루프 절차는 scripts/simulate_mpc.py 의 run() 을 그대로 쓴다 (트윈·예측 서비스는 REST 로 호출).

실행 (저장소 루트에서):
  python scripts/compare_sites.py                 # 전체 (구간마다 프로세스 하나, 약 10분)
  python scripts/compare_sites.py --no-mpc        # 1번만 (약 1분)
  python scripts/compare_sites.py --days 2 --windows 2026-08-03 --sites jeju
결과: output/sites/site_comparison.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트 (services 패키지)

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from scripts import simulate_mpc as SIM
from services.common.settings import REPO_ROOT, repo_path, section

OUT_PATH = repo_path('site_results') / 'site_comparison.json'
# 계절별 1주일 (모두 월요일 시작): 봄, 초여름, 한여름, 가을
WINDOWS = ('2026-04-13', '2026-06-08', '2026-08-03', '2026-10-05')
WINDOW_CONTROLLERS = ('fixed_20', 'rule_pid', 'mpc')
KEEP = (
    'energy_cost_krw',
    'facility_energy_kwh',
    'cooling_energy_kwh',
    'average_pue',
    'max_room_temp_c',
    'violation_hours',
    'fallback_hours',
    'naive_forecast_hours',
)


def load_hourly(site: str) -> pd.DataFrame:
    """지역 데이터의 1시간 평균 IT 부하·외기."""
    raw = pd.read_parquet(REPO_ROOT / section('sites')[site]['data']).set_index('timestamp')
    return raw[['it_power_kw', 'outdoor_temp_c']].resample('h').mean()


def cooling_mode_hours(site: str, start: pd.Timestamp, hours: int) -> dict[str, float]:
    """기준 운전(20℃ 고정)의 냉각 방식별 시간 (5분 데이터 기준)."""
    raw = pd.read_parquet(
        REPO_ROOT / section('sites')[site]['data'], columns=['timestamp', 'cooling_mode']
    ).set_index('timestamp')
    win = raw.loc[start : start + pd.Timedelta(hours=hours) - pd.Timedelta(minutes=5)]
    return {k: round(v / 12, 1) for k, v in win['cooling_mode'].value_counts().items()}


def run_job(job: tuple[str, str, int, tuple[str, ...]]) -> dict:
    """(지역, 시작, 시간 수, 제어기들) 하나를 돌린다. 프로세스마다 서비스 클라이언트를 따로 연다."""
    site, start, hours, controllers = job
    hourly = load_hourly(site)
    t0 = pd.Timestamp(start)
    out = {'site': site, 'start': str(t0), 'hours': hours, 'controllers': {}}
    with SIM.open_clients() as (twin, forecast):
        for name in controllers:
            _, s = SIM.run(hourly, t0, hours, name, twin, forecast)
            out['controllers'][name] = {k: s[k] for k in KEEP}
    win = hourly.loc[t0 : t0 + pd.Timedelta(hours=hours - 1)]
    out['outdoor_mean_c'] = float(win['outdoor_temp_c'].mean())
    out['outdoor_max_c'] = float(win['outdoor_temp_c'].max())
    out['cooling_mode_hours'] = cooling_mode_hours(site, t0, hours)
    rule = out['controllers'].get('rule_pid')
    if rule:
        for s in out['controllers'].values():
            s['cost_saving_vs_rule_pid_pct'] = (
                1 - s['energy_cost_krw'] / rule['energy_cost_krw']
            ) * 100
    return out


def main(argv: list[str] | None = None) -> int:
    sites = section('sites')
    ap = argparse.ArgumentParser(description='지역별(서울·춘천·제주) 운전 비교')
    ap.add_argument('--sites', default=','.join(sites))
    ap.add_argument('--windows', default=','.join(WINDOWS), help='MPC 비교 구간 시작일 (쉼표)')
    ap.add_argument('--days', type=int, default=7, help='MPC 비교 구간 길이(일)')
    ap.add_argument('--no-mpc', action='store_true', help='전체 기간 Rule-based 비교만')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args(argv)
    names = args.sites.split(',')

    cfg = section('dataset')
    full_start = pd.Timestamp(cfg['start']) + pd.Timedelta(hours=SIM.HISTORY_HOURS)
    full_hours = int(
        (pd.Timestamp(cfg['end']) + pd.Timedelta(days=1) - full_start) / pd.Timedelta(hours=1)
    )
    jobs = [(n, str(full_start), full_hours, ('fixed_20', 'rule_pid')) for n in names]
    if not args.no_mpc:
        jobs += [
            (n, w, args.days * 24, WINDOW_CONTROLLERS)
            for w in args.windows.split(',')
            for n in names
        ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(run_job, jobs))

    full = {r['site']: r for r in results[: len(names)]}
    windows = results[len(names) :]
    print(f'\n전체 기간 {full_start:%Y-%m-%d} ~ {cfg["end"]} ({full_hours // 24}일)')
    print(
        f'{"지역":8s}{"외기 평균":>8s}{"외기냉각h":>9s}{"혼합h":>7s}{"칠러h":>7s}'
        f'{"PUE(20℃)":>10s}{"요금 억원":>9s}'
    )
    for n in names:
        r, m = full[n], full[n]['cooling_mode_hours']
        c = r['controllers']['fixed_20']
        print(
            f'{sites[n]["label"]:8s}{r["outdoor_mean_c"]:9.1f}{m.get("free_cooling", 0):10.0f}'
            f'{m.get("hybrid", 0):8.0f}{m.get("chiller", 0):8.0f}{c["average_pue"]:10.3f}'
            f'{c["energy_cost_krw"] / 1e8:10.2f}'
        )
    if windows:
        print(f'\nMPC 요금 절감률 (Rule-based PID 대비, {args.days}일 구간) · 최고 서버실 온도')
        for r in windows:
            mpc = r['controllers']['mpc']
            print(
                f'{r["start"][:10]} {sites[r["site"]]["label"]:6s} '
                f'외기 {r["outdoor_mean_c"]:5.1f}℃  '
                f'절감 {mpc["cost_saving_vs_rule_pid_pct"]:6.2f}%  '
                f'최고 {mpc["max_room_temp_c"]:.1f}℃  위반 {mpc["violation_hours"]}h'
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                'sites': {
                    n: {'label': sites[n]['label'], 'station_id': sites[n]['station_id']}
                    for n in names
                },
                'full_period': full,
                'windows': windows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
        newline='\n',
    )
    print(f'\n저장: {OUT_PATH.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
