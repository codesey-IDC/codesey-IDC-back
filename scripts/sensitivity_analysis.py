"""민감도 분석: 제어 지연·예측 오차가 MPC 의 PUE·전기요금에 미치는 영향.

scripts/simulate_mpc.py 의 폐루프 시뮬레이션(Test 14일)을 조건만 바꿔 여러 번 돌린다.

  제어 지연        MPC 결정이 0 / 1 / 2 / 3 시간 늦게 적용된다
  IT 부하 예측 오차  예측값 × (1 + N(0, σ)), σ = 5 / 10 / 20 %  (시간마다 따로 뽑는 무작위 오차)
  IT 부하 예측 편향  예측값 × 0.9 / 1.1  (계속 낮게 / 높게 예측하는 체계적 오차)
  외기 예보 오차     실제 외기 + N(0, σ), σ = 1 / 2 / 3 ℃  (scripts/evaluate_cooling_forecast.py 와 같은 정의)

비교 기준은 오차·지연이 없는 MPC 와 Rule-based(PID) 이다. 실제 설비(트윈 rollout)에는 항상 실제
부하·외기가 들어가고, MPC 가 계획할 때 보는 정보만 틀리게 준다.

실행:
  python scripts/sensitivity_analysis.py               # Test 14일, 13개 조건 (동시 6개)
  python scripts/sensitivity_analysis.py --days 3 --workers 8
  python scripts/sensitivity_analysis.py --fresh       # 저장된 조건별 결과를 버리고 처음부터
결과: output/mpc/sensitivity.json, sensitivity.csv, sensitivity.png
      output/mpc/sensitivity_cases/<조건>.json  (조건별 결과. 중간에 멈춰도 다시 실행하면 끝난 조건은 건너뛴다)

조건마다 별도 파이썬 프로세스로 돌린다 (프로세스마다 numpy 스레드 1개, 코어를 서로 다투지 않게).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트 (services 패키지)

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass

import pandas as pd

import scripts.simulate_mpc as SIM

RESULT_NAME = 'sensitivity'
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


@dataclass(frozen=True)
class Case:
    name: str
    group: str  # baseline | latency | it_noise | it_bias | outdoor
    value: float  # 그래프 x 축 값 (시간, %, ℃)
    controller: str = 'mpc'
    latency_h: int = 0
    it_noise_pct: float = 0.0
    it_bias_pct: float = 0.0
    outdoor_sigma_c: float = 0.0


def default_cases() -> list[Case]:
    cases = [Case('rule_pid', 'reference', 0, controller='rule_pid'), Case('mpc', 'baseline', 0)]
    cases += [Case(f'latency_{h}h', 'latency', h, latency_h=h) for h in (1, 2, 3)]
    cases += [Case(f'it_noise_{p:g}pct', 'it_noise', p, it_noise_pct=p) for p in (5, 10, 20)]
    cases += [Case(f'it_bias_{p:+g}pct', 'it_bias', p, it_bias_pct=p) for p in (-10, 10)]
    cases += [Case(f'outdoor_{s:g}c', 'outdoor', s, outdoor_sigma_c=s) for s in (1, 2, 3)]
    return cases


def run_case(case: Case, start: str, hours: int, seed: int = 0) -> dict:
    """조건 하나를 돌려 요약 지표를 돌려준다 (프로세스마다 서비스를 새로 띄운다)."""
    t0 = time.time()
    hourly = SIM.load_hourly()
    # with 로 열어야 연결(이벤트 루프)을 하나만 만들어 계속 쓴다 (SIM.open_clients 설명 참고)
    with SIM.open_clients() as (twin, fc):
        forecast = SIM.NoisyForecast(
            fc,
            noise_pct=case.it_noise_pct,
            bias_pct=case.it_bias_pct,
            rng=SIM.np.random.default_rng(seed + 1),
        )
        _, s = SIM.run(
            hourly,
            pd.Timestamp(start),
            hours,
            case.controller,
            twin,
            forecast,
            latency_h=case.latency_h,
            outdoor_sigma_c=case.outdoor_sigma_c,
            seed=seed,
        )
    s['seconds'] = round(time.time() - t0, 1)
    return {**asdict(case), **s}


def summarize(rows: list[dict]) -> pd.DataFrame:
    """기준(오차 없는 MPC)·Rule-based 대비 변화량을 붙인다."""
    df = pd.DataFrame(rows).set_index('name')
    base, rule = df.loc['mpc'], df.loc['rule_pid']
    df['pue_change'] = df.average_pue - base.average_pue
    df['cost_change_pct'] = (df.energy_cost_krw / base.energy_cost_krw - 1) * 100
    df['cost_saving_vs_rule_pid_pct'] = (1 - df.energy_cost_krw / rule.energy_cost_krw) * 100
    df['energy_saving_vs_rule_pid_pct'] = (
        1 - df.facility_energy_kwh / rule.facility_energy_kwh
    ) * 100
    return df


def plot(df: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from matplotlib import font_manager

    # 한글 글꼴: Windows / macOS / Linux 중 설치된 것
    have = {f.name for f in font_manager.fontManager.ttflist}
    korean = [f for f in ('Malgun Gothic', 'AppleGothic', 'NanumGothic') if f in have]
    plt.rcParams['font.family'] = korean + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    base = df.loc['mpc']
    panels = [
        ('latency', '제어 지연 (시간)'),
        ('it_noise', 'IT 부하 예측 무작위 오차 σ (%)'),
        ('it_bias', 'IT 부하 예측 편향 (%)'),
        ('outdoor', '외기 예보 오차 σ (℃)'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, (group, label) in zip(axes, panels):
        sub = df[df.group == group]
        # 오차·지연 0 인 점 = 기준 MPC
        pts = sorted(
            [(0.0, base.cost_saving_vs_rule_pid_pct)]
            + list(zip(sub.value, sub.cost_saving_vs_rule_pid_pct))
        )
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, marker='o', color='#2563eb')
        for x, y in zip(xs, ys):
            ax.annotate(
                f'{y:.1f}%', (x, y), textcoords='offset points', xytext=(0, 6), ha='center'
            )
        ax.axhline(0, color='#9ca3af', lw=0.8)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(top=max(df.cost_saving_vs_rule_pid_pct) + 1.5)  # 숫자 표시 여백
    axes[0].set_ylabel('Rule-based(PID) 대비\n요금 절감률 (%)')
    fig.suptitle('MPC 민감도 분석 (Test 14일): 지연·예측 오차가 커질 때 절감 효과')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description='제어 지연·예측 오차 민감도 분석')
    ap.add_argument('--days', type=int, default=SIM.TEST_DAYS)
    ap.add_argument('--start', default=str(SIM.TEST_START))
    ap.add_argument(
        '--workers',
        type=int,
        default=6,
        help='동시에 돌릴 프로세스 수 (1 = 이 프로세스에서 순서대로)',
    )
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fresh', action='store_true', help='저장된 조건별 결과를 무시')
    ap.add_argument('--case', help=argparse.SUPPRESS)  # 내부용: 조건 하나만 돌려 저장
    args = ap.parse_args()

    cases, hours = default_cases(), args.days * 24
    case_dir = Path(SIM.OUT_DIR) / f'{RESULT_NAME}_cases'
    case_dir.mkdir(parents=True, exist_ok=True)

    def case_path(c: Case) -> Path:
        return case_dir / f'{c.name}.json'

    if args.case:  # 하위 프로세스
        c = next(c for c in cases if c.name == args.case)
        row = run_case(c, args.start, hours, args.seed)
        case_path(c).write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
        return

    def cached(c: Case) -> dict | None:
        if args.fresh or not case_path(c).exists():
            return None
        row = json.loads(case_path(c).read_text(encoding='utf-8'))
        return row if row.get('hours') == hours else None

    t0 = time.time()
    todo = [c for c in cases if cached(c) is None]
    print(
        f'조건 {len(cases)}개 중 {len(cases) - len(todo)}개는 저장된 결과 사용, {len(todo)}개 실행'
    )
    if args.workers <= 1:
        for i, c in enumerate(todo, 1):
            row = run_case(c, args.start, hours, args.seed)
            case_path(c).write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
            print(f'  [{i}/{len(todo)}] {c.name} 완료 ({row["seconds"]:.0f}초)', flush=True)
    else:
        env = {**os.environ, **{k: '1' for k in THREAD_ENV}, 'PYTHONIOENCODING': 'utf-8'}
        base = [sys.executable, '-W', 'ignore', __file__, '--days', str(args.days)]
        base += ['--start', args.start, '--seed', str(args.seed)]
        running: dict[str, subprocess.Popen] = {}
        queue, done = list(todo), 0
        while queue or running:
            while queue and len(running) < args.workers:
                c = queue.pop(0)
                running[c.name] = subprocess.Popen([*base, '--case', c.name], env=env)
            time.sleep(2)
            for name, proc in list(running.items()):
                if proc.poll() is None:
                    continue
                del running[name]
                done += 1
                if proc.returncode != 0:
                    raise SystemExit(f'{name} 실패 (종료 코드 {proc.returncode})')
                print(
                    f'  [{done}/{len(todo)}] {name} 완료 ({time.time() - t0:.0f}초 경과)',
                    flush=True,
                )
    rows = [json.loads(case_path(c).read_text(encoding='utf-8')) for c in cases]
    df = summarize(rows)

    print(f'\n{args.start} 부터 {hours}시간, 조건 {len(cases)}개 ({time.time() - t0:.0f}초)')
    print(
        f'{"조건":18s}{"PUE":>7s}{"ΔPUE":>8s}{"요금 백만원":>11s}{"요금 변화%":>10s}'
        f'{"PID 대비 절감%":>13s}{"위반h":>6s}{"최고℃":>7s}'
    )
    for name, r in df.iterrows():
        print(
            f'{name:18s}{r.average_pue:7.3f}{r.pue_change:+8.4f}{r.energy_cost_krw / 1e6:11.2f}'
            f'{r.cost_change_pct:+10.2f}{r.cost_saving_vs_rule_pid_pct:13.2f}'
            f'{int(r.violation_hours):6d}{r.max_room_temp_c:7.1f}'
        )

    out = SIM.OUT_DIR
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f'{RESULT_NAME}.csv'), encoding='utf-8-sig')
    with open(os.path.join(out, f'{RESULT_NAME}.json'), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'start': args.start,
                'hours': hours,
                'seed': args.seed,
                'baseline': 'mpc (지연·오차 없음)',
                'cases': json.loads(df.reset_index().to_json(orient='records', force_ascii=False)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    plot(df, os.path.join(out, f'{RESULT_NAME}.png'))
    print(f'\n저장: {out}/{RESULT_NAME}.json, .csv, .png')


if __name__ == '__main__':
    main()
