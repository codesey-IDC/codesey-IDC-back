"""우리 열역학 모델(디지털 트윈) vs Sinergym(EnergyPlus) 비교 검증. 기준: 오차 10% 이내.

Sinergym 이 계산한 매 시각의 IT 전력·외기온도를 우리 모델에 그대로 넣고 (같은 조건),
우리 모델이 낸 냉각 전력·PUE 를 Sinergym 의 값과 비교한다.

  입력 (Sinergym → 우리 모델)   IT 전력 [kW], 외기온도 [℃]
  비교 (우리 모델 vs Sinergym)  냉각 전력량, PUE

Sinergym 실행 결과는 scripts/sinergym_collect.py 가 만든 CSV 를 읽는다 (Docker 에서 수집).
이 스크립트는 Sinergym 없이도 돈다 → 저장된 CSV 로 언제든 다시 비교할 수 있다.

  docker compose --profile validation run --rm sinergym-validate    # 수집 + 비교
  python scripts/validate_sinergym.py                               # 저장된 CSV 로 비교만

결과: output/validation/sinergym_report.json
종료 코드: 0 통과 / 1 기준 초과 / 2 Sinergym 결과 CSV 없음 (가짜 통과를 만들지 않는다)

규모: Sinergym 건물은 IT 약 100kW, 우리 데이터센터는 약 3,000kW 다. 우리 모델의 냉각 전력은
      "제거열 × 1kW 당 냉각 전력(외기온도의 함수)" 이라 규모에 비례하므로 그대로 비교할 수 있다.
PUE:  기타 손실(UPS·조명)의 정의가 서로 달라, 판정은 냉각만 넣은 PUE = (IT + 냉각) / IT 로 한다.
      기타 손실까지 넣은 전체 PUE 는 참고로 함께 적는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트 (services 패키지)

import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from services.twin import physics as P

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = ROOT / 'output' / 'validation' / 'sinergym_samples.csv'
REPORT_PATH = ROOT / 'output' / 'validation' / 'sinergym_report.json'
TOLERANCE_PCT = 10.0
BASELINE_SETPOINT_C = 25.0  # scripts/sinergym_collect.py 의 기준 운전
REFERENCE_ITE_DESIGN_KW = 98.27  # 건물 모델의 IT 설계 전력 (200 W/㎡ × 491.34 ㎡)
BANDS = (
    (-50.0, 5.0),
    (5.0, 15.0),
    (15.0, 18.0),
    (18.0, 25.0),
    (25.0, 50.0),
)  # 외기온도 구간 [이상, 미만)
CALIBRATED_FIELDS = ('free_cooling_cop', 'chiller_cop_ref', 'chiller_cop_slope', 'chiller_cop_min')


def pct_error(own: float, reference: float) -> float:
    return (own - reference) / reference * 100.0


def own_cooling_kw(
    it_kw: np.ndarray,
    outdoor_c: np.ndarray,
    supply_temp_c: float | None = None,
    p: P.TwinParams = P.DEFAULT_PARAMS,
) -> dict[str, np.ndarray]:
    """같은 IT 전력·외기온도에서 우리 모델이 계산한 값 (정상 상태)."""
    return P.simulate_static(it_kw, outdoor_c, supply_temp_c, p)


def compare(df: pd.DataFrame, p: P.TwinParams = P.DEFAULT_PARAMS) -> dict:
    """한 설정 온도(기준 운전)의 시계열을 비교해 판정 항목·구간별 오차를 돌려준다."""
    it = df['it_power_kw'].to_numpy()
    outdoor = df['outdoor_temp_c'].to_numpy()
    ref_cool = df['hvac_power_kw'].to_numpy()
    own_cool = own_cooling_kw(it, outdoor, p=p)['cooling_power_kw']

    it_mwh, ref_mwh, own_mwh = it.sum() / 1000, ref_cool.sum() / 1000, own_cool.sum() / 1000
    ref_pue, own_pue = 1 + ref_mwh / it_mwh, 1 + own_mwh / it_mwh
    metrics = [
        {'name': '냉각 전력량', 'unit': 'MWh', 'own': own_mwh, 'sinergym': ref_mwh},
        {'name': 'PUE (IT + 냉각)', 'unit': '', 'own': own_pue, 'sinergym': ref_pue},
    ]
    for m in metrics:
        m['error_pct'] = pct_error(m['own'], m['sinergym'])
        m['passed'] = bool(abs(m['error_pct']) <= TOLERANCE_PCT)

    bands = []
    for lo, hi in BANDS:
        mask = (outdoor >= lo) & (outdoor < hi)
        if not mask.any():
            continue
        bands.append(
            {
                'outdoor_from_c': lo,
                'outdoor_to_c': hi,
                'hours': int(mask.sum()),
                'sinergym_cop': float(it[mask].sum() / ref_cool[mask].sum()),
                'own_cop': float(it[mask].sum() / own_cool[mask].sum()),
                'cooling_error_pct': pct_error(own_cool[mask].sum(), ref_cool[mask].sum()),
            }
        )

    ups = df['ups_loss_kw'].to_numpy() if 'ups_loss_kw' in df else np.zeros_like(it)
    return {
        'hours': int(len(df)),
        'outdoor_range_c': [float(outdoor.min()), float(outdoor.max())],
        'it_power_range_kw': [float(it.min()), float(it.max())],
        'metrics': metrics,
        'passed': all(m['passed'] for m in metrics),
        'hourly_cooling_nmae_pct': float(
            np.abs(own_cool - ref_cool).mean() / ref_cool.mean() * 100
        ),
        'by_outdoor_band': bands,
        # 참고: 기타 손실까지 넣은 전체 PUE. Sinergym 은 UPS 손실, 우리 모델은 IT 전력의 고정 비율
        'total_pue': {
            'own': float(own_pue + p.other_overhead_ratio),
            'sinergym': float(ref_pue + ups.sum() / it.sum()),
            'own_other_ratio': float(p.other_overhead_ratio),
            'sinergym_other_ratio': float(ups.sum() / it.sum()),
        },
    }


def loading_approximation(df: pd.DataFrame) -> dict:
    """참고: IT 전력을 '부하율 × 설계 전력' 으로 어림했을 때의 오차 (이전 프로젝트의 방식).

    EnergyPlus 의 실제 IT 전력은 이 어림보다 훨씬 크다. 예전 계수는 이 어림값에 맞춰져 있어서
    실제 IT 전력을 넣으면 냉각 전력을 23% 크게 잡았다. 판정에는 쓰지 않는다.
    """
    approx_it = df['cpu_loading'].to_numpy() * REFERENCE_ITE_DESIGN_KW
    own = own_cooling_kw(approx_it, df['outdoor_temp_c'].to_numpy())['cooling_power_kw']
    return {
        'approx_it_mean_kw': float(approx_it.mean()),
        'actual_it_mean_kw': float(df['it_power_kw'].mean()),
        'cooling_error_pct': pct_error(own.sum(), df['hvac_power_kw'].sum()),
    }


def setpoint_sensitivity(samples: pd.DataFrame) -> list[dict]:
    """참고: 설정 온도를 바꿨을 때의 변화 (기준 운전 대비 %). 판정에는 쓰지 않는다.

    Sinergym: 전산실 냉방 설정 온도를 바꿔 1년을 다시 돌린 결과.
    우리 모델: 기준 운전의 IT 전력을 기본 부하로 넣고, 설정 온도를 같은 폭만큼 바꿔 계산.
    """
    base = samples[samples['cooling_setpoint_c'] == BASELINE_SETPOINT_C]
    it_base, outdoor = base['it_power_kw'].to_numpy(), base['outdoor_temp_c'].to_numpy()
    p = P.DEFAULT_PARAMS

    def own_totals(delta_c: float) -> dict[str, float]:
        r = own_cooling_kw(it_base, outdoor, p.supply_temp_ref_c + delta_c)
        return {
            'it': r['it_power_kw'].sum(),
            'cooling': r['cooling_power_kw'].sum(),
            'facility': r['facility_power_kw'].sum(),
        }

    def ref_totals(g: pd.DataFrame) -> dict[str, float]:
        it, cool = g['it_power_kw'].sum(), g['hvac_power_kw'].sum()
        return {'it': it, 'cooling': cool, 'facility': it + cool + g['ups_loss_kw'].sum()}

    own0, ref0 = own_totals(0.0), ref_totals(base)
    rows = []
    for sp, g in samples.groupby('cooling_setpoint_c'):
        delta = float(sp) - BASELINE_SETPOINT_C
        own, ref = own_totals(delta), ref_totals(g)
        rows.append(
            {
                'sinergym_setpoint_c': float(sp),
                'delta_c': delta,
                'own_setpoint_c': float(p.supply_temp_ref_c + delta),
                **{f'sinergym_{k}_change_pct': pct_error(ref[k], ref0[k]) for k in ref},
                **{f'own_{k}_change_pct': pct_error(own[k], own0[k]) for k in own},
            }
        )
    return rows


def calibrate(df: pd.DataFrame) -> dict:
    """냉각 효율 계수 4개를 Sinergym 의 시간별 냉각 전력에 맞췄을 때의 값과 오차 (오차 제곱 최소).

    config/settings.yaml 의 [보정] 값이 이 결과다. 설정이 이미 맞춘 값이면 current 와 fitted 가 같게 나온다.
    계수를 바꾸면 데이터의 냉각 전력·PUE 열도 다시 계산해야 한다 (scripts/recompute_cooling_columns.py).
    """
    it, outdoor = df['it_power_kw'].to_numpy(), df['outdoor_temp_c'].to_numpy()
    ref = df['hvac_power_kw'].to_numpy()
    start = np.array([getattr(P.DEFAULT_PARAMS, f) for f in CALIBRATED_FIELDS], dtype=float)

    def params(x: np.ndarray) -> P.TwinParams:
        return replace(P.DEFAULT_PARAMS, **dict(zip(CALIBRATED_FIELDS, map(float, x))))

    def loss(x: np.ndarray) -> float:
        if (x <= 0).any():
            return 1e12
        own = own_cooling_kw(it, outdoor, p=params(x))['cooling_power_kw']
        return float(((own - ref) ** 2).mean())

    best = minimize(
        loss, start, method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-6, 'maxiter': 4000}
    )
    fitted = compare(df, params(best.x))
    return {
        'current': dict(zip(CALIBRATED_FIELDS, map(float, start))),
        'fitted': {f: round(float(v), 3) for f, v in zip(CALIBRATED_FIELDS, best.x)},
        'metrics': fitted['metrics'],
        'passed': fitted['passed'],
        'hourly_cooling_nmae_pct': fitted['hourly_cooling_nmae_pct'],
        'worst_band_error_pct': max(
            abs(b['cooling_error_pct']) for b in fitted['by_outdoor_band']
        ),
    }


def build_report(samples: pd.DataFrame) -> dict:
    base = samples[samples['cooling_setpoint_c'] == BASELINE_SETPOINT_C].reset_index(drop=True)
    if base.empty:
        raise ValueError(f'기준 운전(설정 {BASELINE_SETPOINT_C}℃) 결과가 CSV 에 없습니다')
    first = samples.iloc[0]
    return {
        'sinergym_version': str(first.get('sinergym_version', '')),
        'env': str(first.get('env', '')),
        'seed': int(first.get('seed', 0)),
        'baseline_setpoint_c': BASELINE_SETPOINT_C,
        'tolerance_pct': TOLERANCE_PCT,
        **compare(base),
        'loading_approximation': loading_approximation(base),
        'setpoint_sensitivity': setpoint_sensitivity(samples),
        'calibration': calibrate(base),
    }


def print_report(r: dict) -> None:
    print(f'\n=== 열역학 모델 vs Sinergym {r["sinergym_version"]} ({r["env"]}) ===')
    print(
        f'1년 {r["hours"]}시간, 외기 {r["outdoor_range_c"][0]:.1f}~{r["outdoor_range_c"][1]:.1f}℃, '
        f'IT {r["it_power_range_kw"][0]:.0f}~{r["it_power_range_kw"][1]:.0f}kW '
        '(같은 IT 전력·외기온도를 우리 모델에 입력)\n'
    )
    print(
        f'{"구분":<18}{"자체 모델":>12}{"Sinergym":>12}{"오차율":>10}   판정 (기준 ±{r["tolerance_pct"]:.0f}%)'
    )
    for m in r['metrics']:
        print(
            f'{m["name"]:<18}{m["own"]:>12.3f}{m["sinergym"]:>12.3f}{m["error_pct"]:>+9.1f}%   '
            f'{"통과" if m["passed"] else "초과"}'
        )
    print(f'\n시간별 냉각 전력 nMAE {r["hourly_cooling_nmae_pct"]:.1f}%')
    print('\n외기온도 구간별 (COP = IT 전력량 / 냉각 전력량)')
    for b in r['by_outdoor_band']:
        print(
            f'  {b["outdoor_from_c"]:>5.0f} ~ {b["outdoor_to_c"]:>3.0f}℃  {b["hours"]:>5d}시간  '
            f'Sinergym COP {b["sinergym_cop"]:>5.2f}  자체 {b["own_cop"]:>5.2f}  '
            f'냉각 전력 오차 {b["cooling_error_pct"]:>+6.1f}%'
        )
    t = r['total_pue']
    print(
        f'\n참고 1) 기타 손실 포함 PUE: 자체 {t["own"]:.3f} (기타 {t["own_other_ratio"]:.0%}) / '
        f'Sinergym {t["sinergym"]:.3f} (UPS 손실 {t["sinergym_other_ratio"]:.0%})'
    )
    a = r['loading_approximation']
    print(
        f'참고 2) IT 전력을 부하율 × {REFERENCE_ITE_DESIGN_KW}kW 로 어림하면 '
        f'(평균 {a["approx_it_mean_kw"]:.0f}kW, '
        f'실제 {a["actual_it_mean_kw"]:.0f}kW) 냉각 전력 오차 {a["cooling_error_pct"]:+.1f}%'
    )
    print('참고 3) 설정 온도를 바꿨을 때 변화 (기준 운전 대비, Sinergym / 자체)')
    for s in r['setpoint_sensitivity']:
        if s['delta_c'] == 0:
            continue
        print(
            f'  {s["delta_c"]:+.0f}℃:  냉각 {s["sinergym_cooling_change_pct"]:+6.1f}% / '
            f'{s["own_cooling_change_pct"]:+6.1f}%   '
            f'IT {s["sinergym_it_change_pct"]:+6.1f}% / {s["own_it_change_pct"]:+5.1f}%   '
            f'전체 {s["sinergym_facility_change_pct"]:+6.1f}% / '
            f'{s["own_facility_change_pct"]:+5.1f}%'
        )
    c = r['calibration']
    fitted = ', '.join(f'{k} {c["current"][k]:g}→{v:g}' for k, v in c['fitted'].items())
    errors = ', '.join(f'{m["name"]} {m["error_pct"]:+.1f}%' for m in c['metrics'])
    print(f'참고 4) 냉각 효율 계수를 Sinergym 에 다시 맞추면 (지금 설정→맞춘 값: {fitted}): {errors}')
    print(
        '\n[검증 통과] 모든 항목 오차 10% 이내'
        if r['passed']
        else '\n[검증 미달] 오차 10% 를 넘는 항목이 있습니다'
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='열역학 모델 vs Sinergym 비교 검증')
    ap.add_argument(
        '--samples', default=str(SAMPLES_PATH), help='scripts/sinergym_collect.py 결과 CSV'
    )
    ap.add_argument('--out', default=str(REPORT_PATH))
    args = ap.parse_args(argv)

    samples_path = Path(args.samples)
    if not samples_path.exists() or samples_path.stat().st_size < 1000:
        print(
            f'[보류] Sinergym 결과가 없습니다: {samples_path}\n'
            '  수집 (Docker): docker compose --profile validation run --rm sinergym-validate\n'
            '  Git LFS 로 받는 파일이면: git lfs pull',
            file=sys.stderr,
        )
        return 2

    report = build_report(pd.read_csv(samples_path))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=4), encoding='utf-8')
    print_report(report)
    print(f'저장: {out_path}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
