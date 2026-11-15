"""IT 부하 예측 모델 비교: LightGBM(서비스 모델) vs Ridge 선형 회귀 vs 계절 기준선(1주 전 값).

세 모델 모두 scripts/train_forecast.py 와 같은 데이터·피처·Train/Val/Test 분할을 쓴다.
  lightgbm        분위수 회귀(q05/q50/q95) + CQR 보정. output/forecast/models/ 의 저장된 모델 그대로 평가
  ridge           선형 회귀(Ridge). 시각·요일은 범주형(one-hot), 나머지는 표준화.
                  90% 구간 = 예측 ± Val 절대 오차의 90% 분위수 (split conformal)
  seasonal_naive  1주 전 같은 시각 값 (lag_168) 그대로. 구간은 ridge 와 같은 방식

모델 선정은 Val MAPE(24h + 168h)로 한다. Test 는 선정과 무관하게 마지막에 한 번 보고만 한다.

실행:  python scripts/compare_forecast_models.py
결과:  output/forecast/model_comparison.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트 (services 패키지)

import json
import os

import numpy as np
import pandas as pd

from services.forecast.features import HORIZONS, build_direct_dataset, load_hourly
from services.forecast.predictor import DEFAULT_MODEL_DIR, read_booster
from services.forecast.ridge import build_pipeline, conformal_halfwidth
from scripts.train_forecast import DATA_PATH, OUTPUT_DIR, feature_columns, mape, split

RESULT_PATH = os.path.join(OUTPUT_DIR, 'model_comparison.json')


def interval_metrics(y: np.ndarray, mid: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    return {
        'mape': mape(y, mid),
        'coverage_pct': float(np.mean((y >= lo) & (y <= hi)) * 100),
        'width_kw': float(np.mean(hi - lo)),
    }


def evaluate_lightgbm(name: str, parts: dict[str, pd.DataFrame], feats: list[str]) -> dict:
    with open(DEFAULT_MODEL_DIR / 'model_config.json', encoding='utf-8') as f:
        spec = json.load(f)['models'][name]
    boosters = {
        q: read_booster(DEFAULT_MODEL_DIR / spec['files'][q]) for q in ('q05', 'q50', 'q95')
    }
    test, offset = parts['test'], float(spec['cqr_offset_kw'])
    pred = {q: b.predict(test[feats]) for q, b in boosters.items()}
    out = interval_metrics(
        test['y'].to_numpy(), pred['q50'], pred['q05'] - offset, pred['q95'] + offset
    )
    return out | {
        'val_mape': mape(
            parts['val']['y'].to_numpy(), boosters['q50'].predict(parts['val'][feats])
        )
    }


def evaluate_ridge(parts: dict[str, pd.DataFrame], feats: list[str]) -> dict:
    model = build_pipeline(feats).fit(parts['train'][feats], parts['train']['y'])
    val_pred = model.predict(parts['val'][feats])
    half = conformal_halfwidth(parts['val']['y'].to_numpy(), val_pred)
    mid = model.predict(parts['test'][feats])
    out = interval_metrics(parts['test']['y'].to_numpy(), mid, mid - half, mid + half)
    return out | {
        'conformal_halfwidth_kw': half,
        'val_mape': mape(parts['val']['y'].to_numpy(), val_pred),
    }


def evaluate_naive(parts: dict[str, pd.DataFrame]) -> dict:
    val_pred = parts['val']['lag_168'].to_numpy()
    half = conformal_halfwidth(parts['val']['y'].to_numpy(), val_pred)
    mid = parts['test']['lag_168'].to_numpy()
    out = interval_metrics(parts['test']['y'].to_numpy(), mid, mid - half, mid + half)
    return out | {
        'conformal_halfwidth_kw': half,
        'val_mape': mape(parts['val']['y'].to_numpy(), val_pred),
    }


def compare() -> dict:
    hourly = load_hourly(DATA_PATH)
    results: dict[str, dict] = {'lightgbm': {}, 'ridge': {}, 'seasonal_naive': {}}
    for name, max_h in HORIZONS.items():
        data = build_direct_dataset(hourly, max_h)
        feats = feature_columns(data, max_h)
        parts = split(data)
        results['lightgbm'][name] = evaluate_lightgbm(name, parts, feats)
        results['ridge'][name] = evaluate_ridge(parts, feats)
        results['seasonal_naive'][name] = evaluate_naive(parts)
    best = min(
        results, key=lambda m: results[m]['24h']['val_mape'] + results[m]['168h']['val_mape']
    )
    return {
        'criteria': {'mape_24h_max': 5.0, 'mape_168h_max': 8.0, 'coverage_min': 85.0},
        'selection': 'Val MAPE (24h + 168h) 최소',
        'models': results,
        'selected': best,
    }


def main() -> None:
    report = compare()
    models = report['models']
    print('=== IT 부하 예측 모델 비교 (선정: Val / 보고: Test 2026-05-30 ~ 06-12) ===')
    print(
        f'{"모델":16s}{"Val 24h":>9s}{"Val 168h":>10s} |{"24h MAPE":>10s}{"168h MAPE":>11s}'
        f'{"24h 적중률":>11s}{"168h 적중률":>12s}{"24h 구간폭":>11s}'
    )
    for m, r in models.items():
        print(
            f'{m:16s}{r["24h"]["val_mape"]:8.3f}%{r["168h"]["val_mape"]:9.3f}% |'
            f'{r["24h"]["mape"]:9.3f}%{r["168h"]["mape"]:10.3f}%'
            f'{r["24h"]["coverage_pct"]:10.1f}%{r["168h"]["coverage_pct"]:11.1f}%'
            f'{r["24h"]["width_kw"]:9.0f}kW'
        )
    print(
        f'Val 기준 선정: {report["selected"]}  (과제 기준: 24h MAPE 5% / 168h 8% 이내, 적중률 85% 이상)'
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f'저장: {RESULT_PATH}')


if __name__ == '__main__':
    main()
