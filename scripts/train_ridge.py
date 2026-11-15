"""Ridge 선형 회귀 IT 부하 예측 모델 학습·저장 (서비스 기본 모델).

scripts/compare_forecast_models.py 에서 Val MAPE 로 선정된 모델이다. scripts/train_forecast.py 와 같은
데이터·피처·Train/Val/Test 분할을 쓴다.

  1. Train 으로 학습 (24h 모델, 168h 모델)
  2. Val 로 90% 구간 반폭 결정 (split conformal)
  3. Test 에서 1회 평가 (MAPE, 구간 적중률)
  4. output/forecast/models/ 에 ridge_24h.joblib, ridge_168h.joblib, ridge_config.json 저장

실행:  python scripts/train_ridge.py        (약 3초, CPU)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트 (services 패키지)

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np

from services.forecast.features import HORIZONS, build_direct_dataset, load_hourly
from services.forecast.ridge import (
    CONFIG_NAME,
    RIDGE_ALPHA,
    build_pipeline,
    conformal_halfwidth,
    model_file,
)
from scripts.train_forecast import ALPHA, DATA_PATH, MODEL_DIR, feature_columns, mape, split


def main() -> None:
    hourly = load_hourly(DATA_PATH)
    os.makedirs(MODEL_DIR, exist_ok=True)
    config: dict = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': 'ridge',
        'ridge_alpha': RIDGE_ALPHA,
        'alpha': ALPHA,
        'models': {},
    }
    for name, max_h in HORIZONS.items():
        data = build_direct_dataset(hourly, max_h)
        feats = feature_columns(data, max_h)
        parts = split(data)
        model = build_pipeline(feats).fit(parts['train'][feats], parts['train']['y'])
        half = conformal_halfwidth(
            parts['val']['y'].to_numpy(), model.predict(parts['val'][feats]), ALPHA
        )
        y, mid = parts['test']['y'].to_numpy(), model.predict(parts['test'][feats])
        coverage = float(np.mean(np.abs(y - mid) <= half) * 100)
        joblib.dump(model, os.path.join(MODEL_DIR, model_file(name)))
        config['models'][name] = {
            'max_h': max_h,
            'features': feats,
            'file': model_file(name),
            'conformal_halfwidth_kw': half,
            'test': {'mape': mape(y, mid), 'coverage_pct': coverage},
        }
        print(
            f'[{name}] Test MAPE {mape(y, mid):.3f}%  90% 구간 적중률 {coverage:.1f}%  반폭 {half:.1f}kW'
        )
    with open(os.path.join(MODEL_DIR, CONFIG_NAME), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f'저장: {MODEL_DIR}/{CONFIG_NAME}')


if __name__ == '__main__':
    main()
