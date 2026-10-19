"""IT 부하 예측 모델 학습 (LightGBM 분위수 회귀 + CQR 보정)

흐름
  1. 5분 단위 parquet → 1시간 평균 집계 (2,160행)
  2. 직접(direct) 다중 시점 데이터셋 생성: (예측 시점 t, 예측 거리 h) 한 쌍이 한 행
  3. 시간 기준 분할: Train Day 1~63 / Val Day 64~76 / Test Day 77~90
  4. 분위수(q05, q50, q95)별 LightGBM 학습 (Train만 사용)
  5. Val에서 CQR로 90% 예측구간 보정
  6. Test에서 1회 평가 (MAPE, Coverage) + seasonal naive 기준선 비교

냉각 부하(cooling_load_kw)는 이 데이터에서 it_power_kw와 값이 완전히 같으므로
IT 부하 예측값이 곧 냉각 부하 예측값이다.

피처 계산은 services/forecast/features.py 를 쓴다 (서빙 predictor 와 공용).

실행:
  python train_forecast.py                # 학습 + 평가 + 모델/지표/model_config.json 저장
  python train_forecast.py --config-only  # 학습 없이 기존 모델·metrics.json 으로 model_config.json 만 생성
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.forecast.features import (  # noqa: E402
    TARGET, WEATHER_COLS, HORIZONS, MIN_HISTORY_HOURS, RESAMPLE_RULE,
    build_direct_dataset, feature_names, load_hourly,
)

DATA_PATH = 'data/synthetic_idc_90d.parquet'
OUTPUT_DIR = 'output/forecast'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
CONFIG_PATH = os.path.join(MODEL_DIR, 'model_config.json')

# IT 부하(CPU 트레이스)는 날씨와 물리적으로 무관하다. Val에서 기상 피처를 넣으면
# MAPE가 오히려 나빠졌으므로(24h 0.98→1.02%, 168h 0.94→1.03%) 기본은 제외.
# 켤 경우 미래 외기 실측값을 쓰는 것이므로 "perfect forecast 가정"을 명시해야 한다.
USE_WEATHER_FEATURES = False

# 같은 시점 값을 쓰면 정답이 새는 컬럼
#  - cooling_load_kw, server_heat_kw: it_power_kw와 값이 동일
#  - cpu_utilization: it_power_kw를 계산한 원천 (SPECpower 공식)
#  - 나머지: 냉각 제어 결과(사후 산출값)
FORBIDDEN_FEATURES = {
    'it_power_kw', 'cooling_load_kw', 'server_heat_kw', 'cpu_utilization',
    'cooling_power_kw', 'free_cooling_fraction', 'cooling_mode', 'pue',
}

SPLITS = {'train': (1, 63), 'val': (64, 76), 'test': (77, 90)}
QUANTILES = (0.05, 0.5, 0.95)
ALPHA = 0.10  # 90% 예측구간

LGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)


def feature_columns(data: pd.DataFrame, max_h: int) -> list[str]:
    meta = {'origin_time', 'target_time', 'origin_day', 'target_day', 'y'}
    feats = [c for c in data.columns if c not in meta]
    leaked = FORBIDDEN_FEATURES & set(feats)
    assert not leaked, f'누수 위험 피처 포함: {leaked}'
    expected = feature_names(max_h) + (WEATHER_COLS if USE_WEATHER_FEATURES else [])
    assert feats == expected, f'피처 순서가 features.feature_names 와 다름: {feats}'
    return feats


def model_file(name: str, q: float) -> str:
    return f'it_load_{name}_q{int(q * 100):02d}.txt'


def write_model_config(hourly: pd.DataFrame, train_range: dict[str, dict],
                       offsets: dict[str, float], feats: dict[str, list[str]]) -> None:
    """서빙(predictor)이 읽는 설정 파일. CQR offset·피처 순서·학습 범위를 모델 옆에 둔다."""
    config = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'data_path': DATA_PATH,
        'target': TARGET,
        'resample_rule': RESAMPLE_RULE,
        'alpha': ALPHA,
        'min_history_hours': MIN_HISTORY_HOURS,
        'splits_day': SPLITS,
        'data_period': {'start': str(hourly.index[0]), 'end': str(hourly.index[-1])},
        # 범위 밖 입력 경고 기준: 학습 데이터(5분 원본) IT 부하 범위.
        # (1시간 평균 Train 범위는 Val/Test 정상값도 살짝 벗어나므로 경고 기준으로 쓰지 않는다)
        'train_target_range_kw': train_range['raw'],
        'train_hourly_target_range_kw': train_range['hourly'],
        'lgb_params': LGB_PARAMS,
        'models': {
            name: {
                'max_h': HORIZONS[name],
                'features': feats[name],
                'cqr_offset_kw': offsets[name],
                'files': {f'q{int(q * 100):02d}': model_file(name, q) for q in QUANTILES},
            }
            for name in HORIZONS
        },
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f'   model_config.json 저장: {CONFIG_PATH}')


def target_ranges(hourly: pd.DataFrame, train_parts: dict[str, pd.DataFrame]) -> dict:
    train_y = pd.concat([p['y'] for p in train_parts.values()])
    raw = pd.read_parquet(DATA_PATH)[TARGET]
    return {
        'hourly': {'min': float(train_y.min()), 'max': float(train_y.max())},
        'raw': {'min': float(raw.min()), 'max': float(raw.max())},
    }


def config_only() -> None:
    """학습 없이, 이미 저장된 metrics.json 의 offset 으로 model_config.json 만 만든다."""
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), encoding='utf-8') as f:
        metrics = json.load(f)
    hourly = load_hourly(DATA_PATH)
    feats, train_parts = {}, {}
    for name, max_h in HORIZONS.items():
        data = build_direct_dataset(hourly, max_h)
        feats[name] = feature_columns(data, max_h)
        train_parts[name] = split(data)['train']
    offsets = {name: float(metrics[name]['cqr_offset_kw']) for name in HORIZONS}
    write_model_config(hourly, target_ranges(hourly, train_parts), offsets, feats)


def split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """예측 시점과 목표 시점이 모두 같은 구간에 있는 행만 그 구간에 넣는다."""
    parts = {}
    for name, (lo, hi) in SPLITS.items():
        mask = (data['origin_day'].between(lo, hi)
                & data['target_day'].between(lo, hi))
        parts[name] = data[mask].reset_index(drop=True)
    assert parts['train']['target_time'].max() < parts['val']['origin_time'].min()
    assert parts['val']['target_time'].max() < parts['test']['origin_time'].min()
    return parts


def fit_quantiles(train: pd.DataFrame,
                  feats: list[str]) -> dict[float, lgb.LGBMRegressor]:
    models = {}
    for q in QUANTILES:
        model = lgb.LGBMRegressor(objective='quantile', alpha=q, **LGB_PARAMS)
        model.fit(train[feats], train['y'])
        models[q] = model
    return models


def predict(models: dict[float, lgb.LGBMRegressor], data: pd.DataFrame,
            feats: list[str]) -> dict[float, np.ndarray]:
    return {q: m.predict(data[feats]) for q, m in models.items()}


def cqr_offset(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    """CQR 보정폭: 구간 밖으로 벗어난 정도의 (1-α) 분위수."""
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(scores, level, method='higher'))


def mape(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred) / np.abs(y)) * 100)


def evaluate(test: pd.DataFrame, preds: dict[float, np.ndarray],
             offset: float, max_h: int) -> dict:
    y = test['y'].to_numpy()
    lo, mid, hi = preds[0.05], preds[0.5], preds[0.95]
    lo_c, hi_c = lo - offset, hi + offset

    result = {
        'rows': int(len(test)),
        'mape_model': mape(y, mid),
        'mape_naive_lag168': mape(y, test['lag_168'].to_numpy()),
        'coverage_raw': float(np.mean((y >= lo) & (y <= hi)) * 100),
        'coverage_cqr': float(np.mean((y >= lo_c) & (y <= hi_c)) * 100),
        'width_raw_kw': float(np.mean(hi - lo)),
        'width_cqr_kw': float(np.mean(hi_c - lo_c)),
        'cqr_offset_kw': offset,
    }
    if 'lag_24' in test:
        result['mape_naive_lag24'] = mape(y, test['lag_24'].to_numpy())

    if max_h > 24:
        buckets = {'h1-24': (1, 24), 'h25-72': (25, 72), 'h73-168': (73, 168)}
        result['mape_by_horizon'] = {
            name: mape(y[m], mid[m])
            for name, (a, b) in buckets.items()
            if (m := test['h'].between(a, b).to_numpy()).any()
        }
    return result


def main() -> None:
    print('1. 데이터 로드 및 1시간 집계...')
    hourly = load_hourly(DATA_PATH)
    print(f'   {len(hourly):,}행 ({hourly.index[0]} ~ {hourly.index[-1]})')

    os.makedirs(MODEL_DIR, exist_ok=True)
    report = {}
    offsets, feats_by_model, train_parts = {}, {}, {}

    for name, max_h in HORIZONS.items():
        print(f'\n2. [{name}] 데이터셋 생성 (h = 1~{max_h})...')
        data = build_direct_dataset(
            hourly, max_h, exog_cols=WEATHER_COLS if USE_WEATHER_FEATURES else None)
        feats = feature_columns(data, max_h)
        parts = split(data)
        feats_by_model[name], train_parts[name] = feats, parts['train']
        print(f'   피처: {feats}')
        print('   ' + ' / '.join(f'{k} {len(v):,}행' for k, v in parts.items()))

        print(f'3. [{name}] 분위수 모델 학습 (Train)...')
        models = fit_quantiles(parts['train'], feats)

        print(f'4. [{name}] CQR 보정 (Val)...')
        val_pred = predict(models, parts['val'], feats)
        offset = cqr_offset(val_pred[0.05], val_pred[0.95],
                            parts['val']['y'].to_numpy())
        offsets[name] = offset

        print(f'5. [{name}] Test 평가...')
        test_pred = predict(models, parts['test'], feats)
        metrics = evaluate(parts['test'], test_pred, offset, max_h)
        report[name] = metrics
        for k, v in metrics.items():
            print(f'   {k}: {v:.3f}' if isinstance(v, float) else f'   {k}: {v}')

        importance = pd.Series(models[0.5].feature_importances_, index=feats)
        print('   q50 피처 중요도 상위 5:',
              importance.nlargest(5).to_dict())

        for q, model in models.items():
            model.booster_.save_model(os.path.join(MODEL_DIR, model_file(name, q)))

        out = parts['test'][['origin_time', 'target_time', 'h', 'y']].copy()
        out['q05'] = test_pred[0.05] - offset
        out['q50'] = test_pred[0.5]
        out['q95'] = test_pred[0.95] + offset
        out.to_parquet(os.path.join(OUTPUT_DIR, f'test_pred_{name}.parquet'),
                       index=False)

    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print('\n6. model_config.json 저장...')
    write_model_config(hourly, target_ranges(hourly, train_parts), offsets, feats_by_model)
    print(f'\n완료: 모델·예측·지표가 {OUTPUT_DIR}/ 에 저장되었습니다.')


if __name__ == '__main__':
    if '--config-only' in sys.argv[1:]:
        config_only()
    else:
        main()
