"""IT 부하 예측 모델 하이퍼파라미터 튜닝 (Optuna)

흐름
  1. train_forecast.py 와 같은 데이터·피처·분할을 쓴다 (Train / Val / Test)
  2. Optuna 로 q50(중앙값) 모델의 하이퍼파라미터를 탐색
       - 학습: Train 만 사용
       - 평가: Val MAPE  ← 조합 비교는 Val 로만 한다
  3. 가장 좋은 조합으로 q05/q50/q95 를 다시 학습하고 Val 에서 CQR 보정
  4. 마지막에 Test 에서 1회만 평가해서 기존 고정값(LGB_PARAMS)과 비교

Test 는 조합 선택에 절대 쓰지 않는다. Test 를 보면서 고르면
Test 에 맞춘 셈이 되어 점수가 실제보다 좋게 나온다.

운영 모델(output/forecast/models/)은 건드리지 않는다.
결과는 output/forecast/tuning/ 에 저장되며, 채택하려면
best_params.json 값을 train_forecast.py 의 LGB_PARAMS 에 옮기고 다시 학습한다.

실행:
  pip install -r requirements.txt
  python tune_forecast.py                  # 24h, 168h 각 30회 탐색
  python tune_forecast.py --trials 50      # 탐색 횟수 변경
  python tune_forecast.py --horizon 24h    # 한 모델만
"""
from __future__ import annotations

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from train_forecast import (
    DATA_PATH, LGB_PARAMS, OUTPUT_DIR, QUANTILES, WEATHER_COLS,
    USE_WEATHER_FEATURES, cqr_offset, evaluate, feature_columns, mape, split,
)
from services.forecast.features import HORIZONS, build_direct_dataset, load_hourly

TUNING_DIR = os.path.join(OUTPUT_DIR, 'tuning')
SEED = 42


def suggest_params(trial: optuna.Trial) -> dict:
    """탐색 범위. 현재 고정값(LGB_PARAMS)이 범위 안에 들어가도록 잡았다."""
    return dict(
        n_estimators=trial.suggest_int('n_estimators', 100, 1500, step=50),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        num_leaves=trial.suggest_int('num_leaves', 4, 63),
        min_child_samples=trial.suggest_int('min_child_samples', 10, 300, log=True),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        random_state=SEED,
        verbose=-1,
    )


def fit_quantiles(train: pd.DataFrame, feats: list[str],
                  params: dict) -> dict[float, lgb.LGBMRegressor]:
    models = {}
    for q in QUANTILES:
        model = lgb.LGBMRegressor(objective='quantile', alpha=q, **params)
        model.fit(train[feats], train['y'])
        models[q] = model
    return models


def tune(name: str, parts: dict[str, pd.DataFrame], feats: list[str],
         n_trials: int) -> optuna.Study:
    train, val = parts['train'], parts['val']
    y_val = val['y'].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        model = lgb.LGBMRegressor(objective='quantile', alpha=0.5, **params)
        model.fit(train[feats], train['y'])
        return mape(y_val, model.predict(val[feats]))

    study = optuna.create_study(
        study_name=f'it_load_{name}', direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED))
    # 기존 고정값을 첫 시도로 넣어서 "튜닝 전" 기준을 같은 조건으로 잰다
    study.enqueue_trial({k: v for k, v in LGB_PARAMS.items()
                         if k in ('n_estimators', 'learning_rate', 'num_leaves',
                                  'min_child_samples', 'subsample', 'colsample_bytree')}
                        | {'reg_lambda': 1e-3})
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_log_trial])
    return study


def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    mark = ' ← best' if trial.number == study.best_trial.number else ''
    print(f'   trial {trial.number:3d}  Val MAPE {trial.value:.4f}%{mark}')


def final_eval(parts: dict[str, pd.DataFrame], feats: list[str],
               params: dict, max_h: int) -> dict:
    """3개 분위수 학습 → Val CQR → Test 1회 평가 (train_forecast.py 와 같은 절차)."""
    models = fit_quantiles(parts['train'], feats, params)
    val_pred = {q: m.predict(parts['val'][feats]) for q, m in models.items()}
    offset = cqr_offset(val_pred[0.05], val_pred[0.95], parts['val']['y'].to_numpy())
    test_pred = {q: m.predict(parts['test'][feats]) for q, m in models.items()}
    result = evaluate(parts['test'], test_pred, offset, max_h)
    result['val_mape_model'] = mape(parts['val']['y'].to_numpy(), val_pred[0.5])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--trials', type=int, default=30, help='모델당 탐색 횟수 (기본 30)')
    parser.add_argument('--horizon', choices=list(HORIZONS), help='한 모델만 튜닝')
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(TUNING_DIR, exist_ok=True)

    print('1. 데이터 로드...')
    hourly = load_hourly(DATA_PATH)
    names = [args.horizon] if args.horizon else list(HORIZONS)

    best_params, comparison = {}, {}
    for name in names:
        max_h = HORIZONS[name]
        data = build_direct_dataset(
            hourly, max_h, exog_cols=WEATHER_COLS if USE_WEATHER_FEATURES else None)
        feats = feature_columns(data, max_h)
        parts = split(data)

        print(f'\n2. [{name}] Optuna 탐색 {args.trials}회 (Train 학습 → Val MAPE)...')
        start = time.time()
        study = tune(name, parts, feats, args.trials)
        print(f'   소요 {time.time() - start:.0f}초, '
              f'best trial {study.best_trial.number} Val MAPE {study.best_value:.4f}%')

        study.trials_dataframe().to_csv(
            os.path.join(TUNING_DIR, f'trials_{name}.csv'), index=False)
        params = {**suggest_params(optuna.trial.FixedTrial(study.best_params))}
        best_params[name] = params

        print(f'3. [{name}] 기존 고정값 vs 튜닝값 최종 평가 (Test 1회)...')
        comparison[name] = {
            'baseline': final_eval(parts, feats, LGB_PARAMS, max_h),
            'tuned': final_eval(parts, feats, params, max_h),
        }
        _print_comparison(comparison[name])

    with open(os.path.join(TUNING_DIR, 'best_params.json'), 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=4)
    with open(os.path.join(TUNING_DIR, 'comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=4)
    print(f'\n완료: 결과가 {TUNING_DIR}/ 에 저장되었습니다. (운영 모델은 변경 없음)')


def _print_comparison(cmp: dict) -> None:
    rows = [
        ('Val MAPE (%)', 'val_mape_model'),
        ('Test MAPE (%)', 'mape_model'),
        ('Test 기준선 lag168 (%)', 'mape_naive_lag168'),
        ('Test 적중률 CQR (%)', 'coverage_cqr'),
        ('Test 구간 폭 (kW)', 'width_cqr_kw'),
    ]
    print(f'   {"":24s}{"기존":>10s}{"튜닝":>10s}')
    for label, key in rows:
        print(f'   {label:24s}{cmp["baseline"][key]:10.3f}{cmp["tuned"][key]:10.3f}')


if __name__ == '__main__':
    main()
