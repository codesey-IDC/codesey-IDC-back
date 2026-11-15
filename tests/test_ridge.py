"""Ridge 선형 회귀 예측 모델 테스트: 저장 파일 재현, 서비스 모델 선택, 안전장치 공통 적용."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.common.settings import repo_path, section
from services.forecast import main as svc
from services.forecast import ridge
from services.forecast.features import build_direct_dataset, load_hourly
from services.forecast.predictor import DEFAULT_MODEL_DIR, Predictor
from scripts.train_forecast import feature_columns, mape, split

DATA_PATH = repo_path('data')
needs_ridge = pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / ridge.CONFIG_NAME).exists(), reason='ridge 모델 없음'
)


@pytest.fixture(scope='module')
def history() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)[['timestamp', 'it_power_kw']]


def test_pipeline_one_hot_and_scale() -> None:
    feats = ['h', 'last_value', 'hour', 'dayofweek']
    df = pd.DataFrame(
        {
            'h': [1, 2, 3, 4],
            'last_value': [1.0, 2.0, 3.0, 4.0],
            'hour': [0, 1, 0, 1],
            'dayofweek': [0, 0, 1, 1],
        }
    )
    pipe = ridge.build_pipeline(feats).fit(df[feats], [1.0, 2.0, 3.0, 4.0])
    assert pipe.predict(df[feats]).shape == (4,)
    assert (
        pipe.named_steps['pre'].transform(df[feats]).shape[1] == 2 + 2 + 2
    )  # hour 2 + dow 2 + 수치 2


@needs_ridge
def test_saved_ridge_reproduces_test_metrics() -> None:
    """저장된 ridge 모델이 ridge_config.json 에 적힌 Test MAPE 를 그대로 낸다."""
    cfg = json.loads((DEFAULT_MODEL_DIR / ridge.CONFIG_NAME).read_text(encoding='utf-8'))
    data = build_direct_dataset(load_hourly(str(DATA_PATH)), 24)
    feats = feature_columns(data, 24)
    test = split(data)['test']
    p = Predictor(kind='ridge')
    pred = p.ridge['24h']['pipeline'].predict(test[feats])
    assert mape(test['y'].to_numpy(), pred) == pytest.approx(cfg['models']['24h']['test']['mape'])


@needs_ridge
@pytest.mark.parametrize('horizon, key', [(24, '24h'), (48, '168h')])
def test_ridge_interval_is_symmetric(history: pd.DataFrame, horizon: int, key: str) -> None:
    p = Predictor(kind='ridge')
    r = p.forecast(history, horizon)
    hw = p.ridge[key]['halfwidth_kw']
    np.testing.assert_allclose(np.array(r.pi_upper) - np.array(r.predictions), hw, atol=0.11)
    np.testing.assert_allclose(np.array(r.predictions) - np.array(r.pi_lower), hw, atol=0.11)
    assert r.model == key and len(r.predictions) == horizon


@needs_ridge
def test_ridge_safeguards_shared(history: pd.DataFrame) -> None:
    p = Predictor(kind='ridge')
    with pytest.raises(ValueError, match='168'):
        p.forecast(history.tail(12 * 100), 24)
    surge = history.copy()
    surge['it_power_kw'] *= 1.3
    r = p.forecast(surge, 24)
    assert not r.in_training_range and any(w.startswith('input_out_of_range') for w in r.warnings)


def test_bad_kind() -> None:
    with pytest.raises(ValueError):
        Predictor(kind='lstm')


def test_service_default_model_from_config() -> None:
    assert svc.DEFAULT_MODEL == section('forecast')['model']
    body = TestClient(svc.app).get('/health').json()
    assert body['model'] in ('ridge', 'lgbm', 'unavailable')


@needs_ridge
def test_service_model_selection(history: pd.DataFrame) -> None:
    client = TestClient(svc.app)
    for kind in ('ridge', 'lgbm'):
        if svc._predictors.get(kind) is None:
            continue
        r = client.post('/api/v1/forecast', json={'horizon_hours': 24, 'model': kind})
        assert r.status_code == 200, r.text
        assert r.json()['model'] == f'{kind}-24h'
    assert (
        client.post('/api/v1/forecast', json={'horizon_hours': 24, 'model': 'xgb'}).status_code
        == 422
    )
