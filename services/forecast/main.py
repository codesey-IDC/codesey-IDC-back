"""forecast-service (FastAPI :8002).

IT 부하(=냉각 부하) 24h / 168h 예측 + 90% 예측구간.

실행 (저장소 루트에서):
  uvicorn services.forecast.main:app --port 8002 --reload
  → http://localhost:8002/docs

모델이 없으면 503 을 돌려준다. (가짜 값을 돌려주면 MPC 가 그걸 믿고 제어하므로
스텁 폴백은 두지 않는다. Fallback 판단은 호출 쪽이 한다.)
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException

from services.forecast.features import TARGET
from services.forecast.predictor import REPO_ROOT, Predictor
from services.forecast.schemas import ForecastRequest, ForecastResponse

DATA_PATH = REPO_ROOT / 'data' / 'synthetic_idc_90d.parquet'

app = FastAPI(title='IDC Forecast Service', version='0.2.0')
_predictor = Predictor.load()


@lru_cache(maxsize=1)
def _default_history() -> pd.DataFrame:
    """history 생략 시 쓰는 기본 이력: 학습 데이터 전체 (마지막 시각이 예측 시점)."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail=f'기본 데이터 없음: {DATA_PATH}')
    return pd.read_parquet(DATA_PATH)[['timestamp', TARGET]]


@app.get('/health')
def health() -> dict[str, str]:
    return {
        'status': 'ok',
        'service': 'forecast',
        'model': 'lightgbm' if _predictor else 'unavailable',
    }


@app.post('/api/v1/forecast', response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail='모델이 로드되지 않았습니다. train_forecast.py 를 먼저 실행하세요.')

    if req.history is not None:
        history = pd.DataFrame([p.model_dump() for p in req.history])
    else:
        history = _default_history()

    try:
        result = _predictor.forecast(history, req.horizon_hours)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return ForecastResponse(
        target=req.target,
        model=f'lgbm-{result.model}',
        horizon_hours=req.horizon_hours,
        origin_time=result.origin_time.to_pydatetime(),
        timestamps=[t.to_pydatetime() for t in result.timestamps],
        predictions=result.predictions,
        pi_lower=result.pi_lower,
        pi_upper=result.pi_upper,
        in_training_range=result.in_training_range,
        warnings=result.warnings,
    )
