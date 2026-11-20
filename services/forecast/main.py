"""forecast-service (FastAPI :8002).

IT 부하(=냉각 부하) 24h / 168h 예측 + 90% 예측구간.
냉각 수요(냉각 전력) 예측: target='cooling_power'
  IT 부하 예측(q05/q50/q95) + 외기 → twin-service /api/v1/simulate (REST) 로 냉각 전력·냉각 모드·PUE.
  냉각 전력은 IT 부하에 대해 증가 함수라 IT 구간의 하한·상한을 그대로 냉각 전력 구간으로 바꾼다.

실행 (저장소 루트에서):
  uvicorn services.forecast.main:app --port 8002 --reload
  → http://localhost:8002/docs

모델이 없으면 503 을 돌려준다. (가짜 값을 돌려주면 MPC 가 그걸 믿고 제어하므로
스텁 폴백은 두지 않는다. Fallback 판단은 호출 쪽이 한다.)
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException

from services.common.settings import repo_path, section, service_url
from services.forecast.features import TARGET
from services.forecast.predictor import KINDS, ForecastResult, Predictor
from services.forecast.schemas import ForecastRequest, ForecastResponse, InputCheckResult

DATA_PATH = repo_path('data')
DEFAULT_MODEL: str = section('forecast')['model']  # 'ridge' | 'lgbm'

app = FastAPI(title='IDC Forecast Service', version='0.4.0')
_predictors: dict[str, Predictor | None] = {kind: Predictor.load(kind=kind) for kind in KINDS}
_predictor = _predictors.get(DEFAULT_MODEL)  # 기본 모델 (없으면 None → 503)
_twin = httpx.Client(base_url=service_url('twin'), timeout=float(section('services')['timeout_s']))


def set_twin_client(client: httpx.Client) -> None:
    """테스트·평가에서 다른 twin-service 클라이언트를 쓴다."""
    global _twin
    _twin = client


@lru_cache(maxsize=1)
def _hourly_outdoor() -> pd.Series:
    """서버 데이터의 시간별 외기온도 (외기 예보를 생략했을 때 쓴다)."""
    raw = pd.read_parquet(DATA_PATH, columns=['timestamp', 'outdoor_temp_c']).set_index(
        'timestamp'
    )
    return raw['outdoor_temp_c'].resample('h').mean()


def _cooling_power(result: ForecastResult, outdoor: list[float] | None) -> dict:
    """IT 부하 예측 + 외기 → 냉각 전력 예측 (twin-service, 기준 설정 온도 운전)."""
    if outdoor is None:
        series = _hourly_outdoor()
        stamps = pd.DatetimeIndex(result.timestamps)
        if not stamps.isin(series.index).all():
            raise HTTPException(
                status_code=422,
                detail='예측 구간의 외기 데이터가 없습니다. outdoor_forecast_c 를 넣어 주세요',
            )
        outdoor, source = series.loc[stamps].tolist(), 'data'
    else:
        source = 'request'
    points = [
        {'it_power_kw': it, 'outdoor_temp_c': t}
        for series_ in (result.pi_lower, result.predictions, result.pi_upper)
        for it, t in zip(series_, outdoor)
    ]
    try:
        r = _twin.post('/api/v1/simulate', json={'points': points})
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f'twin-service 호출 실패: {e}') from e
    res = r.json()['results']
    n = len(outdoor)
    lo, mid, hi = res[:n], res[n : 2 * n], res[2 * n :]
    return {
        'predictions': [round(x['cooling_power_kw'], 1) for x in mid],
        'pi_lower': [round(x['cooling_power_kw'], 1) for x in lo],
        'pi_upper': [round(x['cooling_power_kw'], 1) for x in hi],
        'it_predictions': result.predictions,
        'outdoor_temp_c': [float(t) for t in outdoor],
        'cooling_mode': [x['cooling_mode'] for x in mid],
        'free_cooling_fraction': [x['free_cooling_fraction'] for x in mid],
        'pue': [x['pue'] for x in mid],
        'outdoor_source': source,
    }


@lru_cache(maxsize=1)
def _default_history() -> pd.DataFrame:
    """history 생략 시 쓰는 기본 이력: 학습 데이터 전체 (마지막 시각이 예측 시점)."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail=f'기본 데이터 없음: {DATA_PATH}')
    return pd.read_parquet(DATA_PATH)[['timestamp', TARGET]]


@app.get('/health')
def health() -> dict:
    return {
        'status': 'ok',
        'service': 'forecast',
        'model': DEFAULT_MODEL if _predictor else 'unavailable',
        'available_models': [k for k, p in _predictors.items() if p is not None],
    }


@app.post('/api/v1/forecast', response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    kind = req.model or DEFAULT_MODEL
    predictor = _predictors.get(kind)
    if predictor is None:
        script = 'scripts/train_ridge.py' if kind == 'ridge' else 'scripts/train_forecast.py'
        raise HTTPException(
            status_code=503,
            detail=f'{kind} 모델이 로드되지 않았습니다. {script} 를 먼저 실행하세요.',
        )

    if req.history is not None:
        history = pd.DataFrame([p.model_dump() for p in req.history])
    else:
        history = _default_history()

    try:
        result = predictor.forecast(history, req.horizon_hours)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    values = {
        'predictions': result.predictions,
        'pi_lower': result.pi_lower,
        'pi_upper': result.pi_upper,
    }
    model = f'{kind}-{result.model}'
    if req.target == 'cooling_power':
        values = _cooling_power(result, req.outdoor_forecast_c)
        model += '+twin'
    return ForecastResponse(
        target=req.target,
        model=model,
        horizon_hours=req.horizon_hours,
        origin_time=result.origin_time.to_pydatetime(),
        timestamps=[t.to_pydatetime() for t in result.timestamps],
        in_training_range=result.in_training_range,
        input_quality_ok=result.input_quality_ok,
        input_checks=[InputCheckResult(**asdict(c)) for c in result.input_checks],
        warnings=result.warnings,
        **values,
    )
