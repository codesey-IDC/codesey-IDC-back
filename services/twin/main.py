"""twin-service (FastAPI :8001). 가상 데이터센터(디지털 트윈).

실행 (저장소 루트에서):
  uvicorn services.twin.main:app --port 8001 --reload
  → http://localhost:8001/docs

  GET  /health
  GET  /api/v1/params     모델 상수 (데이터 복원값 + 가정값)
  POST /api/v1/simulate   한 시점 정상 상태 계산 (부하·외기 → 냉각 전력·PUE)
  POST /api/v1/rollout    설정 온도 계획 평가 (MPC 용: 서버실 온도·냉각 전력 시계열)
  POST /api/v1/rollout/batch  계획 여러 개를 한 번에 평가 (MPC 후보 비교용, 합계만)
  GET  /api/v1/telemetry  시간별 계측값 (IT 부하·외기). 대시보드·MPC 시뮬레이션이 파일 대신 이걸 쓴다
  POST /api/v1/esg        시뮬레이션 결과 시계열 → 탄소·물 사용량 지표
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from services.twin import esg, physics
from services.twin.schemas import (
    EsgRequest, EsgResponse, RolloutBatchRequest, RolloutBatchResponse, RolloutRequest, RolloutResponse,
    SimulateRequest, SimulateResponse, SimulateResult, TelemetryResponse,
)

DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'synthetic_idc_90d.parquet'
TELEMETRY_COLS = ['it_power_kw', 'outdoor_temp_c']

app = FastAPI(title='IDC Digital Twin Service', version='0.2.0')
PARAMS = physics.DEFAULT_PARAMS


@lru_cache(maxsize=1)
def _hourly() -> pd.DataFrame:
    """5분 계측 데이터를 1시간 평균으로 집계 (첫 호출 때 한 번만 읽는다)."""
    raw = pd.read_parquet(DATA_PATH, columns=['timestamp'] + TELEMETRY_COLS).set_index('timestamp')
    return raw[TELEMETRY_COLS].resample('h').mean()


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'twin'}


@app.get('/api/v1/params')
def params() -> dict:
    return PARAMS.to_dict()


@app.post('/api/v1/simulate', response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    it = [p.it_power_kw if p.it_power_kw is not None
          else float(physics.it_power_from_cpu(p.cpu_utilization, PARAMS)) for p in req.points]
    supply = [PARAMS.supply_temp_ref_c if p.supply_temp_c is None else p.supply_temp_c
              for p in req.points]
    r = physics.simulate_static(it, [p.outdoor_temp_c for p in req.points], supply, PARAMS)
    results = [
        SimulateResult(**{k: (str(v[i]) if k == 'cooling_mode' else float(v[i])) for k, v in r.items()})
        for i in range(len(it))
    ]
    return SimulateResponse(results=results)


@app.post('/api/v1/rollout', response_model=RolloutResponse)
def rollout(req: RolloutRequest) -> RolloutResponse:
    try:
        r = physics.rollout(req.it_power_kw, req.outdoor_temp_c, req.supply_temp_c,
                            initial_room_temp_c=req.initial_room_temp_c, dt_hours=req.dt_hours,
                            capacity_factor=req.capacity_factor, p=PARAMS)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return RolloutResponse(**r.__dict__)


@app.post('/api/v1/rollout/batch', response_model=RolloutBatchResponse)
def rollout_batch(req: RolloutBatchRequest) -> RolloutBatchResponse:
    try:
        s = physics.rollout_batch(req.it_power_kw, req.outdoor_temp_c, req.plans,
                                  initial_room_temp_c=req.initial_room_temp_c, dt_hours=req.dt_hours,
                                  capacity_factor=req.capacity_factor, p=PARAMS)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return RolloutBatchResponse(**{k: v.tolist() for k, v in s.items()})


@app.get('/api/v1/telemetry', response_model=TelemetryResponse)
def telemetry(start: datetime | None = Query(default=None, description='시작 시각. 생략 시 데이터 끝에서 hours 시간'),
              hours: int = Query(default=24, ge=1, le=2400)) -> TelemetryResponse:
    """[start, start + hours) 의 시간별 IT 부하·외기온도. 데이터 범위를 벗어난 부분은 빼고 돌려준다."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail=f'계측 데이터 없음: {DATA_PATH}')
    hourly = _hourly()
    if start is None:
        first = hourly.index[-1] - pd.Timedelta(hours=hours - 1)
    else:
        first = pd.Timestamp(start).tz_localize(None).floor('h')
    win = hourly.loc[first:first + pd.Timedelta(hours=hours - 1)].dropna()
    if win.empty:
        raise HTTPException(status_code=404, detail=(
            f'요청 구간에 데이터가 없습니다: {first} 부터 {hours}시간 '
            f'(데이터 범위 {hourly.index[0]} ~ {hourly.index[-1]})'))
    return TelemetryResponse(
        timestamps=[t.to_pydatetime() for t in win.index],
        it_power_kw=win['it_power_kw'].tolist(), outdoor_temp_c=win['outdoor_temp_c'].tolist(),
        data_start=hourly.index[0].to_pydatetime(), data_end=hourly.index[-1].to_pydatetime())


@app.post('/api/v1/esg', response_model=EsgResponse)
def esg_summary(req: EsgRequest) -> EsgResponse:
    try:
        return EsgResponse(**esg.summarize(
            req.it_power_kw, req.cooling_power_kw, req.cooling_mode, dt_hours=req.dt_hours,
            heat_removed_kw=req.heat_removed_kw, other_overhead_ratio=PARAMS.other_overhead_ratio))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
