"""twin-service (FastAPI :8001). 가상 데이터센터(디지털 트윈).

실행 (저장소 루트에서):
  uvicorn services.twin.main:app --port 8001 --reload
  → http://localhost:8001/docs

  GET  /health
  GET  /api/v1/params     모델 상수 (데이터 복원값 + 가정값)
  POST /api/v1/simulate   한 시점 정상 상태 계산 (부하·외기 → 냉각 전력·PUE)
  POST /api/v1/rollout    설정 온도 계획 평가 (MPC 용: 서버실 온도·냉각 전력 시계열)
  POST /api/v1/rollout/batch  계획 여러 개를 한 번에 평가 (MPC 후보 비교용, 합계만)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from services.twin import physics
from services.twin.schemas import (
    RolloutBatchRequest, RolloutBatchResponse, RolloutRequest, RolloutResponse,
    SimulateRequest, SimulateResponse, SimulateResult,
)

app = FastAPI(title='IDC Digital Twin Service', version='0.1.0')
PARAMS = physics.DEFAULT_PARAMS


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
