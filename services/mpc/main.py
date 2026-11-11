"""mpc-service (FastAPI :8003). 공급 공기 설정 온도를 정하는 모델 예측 제어.

forecast-service(:8002), twin-service(:8001) 를 REST 로 호출한다.
주소는 환경변수로 바꿀 수 있다: FORECAST_URL, TWIN_URL

실행 (저장소 루트에서, 다른 두 서비스를 먼저 띄운 뒤):
  uvicorn services.twin.main:app --port 8001
  uvicorn services.forecast.main:app --port 8002
  uvicorn services.mpc.main:app --port 8003
  → http://localhost:8003/docs

  GET  /health
  POST /api/v1/plan    현재 서버실 온도 + 최근 IT 부하 + 외기 예보 → 설정 온도
"""
from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, HTTPException

from services.mpc.controller import MPCController
from services.mpc.schemas import PlanRequest, PlanResponse

# localhost 대신 127.0.0.1: Windows 에서 localhost 는 IPv6 를 먼저 시도해 연결마다 약 2초 지연된다
FORECAST_URL = os.environ.get('FORECAST_URL', 'http://127.0.0.1:8002')
TWIN_URL = os.environ.get('TWIN_URL', 'http://127.0.0.1:8001')

app = FastAPI(title='IDC MPC Service', version='0.1.0')
_controller = MPCController(
    twin=httpx.Client(base_url=TWIN_URL, timeout=30.0),
    forecast=httpx.Client(base_url=FORECAST_URL, timeout=30.0),
)


def set_controller(controller: MPCController) -> None:
    """테스트·시뮬레이션에서 다른 클라이언트를 쓰는 제어기로 바꾼다."""
    global _controller
    _controller = controller


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'mpc', 'forecast_url': FORECAST_URL, 'twin_url': TWIN_URL}


@app.post('/api/v1/plan', response_model=PlanResponse)
def plan(req: PlanRequest) -> PlanResponse:
    history = [{'timestamp': p.timestamp.isoformat(), 'it_power_kw': p.it_power_kw} for p in req.history]
    try:
        r = _controller.plan(req.room_temp_c, history, req.outdoor_forecast_c, req.capacity_factor)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PlanResponse(**{k: v for k, v in r.__dict__.items() if k != 'cost'})
