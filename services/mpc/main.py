"""mpc-service (FastAPI :8003). 공급 공기 설정 온도를 정하는 모델 예측 제어.

forecast-service(:8002), twin-service(:8001) 를 REST 로 호출한다.
주소는 환경변수로 바꿀 수 있다: FORECAST_URL, TWIN_URL

실행 (저장소 루트에서, 다른 두 서비스를 먼저 띄운 뒤):
  uvicorn services.twin.main:app --port 8001
  uvicorn services.forecast.main:app --port 8002
  uvicorn services.mpc.main:app --port 8003
  → http://localhost:8003/docs

  GET  /health
  POST /api/v1/plan       현재 서버실 온도 + 최근 IT 부하 + 외기 예보 → 설정 온도
  POST /api/v1/simulate   폐루프 시뮬레이션 (구간·제어기·위기 시나리오 선택) → 시간별 결과 + 요약
  GET  /api/v1/scenarios  위기 시나리오 목록
  GET  /api/v1/results    미리 계산해 둔 장기 비교 실험 (output/mpc/)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from services.mpc import simulation
from services.mpc.controller import MPCController
from services.mpc.schemas import (
    PlanRequest, PlanResponse, ResultsResponse, ScenarioInfo, SimulateRequest, SimulateResponse,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / 'output' / 'mpc'

# localhost 대신 127.0.0.1: Windows 에서 localhost 는 IPv6 를 먼저 시도해 연결마다 약 2초 지연된다
FORECAST_URL = os.environ.get('FORECAST_URL', 'http://127.0.0.1:8002')
TWIN_URL = os.environ.get('TWIN_URL', 'http://127.0.0.1:8001')

app = FastAPI(title='IDC MPC Service', version='0.2.0')
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


@app.post('/api/v1/simulate', response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    """매시간 설정 온도 결정 → 트윈 1시간 진행을 hours 번 반복한다. MPC 는 1시간당 약 0.3초."""
    try:
        # 계획 상태(직전 해)가 /plan 호출과 섞이지 않게 시뮬레이션마다 새 제어기를 만든다
        r = simulation.run_closed_loop(_controller.twin, _controller.forecast, pd.Timestamp(req.start),
                                       req.hours, req.controller, req.scenario, _controller.config)
    except simulation.SimulationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f'twin-service 호출 실패: {e}') from e
    return SimulateResponse(**r)


@app.get('/api/v1/scenarios', response_model=list[ScenarioInfo])
def scenarios() -> list[ScenarioInfo]:
    return [ScenarioInfo(name=name, label=s.label, strategy=s.strategy, it_scale=s.it_scale,
                         outdoor_min_c=s.outdoor_min_c, capacity_factor=s.capacity_factor)
            for name, s in simulation.SCENARIOS.items()]


@app.get('/api/v1/results', response_model=ResultsResponse)
def results(scenario: str = Query(default='base', pattern='^[a-z0-9_]+$',
                                  description="'base'(Test 14일) | 'surge'(서버 급증 3일)")) -> ResultsResponse:
    metrics_path = RESULTS_DIR / f'metrics_{scenario}.json'
    series_path = RESULTS_DIR / f'timeseries_{scenario}.csv'
    if not metrics_path.exists() or not series_path.exists():
        raise HTTPException(status_code=404, detail=f"결과 파일 없음: '{scenario}'. simulate_mpc.py 를 먼저 실행하세요.")
    meta = json.loads(metrics_path.read_text(encoding='utf-8'))
    ts = pd.read_csv(series_path)
    series = {name: {c: g[c].tolist() for c in g.columns if c != 'controller'}
              for name, g in ts.groupby('controller', sort=False)}
    return ResultsResponse(start=meta['start'], hours=meta['hours'], scenario=meta['scenario'],
                           controllers=meta['controllers'], timeseries=series)
