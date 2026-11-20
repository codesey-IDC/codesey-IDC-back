"""mpc-service (FastAPI :8003). 공급 공기 설정 온도를 정하는 모델 예측 제어.

forecast-service(:8002), twin-service(:8001) 를 REST 로 호출한다.
주소는 config/settings.yaml 의 services 항목. 환경변수 FORECAST_URL, TWIN_URL 이 있으면 그 값이 우선한다.

실행 (저장소 루트에서, 다른 두 서비스를 먼저 띄운 뒤):
  uvicorn services.twin.main:app --port 8001
  uvicorn services.forecast.main:app --port 8002
  uvicorn services.mpc.main:app --port 8003
  → http://localhost:8003/docs

  GET  /health            status: ok | degraded (예측·트윈에 닿지 않거나 직전 계획이 예측 불신·Fallback)
  POST /api/v1/control/optimize  현재 상태 → 시간대별 최적 냉각 설정값(설정 온도·축열조) + 예상 PUE
  POST /api/v1/control/mpc       현재 상태 → MPC 제어 결과 전체 (예측 부하·단가·예상 상태·계산 시간)
  POST /api/v1/plan       (대시보드용) 서버실 온도 + 최근 IT 부하 + 외기 예보 → MPC 결과. control/mpc 와 같은 계산
  POST /api/v1/simulate/jobs           폐루프 시뮬레이션 작업 등록 → 바로 응답 (작업 번호, 진행률)
  GET  /api/v1/simulate/jobs/{job_id}  작업 상태·진행률, 끝나면 결과 (같은 요청은 결과 재사용)
  POST /api/v1/simulate   폐루프 시뮬레이션을 끝날 때까지 기다려 결과 (짧은 계산·스크립트용.
                          MPC 는 1시간당 약 0.4초라 24시간 이상이면 jobs 를 쓴다)
  GET  /api/v1/scenarios  위기 시나리오 목록
  GET  /api/v1/results    미리 계산해 둔 장기 비교 실험 (output/mpc/)
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from services.common.settings import repo_path, section, service_url
from services.mpc import simulation
from services.mpc.controller import MPCController, PlanResult
from services.mpc.jobs import Job, SimulationJobs
from services.mpc.schemas import (
    ControlRequest,
    ControlSchedule,
    PlanRequest,
    PlanResponse,
    ResultsResponse,
    ScenarioInfo,
    SimulateRequest,
    SimulateResponse,
    SimulationJob,
)

RESULTS_DIR = repo_path('mpc_results')

# 주소는 config/settings.yaml 의 services 항목 (환경변수 FORECAST_URL, TWIN_URL 이 있으면 그 값)
FORECAST_URL = service_url('forecast')
TWIN_URL = service_url('twin')
TIMEOUT_S = float(section('services')['timeout_s'])
HEALTH_TIMEOUT_S = 2.0  # /health 가 의존 서비스 상태를 물어볼 때 기다리는 시간

app = FastAPI(title='IDC MPC Service', version='0.4.0')
_controller = MPCController(
    twin=httpx.Client(base_url=TWIN_URL, timeout=TIMEOUT_S),
    forecast=httpx.Client(base_url=FORECAST_URL, timeout=TIMEOUT_S),
)


def set_controller(controller: MPCController) -> None:
    """테스트·시뮬레이션에서 다른 클라이언트를 쓰는 제어기로 바꾼다."""
    global _controller
    _controller = controller


def _run_simulation(params: dict, progress: Callable[[int, int], None] | None = None) -> dict:
    """폐루프 시뮬레이션 1회. 계획 상태가 /plan 호출과 섞이지 않게 시뮬레이션마다 새 제어기를 만든다."""
    return simulation.run_closed_loop(
        _controller.twin,
        _controller.forecast,
        pd.Timestamp(params['start']),
        params['hours'],
        params['controller'],
        params['scenario'],
        _controller.config,
        initial_room_temp_c=params.get('initial_room_temp_c'),
        progress=progress,
    )


_jobs = SimulationJobs(_run_simulation)


def _dependency_state(client: httpx.Client) -> str:
    """의존 서비스 상태: 'ok' | 'model_unavailable'(예측 모델 파일 없음) | 'unreachable'."""
    try:
        r = client.get('/health', timeout=HEALTH_TIMEOUT_S)
        if r.status_code != 200:
            return 'unreachable'
        return 'model_unavailable' if r.json().get('model') == 'unavailable' else 'ok'
    except (httpx.HTTPError, ValueError):
        return 'unreachable'


@app.get('/health')
def health() -> dict:
    """프로세스가 살아 있으면 항상 200. status 는 제어가 정상 경로(mpc)로 돌 수 있는지를 말한다.

    ok        예측·트윈 서비스가 응답하고, 직전 계획이 정상 모드(mpc)였거나 아직 계획한 적이 없다
    degraded  예측 또는 트윈에 닿지 않거나, 직전 계획이 예측 불신(mpc_naive)·Fallback 이었다.
              안전하게 물러난 상태지만 "물러났다"는 사실을 알리기 위해 ok 와 구분한다 (reasons 에 이유)
    """
    deps = {
        'forecast': _dependency_state(_controller.forecast),
        'twin': _dependency_state(_controller.twin),
    }
    reasons = [f'{name}_{state}' for name, state in deps.items() if state != 'ok']
    last_mode = _controller.last_mode
    if last_mode not in (None, 'mpc'):
        reasons.append(f'last_plan_{last_mode}')
    return {
        'status': 'degraded' if reasons else 'ok',
        'service': 'mpc',
        'forecast_url': FORECAST_URL,
        'twin_url': TWIN_URL,
        'dependencies': deps,
        'last_plan_mode': last_mode,
        'reasons': reasons,
    }


def _plan_response(r: PlanResult) -> PlanResponse:
    return PlanResponse(**{k: v for k, v in r.__dict__.items() if k != 'cost'})


def _resolve_inputs(req: ControlRequest) -> tuple[list[dict], list[float], str]:
    """이력·외기 예보: 요청에 있으면 그대로, 없으면 twin-service 계측값에서 (timestamp 기준)."""
    history = (
        None
        if req.history is None
        else [
            {'timestamp': h.timestamp.isoformat(), 'it_power_kw': h.it_power_kw}
            for h in req.history
        ]
    )
    outdoor = req.outdoor_forecast_c
    if history is not None and outdoor is not None:
        return history, outdoor, 'request'

    horizon = _controller.config.horizon_hours
    now = pd.Timestamp(req.timestamp).tz_localize(None).floor('h')
    first = now - pd.Timedelta(hours=simulation.HISTORY_HOURS)
    r = _controller.twin.get(
        '/api/v1/telemetry',
        params={'start': first.isoformat(), 'hours': simulation.HISTORY_HOURS + horizon},
    )
    if r.status_code != 200:
        raise HTTPException(
            status_code=422,
            detail=f'계측값을 받을 수 없습니다 (twin telemetry {r.status_code}): '
            f'{r.text[:200]}',
        )
    body = r.json()
    stamps = pd.to_datetime(body['timestamps'])
    past, future = stamps < now, stamps >= now
    if history is None:
        history = [
            {'timestamp': t.isoformat(), 'it_power_kw': float(v)}
            for t, v, m in zip(stamps, body['it_power_kw'], past)
            if m
        ]
        if len(history) < 168:
            raise HTTPException(
                status_code=422,
                detail=f'{now} 이전 계측 이력이 168시간보다 짧습니다: {len(history)}시간',
            )
    if outdoor is None:
        outdoor = [float(v) for v, m in zip(body['outdoor_temp_c'], future) if m][:horizon]
        if not outdoor:
            raise HTTPException(status_code=422, detail=f'{now} 이후 외기 계측값이 없습니다')
        outdoor += [outdoor[-1]] * (horizon - len(outdoor))  # 데이터 끝: 마지막 값 유지
    return history, outdoor, 'twin_telemetry'


def _run_control(req: ControlRequest) -> tuple[PlanResult, str]:
    history, outdoor, source = _resolve_inputs(req)
    try:
        r = _controller.plan(
            req.room_temp_c,
            history,
            outdoor,
            req.capacity_factor,
            req.tes_soc_kwh,
            current_setpoint_c=req.current_setpoint_c,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return r, source


@app.post('/api/v1/control/optimize', response_model=ControlSchedule)
def control_optimize(req: ControlRequest) -> ControlSchedule:
    """현재 상태 → 시간대별 최적 냉각 설정값(공급 온도·축열조 출력)과 예상 PUE. 첫 시간 값만 적용하고 매시간 다시 부른다."""
    r, source = _run_control(req)
    hours = len(r.plan_c)
    energy, it_energy = r.expected_facility_energy_kwh, None
    if r.predicted_pue and r.predicted_facility_kw:
        it_energy = sum(f / p for f, p in zip(r.predicted_facility_kw, r.predicted_pue))
    return ControlSchedule(
        mode=r.mode,
        objective=r.objective,
        timestamps=r.timestamps,
        supply_temp_c=r.plan_c,
        tes_kw=r.tes_plan_kw or [0.0] * hours,
        cooling_mode=r.predicted_cooling_mode,
        expected_pue=r.predicted_pue,
        expected_room_temp_c=r.predicted_room_temp_c,
        expected_tes_soc_kwh=r.predicted_tes_soc_kwh,
        price_krw_per_kwh=r.price_krw_per_kwh,
        expected_energy_cost_krw=r.expected_energy_cost_krw,
        baseline_energy_cost_krw=r.baseline_energy_cost_krw,
        expected_cost_saving_pct=r.expected_cost_saving_pct,
        expected_average_pue=None if not it_energy else energy / it_energy,
        computation_time_s=r.computation_time_s,
        inputs_from=source,
        warnings=r.warnings,
    )


@app.post('/api/v1/control/mpc', response_model=PlanResponse)
def control_mpc(req: ControlRequest) -> PlanResponse:
    """현재 상태 → MPC 제어 결과 전체 (지금 적용할 값, 24시간 계획, 예측 부하, 단가, 예상 상태, 계산 시간)."""
    r, _ = _run_control(req)
    return _plan_response(r)


@app.post('/api/v1/plan', response_model=PlanResponse)
def plan(req: PlanRequest) -> PlanResponse:
    history = [
        {'timestamp': p.timestamp.isoformat(), 'it_power_kw': p.it_power_kw} for p in req.history
    ]
    try:
        r = _controller.plan(
            req.room_temp_c,
            history,
            req.outdoor_forecast_c,
            req.capacity_factor,
            req.tes_soc_kwh,
            current_setpoint_c=req.current_setpoint_c,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _plan_response(r)


@app.post('/api/v1/simulate', response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    """매시간 설정 온도 결정 → 트윈 1시간 진행을 hours 번 반복하고 끝날 때까지 기다린다.

    MPC 는 1시간당 약 0.4초라 오래 걸린다. 같은 요청의 작업이 이미 끝났으면 그 결과를 바로 돌려준다.
    """
    params = req.model_dump(mode='json')
    done = _jobs.find_done(params)
    if done is not None:
        return SimulateResponse(**done.result)
    try:
        r = _run_simulation(params)
    except simulation.SimulationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f'twin-service 호출 실패: {e}') from e
    return SimulateResponse(**r)


def _job_response(job: Job) -> SimulationJob:
    return SimulationJob(
        job_id=job.job_id,
        status=job.status,
        progress_hours=job.progress_hours,
        total_hours=job.total_hours,
        elapsed_s=job.elapsed_s,
        request=SimulateRequest(**job.params),
        result=None if job.result is None else SimulateResponse(**job.result),
        error=job.error,
    )


@app.post('/api/v1/simulate/jobs', response_model=SimulationJob, status_code=202)
def submit_simulation(req: SimulateRequest) -> SimulationJob:
    """시뮬레이션을 백그라운드 작업으로 등록하고 바로 응답한다. 같은 요청이면 기존 작업(결과)을 돌려준다."""
    try:
        simulation.validate_request(req.controller, req.scenario)
    except simulation.SimulationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _job_response(_jobs.submit(req.model_dump(mode='json')))


@app.get('/api/v1/simulate/jobs/{job_id}', response_model=SimulationJob)
def simulation_status(job_id: str) -> SimulationJob:
    """작업 상태: queued → running (progress_hours / total_hours) → done (result) | failed (error)."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f'작업이 없습니다: {job_id} (서비스를 다시 띄우면 작업 목록이 비워집니다)',
        )
    return _job_response(job)


@app.get('/api/v1/scenarios', response_model=list[ScenarioInfo])
def scenarios() -> list[ScenarioInfo]:
    return [
        ScenarioInfo(
            name=name,
            label=s.label,
            strategy=s.strategy,
            it_scale=s.it_scale,
            outdoor_min_c=s.outdoor_min_c,
            capacity_factor=s.capacity_factor,
        )
        for name, s in simulation.SCENARIOS.items()
    ]


@app.get('/api/v1/results', response_model=ResultsResponse)
def results(
    scenario: str = Query(
        default='base',
        pattern='^[a-z0-9_]+$',
        description="'base'(Test 14일) | 'surge'(서버 급증 3일)",
    )
) -> ResultsResponse:
    metrics_path = RESULTS_DIR / f'metrics_{scenario}.json'
    series_path = RESULTS_DIR / f'timeseries_{scenario}.csv'
    if not metrics_path.exists() or not series_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"결과 파일 없음: '{scenario}'. scripts/simulate_mpc.py 를 먼저 실행하세요.",
        )
    meta = json.loads(metrics_path.read_text(encoding='utf-8'))
    ts = pd.read_csv(series_path)
    series = {
        name: {c: g[c].tolist() for c in g.columns if c != 'controller'}
        for name, g in ts.groupby('controller', sort=False)
    }
    return ResultsResponse(
        start=meta['start'],
        hours=meta['hours'],
        scenario=meta['scenario'],
        controllers=meta['controllers'],
        timeseries=series,
    )
