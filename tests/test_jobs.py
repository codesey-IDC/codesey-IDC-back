"""백그라운드 시뮬레이션 작업 테스트: 즉시 응답, 진행률, 결과 = 동기 API, 같은 요청 재사용, 실패 처리."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from services.forecast import main as forecast_svc
from services.mpc import main as mpc_svc
from services.mpc.controller import MPCConfig, MPCController
from services.mpc.jobs import SimulationJobs
from services.twin.main import app as twin_app

needs_model = pytest.mark.skipif(forecast_svc._predictor is None, reason='예측 모델 없음')
Progress = Callable[[int, int], None]
REQ = {'start': '2026-06-04T00:00:00', 'hours': 4, 'controller': 'rule_pid'}


@pytest.fixture()
def client() -> TestClient:
    mpc_svc.set_controller(
        MPCController(
            TestClient(twin_app),
            TestClient(forecast_svc.app),
            MPCConfig(de_maxiter=2, de_popsize=2),
        )
    )
    return TestClient(mpc_svc.app)


def _wait(client: TestClient, job_id: str, timeout_s: float = 60.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = client.get(f'/api/v1/simulate/jobs/{job_id}').json()
        if job['status'] in ('done', 'failed'):
            return job
        time.sleep(0.05)
    raise AssertionError('작업이 끝나지 않았습니다')


# ---------------------------------------------------------------- 작업 큐 단위


def test_job_queue_runs_and_reports_progress() -> None:
    gate = threading.Event()

    def runner(params: dict, progress: Progress) -> dict:
        progress(1, params['hours'])
        gate.wait(5)
        progress(params['hours'], params['hours'])
        return {'ok': params['hours']}

    jobs = SimulationJobs(runner, max_workers=1)
    job = jobs.submit({'hours': 3})
    time.sleep(0.1)
    assert job.status == 'running' and job.progress_hours == 1
    assert jobs.submit({'hours': 3}) is job  # 같은 요청은 같은 작업
    gate.set()
    for _ in range(100):
        if job.status == 'done':
            break
        time.sleep(0.02)
    assert job.result == {'ok': 3} and job.progress_hours == 3
    assert jobs.find_done({'hours': 3}) is job and jobs.get(job.job_id) is job


def test_failed_job_is_retried_on_resubmit() -> None:
    calls = []

    def runner(params: dict, progress: Progress) -> dict:
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('첫 시도 실패')
        return {}

    jobs = SimulationJobs(runner, max_workers=1)
    first = jobs.submit({'hours': 1})
    for _ in range(100):
        if first.status == 'failed':
            break
        time.sleep(0.02)
    assert first.status == 'failed' and 'ValueError' in first.error
    second = jobs.submit({'hours': 1})
    assert second is not first


def test_old_finished_jobs_evicted() -> None:
    jobs = SimulationJobs(lambda params, progress: {}, max_workers=1, max_jobs=3)
    submitted = [jobs.submit({'hours': i}) for i in range(1, 6)]
    time.sleep(0.3)
    jobs.submit({'hours': 99})
    assert jobs.get(submitted[0].job_id) is None


# ---------------------------------------------------------------- API


@needs_model
def test_submit_returns_immediately_and_matches_sync(client: TestClient) -> None:
    t = time.perf_counter()
    r = client.post('/api/v1/simulate/jobs', json=REQ)
    assert r.status_code == 202 and time.perf_counter() - t < 1.0
    job = _wait(client, r.json()['job_id'])
    assert job['status'] == 'done' and job['progress_hours'] == job['total_hours'] == 4
    sync = client.post('/api/v1/simulate', json=REQ).json()
    assert job['result']['summary'] == sync['summary']


@needs_model
def test_same_request_reuses_job(client: TestClient) -> None:
    req = REQ | {'hours': 3}
    a = client.post('/api/v1/simulate/jobs', json=req).json()
    b = client.post('/api/v1/simulate/jobs', json=req).json()
    assert a['job_id'] == b['job_id']


@needs_model
def test_mpc_job(client: TestClient) -> None:
    job = _wait(
        client,
        client.post('/api/v1/simulate/jobs', json=REQ | {'controller': 'mpc', 'hours': 2}).json()[
            'job_id'
        ],
    )
    assert job['status'] == 'done' and set(job['result']['summary']['mode_hours']) == {'mpc'}


def test_bad_request_rejected_before_queue(client: TestClient) -> None:
    assert (
        client.post('/api/v1/simulate/jobs', json=REQ | {'scenario': 'meteor'}).status_code == 422
    )
    assert client.get('/api/v1/simulate/jobs/nope').status_code == 404


@needs_model
def test_out_of_data_job_fails_with_message(client: TestClient) -> None:
    job = _wait(
        client,
        client.post('/api/v1/simulate/jobs', json=REQ | {'start': '2026-03-16T00:00:00'}).json()[
            'job_id'
        ],
    )
    assert job['status'] == 'failed' and '데이터 범위' in job['error']
