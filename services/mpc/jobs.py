"""오래 걸리는 폐루프 시뮬레이션을 백그라운드 작업으로 돌린다 (API 응답 10초 이내 유지).

MPC 시뮬레이션은 1시간마다 MPC 계산(약 0.4초)을 반복해 24시간에 약 10초, 72시간에 약 30초가 걸린다.
요청은 작업만 등록하고 바로 응답하며, 호출 쪽은 작업 상태를 다시 물어 진행률과 결과를 받는다.

  submit(params)  같은 요청이면 기존 작업을 돌려준다 (끝났으면 결과 재사용)
  get(job_id)     상태: queued → running → done | failed, 진행 시간 수
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

Runner = Callable[[dict, Callable[[int, int], None]], dict]


@dataclass
class Job:
    job_id: str
    key: str
    params: dict
    total_hours: int
    status: str = 'queued'  # queued | running | done | failed
    progress_hours: int = 0
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def elapsed_s(self) -> float:
        return (self.finished_at or time.time()) - self.created_at


class SimulationJobs:
    """요청 내용이 같으면 같은 작업으로 묶는 작은 작업 큐 (프로세스 메모리, 재시작하면 비워진다)."""

    def __init__(self, runner: Runner, max_workers: int = 2, max_jobs: int = 64) -> None:
        self._runner = runner
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='simulation')
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._by_key: dict[str, str] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    @staticmethod
    def key_of(params: dict) -> str:
        return json.dumps(params, sort_keys=True, default=str)

    def submit(self, params: dict) -> Job:
        key = self.key_of(params)
        with self._lock:
            existing = self._jobs.get(self._by_key.get(key, ''))
            if existing is not None and existing.status != 'failed':
                return existing
            job = Job(
                job_id=uuid.uuid4().hex[:12],
                key=key,
                params=params,
                total_hours=int(params['hours']),
            )
            self._jobs[job.job_id] = job
            self._by_key[key] = job.job_id
            self._evict()
        self._pool.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def find_done(self, params: dict) -> Job | None:
        """같은 요청으로 끝난 작업이 있으면 그 작업 (동기 API 가 결과를 재사용할 때)."""
        with self._lock:
            job = self._jobs.get(self._by_key.get(self.key_of(params), ''))
            return job if job is not None and job.status == 'done' else None

    def _run(self, job: Job) -> None:
        job.status = 'running'

        def progress(done: int, total: int) -> None:
            job.progress_hours, job.total_hours = done, total

        try:
            job.result = self._runner(job.params, progress)
            job.status = 'done'
        except Exception as e:  # noqa: BLE001  작업 실패는 상태로 알린다 (서비스는 계속 동작)
            job.error = f'{type(e).__name__}: {e}'
            job.status = 'failed'
        finally:
            job.finished_at = time.time()

    def _evict(self) -> None:
        """작업 수가 한도를 넘으면 끝난 작업부터 오래된 순으로 지운다."""
        while len(self._jobs) > self._max_jobs:
            old_id = next(
                (i for i, j in self._jobs.items() if j.status in ('done', 'failed')), None
            )
            if old_id is None:
                break
            old = self._jobs.pop(old_id)
            if self._by_key.get(old.key) == old_id:
                del self._by_key[old.key]
