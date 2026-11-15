"""공용 픽스처: 세 서비스를 실제 HTTP 서버로 띄운다 (대시보드·실행 스크립트 테스트용).

포트 8001~8003 은 config/settings.yaml 의 서비스 주소와 같다. 이미 다른 프로그램이 쓰고 있으면
(예: start.bat 으로 서비스를 띄워 둔 경우) 이 픽스처를 쓰는 테스트는 건너뛴다.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn

from services.common.settings import service_url
from services.forecast import main as forecast_svc
from services.mpc import main as mpc_svc
from services.mpc.controller import MPCConfig, MPCController
from services.twin.main import app as twin_app

# 테스트에서는 MPC 최적화를 가볍게 (결정 1회 약 0.05초)
FAST_MPC = MPCConfig(de_maxiter=2, de_popsize=2)
PORTS = {'twin': 8001, 'forecast': 8002, 'mpc': 8003}


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) != 0


def _start(app: object, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError(f'{port} 포트 서버가 시작되지 않았습니다')
        time.sleep(0.05)
    return server, thread


@pytest.fixture(scope='session')
def live_services() -> Iterator[dict[str, str]]:
    """twin·forecast·mpc 를 127.0.0.1:8001~8003 에 띄우고 주소를 돌려준다."""
    if not all(_port_free(p) for p in PORTS.values()):
        pytest.skip(
            '8001~8003 포트를 다른 프로그램이 쓰고 있습니다 (start.bat 으로 띄운 서비스 등)'
        )
    if forecast_svc._predictor is None:
        pytest.skip('예측 모델 없음')
    urls = {name: service_url(name) for name in PORTS}
    # 다른 테스트가 바꿔 둔 장애 흉내 클라이언트를 실제 HTTP 클라이언트로 되돌린다
    forecast_svc.set_twin_client(httpx.Client(base_url=urls['twin'], timeout=30))
    mpc_svc.set_controller(
        MPCController(
            httpx.Client(base_url=urls['twin'], timeout=30),
            httpx.Client(base_url=urls['forecast'], timeout=30),
            FAST_MPC,
        )
    )
    servers = [
        _start(twin_app, PORTS['twin']),
        _start(forecast_svc.app, PORTS['forecast']),
        _start(mpc_svc.app, PORTS['mpc']),
    ]
    yield urls
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=10)
