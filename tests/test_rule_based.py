"""Rule-based(PID) 컨트롤러 테스트: PID 계산, 출력 한계·적분 누적 방지, 트윈과 붙였을 때 온도 유지."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.common.settings import section
from services.forecast import main as forecast_svc
from services.mpc import simulation
from services.mpc.rule_based import PIDConfig, PIDController
from services.twin import physics as P
from services.twin.main import app as twin_app

needs_model = pytest.mark.skipif(forecast_svc._predictor is None, reason='예측 모델 없음')


def _closed_loop(
    start_room_c: float, hours: int = 8, pid: PIDController | None = None
) -> np.ndarray:
    """PID 가 매시간 설정 온도를 정하고 트윈을 1시간 진행 (부하 3,000kW, 외기 25℃)."""
    pid = pid or PIDController()
    room, out = start_room_c, []
    for _ in range(hours):
        setpoint = pid.update(room)
        room = P.rollout([3000.0], [25.0], [setpoint], initial_room_temp_c=room).room_temp_c[0]
        out.append(room)
    return np.array(out)


def test_config_from_settings() -> None:
    rb = section('rule_based')
    c = PIDConfig()
    assert (c.target_room_temp_c, c.kp, c.ki_per_h, c.kd_h) == (
        rb['target_room_temp_c'],
        rb['kp'],
        rb['ki_per_h'],
        rb['kd_h'],
    )
    assert (c.output_min_c, c.output_max_c) == (
        section('datacenter')['supply_temp_min_c'],
        section('datacenter')['supply_temp_max_c'],
    )


def test_at_target_outputs_target() -> None:
    pid = PIDController()
    assert [pid.update(20.0) for _ in range(5)] == [20.0] * 5
    assert pid.integral_c == 0.0


def test_pid_terms() -> None:
    """u = 목표 + Kp·e + Ki·Σe·dt - Kd·(측정 변화율)."""
    pid = PIDController(PIDConfig(kp=0.5, ki_per_h=0.2, kd_h=1.0))
    assert pid.update(22.0) == pytest.approx(20 + 0.5 * -2 + 0.2 * -2)  # 첫 호출: 미분 없음
    assert pid.update(21.0) == pytest.approx(20 + 0.5 * -1 + 0.2 * (-2 - 1) - 1.0 * (21 - 22))


def test_output_clamped_and_anti_windup() -> None:
    """출력이 하한에 걸린 동안 적분이 쌓이지 않아, 온도가 돌아오면 바로 정상 출력으로 돌아간다."""
    pid = PIDController(PIDConfig(kp=2.0, ki_per_h=1.0))
    outputs = [pid.update(30.0) for _ in range(10)]
    assert outputs == [16.0] * 10
    assert pid.integral_c == pytest.approx(0.0)
    assert pid.update(20.0) == pytest.approx(20.0)


def test_reset() -> None:
    pid = PIDController()
    pid.update(25.0)
    pid.reset()
    assert pid.integral_c == 0.0 and pid.update(20.0) == 20.0


def test_invalid_config() -> None:
    with pytest.raises(ValueError):
        PIDConfig(target_room_temp_c=40.0)
    with pytest.raises(ValueError):
        PIDConfig(dt_hours=0)


def test_recovers_faster_than_fixed_setpoint() -> None:
    """서버실 25℃ 에서 시작: PID 는 1시간 만에 목표 ±0.5℃, 고정 20℃ 는 아직 1℃ 이상 높다."""
    pid = _closed_loop(25.0)
    fixed = P.rollout([3000.0] * 8, [25.0] * 8, [20.0] * 8, initial_room_temp_c=25.0).room_temp_c
    assert abs(pid[0] - 20.0) < 0.5 and fixed[0] - 20.0 > 1.0
    assert np.abs(pid[2:] - 20.0).max() < 0.5


@pytest.mark.parametrize('start_room_c', [17.0, 25.0])
def test_keeps_room_in_allowed_range(start_room_c: float) -> None:
    rooms = _closed_loop(start_room_c, hours=12)
    assert ((rooms >= 18.0) & (rooms <= 27.0)).all()


@needs_model
@pytest.mark.parametrize('scenario', ['normal', 'server_surge', 'chiller_failure', 'heatwave'])
def test_simulation_rule_pid_maintains_temperature(scenario: str) -> None:
    """과제 기준: Rule-based 온도 범위(18~27℃) 유지율 95% 이상, 위기 시나리오 27℃ 이하."""
    r = simulation.run_closed_loop(
        TestClient(twin_app),
        TestClient(forecast_svc.app),
        pd.Timestamp('2026-06-03 00:00'),
        24,
        'rule_pid',
        scenario,
        initial_room_temp_c=24.0,
    )
    s = r['summary']
    assert s['temp_maintenance_pct'] >= 95.0
    assert s['max_room_temp_c'] <= 27.0
    assert set(s['mode_hours']) == {'rule_pid'}
