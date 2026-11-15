"""Rule-based 컨트롤러: 고정 목표 온도 + PID 로 공급 공기 설정 온도를 정한다.

MPC 와 비교하는 기준(베이스라인)이다. 미래 예측을 쓰지 않고 지금 측정한 서버실 온도만 본다.

  오차        e(t) = 목표 온도 - 측정 서버실 온도         (양수 = 목표보다 차갑다 → 설정 온도를 올린다)
  설정 온도   u(t) = 목표 + Kp·e + Ki·∫e dt - Kd·d(측정값)/dt
              ※ 미분은 오차가 아니라 측정값에 건다 (목표를 바꿀 때 출력이 튀지 않게)
  출력 한계   설정 가능 범위(supply_temp_min_c ~ max_c)로 자른다.
  적분 누적 방지(anti-windup)  출력이 한계에 걸린 채 오차가 같은 방향이면 적분하지 않는다.

외기 온도 기반 Free Cooling 전환(외기 15℃ 이하 외기 냉각 100%, 18℃ 이상 칠러)은 트윈의 냉각 설비
모델이 한다. 이 컨트롤러는 그 위에서 서버실 온도만 맞춘다.

축열조 시간표 운전 (tes_schedule_kw): 요금 시간대만 보고 경부하에 최대 충전, 최대부하에 최대 방전.
미래 부하·외기를 보지 않는 Rule-based 축열조 운전으로, MPC 의 축열조 운전과 비교하는 기준이다.

목표 온도·계수는 config/settings.yaml 의 rule_based 항목.
"""

from __future__ import annotations

from dataclasses import dataclass

from services.common.settings import section

_RB = section('rule_based')
_DC = section('datacenter')


@dataclass(frozen=True)
class PIDConfig:
    target_room_temp_c: float = _RB['target_room_temp_c']
    kp: float = _RB['kp']
    ki_per_h: float = _RB['ki_per_h']
    kd_h: float = _RB['kd_h']
    output_min_c: float = _DC['supply_temp_min_c']
    output_max_c: float = _DC['supply_temp_max_c']
    dt_hours: float = _RB['dt_hours']

    def __post_init__(self) -> None:
        if self.output_min_c >= self.output_max_c:
            raise ValueError('output_min_c 는 output_max_c 보다 작아야 합니다')
        if not self.output_min_c <= self.target_room_temp_c <= self.output_max_c:
            raise ValueError(f'목표 온도 {self.target_room_temp_c}℃ 가 설정 가능 범위 밖입니다')
        if self.dt_hours <= 0:
            raise ValueError('dt_hours 는 0보다 커야 합니다')


class PIDController:
    """서버실 온도를 목표에 맞추는 PID. update() 를 제어 주기(dt_hours)마다 한 번 부른다."""

    def __init__(self, config: PIDConfig = PIDConfig()) -> None:
        self.config = config
        self.integral_c = 0.0
        self._prev_measured: float | None = None

    def reset(self) -> None:
        self.integral_c = 0.0
        self._prev_measured = None

    def update(self, measured_room_c: float) -> float:
        """측정한 서버실 온도 → 이번 주기에 적용할 공급 공기 설정 온도(℃)."""
        c = self.config
        error = c.target_room_temp_c - measured_room_c
        slope = (
            0.0
            if self._prev_measured is None
            else (measured_room_c - self._prev_measured) / c.dt_hours
        )
        self._prev_measured = measured_room_c

        candidate = self.integral_c + c.ki_per_h * error * c.dt_hours
        raw = c.target_room_temp_c + c.kp * error + candidate - c.kd_h * slope
        output = min(max(raw, c.output_min_c), c.output_max_c)
        saturated_same_way = (raw > c.output_max_c and error > 0) or (
            raw < c.output_min_c and error < 0
        )
        if not saturated_same_way:
            self.integral_c = candidate
        return output


def tes_schedule_kw(
    tariff_period: str,
    max_charge_kw: float = _DC['tes_max_charge_kw'],
    max_discharge_kw: float = _DC['tes_max_discharge_kw'],
) -> float:
    """요금 시간대별 축열조 출력 명령: 경부하 최대 충전(+), 최대부하 최대 방전(-), 중간부하 0."""
    return {'off_peak': max_charge_kw, 'peak': -max_discharge_kw}.get(tariff_period, 0.0)
