"""mpc-service Pydantic 스키마."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HistoryPoint(BaseModel):
    timestamp: datetime
    it_power_kw: float


class PlanRequest(BaseModel):
    """지금 상태 + 최근 이력 + 외기 예보를 받아 설정 온도를 정한다."""

    room_temp_c: float = Field(ge=0, le=60, description='현재 서버실 온도')
    history: list[HistoryPoint] = Field(min_length=168, description='최근 IT 부하 (1시간 또는 5분, 최소 168시간)')
    outdoor_forecast_c: list[float] = Field(min_length=24, max_length=24, description='앞으로 24시간 외기온도 예보')
    capacity_factor: float = Field(default=1.0, ge=0, le=2, description='냉각 용량 배율 (0.5 = 칠러 일부 고장)')


class PlanResponse(BaseModel):
    mode: str                               # 'mpc' | 'mpc_naive'(예측 불신, 최근 최대 부하 가정) | 'fallback'
    setpoint_c: float                       # 지금 적용할 공급 공기 설정 온도
    plan_c: list[float]                     # 24시간 계획 (첫 값만 적용, 다음 시간에 다시 계산)
    timestamps: list[datetime] = Field(default_factory=list)
    forecast_it_kw: list[float] = Field(default_factory=list)
    predicted_room_temp_c: list[float] = Field(default_factory=list)
    predicted_facility_kw: list[float] = Field(default_factory=list)
    expected_facility_energy_kwh: float = 0.0
    baseline_facility_energy_kwh: float = 0.0
    expected_saving_pct: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class SimulateRequest(BaseModel):
    """폐루프 시뮬레이션: start 부터 hours 시간을 1시간씩 진행한다."""

    start: datetime = Field(description='시작 시각. 이전 192시간의 계측 이력이 있어야 한다')
    hours: int = Field(default=24, ge=1, le=168)
    controller: Literal['fixed_20', 'reactive', 'mpc'] = 'mpc'
    scenario: Literal['normal', 'server_surge', 'chiller_failure', 'heatwave'] = 'normal'


class SimulateSummary(BaseModel):
    hours: int
    facility_energy_kwh: float
    cooling_energy_kwh: float
    fan_extra_energy_kwh: float
    it_energy_kwh: float
    average_pue: float
    average_xfactor: float
    average_setpoint_c: float
    max_room_temp_c: float
    min_room_temp_c: float
    temp_maintenance_pct: float         # 서버실 18~27℃ 유지율
    temp_ok: bool                       # 최고 온도가 허용 상한 이하
    min_ride_through_min: float
    violation_hours: int                # 온도 범위 또는 비상 여유 시간 위반
    capacity_limited_hours: int
    mode_hours: dict[str, int]
    diverged: bool                      # 서버실 온도 발산으로 중단됨


class SimulateResponse(BaseModel):
    start: datetime
    hours: int
    controller: str
    scenario: str
    scenario_label: str
    strategy: str
    summary: SimulateSummary
    timeseries: dict[str, list]         # 컬럼 이름 → 시간별 값 (timestamp, setpoint_c, room_temp_c, ...)
    warnings: list[str] = Field(default_factory=list)


class ScenarioInfo(BaseModel):
    name: str
    label: str
    strategy: str
    it_scale: float
    outdoor_min_c: float | None
    capacity_factor: float


class ResultsResponse(BaseModel):
    """simulate_mpc.py 로 미리 계산해 둔 장기 비교 실험 (output/mpc/)."""

    start: datetime
    hours: int
    scenario: str
    controllers: dict[str, dict]                    # 제어기 → 요약 지표
    timeseries: dict[str, dict[str, list]]          # 제어기 → 컬럼 → 시간별 값
