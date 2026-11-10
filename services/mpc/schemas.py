"""mpc-service Pydantic 스키마."""
from __future__ import annotations

from datetime import datetime

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
