"""twin-service Pydantic 스키마."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SimulatePoint(BaseModel):
    """한 시점 조건. it_power_kw 와 cpu_utilization 중 하나는 있어야 한다."""

    outdoor_temp_c: float = Field(ge=-40, le=60)
    it_power_kw: float | None = Field(default=None, gt=0)
    cpu_utilization: float | None = Field(default=None, ge=0, le=1)
    supply_temp_c: float | None = Field(default=None, description='생략 시 기준 설정 온도 (원본 데이터와 같음)')

    @model_validator(mode='after')
    def _need_load(self) -> 'SimulatePoint':
        if self.it_power_kw is None and self.cpu_utilization is None:
            raise ValueError('it_power_kw 또는 cpu_utilization 중 하나가 필요합니다')
        return self


class SimulateRequest(BaseModel):
    points: list[SimulatePoint] = Field(min_length=1, max_length=50_000)


class SimulateResult(BaseModel):
    it_power_kw: float              # 실제 IT 전력 (팬 증가분 포함)
    server_fan_extra_kw: float
    server_heat_kw: float
    free_cooling_fraction: float
    chiller_cop: float
    cooling_mode: str
    cooling_power_kw: float
    facility_power_kw: float        # IT + 냉각 + 기타
    pue: float
    ride_through_min: float         # 냉각 정지 시 위험 온도까지 걸리는 시간
    xfactor: float                  # 상대 고장률 (20℃ = 1.0)


class SimulateResponse(BaseModel):
    """정상 상태 계산 결과 (제거열 = 발열)."""

    results: list[SimulateResult]
    unit: str = 'kW'


class RolloutRequest(BaseModel):
    """설정 온도 계획 평가 (MPC 용). 세 배열은 길이가 같아야 한다."""

    it_power_kw: list[float] = Field(min_length=1, max_length=2016, description='구간별 IT 부하 (예측값)')
    outdoor_temp_c: list[float] = Field(min_length=1, max_length=2016)
    supply_temp_c: list[float] = Field(min_length=1, max_length=2016, description='구간별 공급 공기 설정 온도 계획')
    initial_room_temp_c: float | None = Field(default=None, ge=0, le=60,
                                              description='생략 시 첫 설정 온도에서 시작')
    dt_hours: float = Field(default=1.0, gt=0, le=24, description='구간 길이(시간)')
    capacity_factor: float = Field(default=1.0, ge=0, le=2,
                                   description='냉각 용량 배율 (0.5 = 칠러 일부 고장)')

    @model_validator(mode='after')
    def _same_length(self) -> 'RolloutRequest':
        n = {len(self.it_power_kw), len(self.outdoor_temp_c), len(self.supply_temp_c)}
        if len(n) != 1:
            raise ValueError('it_power_kw / outdoor_temp_c / supply_temp_c 길이가 같아야 합니다')
        return self


class RolloutBatchRequest(BaseModel):
    """설정 온도 계획 여러 개를 한 번에 평가 (MPC 후보 비교용)."""

    it_power_kw: list[float] = Field(min_length=1, max_length=2016)
    outdoor_temp_c: list[float] = Field(min_length=1, max_length=2016)
    plans: list[list[float]] = Field(min_length=1, max_length=1024, description='계획별 구간 설정 온도')
    initial_room_temp_c: float = Field(ge=0, le=60)
    dt_hours: float = Field(default=1.0, gt=0, le=24)
    capacity_factor: float = Field(default=1.0, ge=0, le=2)

    @model_validator(mode='after')
    def _same_length(self) -> 'RolloutBatchRequest':
        n = len(self.it_power_kw)
        if len(self.outdoor_temp_c) != n or any(len(plan) != n for plan in self.plans):
            raise ValueError('it_power_kw / outdoor_temp_c / 각 plan 길이가 같아야 합니다')
        return self


class RolloutBatchResponse(BaseModel):
    """계획별 합계·지표 (plans 순서대로)."""

    total_it_energy_kwh: list[float]
    total_fan_extra_energy_kwh: list[float]
    total_cooling_energy_kwh: list[float]
    total_facility_energy_kwh: list[float]
    average_pue: list[float]
    max_room_temp_c: list[float]
    min_room_temp_c: list[float]
    min_ride_through_min: list[float]
    average_xfactor: list[float]
    violation_steps: list[int]
    capacity_limited_steps: list[int]


class RolloutResponse(BaseModel):
    room_temp_c: list[float]
    it_power_kw: list[float]
    server_fan_extra_kw: list[float]
    heat_removed_kw: list[float]
    cooling_power_kw: list[float]
    facility_power_kw: list[float]
    pue: list[float]
    free_cooling_fraction: list[float]
    cooling_mode: list[str]
    ride_through_min: list[float]
    total_it_energy_kwh: float
    total_fan_extra_energy_kwh: float
    total_cooling_energy_kwh: float
    total_facility_energy_kwh: float
    average_pue: float
    max_room_temp_c: float
    min_room_temp_c: float
    min_ride_through_min: float
    average_xfactor: float
    violation_steps: int
    capacity_limited_steps: int
    warnings: list[str] = Field(default_factory=list)
