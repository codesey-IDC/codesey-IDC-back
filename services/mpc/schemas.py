"""mpc-service Pydantic 스키마."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HistoryPoint(BaseModel):
    timestamp: datetime
    it_power_kw: float


class PlanRequest(BaseModel):
    """지금 상태 + 최근 이력 + 외기 예보를 받아 설정 온도를 정한다."""

    room_temp_c: float = Field(ge=0, le=60, description='현재 서버실 온도')
    history: list[HistoryPoint] = Field(
        min_length=168, description='최근 IT 부하 (1시간 또는 5분, 최소 168시간)'
    )
    outdoor_forecast_c: list[float] = Field(
        min_length=24, max_length=24, description='앞으로 24시간 외기온도 예보'
    )
    capacity_factor: float = Field(
        default=1.0, ge=0, le=2, description='냉각 용량 배율 (0.5 = 칠러 일부 고장)'
    )
    tes_soc_kwh: float | None = Field(
        default=None, ge=0, description='현재 축열조 저장량. 생략 시 시작 저장량'
    )
    current_setpoint_c: float | None = Field(
        default=None, ge=0, le=60, description='지금 적용 중인 설정 온도 (시간당 변화율 제약 기준)'
    )


class ControlRequest(BaseModel):
    """현재 상태를 받아 최적 냉각 제어 계획을 만든다 (POST /api/v1/control/optimize, /control/mpc).

    history·outdoor_forecast_c 를 생략하면 MPC 서비스가 twin-service /api/v1/telemetry 에서
    timestamp 이전 192시간 IT 부하와 이후 24시간 외기온도를 받아 쓴다 (외기는 실측을 예보로 가정).
    """

    timestamp: datetime | None = Field(
        default=None, description='현재 시각 (계획 시작). history 나 외기 예보를 ' '생략하면 필요'
    )
    room_temp_c: float = Field(ge=0, le=60, description='현재 서버실(서버 흡기) 온도')
    current_setpoint_c: float | None = Field(
        default=None,
        ge=0,
        le=60,
        description='지금 적용 중인 공급 공기 설정 온도 (변화율 제약 기준)',
    )
    tes_soc_kwh: float | None = Field(
        default=None, ge=0, description='현재 축열조 저장량 (kWh). 생략 시 시작 저장량'
    )
    capacity_factor: float = Field(
        default=1.0, ge=0, le=2, description='냉각 용량 배율 (0.75 = 칠러 4대 중 1대 고장)'
    )
    history: list[HistoryPoint] | None = Field(
        default=None, min_length=168, description='최근 IT 부하 (1시간 또는 5분, 최소 168시간)'
    )
    outdoor_forecast_c: list[float] | None = Field(
        default=None, min_length=24, max_length=24, description='앞으로 24시간 외기온도 예보'
    )

    @model_validator(mode='after')
    def _need_timestamp(self) -> 'ControlRequest':
        if (self.history is None or self.outdoor_forecast_c is None) and self.timestamp is None:
            raise ValueError('history 나 outdoor_forecast_c 를 생략하면 timestamp 가 필요합니다')
        return self


class ControlSchedule(BaseModel):
    """시간대별 최적 냉각 설정값과 예상 상태 (POST /api/v1/control/optimize)."""

    mode: str  # 'mpc' | 'mpc_naive' | 'fallback'
    method: str = 'mpc'  # 최적화 방식
    objective: str = 'cost'
    timestamps: list[datetime]
    supply_temp_c: list[float]  # 시간대별 최적 공급 공기 설정 온도
    tes_kw: list[float]  # 시간대별 축열조 출력 (+충전 / -방전, 냉열 kW)
    cooling_mode: list[str]  # 예상 냉각 모드 'free_cooling' | 'hybrid' | 'chiller'
    expected_pue: list[float]  # 시간대별 예상 PUE
    expected_room_temp_c: list[float]
    expected_tes_soc_kwh: list[float]
    price_krw_per_kwh: list[float]
    expected_energy_cost_krw: float | None = None
    baseline_energy_cost_krw: float | None = None  # 같은 조건에서 기준 설정(20℃) 고정 시
    expected_cost_saving_pct: float | None = None
    expected_average_pue: float | None = None
    computation_time_s: float
    inputs_from: str  # 'request' | 'twin_telemetry' (이력·외기 출처)
    warnings: list[str] = Field(default_factory=list)


class PlanResponse(BaseModel):
    mode: str  # 'mpc' | 'mpc_naive'(예측 불신, 최근 최대 부하 가정) | 'fallback'
    setpoint_c: float  # 지금 적용할 공급 공기 설정 온도
    plan_c: list[float]  # 24시간 계획 (첫 값만 적용, 다음 시간에 다시 계산)
    tes_kw: float = 0.0  # 지금 적용할 축열조 출력 (+충전 / -방전, 냉열 kW)
    tes_plan_kw: list[float] = Field(default_factory=list)  # 24시간 축열조 계획
    predicted_tes_soc_kwh: list[float] = Field(default_factory=list)  # 예상 저장량
    objective: str = 'cost'  # 'cost'(전기요금) | 'energy'(전력량)
    timestamps: list[datetime] = Field(default_factory=list)
    forecast_it_kw: list[float] = Field(default_factory=list)
    price_krw_per_kwh: list[float] = Field(default_factory=list)  # 시간대별 전기 단가
    predicted_room_temp_c: list[float] = Field(default_factory=list)
    predicted_facility_kw: list[float] = Field(default_factory=list)
    predicted_pue: list[float] = Field(default_factory=list)  # 시간대별 예상 PUE
    predicted_cooling_mode: list[str] = Field(default_factory=list)
    expected_facility_energy_kwh: float = 0.0
    baseline_facility_energy_kwh: float = 0.0  # 같은 예측으로 기준 설정 고정 시
    expected_saving_pct: float = 0.0  # 전력량 기준 절감률
    expected_energy_cost_krw: float | None = None
    baseline_energy_cost_krw: float | None = None
    expected_cost_saving_pct: float | None = None  # 전기요금 기준 절감률
    computation_time_s: float = 0.0  # 예측 + 최적화 + 결과 계산 시간
    warnings: list[str] = Field(default_factory=list)


class SimulateRequest(BaseModel):
    """폐루프 시뮬레이션: start 부터 hours 시간을 1시간씩 진행한다."""

    start: datetime = Field(description='시작 시각. 이전 192시간의 계측 이력이 있어야 한다')
    hours: int = Field(default=24, ge=1, le=168)
    controller: Literal['fixed_20', 'rule_pid', 'rule_tes', 'reactive', 'mpc'] = 'mpc'
    scenario: str = Field(
        default='normal',
        description='GET /api/v1/scenarios 의 name (config/settings.yaml 의 scenarios)',
    )
    initial_room_temp_c: float | None = Field(
        default=None, ge=0, le=60, description='시작 서버실 온도. 생략 시 기준 설정 온도(20℃)'
    )


class SimulateSummary(BaseModel):
    hours: int
    energy_cost_krw: float  # 전력량요금 합계 (시간대별 단가, 기본요금·부가세 제외)
    facility_energy_kwh: float
    cooling_energy_kwh: float
    fan_extra_energy_kwh: float
    it_energy_kwh: float
    average_pue: float
    average_xfactor: float
    average_setpoint_c: float
    max_room_temp_c: float
    min_room_temp_c: float
    temp_maintenance_pct: float  # 서버실 18~27℃ 유지율
    temp_ok: bool  # 최고 온도가 허용 상한 이하
    min_ride_through_min: float
    violation_hours: int  # 온도 범위 또는 비상 여유 시간 위반
    capacity_limited_hours: int
    mode_hours: dict[str, int]
    final_tes_soc_kwh: float  # 마지막 축열조 저장량
    diverged: bool  # 서버실 온도 발산으로 중단됨


class SimulateResponse(BaseModel):
    start: datetime
    hours: int
    controller: str
    scenario: str
    scenario_label: str
    strategy: str
    summary: SimulateSummary
    timeseries: dict[str, list]  # 컬럼 이름 → 시간별 값 (timestamp, setpoint_c, room_temp_c, ...)
    warnings: list[str] = Field(default_factory=list)


class SimulationJob(BaseModel):
    """백그라운드 시뮬레이션 작업 (POST /api/v1/simulate/jobs, GET /api/v1/simulate/jobs/{job_id})."""

    job_id: str
    status: Literal['queued', 'running', 'done', 'failed']
    progress_hours: int  # 지금까지 진행한 시간 수
    total_hours: int
    elapsed_s: float
    request: SimulateRequest
    result: SimulateResponse | None = None  # status = 'done' 일 때
    error: str | None = None  # status = 'failed' 일 때


class ScenarioInfo(BaseModel):
    name: str
    label: str
    strategy: str
    it_scale: float
    outdoor_min_c: float | None
    capacity_factor: float


class ResultsResponse(BaseModel):
    """scripts/simulate_mpc.py 로 미리 계산해 둔 장기 비교 실험 (output/mpc/)."""

    start: datetime
    hours: int
    scenario: str
    controllers: dict[str, dict]  # 제어기 → 요약 지표
    timeseries: dict[str, dict[str, list]]  # 제어기 → 컬럼 → 시간별 값
