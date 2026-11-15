"""twin-service Pydantic 스키마."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class TelemetryResponse(BaseModel):
    """가상 데이터센터의 시간별 계측값 (1시간 평균). 대시보드·MPC 시뮬레이션의 입력."""

    timestamps: list[datetime]
    it_power_kw: list[float]  # 실제 IT 부하
    outdoor_temp_c: list[float]  # 외기온도
    data_start: datetime  # 계측 데이터 전체 범위
    data_end: datetime
    unit: str = 'kW'


class TariffResponse(BaseModel):
    """시간대별 전기 단가 (한전 일반용전력(을) 고압A 선택Ⅲ). 시각은 그 1시간 구간의 시작."""

    name: str
    source: str
    effective_date: str
    timestamps: list[datetime]
    season: list[str]  # 'summer' | 'spring_fall' | 'winter'
    period: list[str]  # 'off_peak'(경부하) | 'mid'(중간부하) | 'peak'(최대부하)
    price_krw_per_kwh: list[float]
    basic_charge_krw_per_kw: float  # 참고용 (계약전력 고정비, 비용 비교에서는 제외)


class EsgRequest(BaseModel):
    """시뮬레이션 결과 시계열 → ESG 지표. 배열 길이는 모두 같아야 한다."""

    it_power_kw: list[float] = Field(min_length=1, max_length=50_000)
    cooling_power_kw: list[float] = Field(min_length=1, max_length=50_000)
    cooling_mode: list[str] = Field(
        min_length=1, max_length=50_000, description="'free_cooling' | 'hybrid' | 'chiller'"
    )
    heat_removed_kw: list[float] | None = Field(
        default=None, description='생략 시 IT 전력과 같다고 본다'
    )
    price_krw_per_kwh: list[float] | None = Field(
        default=None, description='구간별 전기 단가. 주면 전기요금도 계산'
    )
    dt_hours: float = Field(default=1.0, gt=0, le=24)


class EsgResponse(BaseModel):
    it_energy_mwh: float
    cooling_energy_mwh: float
    total_energy_mwh: float  # IT + 냉각 + 기타 손실
    average_pue: float
    carbon_tco2: float
    cue_tco2_per_mwh: float  # 탄소 배출 / IT 전력량
    water_usage_l: float
    wue_l_per_kwh: float  # 물 사용량 / IT 전력량
    water_by_mode_l: dict[str, float]
    cooling_energy_by_mode_mwh: dict[str, float]
    hours_by_mode: dict[str, float]
    carbon_factor_tco2_per_mwh: float
    energy_cost_krw: float | None = None  # 전체 전력 × 단가 (단가를 줬을 때만)
    cost_krw_per_mwh: float | None = None  # 평균 단가 (원/MWh)


class SimulatePoint(BaseModel):
    """한 시점 조건. it_power_kw 와 cpu_utilization 중 하나는 있어야 한다."""

    outdoor_temp_c: float = Field(ge=-40, le=60)
    it_power_kw: float | None = Field(default=None, gt=0)
    cpu_utilization: float | None = Field(default=None, ge=0, le=1)
    supply_temp_c: float | None = Field(
        default=None, description='생략 시 기준 설정 온도 (원본 데이터와 같음)'
    )

    @model_validator(mode='after')
    def _need_load(self) -> 'SimulatePoint':
        if self.it_power_kw is None and self.cpu_utilization is None:
            raise ValueError('it_power_kw 또는 cpu_utilization 중 하나가 필요합니다')
        return self


class SimulateRequest(BaseModel):
    points: list[SimulatePoint] = Field(min_length=1, max_length=50_000)


class SimulateResult(BaseModel):
    it_power_kw: float  # 실제 IT 전력 (팬 증가분 포함)
    server_fan_extra_kw: float
    server_heat_kw: float
    air_mass_flow_kg_s: float  # CRAH 공기 유량 ṁ
    return_air_temp_c: float  # 환기 온도 = 서버실 온도 + 발열 / (ṁ c_p)
    cooling_load_kw: float  # 냉각 부하 Q = ṁ c_p (환기 온도 - 공급 온도)
    free_cooling_fraction: float
    chiller_cop: float
    cooling_mode: str
    cooling_power_kw: float
    facility_power_kw: float  # IT + 냉각 + 기타
    pue: float
    ride_through_min: float  # 냉각 정지 시 위험 온도까지 걸리는 시간
    xfactor: float  # 상대 고장률 (20℃ = 1.0)


class SimulateResponse(BaseModel):
    """정상 상태 계산 결과 (제거열 = 발열)."""

    results: list[SimulateResult]
    unit: str = 'kW'


class RolloutRequest(BaseModel):
    """설정 온도 계획 평가 (MPC 용). 세 배열은 길이가 같아야 한다."""

    it_power_kw: list[float] = Field(
        min_length=1, max_length=2016, description='구간별 IT 부하 (예측값)'
    )
    outdoor_temp_c: list[float] = Field(min_length=1, max_length=2016)
    supply_temp_c: list[float] = Field(
        min_length=1, max_length=2016, description='구간별 공급 공기 설정 온도 계획'
    )
    initial_room_temp_c: float | None = Field(
        default=None, ge=0, le=60, description='생략 시 첫 설정 온도에서 시작'
    )
    dt_hours: float = Field(default=1.0, gt=0, le=24, description='구간 길이(시간)')
    capacity_factor: float = Field(
        default=1.0, ge=0, le=2, description='냉각 용량 배율 (0.5 = 칠러 일부 고장)'
    )
    price_krw_per_kwh: list[float] | None = Field(
        default=None, description='구간별 전기 단가 (GET /api/v1/tariff). ' '주면 전기요금도 계산'
    )
    tes_kw: list[float] | None = Field(
        default=None,
        description='구간별 축열조 출력 명령 (+충전 / -방전, 냉열 kW). '
        '생략하면 축열조를 쓰지 않는다',
    )
    initial_tes_kwh: float | None = Field(
        default=None, ge=0, description='시작 저장량. 생략 시 용량 × tes_initial_soc'
    )

    @model_validator(mode='after')
    def _same_length(self) -> 'RolloutRequest':
        n = {len(self.it_power_kw), len(self.outdoor_temp_c), len(self.supply_temp_c)}
        for extra in (self.price_krw_per_kwh, self.tes_kw):
            if extra is not None:
                n.add(len(extra))
        if len(n) != 1:
            raise ValueError(
                'it_power_kw / outdoor_temp_c / supply_temp_c / price_krw_per_kwh / tes_kw '
                '길이가 같아야 합니다'
            )
        return self


class RolloutBatchRequest(BaseModel):
    """설정 온도 계획 여러 개를 한 번에 평가 (MPC 후보 비교용)."""

    it_power_kw: list[float] = Field(min_length=1, max_length=2016)
    outdoor_temp_c: list[float] = Field(min_length=1, max_length=2016)
    plans: list[list[float]] = Field(
        min_length=1, max_length=1024, description='계획별 구간 설정 온도'
    )
    initial_room_temp_c: float = Field(ge=0, le=60)
    dt_hours: float = Field(default=1.0, gt=0, le=24)
    capacity_factor: float = Field(default=1.0, ge=0, le=2)
    price_krw_per_kwh: list[float] | None = Field(
        default=None, description='구간별 전기 단가. 주면 요금 합계도 계산'
    )
    tes_plans: list[list[float]] | None = Field(
        default=None, description='계획별 축열조 출력 (plans 와 같은 개수·길이)'
    )
    initial_tes_kwh: float | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def _same_length(self) -> 'RolloutBatchRequest':
        n = len(self.it_power_kw)
        if len(self.outdoor_temp_c) != n or any(len(plan) != n for plan in self.plans):
            raise ValueError('it_power_kw / outdoor_temp_c / 각 plan 길이가 같아야 합니다')
        if self.price_krw_per_kwh is not None and len(self.price_krw_per_kwh) != n:
            raise ValueError('price_krw_per_kwh 길이가 it_power_kw 와 같아야 합니다')
        if self.tes_plans is not None and (
            len(self.tes_plans) != len(self.plans) or any(len(t) != n for t in self.tes_plans)
        ):
            raise ValueError('tes_plans 는 plans 와 개수·길이가 같아야 합니다')
        return self


class RolloutBatchResponse(BaseModel):
    """계획별 합계·지표 (plans 순서대로)."""

    total_energy_cost_krw: list[float] | None = None  # 단가를 줬을 때만
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
    final_tes_soc_kwh: list[float]


class RolloutResponse(BaseModel):
    room_temp_c: list[float]
    return_air_temp_c: list[float]
    it_power_kw: list[float]
    server_fan_extra_kw: list[float]
    heat_removed_kw: list[float]
    cooling_power_kw: list[float]
    facility_power_kw: list[float]
    pue: list[float]
    free_cooling_fraction: list[float]
    cooling_mode: list[str]
    ride_through_min: list[float]
    tes_kw: list[float]  # 축열조 실제 출력 (+충전 / -방전)
    tes_soc_kwh: list[float]  # 구간 끝 저장량
    energy_cost_krw: list[float] | None = None  # 단가를 줬을 때만
    total_energy_cost_krw: float | None = None
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
    final_tes_soc_kwh: float
    warnings: list[str] = Field(default_factory=list)
