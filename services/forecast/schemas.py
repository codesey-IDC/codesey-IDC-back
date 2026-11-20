"""forecast-service Pydantic 스키마."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HistoryPoint(BaseModel):
    timestamp: datetime
    it_power_kw: float


class ForecastRequest(BaseModel):
    """예측 요청.

    history 를 생략하면 서버가 가진 기본 데이터(config/settings.yaml 의 paths.data)의
    마지막 시점을 예측 시점으로 쓴다. 넘길 때는 5분 또는 1시간 간격으로
    최소 168시간(5분 기준 2,016행) 이상이어야 한다.

    target
      it_load        IT 부하 (kW)
      cooling_load   냉각 부하(열량). 데이터상 IT 부하와 같아 같은 값을 돌려준다
      cooling_power  냉각 전력(전기, kW) = IT 부하 예측 + 외기 → 트윈 물리 모델 (기준 설정 온도 운전)
                     외기 예보(outdoor_forecast_c)를 생략하면 서버 데이터의 외기 실측값을 쓴다
    """

    horizon_hours: int = Field(default=24, ge=1, le=168, description='예측 기간(시간)')
    target: Literal['it_load', 'cooling_load', 'cooling_power'] = Field(default='it_load')
    model: Literal['lgbm', 'ridge'] | None = Field(
        default=None,
        description='IT 부하 예측 모델. 생략 시 config/settings.yaml 의 forecast.model (기본 ridge)',
    )
    history: list[HistoryPoint] | None = Field(
        default=None,
        description='최근 이력 (timestamp, it_power_kw). 생략 시 서버 기본 데이터 사용',
    )
    outdoor_forecast_c: list[float] | None = Field(
        default=None,
        description="target='cooling_power' 일 때 시간별 외기온도 예보 (horizon_hours 개)",
    )

    @model_validator(mode='after')
    def _outdoor_length(self) -> 'ForecastRequest':
        if (
            self.outdoor_forecast_c is not None
            and len(self.outdoor_forecast_c) != self.horizon_hours
        ):
            raise ValueError(
                f'outdoor_forecast_c 는 horizon_hours({self.horizon_hours})개여야 합니다'
            )
        return self


class InputCheckResult(BaseModel):
    """입력 이력 품질 검사 한 항목 (config/settings.yaml 의 forecast.input_quality)."""

    rule: str  # 'stuck'(값 고착) | 'jump'(1시간 급변) | 'level_shift'(1주 전 대비 평균 이동)
    label: str
    ok: bool | None  # None = 이력이 짧아 검사하지 못함
    measured: str
    limit: str


class ForecastResponse(BaseModel):
    """예측 결과 + 90% 예측구간 (CQR 보정 적용).

    target='cooling_power' 면 predictions/pi_* 는 냉각 전력(kW)이고, 아래 항목이 추가된다.
    """

    target: str
    model: str  # '{ridge|lgbm}-{24h|168h}' (cooling_power 는 뒤에 '+twin')
    horizon_hours: int
    origin_time: datetime  # 예측 시점 t
    timestamps: list[datetime]  # t+1 ... t+horizon
    predictions: list[float]  # q50 (kW)
    pi_lower: list[float]  # q05 - offset (kW)
    pi_upper: list[float]  # q95 + offset (kW)
    unit: str = 'kW'
    in_training_range: bool = True
    # 학습 범위 안에서 생기는 입력 이상(고착·급변·평균 이동)이 없으면 true. false 면 MPC 가 예측을 믿지 않는다
    input_quality_ok: bool = True
    input_checks: list[InputCheckResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # ---- target='cooling_power' 일 때만
    it_predictions: list[float] | None = None  # IT 부하 예측 (q50)
    outdoor_temp_c: list[float] | None = None  # 사용한 외기온도
    cooling_mode: list[str] | None = None  # 예상 냉각 모드 (외기 15℃ 이하 외기 냉각)
    free_cooling_fraction: list[float] | None = None
    pue: list[float] | None = None
    outdoor_source: str | None = None  # 'request' | 'data'
