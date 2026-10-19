"""forecast-service Pydantic 스키마."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HistoryPoint(BaseModel):
    timestamp: datetime
    it_power_kw: float


class ForecastRequest(BaseModel):
    """예측 요청.

    history 를 생략하면 서버가 가진 기본 데이터(data/synthetic_idc_90d.parquet)의
    마지막 시점을 예측 시점으로 쓴다. 넘길 때는 5분 또는 1시간 간격으로
    최소 168시간(5분 기준 2,016행) 이상이어야 한다.
    """

    horizon_hours: int = Field(default=24, ge=1, le=168, description='예측 기간(시간)')
    target: Literal['it_load', 'cooling_load'] = Field(
        default='it_load',
        description='냉각 부하는 데이터상 IT 부하와 동일하므로 같은 값을 돌려준다')
    model: Literal['lgbm'] = Field(default='lgbm')
    history: list[HistoryPoint] | None = Field(
        default=None, description='최근 이력 (timestamp, it_power_kw). 생략 시 서버 기본 데이터 사용')


class ForecastResponse(BaseModel):
    """예측 결과 + 90% 예측구간 (CQR 보정 적용)."""

    target: str
    model: str                    # 'lgbm-24h' | 'lgbm-168h'
    horizon_hours: int
    origin_time: datetime         # 예측 시점 t
    timestamps: list[datetime]    # t+1 ... t+horizon
    predictions: list[float]      # q50 (kW)
    pi_lower: list[float]         # q05 - offset (kW)
    pi_upper: list[float]         # q95 + offset (kW)
    unit: str = 'kW'
    in_training_range: bool = True
    warnings: list[str] = Field(default_factory=list)
