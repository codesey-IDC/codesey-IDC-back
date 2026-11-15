"""ESG 지표 계산: 탄소 배출, 물 사용량(WUE), 냉각 방식별 비교.

입력은 시뮬레이션 결과 시계열(구간별 IT 전력·냉각 전력·냉각 모드)이다.

  탄소 배출  = 전체 전력량(MWh) × 배출계수 (기본 0.459 tCO2/MWh, 국내 전력 배출계수)
  CUE        = 탄소 배출 / IT 전력량  (tCO2/MWh)
  물 사용량  = Σ 제거열(kWh) × 냉각 방식별 물 원단위 (L/kWh)
  WUE        = 물 사용량 / IT 전력량  (L/kWh)

※ 물 원단위는 데이터에 없는 가정값이다 (칠러의 냉각탑 증발 > 혼합 > 외기 냉각).
배출계수와 물 원단위는 config/settings.yaml 의 esg 항목에서 읽는다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from services.common.settings import section

_ESG = section('esg')
CARBON_FACTOR_TCO2_PER_MWH: float = float(_ESG['carbon_factor_tco2_per_mwh'])
WATER_INTENSITY_L_PER_KWH: dict[str, float] = {
    k: float(v) for k, v in _ESG['water_intensity_l_per_kwh'].items()
}
MODES = tuple(WATER_INTENSITY_L_PER_KWH)


def summarize(
    it_power_kw: ArrayLike,
    cooling_power_kw: ArrayLike,
    cooling_mode: ArrayLike,
    dt_hours: float = 1.0,
    heat_removed_kw: ArrayLike | None = None,
    other_overhead_ratio: float = 0.05,
    carbon_factor: float = CARBON_FACTOR_TCO2_PER_MWH,
    price_krw_per_kwh: ArrayLike | None = None,
) -> dict:
    """구간별 시계열 → ESG 지표. heat_removed_kw 생략 시 제거열 = IT 전력으로 본다.

    price_krw_per_kwh(구간별 단가)를 주면 전기요금(비용)도 계산한다 → 비용·에너지·탄소 비교.
    """
    it = np.asarray(it_power_kw, dtype=float)
    cooling = np.asarray(cooling_power_kw, dtype=float)
    mode = np.asarray(cooling_mode, dtype=str)
    heat = it if heat_removed_kw is None else np.asarray(heat_removed_kw, dtype=float)
    price = None if price_krw_per_kwh is None else np.asarray(price_krw_per_kwh, dtype=float)
    if not (len(it) == len(cooling) == len(mode) == len(heat)) or len(it) == 0:
        raise ValueError('it_power_kw / cooling_power_kw / cooling_mode 길이가 같아야 합니다')
    if price is not None and len(price) != len(it):
        raise ValueError('price_krw_per_kwh 길이가 it_power_kw 와 같아야 합니다')
    unknown = sorted(set(mode) - set(MODES))
    if unknown:
        raise ValueError(f'알 수 없는 cooling_mode: {unknown} (가능: {list(MODES)})')

    it_mwh = float(it.sum() * dt_hours / 1000)
    cooling_mwh = float(cooling.sum() * dt_hours / 1000)
    total_mwh = it_mwh * (1.0 + other_overhead_ratio) + cooling_mwh
    water_by_mode = {
        m: float(heat[mode == m].sum() * dt_hours * WATER_INTENSITY_L_PER_KWH[m]) for m in MODES
    }
    water = sum(water_by_mode.values())
    carbon = total_mwh * carbon_factor
    facility_kw = it * (1.0 + other_overhead_ratio) + cooling
    cost = None if price is None else float((facility_kw * dt_hours * price).sum())
    return {
        'energy_cost_krw': cost,
        'cost_krw_per_mwh': None if cost is None else cost / total_mwh,
        'it_energy_mwh': it_mwh,
        'cooling_energy_mwh': cooling_mwh,
        'total_energy_mwh': total_mwh,
        'average_pue': total_mwh / it_mwh,
        'carbon_tco2': carbon,
        'cue_tco2_per_mwh': carbon / it_mwh,
        'water_usage_l': water,
        'wue_l_per_kwh': water / (it_mwh * 1000),
        'water_by_mode_l': water_by_mode,
        'cooling_energy_by_mode_mwh': {
            m: float(cooling[mode == m].sum() * dt_hours / 1000) for m in MODES
        },
        'hours_by_mode': {m: float((mode == m).sum() * dt_hours) for m in MODES},
        'carbon_factor_tco2_per_mwh': carbon_factor,
    }
