"""가상 데이터센터 물리 모델 (디지털 트윈 핵심).

에너지 흐름
  CPU 사용률 → IT 전력 → 발열(=IT 전력) → 서버실 공기 → 냉각 설비(외기 냉각/칠러) → 냉각 전력

1단계: 정적 계산 (simulate_static)
  data/synthetic_idc_90d.parquet 를 만든 열역학 모델을 데이터에서 역으로 복원한 공식.
  25,920행 전부에서 it_power_kw / cooling_power_kw / free_cooling_fraction /
  cooling_mode / pue 가 오차 없이 일치한다 (tests/test_twin.py).

2단계: 동적 계산 (rollout) — MPC 용
  데이터에 없는 것들을 "가정"으로 추가했다. 값은 TwinParams 에 모여 있다.
    - 제어 변수: 공급 공기 설정 온도 supply_temp_c
        설정 온도를 기준(20℃)보다 Δ 올리면 외기와의 온도차가 Δ 줄어든 것과 같다고 본다.
        → 유효 외기온도 = 외기 - Δ 로 외기 냉각 비율과 칠러 COP 를 계산.
          (외기 냉각 가능 범위가 넓어지고 칠러 효율이 좋아짐)
    - 상태: 서버실(서버 흡기) 온도 room_temp_c (집중 열용량 모델)
        열용량 × dT/dt = 발열 - 냉각 부하
    - 냉각 부하 (공기측 열수지, CRAH)
        환기 온도  T_return = 서버실 온도 + 발열 / (ṁ c_p)     (서버를 지나며 데워진 공기)
        냉각 부하  Q = ṁ × c_p × (T_return - T_supply)          (CRAH 가 공급 설정 온도까지 식힌다)
        → 서버실이 설정 온도보다 더우면 그만큼 더 빼내고, 차가우면 덜 빼낸다. 냉각 용량 한계로 자른다.
        정상 상태(서버실 = 설정 온도)에서는 Q = 발열.
    - 설정 온도를 올릴 때의 단점
        ① 서버 팬 전력: 서버실이 fan_threshold_c 를 넘으면 IT 전력이 (T - 기준)² 비율로 증가.
           늘어난 전력도 열이 되어 냉각 부하에 더해진다.
        ② 비상 여유 시간: 냉각이 멈췄을 때 critical_temp_c 에 닿기까지 걸리는 시간.
           min_ride_through_min 미만이면 위반.
        ③ 상대 고장률 (ASHRAE x-factor, 20℃ = 1.0): 지표로만 보고, 비용에는 넣지 않는다.
  설정 온도 = 기준, 서버실 온도 = 설정 온도에서 시작하면 단점 효과가 모두 0 이고
  제거열 = 발열 이 되어 1단계와 결과가 똑같다.

  ※ 팬 전력은 IT 전력에 들어가므로, 설정 온도를 올리면 PUE 는 좋아져도
    전체 전력은 늘어날 수 있다. 최적화 목표는 PUE 가 아니라 전체 전력(facility)으로 잡아야 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
from numpy.typing import ArrayLike

from services.common.settings import section

SUBSTEP_HOURS = 5 / 60  # 적분 간격 5분 (원본 데이터 해상도)

# ASHRAE TC9.9 (2011) 흡기 온도별 상대 고장률 x-factor 평균값 (근사, 20℃ = 1.00)
XFACTOR_TEMPS_C = (15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0)
XFACTOR_VALUES = (0.72, 0.87, 1.00, 1.13, 1.24, 1.34, 1.42, 1.48, 1.55)


_DC = section(
    'datacenter'
)  # 기본값은 config/settings.yaml 의 datacenter 항목 (값의 의미·출처도 거기 있다)


@dataclass(frozen=True)
class TwinParams:
    # ---- 데이터에서 복원 (synthetic_idc_90d.parquet 와 정확히 일치) ----
    num_servers: int = _DC['num_servers']
    server_idle_power_w: float = _DC['server_idle_power_w']  # SPECpower P_idle
    server_max_power_w: float = _DC['server_max_power_w']  # SPECpower P_max
    free_cooling_full_below_c: float = _DC['free_cooling_full_below_c']
    free_cooling_none_above_c: float = _DC['free_cooling_none_above_c']
    free_cooling_cop: float = _DC['free_cooling_cop']
    chiller_cop_ref: float = _DC['chiller_cop_ref']
    chiller_cop_ref_temp_c: float = _DC['chiller_cop_ref_temp_c']
    chiller_cop_slope: float = _DC['chiller_cop_slope']
    chiller_cop_min: float = _DC['chiller_cop_min']
    other_overhead_ratio: float = _DC['other_overhead_ratio']

    # ---- 가정 (2단계, MPC 용. 데이터에 없음) ----
    supply_temp_ref_c: float = _DC['supply_temp_ref_c']
    supply_temp_min_c: float = _DC['supply_temp_min_c']
    supply_temp_max_c: float = _DC['supply_temp_max_c']
    room_temp_min_c: float = _DC['room_temp_min_c']
    room_temp_max_c: float = _DC['room_temp_max_c']
    thermal_capacity_kwh_per_c: float = _DC['thermal_capacity_kwh_per_c']
    air_mass_flow_kg_s: float = _DC['air_mass_flow_kg_s']  # CRAH 공기 유량 ṁ
    air_cp_kj_per_kg_k: float = _DC['air_cp_kj_per_kg_k']  # 공기 정압비열 c_p
    chiller_units: int = _DC['chiller_units']
    cooling_capacity_kw: float = _DC['cooling_capacity_kw']

    # ---- 가정: 냉수 축열조 (TES) ----
    tes_capacity_kwh: float = _DC['tes_capacity_kwh']
    tes_max_charge_kw: float = _DC['tes_max_charge_kw']
    tes_max_discharge_kw: float = _DC['tes_max_discharge_kw']
    tes_charge_efficiency: float = _DC['tes_charge_efficiency']
    tes_loss_per_h: float = _DC['tes_loss_per_h']
    tes_initial_soc: float = _DC['tes_initial_soc']

    # ---- 가정: 설정 온도를 올릴 때의 단점 ----
    fan_threshold_c: float = _DC['fan_threshold_c']
    fan_power_coef_per_c2: float = _DC['fan_power_coef_per_c2']
    critical_temp_c: float = _DC['critical_temp_c']
    min_ride_through_min: float = _DC['min_ride_through_min']

    @property
    def idle_power_kw(self) -> float:
        """CPU 0% 일 때 IT 전력 = 서버 수 × P_idle."""
        return self.num_servers * self.server_idle_power_w / 1000.0

    @property
    def cpu_power_range_kw(self) -> float:
        """CPU 0% → 100% 증가분 = 서버 수 × (P_max - P_idle)."""
        return self.num_servers * (self.server_max_power_w - self.server_idle_power_w) / 1000.0

    @property
    def air_heat_rate_kw_per_c(self) -> float:
        """ṁ × c_p (kW/℃): 공기 온도 1℃ 차이당 실어 나르는 열량. kg/s × kJ/(kg·K) = kW/K."""
        return self.air_mass_flow_kg_s * self.air_cp_kj_per_kg_k

    def to_dict(self) -> dict:
        return asdict(self) | {
            'idle_power_kw': self.idle_power_kw,
            'cpu_power_range_kw': self.cpu_power_range_kw,
            'air_heat_rate_kw_per_c': self.air_heat_rate_kw_per_c,
        }


DEFAULT_PARAMS = TwinParams()


# ---------------------------------------------------------------- 공통 계산


def it_power_from_cpu(cpu_utilization: ArrayLike, p: TwinParams = DEFAULT_PARAMS) -> np.ndarray:
    """CPU 사용률(0~1) → IT 전력(kW). 서버 팬 증가분 제외 (기준 온도 기준)."""
    return p.idle_power_kw + p.cpu_power_range_kw * np.asarray(cpu_utilization, dtype=float)


def fan_extra_ratio(room_temp_c: ArrayLike, p: TwinParams = DEFAULT_PARAMS) -> np.ndarray:
    """서버실 온도에 따른 IT 전력 증가 비율 (팬·누설 전력)."""
    over = np.maximum(np.asarray(room_temp_c, dtype=float) - p.fan_threshold_c, 0.0)
    return p.fan_power_coef_per_c2 * over**2


def xfactor(room_temp_c: ArrayLike) -> np.ndarray:
    """ASHRAE 상대 고장률 (20℃ = 1.0). 표 범위 밖은 끝값."""
    return np.interp(np.asarray(room_temp_c, dtype=float), XFACTOR_TEMPS_C, XFACTOR_VALUES)


def ride_through_min(
    room_temp_c: ArrayLike, heat_kw: ArrayLike, p: TwinParams = DEFAULT_PARAMS
) -> np.ndarray:
    """냉각이 멈췄을 때 위험 온도까지 걸리는 시간(분)."""
    margin = np.maximum(p.critical_temp_c - np.asarray(room_temp_c, dtype=float), 0.0)
    return margin * p.thermal_capacity_kwh_per_c / np.asarray(heat_kw, dtype=float) * 60.0


def rack_delta_t(heat_kw: ArrayLike, p: TwinParams = DEFAULT_PARAMS) -> np.ndarray:
    """서버 앞뒤 공기 온도차 ΔT_rack = Q_IT / (ṁ × c_p)."""
    return np.asarray(heat_kw, dtype=float) / p.air_heat_rate_kw_per_c


def return_air_temp(
    room_temp_c: ArrayLike, heat_kw: ArrayLike, p: TwinParams = DEFAULT_PARAMS
) -> np.ndarray:
    """서버를 지나 CRAH 로 돌아가는 공기 온도 = 서버실(흡기) 온도 + ΔT_rack."""
    return np.asarray(room_temp_c, dtype=float) + rack_delta_t(heat_kw, p)


def cooling_load(
    return_temp_c: ArrayLike, supply_temp_c: ArrayLike, p: TwinParams = DEFAULT_PARAMS
) -> np.ndarray:
    """냉각 부하 Q = ṁ × c_p × (환기 온도 - 공급 공기 온도) (kW). CRAH 가 공기에서 빼내는 열량."""
    return p.air_heat_rate_kw_per_c * (
        np.asarray(return_temp_c, dtype=float) - np.asarray(supply_temp_c, dtype=float)
    )


def cooling_efficiency(
    outdoor_temp_c: ArrayLike,
    supply_temp_c: ArrayLike | None = None,
    p: TwinParams = DEFAULT_PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """외기 냉각 비율과 칠러 COP. supply_temp_c 가 기준보다 높으면 유효 외기온도가 내려간다."""
    t_out = np.asarray(outdoor_temp_c, dtype=float)
    delta = (
        0.0
        if supply_temp_c is None
        else np.asarray(supply_temp_c, dtype=float) - p.supply_temp_ref_c
    )
    t_eff = t_out - delta
    span = p.free_cooling_none_above_c - p.free_cooling_full_below_c
    fc = np.clip((p.free_cooling_none_above_c - t_eff) / span, 0.0, 1.0)
    cop = np.maximum(
        p.chiller_cop_ref - p.chiller_cop_slope * (t_eff - p.chiller_cop_ref_temp_c),
        p.chiller_cop_min,
    )
    return fc, cop


def cooling_power_per_kw(
    fc: ArrayLike, cop: ArrayLike, p: TwinParams = DEFAULT_PARAMS
) -> np.ndarray:
    """제거열 1kW 당 냉각 전력(kW)."""
    return fc / p.free_cooling_cop + (1.0 - fc) / cop


def cooling_mode(fc: ArrayLike) -> np.ndarray:
    fc = np.asarray(fc, dtype=float)
    return np.where(fc >= 1.0, 'free_cooling', np.where(fc <= 0.0, 'chiller', 'hybrid'))


def pue(it_kw: ArrayLike, cooling_kw: ArrayLike, p: TwinParams = DEFAULT_PARAMS) -> np.ndarray:
    it_kw = np.asarray(it_kw, dtype=float)
    return 1.0 + np.asarray(cooling_kw, dtype=float) / it_kw + p.other_overhead_ratio


def facility_power(
    it_kw: ArrayLike, cooling_kw: ArrayLike, p: TwinParams = DEFAULT_PARAMS
) -> np.ndarray:
    """전체 전력 = IT + 냉각 + 기타 손실."""
    it_kw = np.asarray(it_kw, dtype=float)
    return it_kw * (1.0 + p.other_overhead_ratio) + np.asarray(cooling_kw, dtype=float)


# ---------------------------------------------------------------- 1단계: 정적


def simulate_static(
    it_power_kw: ArrayLike,
    outdoor_temp_c: ArrayLike,
    supply_temp_c: ArrayLike | None = None,
    p: TwinParams = DEFAULT_PARAMS,
) -> dict[str, np.ndarray]:
    """정상 상태(서버실 = 설정 온도) 계산. 배열을 넣으면 배열로 돌려준다.

    it_power_kw 는 팬 증가분을 뺀 기본 IT 부하(예측 모듈이 내는 값)다.
    결과의 it_power_kw 는 팬 증가분을 더한 실제 IT 전력이다.
    냉각 부하는 Q = ṁ c_p (환기 온도 - 공급 온도) 로 계산하며, 정상 상태에서는 발열과 같다.
    """
    base = np.asarray(it_power_kw, dtype=float)
    room = (
        np.full_like(base, p.supply_temp_ref_c)
        if supply_temp_c is None
        else np.broadcast_to(np.asarray(supply_temp_c, dtype=float), base.shape)
    )
    fan = base * fan_extra_ratio(room, p)
    it = base + fan
    t_return = return_air_temp(room, it, p)
    load = cooling_load(t_return, room, p)
    fc, cop = cooling_efficiency(outdoor_temp_c, supply_temp_c, p)
    cooling = load * cooling_power_per_kw(fc, cop, p)
    return {
        'it_power_kw': it,
        'server_fan_extra_kw': fan,
        'server_heat_kw': it,
        'air_mass_flow_kg_s': np.full_like(base, p.air_mass_flow_kg_s),
        'return_air_temp_c': t_return,
        'cooling_load_kw': load,
        'free_cooling_fraction': fc,
        'chiller_cop': cop,
        'cooling_mode': cooling_mode(fc),
        'cooling_power_kw': cooling,
        'facility_power_kw': facility_power(it, cooling, p),
        'pue': pue(it, cooling, p),
        'ride_through_min': ride_through_min(room, it, p),
        'xfactor': xfactor(room),
    }


# ---------------------------------------------------------------- 2단계: 동적


@dataclass
class RolloutResult:
    room_temp_c: list[float]  # 각 구간 끝의 서버실(서버 흡기) 온도
    return_air_temp_c: list[float]  # 각 구간 끝의 환기 온도 (서버를 지난 공기)
    it_power_kw: list[float]  # 구간 평균 실제 IT 전력 (팬 증가분 포함)
    server_fan_extra_kw: list[float]  # 구간 평균 팬 증가분
    heat_removed_kw: list[float]  # 구간 평균 냉각 부하 Q = ṁ c_p ΔT (용량 한계 적용 후)
    cooling_power_kw: list[float]  # 구간 평균 냉각 전력
    facility_power_kw: list[float]  # 구간 평균 전체 전력
    pue: list[float]  # 구간 에너지 기준 PUE
    free_cooling_fraction: list[float]
    cooling_mode: list[str]
    ride_through_min: list[float]  # 각 구간 끝에서 냉각 정지 시 버티는 시간
    tes_kw: list[float]  # 구간 평균 축열조 실제 출력 (+충전 / -방전, 냉열 kW)
    tes_soc_kwh: list[float]  # 각 구간 끝의 축열조 저장량
    energy_cost_krw: list[float] | None = None  # 구간별 전기요금 (단가를 줬을 때만)
    total_energy_cost_krw: float | None = None  # 전기요금 합계 (단가를 줬을 때만)
    total_it_energy_kwh: float = 0.0
    total_fan_extra_energy_kwh: float = 0.0
    total_cooling_energy_kwh: float = 0.0
    total_facility_energy_kwh: float = 0.0  # 최적화 목표로 쓸 값
    average_pue: float = 0.0
    max_room_temp_c: float = 0.0
    min_room_temp_c: float = 0.0
    min_ride_through_min: float = 0.0
    average_xfactor: float = 0.0
    violation_steps: int = 0  # 온도 범위 또는 비상 여유 시간을 벗어난 구간 수
    capacity_limited_steps: int = 0  # 냉각 용량 한계에 걸린 구간 수
    final_tes_soc_kwh: float = 0.0  # 마지막 구간 끝 축열조 저장량
    warnings: list[str] = field(default_factory=list)


def _check_inputs(base: np.ndarray, t_out: np.ndarray, n_plan_steps: int, dt_hours: float) -> None:
    n = len(base)
    if not (len(t_out) == n_plan_steps == n) or n == 0:
        raise ValueError(
            f'it_power_kw / outdoor_temp_c / supply_temp_c 길이가 같아야 합니다: '
            f'{len(base)}, {len(t_out)}, {n_plan_steps}'
        )
    if dt_hours <= 0:
        raise ValueError('dt_hours 는 0보다 커야 합니다')
    if (base <= 0).any():
        raise ValueError('it_power_kw 는 0보다 커야 합니다')


def _integrate(
    base: np.ndarray,
    t_out: np.ndarray,
    t_set: np.ndarray,
    room0: np.ndarray,
    dt_hours: float,
    capacity_factor: float,
    p: TwinParams,
    tes: np.ndarray | None = None,
    soc0: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """설정 온도 계획 N개를 한 번에 적분한다. t_set: (N, n), 반환 배열은 (N, n) 또는 (N,).

    tes: 축열조 출력 계획 (N, n), +충전 / -방전 (냉열 kW). None 이면 축열조를 쓰지 않는다.
    soc0: 시작 저장량 (N,) kWh. None 이면 용량 × tes_initial_soc.
    """
    n_sub = max(1, math.ceil(dt_hours / SUBSTEP_HOURS - 1e-9))
    h = dt_hours / n_sub
    q_max = p.cooling_capacity_kw * capacity_factor
    fc, cop = cooling_efficiency(t_out[None, :], t_set, p)

    n_plans, n = t_set.shape
    tes_cmd = (
        np.zeros((n_plans, n))
        if tes is None
        else np.clip(tes, -p.tes_max_discharge_kw, p.tes_max_charge_kw)
    )
    soc = (
        np.full(n_plans, p.tes_capacity_kwh * p.tes_initial_soc)
        if soc0 is None
        else soc0.astype(float).copy()
    )
    room = room0.astype(float).copy()
    room_end, it_avg, q_avg, cooling, soc_end, tes_avg = (np.empty((n_plans, n)) for _ in range(6))
    limited = np.zeros((n_plans, n), dtype=bool)
    xf_sum = np.zeros(n_plans)
    for i in range(n):
        removed = np.zeros(n_plans)
        it_kwh = np.zeros(n_plans)
        cooling_kwh = np.zeros(n_plans)
        tes_kwh = np.zeros(n_plans)
        f, c_op = fc[:, i], cop[:, i]
        for _ in range(n_sub):
            heat = base[i] * (1.0 + fan_extra_ratio(room, p))
            # CRAH: 서버를 지난 공기(환기)를 공급 설정 온도까지 식힌다. Q = ṁ c_p (T_return - T_supply)
            q_raw = cooling_load(return_air_temp(room, heat, p), t_set[:, i], p)
            # 축열조 방전: 칠러가 맡을 부분((1 - 외기 냉각 비율) × Q)만, 저장량 안에서 대신한다
            discharge = np.minimum(
                np.minimum(np.maximum(-tes_cmd[:, i], 0.0), soc / h),
                (1.0 - f) * np.maximum(q_raw, 0.0),
            )
            limited[:, i] |= q_raw > q_max + discharge
            q = np.clip(q_raw, 0.0, q_max + discharge)
            discharge = np.minimum(discharge, (1.0 - f) * q)
            # 축열조 충전: 칠러 남는 용량과 빈 저장 공간 안에서
            headroom = np.maximum(q_max - (q - discharge), 0.0)
            charge = np.minimum(
                np.minimum(np.maximum(tes_cmd[:, i], 0.0), (p.tes_capacity_kwh - soc) / h),
                headroom,
            )
            # 냉각 전력 = 외기 냉각분 + 칠러분(방전만큼 줄고) + 충전분(효율 저하)
            power = (
                f * q / p.free_cooling_cop
                + ((1.0 - f) * q - discharge) / c_op
                + charge / (c_op * p.tes_charge_efficiency)
            )
            soc = np.clip(
                soc + (charge - discharge - p.tes_loss_per_h * soc) * h, 0.0, p.tes_capacity_kwh
            )
            xf_sum += xfactor(room) * h
            room = room + (heat - q) * h / p.thermal_capacity_kwh_per_c
            removed += q * h
            it_kwh += heat * h
            cooling_kwh += power * h
            tes_kwh += (charge - discharge) * h
        room_end[:, i] = room
        it_avg[:, i] = it_kwh / dt_hours
        q_avg[:, i] = removed / dt_hours
        cooling[:, i] = cooling_kwh / dt_hours
        soc_end[:, i] = soc
        tes_avg[:, i] = tes_kwh / dt_hours

    heat_end = base[None, :] * (1.0 + fan_extra_ratio(room_end, p))
    ride = ride_through_min(room_end, heat_end, p)
    bad_temp = (room_end < p.room_temp_min_c) | (room_end > p.room_temp_max_c)
    bad_ride = ride < p.min_ride_through_min
    return {
        'room_temp_c': room_end,
        'return_air_temp_c': return_air_temp(room_end, heat_end, p),
        'it_power_kw': it_avg,
        'server_fan_extra_kw': it_avg - base[None, :],
        'heat_removed_kw': q_avg,
        'cooling_power_kw': cooling,
        'facility_power_kw': facility_power(it_avg, cooling, p),
        'pue': pue(it_avg, cooling, p),
        'free_cooling_fraction': fc,
        'ride_through_min': ride,
        'tes_kw': tes_avg,
        'tes_soc_kwh': soc_end,
        'bad_temp': bad_temp,
        'bad_ride': bad_ride,
        'limited': limited,
        'average_xfactor': xf_sum / (n * dt_hours),
    }


def _check_prices(prices: ArrayLike | None, n: int) -> np.ndarray | None:
    if prices is None:
        return None
    prices = np.asarray(prices, dtype=float)
    if prices.shape != (n,):
        raise ValueError(f'price_krw_per_kwh 는 구간 수({n})만큼 있어야 합니다: {prices.shape}')
    if (prices < 0).any():
        raise ValueError('price_krw_per_kwh 는 0 이상이어야 합니다')
    return prices


def energy_cost(
    facility_power_kw: ArrayLike, prices_krw_per_kwh: ArrayLike, dt_hours: float
) -> np.ndarray:
    """구간별 전기요금(원) = 전체 전력(kW) × 시간(h) × 단가(원/kWh)."""
    return (
        np.asarray(facility_power_kw, dtype=float)
        * dt_hours
        * np.asarray(prices_krw_per_kwh, dtype=float)
    )


def _summaries(
    r: dict[str, np.ndarray], dt_hours: float, prices: np.ndarray | None = None
) -> dict:
    """_integrate 결과에서 계획별 합계·지표 (N,). prices 가 있으면 전기요금 합계도 (없으면 None)."""
    it_e = r['it_power_kw'].sum(axis=1) * dt_hours
    facility_e = r['facility_power_kw'].sum(axis=1) * dt_hours
    cost = (
        None
        if prices is None
        else energy_cost(r['facility_power_kw'], prices[None, :], dt_hours).sum(axis=1)
    )
    return {
        'total_energy_cost_krw': cost,
        'total_it_energy_kwh': it_e,
        'total_fan_extra_energy_kwh': r['server_fan_extra_kw'].sum(axis=1) * dt_hours,
        'total_cooling_energy_kwh': r['cooling_power_kw'].sum(axis=1) * dt_hours,
        'total_facility_energy_kwh': facility_e,
        'average_pue': facility_e / it_e,
        'max_room_temp_c': r['room_temp_c'].max(axis=1),
        'min_room_temp_c': r['room_temp_c'].min(axis=1),
        'min_ride_through_min': r['ride_through_min'].min(axis=1),
        'average_xfactor': r['average_xfactor'],
        'violation_steps': (r['bad_temp'] | r['bad_ride']).sum(axis=1),
        'capacity_limited_steps': r['limited'].sum(axis=1),
        'final_tes_soc_kwh': r['tes_soc_kwh'][:, -1],
    }


def _check_tes(
    tes_kw: ArrayLike | None, n: int, initial_tes_kwh: float | None, p: TwinParams
) -> tuple[np.ndarray | None, float]:
    """축열조 입력 검증. 반환: (출력 배열 또는 None, 시작 저장량)."""
    soc0 = (
        p.tes_capacity_kwh * p.tes_initial_soc
        if initial_tes_kwh is None
        else float(initial_tes_kwh)
    )
    if not 0.0 <= soc0 <= p.tes_capacity_kwh:
        raise ValueError(
            f'initial_tes_kwh 는 0 ~ {p.tes_capacity_kwh:.0f} 사이여야 합니다: {soc0}'
        )
    if tes_kw is None:
        return None, soc0
    tes = np.asarray(tes_kw, dtype=float)
    if tes.shape[-1] != n:
        raise ValueError(f'tes_kw 는 구간 수({n})만큼 있어야 합니다: {tes.shape}')
    return tes, soc0


def rollout(
    it_power_kw: ArrayLike,
    outdoor_temp_c: ArrayLike,
    supply_temp_c: ArrayLike,
    initial_room_temp_c: float | None = None,
    dt_hours: float = 1.0,
    capacity_factor: float = 1.0,
    price_krw_per_kwh: ArrayLike | None = None,
    tes_kw: ArrayLike | None = None,
    initial_tes_kwh: float | None = None,
    p: TwinParams = DEFAULT_PARAMS,
) -> RolloutResult:
    """시간 구간별 IT 부하·외기온도·설정 온도 계획을 받아 서버실 온도와 전력을 계산한다.

    MPC 는 이 함수로 "이 설정 계획대로 돌리면 어떻게 되나"를 평가한다.
    it_power_kw 는 팬 증가분을 뺀 기본 IT 부하(예측 모듈 출력)다.
    각 구간은 5분 이하 간격으로 나눠 적분한다.
    capacity_factor: 냉각 용량 배율 (0.5 = 칠러 일부 고장 시나리오).
    price_krw_per_kwh: 구간별 전기 단가. 주면 전기요금도 계산한다.
    tes_kw: 구간별 축열조 출력 명령 (+충전 / -방전, 냉열 kW). 생략하면 축열조를 쓰지 않는다.
    initial_tes_kwh: 시작 저장량. 생략하면 용량 × tes_initial_soc.
    """
    base = np.asarray(it_power_kw, dtype=float)
    t_out = np.asarray(outdoor_temp_c, dtype=float)
    t_set = np.asarray(supply_temp_c, dtype=float)
    _check_inputs(base, t_out, len(t_set), dt_hours)
    prices = _check_prices(price_krw_per_kwh, len(base))
    tes, soc0 = _check_tes(tes_kw, len(base), initial_tes_kwh, p)

    warnings: list[str] = []
    clipped = np.clip(t_set, p.supply_temp_min_c, p.supply_temp_max_c)
    if (clipped != t_set).any():
        warnings.append(
            f'supply_temp_clipped: 설정 온도를 {p.supply_temp_min_c}~{p.supply_temp_max_c}℃ 로 제한'
        )
        t_set = clipped

    room0 = t_set[0] if initial_room_temp_c is None else initial_room_temp_c
    r = _integrate(
        base,
        t_out,
        t_set[None, :],
        np.array([room0]),
        dt_hours,
        capacity_factor,
        p,
        tes=None if tes is None else tes[None, :],
        soc0=np.array([soc0]),
    )
    s = {k: (None if v is None else v[0]) for k, v in _summaries(r, dt_hours, prices).items()}
    fc = r['free_cooling_fraction'][0]
    out = RolloutResult(
        **{
            k: r[k][0].tolist()
            for k in (
                'room_temp_c',
                'return_air_temp_c',
                'it_power_kw',
                'server_fan_extra_kw',
                'heat_removed_kw',
                'cooling_power_kw',
                'facility_power_kw',
                'pue',
                'ride_through_min',
                'tes_kw',
                'tes_soc_kwh',
            )
        },
        free_cooling_fraction=fc.tolist(),
        cooling_mode=cooling_mode(fc).tolist(),
        energy_cost_krw=(
            None
            if prices is None
            else energy_cost(r['facility_power_kw'][0], prices, dt_hours).tolist()
        ),
        **{
            k: (v if v is None else int(v) if k.endswith('_steps') else float(v))
            for k, v in s.items()
        },
    )

    temp_violations, ride_violations = int(r['bad_temp'].sum()), int(r['bad_ride'].sum())
    if temp_violations:
        warnings.append(
            f'room_temp_violation: {temp_violations}개 구간에서 서버실 온도가 '
            f'{p.room_temp_min_c}~{p.room_temp_max_c}℃ 밖'
        )
    if ride_violations:
        warnings.append(
            f'ride_through_violation: {ride_violations}개 구간에서 냉각 정지 시 '
            f'{p.critical_temp_c}℃ 도달까지 {p.min_ride_through_min:.0f}분 미만'
        )
    if out.capacity_limited_steps:
        warnings.append(
            f'cooling_capacity_limited: {out.capacity_limited_steps}개 구간에서 냉각 용량 '
            f'{p.cooling_capacity_kw * capacity_factor:.0f}kW 한계 도달'
        )
    out.warnings = warnings
    return out


def rollout_batch(
    it_power_kw: ArrayLike,
    outdoor_temp_c: ArrayLike,
    plans: ArrayLike,
    initial_room_temp_c: float,
    dt_hours: float = 1.0,
    capacity_factor: float = 1.0,
    price_krw_per_kwh: ArrayLike | None = None,
    tes_plans: ArrayLike | None = None,
    initial_tes_kwh: float | None = None,
    p: TwinParams = DEFAULT_PARAMS,
) -> dict:
    """설정 온도 계획 여러 개(plans: (N, n))를 한 번에 평가해 계획별 합계·지표를 돌려준다.

    MPC 가 후보 계획 수십 개를 API 호출 한 번으로 비교할 때 쓴다.
    계획은 설정 가능 범위로 잘라서 계산한다. 결과는 rollout() 의 합계와 같다.
    price_krw_per_kwh 를 주면 total_energy_cost_krw 도 계산한다 (없으면 None).
    tes_plans: 계획별 축열조 출력 (N, n). plans 와 개수가 같아야 한다.
    """
    base = np.asarray(it_power_kw, dtype=float)
    t_out = np.asarray(outdoor_temp_c, dtype=float)
    t_set = np.atleast_2d(np.asarray(plans, dtype=float))
    _check_inputs(base, t_out, t_set.shape[1], dt_hours)
    prices = _check_prices(price_krw_per_kwh, len(base))
    tes, soc0 = _check_tes(tes_plans, len(base), initial_tes_kwh, p)
    if tes is not None:
        tes = np.atleast_2d(tes)
        if tes.shape != t_set.shape:
            raise ValueError(f'tes_plans 모양 {tes.shape} 이 plans {t_set.shape} 와 같아야 합니다')
    t_set = np.clip(t_set, p.supply_temp_min_c, p.supply_temp_max_c)
    room0 = np.full(len(t_set), float(initial_room_temp_c))
    r = _integrate(
        base,
        t_out,
        t_set,
        room0,
        dt_hours,
        capacity_factor,
        p,
        tes=tes,
        soc0=np.full(len(t_set), soc0),
    )
    return _summaries(r, dt_hours, prices)
