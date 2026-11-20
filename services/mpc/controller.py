"""MPC 제어기: 예측(forecast-service) → 후보 계획 평가(twin-service) → 설정 온도 결정.

과제 제약(모듈 간 직접 함수 호출 금지)에 따라 다른 모듈은 REST API 로만 부른다.
twin / forecast 는 httpx.Client 호환 객체다. 실서비스는 base_url 을 준 httpx.Client,
테스트·시뮬레이션은 fastapi.testclient.TestClient 를 넘긴다 (둘 다 HTTP 요청을 거친다).

MPC 5요소
  모델        twin-service /api/v1/rollout/batch
  예측 구간   horizon_hours (기본 24시간)
  목적 함수   objective=cost: 전기요금(원) = Σ 전체 전력 × 시간대별 단가 (twin-service /api/v1/tariff)
              objective=energy: 전체 전력량(kWh)
              + 위반 벌점 + 고장률 벌점(작게, 같은 비용이면 낮은 온도 선호)
              + 설정 변경 벌점(1℃ 바꿀 때마다 조금, 비용이 비슷할 때 들쭉날쭉한 계획 방지)
              벌점은 kWh 로 정의하고, cost 목적이면 구간 평균 단가를 곱해 원으로 바꾼다.
              ※ PUE 가 아니라 전체 전력·요금: 팬 전력은 IT 로 잡혀서 PUE 만 보면 과하게 올린다
  제약 조건   서버실 18~27℃, 냉각 정지 시 15분 이상 버팀 → 큰 벌점 (트윈이 위반 구간 수를 센다)
              냉각 용량 → 트윈 물리 한계 (칠러·축열조 용량을 넘는 냉각은 일어나지 않는다)
              제어 변화율 → 설정 온도 시간당 ±max_setpoint_rate_c_per_h ℃ 이내.
                            모든 후보 계획을 이 범위로 고친 뒤 평가하므로 결과 계획은 항상 지킨다
  후퇴 구간   계획 중 첫 1시간 설정만 적용, 다음 시간에 다시 계산

최적화: scipy.optimize.differential_evolution (과제 허용 라이브러리)
  결정 변수: 설정 온도(블록) [+ 축열조 출력(블록)]. 블록은 앞쪽 1시간, 뒤로 갈수록 길게 묶어 변수를 줄인다.
  vectorized=True 로 한 세대(후보 popsize × 변수 수 개)를 트윈 batch API 한 번에 평가한다.
  초기 세대에 기준 온도·일정 온도·축열 시간표·직전 해를 넣어 그보다 나쁜 답이 나오지 않게 한다.

예측을 믿을 수 없을 때 (mode = 'mpc_naive')
  예측 서비스 호출 실패, in_training_range=false (학습 범위 밖 부하),
  또는 input_quality_ok=false (학습 범위 안이지만 입력 이력이 고착·급변·평균 이동).
  믿을 수 없는 건 예측 모델이지 트윈의 물리 계산이 아니므로, "최근 3시간 최대 부하가
  계속된다"는 보수적 가정으로 바꿔 계속 최적화한다.

Fallback (mode = 'fallback', 기준 설정 온도 고정)
  트윈 서비스 호출 실패 — 계획을 평가할 수 없으므로 원본 데이터에서 검증된 기준 설정으로 돌아간다.
  트윈이 모든 후보 계획을 계산하지 못하는 경우(수치 발산)도 같다.

냉각 용량 < 발열 (예: 칠러 2대 탈락)
  트윈의 팬 전력 상한 덕분에 온도가 발산하지 않고 유한하게 오른다. 이때는 Fallback 하지 않고
  계속 최적화한다: 위반 벌점이 "위반 구간 수"라서 축열조를 방전해 온도 상승을 최대한 늦추는 계획이 뽑힌다.
  계획 구간 안에 위험 온도에 닿으면 경고 twin.critical_temp_reached 가 붙는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import time

import httpx
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import differential_evolution

from services.common.settings import section

_MPC = section('mpc')  # 기본값은 config/settings.yaml 의 mpc 항목
_INFEASIBLE = 1e18  # 서버실 온도가 발산한 계획의 목적 함수 값


@dataclass(frozen=True)
class MPCConfig:
    objective: str = _MPC['objective']  # 'cost' | 'energy'
    horizon_hours: int = _MPC['horizon_hours']
    blocks: tuple[int, ...] = tuple(_MPC['blocks'])  # 합 = horizon_hours
    max_setpoint_rate_c_per_h: float = _MPC['max_setpoint_rate_c_per_h']  # 제어 변화율 제약
    naive_window_hours: int = _MPC[
        'naive_window_hours'
    ]  # 예측 불신 시: 최근 N시간 최대 부하 유지 가정
    use_tes: bool = _MPC['use_tes']  # 냉수 축열조 출력도 함께 최적화
    de_popsize: int = _MPC['de_popsize']
    de_maxiter: int = _MPC['de_maxiter']
    de_mutation: tuple[float, float] = tuple(_MPC['de_mutation'])
    de_recombination: float = _MPC['de_recombination']
    de_tol: float = _MPC['de_tol']
    violation_penalty_kwh: float = _MPC['violation_penalty_kwh']  # 위반 구간 1개당 벌점
    # (고장률 - 1) × 시간당 벌점
    reliability_weight_kwh_per_h: float = _MPC['reliability_weight_kwh_per_h']
    move_weight_kwh_per_c: float = _MPC['move_weight_kwh_per_c']  # 설정 온도 1℃ 변경당 벌점
    seed: int = _MPC['seed']

    def __post_init__(self) -> None:
        if sum(self.blocks) != self.horizon_hours:
            raise ValueError(
                f'blocks 합({sum(self.blocks)})이 horizon_hours({self.horizon_hours})와 같아야 합니다'
            )
        if self.objective not in ('cost', 'energy'):
            raise ValueError(f"objective 는 'cost' 또는 'energy' 여야 합니다: {self.objective}")
        if self.max_setpoint_rate_c_per_h <= 0:
            raise ValueError('max_setpoint_rate_c_per_h 는 0보다 커야 합니다')
        if self.de_popsize < 1 or self.de_maxiter < 1:
            raise ValueError('de_popsize, de_maxiter 는 1 이상이어야 합니다')


@dataclass
class PlanResult:
    mode: str  # 'mpc' | 'mpc_naive' | 'fallback'
    setpoint_c: float  # 지금 적용할 설정 온도 (계획의 첫 구간)
    plan_c: list[float]  # horizon 전체 계획
    tes_kw: float = 0.0  # 지금 적용할 축열조 출력 (+충전 / -방전, 냉열 kW)
    tes_plan_kw: list[float] = field(default_factory=list)
    predicted_tes_soc_kwh: list[float] = field(default_factory=list)
    objective: str = 'cost'
    timestamps: list[str] = field(default_factory=list)
    forecast_it_kw: list[float] = field(default_factory=list)
    price_krw_per_kwh: list[float] = field(default_factory=list)
    predicted_room_temp_c: list[float] = field(default_factory=list)
    predicted_facility_kw: list[float] = field(default_factory=list)
    predicted_pue: list[float] = field(default_factory=list)
    predicted_cooling_mode: list[str] = field(default_factory=list)
    expected_facility_energy_kwh: float = 0.0
    baseline_facility_energy_kwh: float | None = 0.0  # 같은 예측으로 기준 설정 고정 시
    expected_saving_pct: float | None = 0.0  # 전력량 기준 절감률 (기준 운전 계산 불가 시 None)
    expected_energy_cost_krw: float | None = None
    baseline_energy_cost_krw: float | None = None
    expected_cost_saving_pct: float | None = None  # 전기요금 기준 절감률
    cost: float = 0.0
    computation_time_s: float = 0.0  # 예측 호출 + 최적화 + 결과 계산 시간
    warnings: list[str] = field(default_factory=list)


class TwinUnavailable(RuntimeError):
    pass


def _saving_pct(value: float | None, baseline: float | None) -> float | None:
    """기준 대비 절감률(%). 어느 한쪽을 계산할 수 없으면 None (트윈은 발산한 합계를 null 로 준다)."""
    if value is None or baseline is None or baseline <= 0:
        return None
    return (1 - value / baseline) * 100


class MPCController:
    def __init__(
        self, twin: httpx.Client, forecast: httpx.Client, config: MPCConfig = MPCConfig()
    ) -> None:
        self.twin = twin
        self.forecast = forecast
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._params: dict | None = None
        self._warm: np.ndarray | None = None  # 직전 해 (블록 값), 다음 계산 시작점
        self._last_setpoint: float | None = None  # 직전에 적용한 설정 온도 (변경 벌점 기준)
        self.last_mode: str | None = None  # 직전 계획의 동작 모드 (/health 가 보여준다)

    # ------------------------------------------------------------ 서비스 호출
    @property
    def params(self) -> dict:
        if self._params is None:
            r = self.twin.get('/api/v1/params')
            r.raise_for_status()
            self._params = r.json()
        return self._params

    def _forecast(self, history: list[dict]) -> dict:
        r = self.forecast.post(
            '/api/v1/forecast',
            json={'horizon_hours': self.config.horizon_hours, 'history': history},
        )
        r.raise_for_status()
        return r.json()

    def _tariff(self, first_timestamp: str) -> list[float]:
        """계획 구간(horizon_hours 시간)의 시간별 전기 단가 (원/kWh)."""
        r = self.twin.get(
            '/api/v1/tariff', params={'start': first_timestamp, 'hours': self.config.horizon_hours}
        )
        if r.status_code != 200:
            raise TwinUnavailable(f'twin tariff {r.status_code}: {r.text[:200]}')
        return r.json()['price_krw_per_kwh']

    def _evaluate(
        self,
        it_kw: ArrayLike,
        outdoor_c: ArrayLike,
        plans: np.ndarray,
        room_c: float,
        capacity_factor: float,
        prices: list[float] | None = None,
        tes_plans: np.ndarray | None = None,
        tes_soc_kwh: float | None = None,
    ) -> dict:
        r = self.twin.post(
            '/api/v1/rollout/batch',
            json={
                'it_power_kw': list(map(float, it_kw)),
                'outdoor_temp_c': list(map(float, outdoor_c)),
                'plans': plans.round(3).tolist(),
                'initial_room_temp_c': float(room_c),
                'capacity_factor': capacity_factor,
                'price_krw_per_kwh': prices,
                'tes_plans': None if tes_plans is None else tes_plans.round(1).tolist(),
                'initial_tes_kwh': tes_soc_kwh,
            },
        )
        if r.status_code != 200:
            raise TwinUnavailable(f'twin rollout/batch {r.status_code}: {r.text[:200]}')
        # 서버실 온도가 발산한 계획은 트윈이 null 을 돌려준다 → NaN 으로 받아 _cost 에서 탈락시킨다
        return {k: np.asarray(v, dtype=float) for k, v in r.json().items() if v is not None}

    def _trajectory(
        self,
        it_kw: ArrayLike,
        outdoor_c: ArrayLike,
        plan: ArrayLike,
        room_c: float,
        capacity_factor: float,
        prices: list[float] | None = None,
        tes_plan: ArrayLike | None = None,
        tes_soc_kwh: float | None = None,
    ) -> dict:
        r = self.twin.post(
            '/api/v1/rollout',
            json={
                'it_power_kw': list(map(float, it_kw)),
                'outdoor_temp_c': list(map(float, outdoor_c)),
                'supply_temp_c': list(map(float, plan)),
                'initial_room_temp_c': float(room_c),
                'capacity_factor': capacity_factor,
                'price_krw_per_kwh': prices,
                'tes_kw': None if tes_plan is None else list(map(float, tes_plan)),
                'initial_tes_kwh': tes_soc_kwh,
            },
        )
        if r.status_code != 200:
            raise TwinUnavailable(f'twin rollout {r.status_code}: {r.text[:200]}')
        return r.json()

    # ------------------------------------------------------------ 최적화
    def _expand(self, block_values: np.ndarray) -> np.ndarray:
        """(N, 블록 수) → (N, horizon) 시간별 계획."""
        return np.repeat(block_values, self.config.blocks, axis=-1)

    def _limit_rate(self, setpoints: np.ndarray) -> np.ndarray:
        """제어 변화율 제약: 시간별 설정 온도 계획 (N, horizon) 을 직전 값 ±max_setpoint_rate_c_per_h 로 고친다.

        첫 시간은 직전에 적용한 설정 온도 기준 (처음 계산이면 제한 없음).
        """
        rate = self.config.max_setpoint_rate_c_per_h
        out = np.array(setpoints, dtype=float, copy=True)
        prev = out[:, 0] if self._last_setpoint is None else np.full(len(out), self._last_setpoint)
        for h in range(out.shape[1]):
            out[:, h] = np.clip(out[:, h], prev - rate, prev + rate)
            prev = out[:, h]
        return out

    def _cost(
        self,
        s: dict,
        setpoints: np.ndarray,
        prices: list[float] | None = None,
        tes_soc_kwh: float | None = None,
    ) -> np.ndarray:
        """목적 함수. cost 목적이면 전기요금(원) + 벌점(kWh × 구간 평균 단가), energy 목적이면 kWh.

        setpoints: (N, horizon) 시간별 설정 온도 (변화율 제약 적용 후).
        축열조를 쓰면 끝 저장량이 시작보다 적은 만큼 "구간 최저 단가로 다시 채우는 비용"을 더한다
        (계획 끝에 탱크를 비워 버리는 꼼수 방지).
        """
        c, p = self.config, self.params
        start = (
            setpoints[:, :1]
            if self._last_setpoint is None
            else np.full((len(setpoints), 1), self._last_setpoint)
        )
        moves = np.abs(np.diff(np.hstack([start, setpoints]), axis=1)).sum(axis=1)
        if c.objective == 'cost' and prices is not None:
            base, krw_per_kwh, refill_krw_per_kwh = (
                s['total_energy_cost_krw'],
                float(np.mean(prices)),
                float(min(prices)),
            )
        else:
            base, krw_per_kwh, refill_krw_per_kwh = s['total_facility_energy_kwh'], 1.0, 1.0
        penalty_kwh = (
            c.violation_penalty_kwh * s['violation_steps']
            + c.reliability_weight_kwh_per_h * (s['average_xfactor'] - 1.0) * c.horizon_hours
            + c.move_weight_kwh_per_c * moves
        )
        cost = base + krw_per_kwh * penalty_kwh
        if tes_soc_kwh is not None and 'final_tes_soc_kwh' in s:
            deficit_kwh_th = np.maximum(tes_soc_kwh - s['final_tes_soc_kwh'], 0.0)
            refill_kwh = deficit_kwh_th / (p['chiller_cop_ref'] * p['tes_charge_efficiency'])
            cost = cost + refill_krw_per_kwh * refill_kwh
        return np.where(np.isfinite(cost), cost, np.inf)

    def _tes_schedule(self, prices: list[float] | None) -> np.ndarray:
        """단가 시간표 축열조 후보 (블록 값): 최저 단가 시간 충전, 최고 단가 시간 방전."""
        p, nb = self.params, len(self.config.blocks)
        if not prices:
            return np.zeros(nb)
        pr = np.asarray(prices, dtype=float)
        hourly = np.where(
            pr <= pr.min() + 1e-9,
            p['tes_max_charge_kw'],
            np.where(pr >= pr.max() - 1e-9, -p['tes_max_discharge_kw'], 0.0),
        )
        edges = np.cumsum((0,) + self.config.blocks)
        return np.array([hourly[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])

    def optimize(
        self,
        it_kw: ArrayLike,
        outdoor_c: ArrayLike,
        room_c: float,
        capacity_factor: float = 1.0,
        prices: list[float] | None = None,
        tes_soc_kwh: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        """scipy.optimize.differential_evolution 으로 목적 함수가 가장 작은 계획을 찾는다.

        반환: (시간별 설정 온도 (변화율 제약 적용), 시간별 축열조 출력 또는 None, 목적 함수 값).
        prices: 시간별 단가 (cost 목적). tes_soc_kwh: 현재 축열조 저장량 (use_tes 일 때).
        """
        c, p = self.config, self.params
        use_tes = c.use_tes and tes_soc_kwh is not None
        soc = tes_soc_kwh if use_tes else None
        nb = len(c.blocks)
        bounds = [(p['supply_temp_min_c'], p['supply_temp_max_c'])] * nb
        if use_tes:
            bounds += [(-p['tes_max_discharge_kw'], p['tes_max_charge_kw'])] * nb
        lo, hi = np.array(bounds).T
        dim = len(bounds)

        def decode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
            """후보 (S, dim) → 시간별 설정 온도 (변화율 제약 적용), 시간별 축열조 출력."""
            setpoints = self._limit_rate(self._expand(x[:, :nb]))
            return setpoints, (self._expand(x[:, nb:]) if use_tes else None)

        def objective(xt: np.ndarray) -> np.ndarray:
            """vectorized: xt (dim, S) → 비용 (S,). 한 세대를 트윈 batch API 한 번으로 평가."""
            x = np.atleast_2d(xt.T)
            setpoints, tes_plans = decode(x)
            s = self._evaluate(
                it_kw, outdoor_c, setpoints, room_c, capacity_factor, prices, tes_plans, soc
            )
            cost = self._cost(s, setpoints, prices, soc)
            return np.where(np.isfinite(cost), cost, _INFEASIBLE)  # 발산한 계획은 아주 큰 값

        # 초기 세대: 기준 온도·일정 온도 (축열조 미사용 / 단가 시간표), 직전 해 + 나머지는 균등 무작위
        fixed_sp = [np.full(nb, v) for v in (p['supply_temp_ref_c'], 22.0, 24.0, 25.0, 26.0)]
        if use_tes:
            schedule = self._tes_schedule(prices)
            fixed = [np.append(sp, np.zeros(nb)) for sp in fixed_sp] + [
                np.append(sp, schedule) for sp in fixed_sp[2:]
            ]
        else:
            fixed = list(fixed_sp)
        if self._warm is not None and len(self._warm) == dim:
            fixed.append(self._warm)
        n_pop = max(c.de_popsize * dim, len(fixed) + 5)
        init = self.rng.uniform(lo, hi, (n_pop, dim))
        init[: len(fixed)] = np.clip(np.array(fixed), lo, hi)

        res = differential_evolution(
            objective,
            bounds,
            init=init,
            maxiter=c.de_maxiter,
            mutation=c.de_mutation,
            recombination=c.de_recombination,
            tol=c.de_tol,
            polish=False,
            vectorized=True,
            updating='deferred',
            rng=int(self.rng.integers(2**31)),
        )
        if not res.fun < _INFEASIBLE:
            raise TwinUnavailable(
                '모든 후보 계획에서 서버실 온도가 발산 (냉각 용량이 발열보다 작음)'
            )

        best_x = np.asarray(res.x, dtype=float)
        # 다음 시간 시작점: 한 시간 당긴 계획 (설정 온도·축열조 각각 첫 블록 소비)
        if c.blocks[0] == 1:
            parts = [best_x[:nb]] + ([best_x[nb:]] if use_tes else [])
            self._warm = np.concatenate([np.append(x[1:], x[-1]) for x in parts])
        else:
            self._warm = best_x
        setpoints, tes_plans = decode(best_x[None, :])
        return setpoints[0], (None if tes_plans is None else tes_plans[0]), float(res.fun)

    # ------------------------------------------------------------ 한 번의 결정
    def plan(
        self,
        room_temp_c: float,
        history: list[dict],
        outdoor_forecast_c: list[float],
        capacity_factor: float = 1.0,
        tes_soc_kwh: float | None = None,
        current_setpoint_c: float | None = None,
    ) -> PlanResult:
        """history: 최근 IT 부하 [{'timestamp', 'it_power_kw'}, ...] (1시간 또는 5분, 최소 168시간).
        outdoor_forecast_c: 앞으로 horizon_hours 시간의 외기온도 예보.
        tes_soc_kwh: 현재 축열조 저장량. 생략하면 트윈 설정의 시작 저장량(용량 × tes_initial_soc).
        current_setpoint_c: 지금 적용 중인 설정 온도 (변화율 제약 기준). 생략하면 직전 계획의 값을 쓴다."""
        c = self.config
        if len(outdoor_forecast_c) != c.horizon_hours:
            raise ValueError(
                f'outdoor_forecast_c 는 {c.horizon_hours}개여야 합니다: {len(outdoor_forecast_c)}'
            )
        started = time.perf_counter()
        if current_setpoint_c is not None:
            self._last_setpoint = float(current_setpoint_c)

        warnings: list[str] = []
        mode, fc = 'mpc', None
        try:
            fc = self._forecast(history)
            warnings += [f'forecast.{w}' for w in fc.get('warnings', [])]
            if not fc.get('in_training_range', True):
                mode = 'mpc_naive'
                warnings.append(
                    'forecast_out_of_range: 학습 범위 밖 부하라 예측 대신 최근 최대 부하 유지 가정'
                )
            elif not fc.get('input_quality_ok', True):
                mode = 'mpc_naive'
                warnings.append(
                    'forecast_input_suspect: 입력 이력에 이상(고착·급변·평균 이동)이 있어 '
                    '예측 대신 최근 최대 부하 유지 가정'
                )
        except httpx.HTTPError as e:
            mode = 'mpc_naive'
            warnings.append(f'forecast_unavailable: {e}. 최근 최대 부하 유지 가정')

        if mode == 'mpc':
            it_kw, timestamps = fc['predictions'], fc['timestamps']
        else:
            it_kw, timestamps = self._naive_forecast(history)
        try:
            p = self.params
            ref = float(p['supply_temp_ref_c'])
            soc = None
            if c.use_tes:
                soc = (
                    p['tes_capacity_kwh'] * p['tes_initial_soc']
                    if tes_soc_kwh is None
                    else float(tes_soc_kwh)
                )
            prices = self._tariff(timestamps[0])
            plan, tes_plan, cost = self.optimize(
                it_kw, outdoor_forecast_c, room_temp_c, capacity_factor, prices, soc
            )
            traj = self._trajectory(
                it_kw,
                outdoor_forecast_c,
                plan,
                room_temp_c,
                capacity_factor,
                prices,
                tes_plan,
                soc,
            )
            base = self._trajectory(
                it_kw,
                outdoor_forecast_c,
                [ref] * c.horizon_hours,
                room_temp_c,
                capacity_factor,
                prices,
            )
        except (TwinUnavailable, httpx.HTTPError) as e:
            return self._fallback(f'twin_unavailable: {e}', warnings)

        e_mpc, e_base = traj['total_facility_energy_kwh'], base['total_facility_energy_kwh']
        w_mpc, w_base = traj['total_energy_cost_krw'], base['total_energy_cost_krw']
        if e_mpc is None:  # 고른 계획조차 트윈이 계산하지 못함 (수치 발산)
            return self._fallback('twin_unavailable: 선택한 계획의 트윈 계산이 발산', warnings)
        warnings += [f'twin.{w}' for w in traj['warnings']]
        if e_base is None:
            # 기준 운전(기준 설정 온도 고정, 축열조 없음)만 발산한 경우: 계획은 유효하니 절감률만 비운다
            warnings.append(
                'baseline_diverged: 기준 운전(설정 온도 고정)은 이 조건에서 계산할 수 없어 '
                '절감률을 내지 않음'
            )
        self._last_setpoint = float(plan[0])
        self.last_mode = mode
        return PlanResult(
            mode=mode,
            setpoint_c=float(plan[0]),
            plan_c=[round(float(v), 2) for v in plan],
            tes_kw=0.0 if tes_plan is None else float(tes_plan[0]),
            tes_plan_kw=[] if tes_plan is None else [round(float(v), 1) for v in tes_plan],
            predicted_tes_soc_kwh=traj['tes_soc_kwh'] if tes_plan is not None else [],
            objective=c.objective,
            timestamps=timestamps,
            forecast_it_kw=list(map(float, it_kw)),
            price_krw_per_kwh=prices,
            predicted_room_temp_c=traj['room_temp_c'],
            predicted_facility_kw=traj['facility_power_kw'],
            predicted_pue=traj['pue'],
            predicted_cooling_mode=traj['cooling_mode'],
            expected_facility_energy_kwh=e_mpc,
            baseline_facility_energy_kwh=e_base,
            expected_saving_pct=_saving_pct(e_mpc, e_base),
            expected_energy_cost_krw=w_mpc,
            baseline_energy_cost_krw=w_base,
            expected_cost_saving_pct=_saving_pct(w_mpc, w_base),
            cost=cost,
            computation_time_s=time.perf_counter() - started,
            warnings=warnings,
        )

    def _naive_forecast(self, history: list[dict]) -> tuple[list[float], list[str]]:
        """최근 naive_window_hours 시간의 최대(1시간 평균) 부하가 계속된다고 가정."""
        df = pd.DataFrame(history)
        hourly = (
            df.set_index(pd.to_datetime(df['timestamp']))['it_power_kw']
            .resample('h')
            .mean()
            .dropna()
        )
        level = float(hourly.iloc[-self.config.naive_window_hours :].max())
        last = hourly.index[-1]
        stamps = [
            (last + pd.Timedelta(hours=h)).isoformat()
            for h in range(1, self.config.horizon_hours + 1)
        ]
        return [level] * self.config.horizon_hours, stamps

    def _fallback(self, reason: str, warnings: list[str]) -> PlanResult:
        self._warm = None
        try:
            ref = float(self.params['supply_temp_ref_c'])
        except httpx.HTTPError:
            ref = 20.0
        self._last_setpoint = ref
        self.last_mode = 'fallback'
        return PlanResult(
            mode='fallback',
            setpoint_c=ref,
            plan_c=[ref] * self.config.horizon_hours,
            warnings=warnings + [reason],
        )
