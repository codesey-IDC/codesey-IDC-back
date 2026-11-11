"""MPC 제어기: 예측(forecast-service) → 후보 계획 평가(twin-service) → 설정 온도 결정.

과제 제약(모듈 간 직접 함수 호출 금지)에 따라 다른 모듈은 REST API 로만 부른다.
twin / forecast 는 httpx.Client 호환 객체다. 실서비스는 base_url 을 준 httpx.Client,
테스트·시뮬레이션은 fastapi.testclient.TestClient 를 넘긴다 (둘 다 HTTP 요청을 거친다).

MPC 5요소
  모델        twin-service /api/v1/rollout/batch
  예측 구간   horizon_hours (기본 24시간)
  목적 함수   전체 전력량(kWh) + 위반 벌점 + 고장률 벌점(작게, 같은 전력이면 낮은 온도 선호)
              + 설정 변경 벌점(1℃ 바꿀 때마다 조금, 비용이 비슷할 때 들쭉날쭉한 계획 방지)
              ※ PUE 가 아니라 전체 전력: 팬 전력은 IT 로 잡혀서 PUE 만 보면 과하게 올린다
  제약 조건   서버실 18~27℃, 냉각 정지 시 15분 이상 버팀 → 벌점으로 처리
  후퇴 구간   계획 중 첫 1시간 설정만 적용, 다음 시간에 다시 계산

최적화: CEM(Cross-Entropy Method)
  설정 온도 계획을 정규분포에서 여러 개 뽑아 트윈으로 한 번에 평가 → 좋은 상위 계획들로
  분포를 좁혀 다시 뽑기를 반복. 변수는 블록 단위(앞쪽은 1시간, 뒤로 갈수록 길게)로 묶어 줄인다.

예측을 믿을 수 없을 때 (mode = 'mpc_naive')
  예측 서비스 호출 실패, 또는 in_training_range=false (학습 범위 밖 부하).
  믿을 수 없는 건 예측 모델이지 트윈의 물리 계산이 아니므로, "최근 3시간 최대 부하가
  계속된다"는 보수적 가정으로 바꿔 계속 최적화한다.

Fallback (mode = 'fallback', 기준 설정 온도 고정)
  트윈 서비스 호출 실패 — 계획을 평가할 수 없으므로 원본 데이터에서 검증된 기준 설정으로 돌아간다.
  어떤 계획으로도 서버실 온도가 발산하는 경우(냉각 용량 < 발열)도 같다. 설정 온도로 해결할 수 없는 상황이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MPCConfig:
    horizon_hours: int = 24
    blocks: tuple[int, ...] = (1, 1, 1, 1, 2, 2, 4, 4, 8)   # 합 = horizon_hours
    cem_iterations: int = 6
    cem_samples: int = 96
    elite_fraction: float = 0.15
    initial_std_c: float = 3.0
    min_std_c: float = 0.2
    naive_window_hours: int = 3                  # 예측 불신 시: 최근 N시간 최대 부하 유지 가정
    violation_penalty_kwh: float = 100_000.0     # 위반 구간 1개당 벌점
    reliability_weight_kwh_per_h: float = 20.0   # (고장률 - 1) × 시간당 벌점
    move_weight_kwh_per_c: float = 30.0          # 설정 온도 1℃ 변경당 벌점
    seed: int = 0

    def __post_init__(self) -> None:
        if sum(self.blocks) != self.horizon_hours:
            raise ValueError(f'blocks 합({sum(self.blocks)})이 horizon_hours({self.horizon_hours})와 같아야 합니다')


@dataclass
class PlanResult:
    mode: str                           # 'mpc' | 'mpc_naive' | 'fallback'
    setpoint_c: float                   # 지금 적용할 설정 온도 (계획의 첫 구간)
    plan_c: list[float]                 # horizon 전체 계획
    timestamps: list[str] = field(default_factory=list)
    forecast_it_kw: list[float] = field(default_factory=list)
    predicted_room_temp_c: list[float] = field(default_factory=list)
    predicted_facility_kw: list[float] = field(default_factory=list)
    expected_facility_energy_kwh: float = 0.0
    baseline_facility_energy_kwh: float = 0.0     # 같은 예측으로 기준 설정 고정 시
    expected_saving_pct: float = 0.0
    cost: float = 0.0
    warnings: list[str] = field(default_factory=list)


class TwinUnavailable(RuntimeError):
    pass


class MPCController:
    def __init__(self, twin: httpx.Client, forecast: httpx.Client, config: MPCConfig = MPCConfig()) -> None:
        self.twin = twin
        self.forecast = forecast
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._params: dict | None = None
        self._warm: np.ndarray | None = None      # 직전 해 (블록 값), 다음 계산 시작점
        self._last_setpoint: float | None = None  # 직전에 적용한 설정 온도 (변경 벌점 기준)

    # ------------------------------------------------------------ 서비스 호출
    @property
    def params(self) -> dict:
        if self._params is None:
            r = self.twin.get('/api/v1/params')
            r.raise_for_status()
            self._params = r.json()
        return self._params

    def _forecast(self, history: list[dict]) -> dict:
        r = self.forecast.post('/api/v1/forecast', json={
            'horizon_hours': self.config.horizon_hours, 'history': history})
        r.raise_for_status()
        return r.json()

    def _evaluate(self, it_kw, outdoor_c, plans: np.ndarray, room_c: float, capacity_factor: float) -> dict:
        r = self.twin.post('/api/v1/rollout/batch', json={
            'it_power_kw': list(map(float, it_kw)), 'outdoor_temp_c': list(map(float, outdoor_c)),
            'plans': plans.round(3).tolist(), 'initial_room_temp_c': float(room_c),
            'capacity_factor': capacity_factor})
        if r.status_code != 200:
            raise TwinUnavailable(f'twin rollout/batch {r.status_code}: {r.text[:200]}')
        # 서버실 온도가 발산한 계획은 트윈이 null 을 돌려준다 → NaN 으로 받아 _cost 에서 탈락시킨다
        return {k: np.asarray(v, dtype=float) for k, v in r.json().items()}

    def _trajectory(self, it_kw, outdoor_c, plan, room_c: float, capacity_factor: float) -> dict:
        r = self.twin.post('/api/v1/rollout', json={
            'it_power_kw': list(map(float, it_kw)), 'outdoor_temp_c': list(map(float, outdoor_c)),
            'supply_temp_c': list(map(float, plan)), 'initial_room_temp_c': float(room_c),
            'capacity_factor': capacity_factor})
        if r.status_code != 200:
            raise TwinUnavailable(f'twin rollout {r.status_code}: {r.text[:200]}')
        return r.json()

    # ------------------------------------------------------------ 최적화
    def _expand(self, block_values: np.ndarray) -> np.ndarray:
        """(N, 블록 수) → (N, horizon) 시간별 계획."""
        return np.repeat(block_values, self.config.blocks, axis=-1)

    def _cost(self, s: dict, block_values: np.ndarray) -> np.ndarray:
        c = self.config
        start = block_values[:, :1] if self._last_setpoint is None else np.full((len(block_values), 1), self._last_setpoint)
        moves = np.abs(np.diff(np.hstack([start, block_values]), axis=1)).sum(axis=1)
        cost = (s['total_facility_energy_kwh']
                + c.violation_penalty_kwh * s['violation_steps']
                + c.reliability_weight_kwh_per_h * (s['average_xfactor'] - 1.0) * c.horizon_hours
                + c.move_weight_kwh_per_c * moves)
        return np.where(np.isfinite(cost), cost, np.inf)

    def optimize(self, it_kw, outdoor_c, room_c: float, capacity_factor: float = 1.0
                 ) -> tuple[np.ndarray, float]:
        c, p = self.config, self.params
        lo, hi = p['supply_temp_min_c'], p['supply_temp_max_c']
        nb = len(c.blocks)
        mean = np.full(nb, p['supply_temp_ref_c']) if self._warm is None else self._warm.copy()
        std = np.full(nb, c.initial_std_c)
        n_elite = max(2, int(c.cem_samples * c.elite_fraction))
        # 항상 비교하는 고정 후보: 기준 온도, 몇 가지 일정 온도, 직전 해
        fixed = [np.full(nb, v) for v in (p['supply_temp_ref_c'], 22.0, 24.0, 25.0, 26.0)]
        if self._warm is not None:
            fixed.append(self._warm)

        best_x, best_cost = None, np.inf
        for it in range(c.cem_iterations):
            samples = np.clip(self.rng.normal(mean, std, (c.cem_samples, nb)), lo, hi)
            if it == 0:
                samples[:len(fixed)] = np.clip(np.array(fixed), lo, hi)
            if best_x is not None:
                samples[-1] = best_x
            cost = self._cost(self._evaluate(it_kw, outdoor_c, self._expand(samples), room_c, capacity_factor),
                              samples)
            order = np.argsort(cost)
            if cost[order[0]] < best_cost:
                best_cost, best_x = float(cost[order[0]]), samples[order[0]].copy()
            elite = samples[order[:n_elite]]
            mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), c.min_std_c)

        if best_x is None:
            raise TwinUnavailable('모든 후보 계획에서 서버실 온도가 발산 (냉각 용량이 발열보다 작음)')
        # 다음 시간 시작점: 한 시간 당긴 계획 (첫 블록 소비)
        self._warm = np.append(best_x[1:], best_x[-1]) if c.blocks[0] == 1 else best_x
        return self._expand(best_x[None, :])[0], best_cost

    # ------------------------------------------------------------ 한 번의 결정
    def plan(self, room_temp_c: float, history: list[dict], outdoor_forecast_c: list[float],
             capacity_factor: float = 1.0) -> PlanResult:
        """history: 최근 IT 부하 [{'timestamp', 'it_power_kw'}, ...] (1시간 또는 5분, 최소 168시간).
        outdoor_forecast_c: 앞으로 horizon_hours 시간의 외기온도 예보."""
        c = self.config
        if len(outdoor_forecast_c) != c.horizon_hours:
            raise ValueError(f'outdoor_forecast_c 는 {c.horizon_hours}개여야 합니다: {len(outdoor_forecast_c)}')

        warnings: list[str] = []
        mode, fc = 'mpc', None
        try:
            fc = self._forecast(history)
            warnings += [f'forecast.{w}' for w in fc.get('warnings', [])]
            if not fc.get('in_training_range', True):
                mode = 'mpc_naive'
                warnings.append('forecast_out_of_range: 학습 범위 밖 부하라 예측 대신 최근 최대 부하 유지 가정')
        except httpx.HTTPError as e:
            mode = 'mpc_naive'
            warnings.append(f'forecast_unavailable: {e}. 최근 최대 부하 유지 가정')

        if mode == 'mpc':
            it_kw, timestamps = fc['predictions'], fc['timestamps']
        else:
            it_kw, timestamps = self._naive_forecast(history)
        try:
            ref = float(self.params['supply_temp_ref_c'])
            plan, cost = self.optimize(it_kw, outdoor_forecast_c, room_temp_c, capacity_factor)
            traj = self._trajectory(it_kw, outdoor_forecast_c, plan, room_temp_c, capacity_factor)
            base = self._trajectory(it_kw, outdoor_forecast_c, [ref] * c.horizon_hours, room_temp_c,
                                    capacity_factor)
        except (TwinUnavailable, httpx.HTTPError) as e:
            return self._fallback(f'twin_unavailable: {e}', warnings)

        warnings += [f'twin.{w}' for w in traj['warnings']]
        self._last_setpoint = float(plan[0])
        e_mpc, e_base = traj['total_facility_energy_kwh'], base['total_facility_energy_kwh']
        return PlanResult(
            mode=mode, setpoint_c=float(plan[0]), plan_c=[round(float(v), 2) for v in plan],
            timestamps=timestamps, forecast_it_kw=list(map(float, it_kw)),
            predicted_room_temp_c=traj['room_temp_c'], predicted_facility_kw=traj['facility_power_kw'],
            expected_facility_energy_kwh=e_mpc, baseline_facility_energy_kwh=e_base,
            expected_saving_pct=(1 - e_mpc / e_base) * 100, cost=cost, warnings=warnings,
        )

    def _naive_forecast(self, history: list[dict]) -> tuple[list[float], list[str]]:
        """최근 naive_window_hours 시간의 최대(1시간 평균) 부하가 계속된다고 가정."""
        df = pd.DataFrame(history)
        hourly = df.set_index(pd.to_datetime(df['timestamp']))['it_power_kw'].resample('h').mean().dropna()
        level = float(hourly.iloc[-self.config.naive_window_hours:].max())
        last = hourly.index[-1]
        stamps = [(last + pd.Timedelta(hours=h)).isoformat() for h in range(1, self.config.horizon_hours + 1)]
        return [level] * self.config.horizon_hours, stamps

    def _fallback(self, reason: str, warnings: list[str]) -> PlanResult:
        self._warm = None
        try:
            ref = float(self.params['supply_temp_ref_c'])
        except httpx.HTTPError:
            ref = 20.0
        self._last_setpoint = ref
        return PlanResult(mode='fallback', setpoint_c=ref, plan_c=[ref] * self.config.horizon_hours,
                          warnings=warnings + [reason])
