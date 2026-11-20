# API 명세서

세 서비스는 FastAPI 로 구현했고 서로 **REST API 로만** 통신한다 (모듈 간 직접 함수 호출 없음).
기계용 명세(OpenAPI 3)는 `docs/openapi/{twin,forecast,mpc}.json` 에 있고, 서비스를 띄우면
`http://127.0.0.1:<포트>/docs` 에서 직접 호출해 볼 수 있다. 다시 만들기: `python docs/export_openapi.py`

| 서비스 | 포트 | 기본 주소 | 역할 |
|---|---|---|---|
| twin-service | 8001 | `http://127.0.0.1:8001` | 가상 데이터센터(디지털 트윈): 열역학 계산, 계측값, 요금, ESG |
| forecast-service | 8002 | `http://127.0.0.1:8002` | IT 부하·냉각 수요 예측 |
| mpc-service | 8003 | `http://127.0.0.1:8003` | 최적 냉각 제어(MPC), 폐루프 시뮬레이션 |

공통 규칙
- 요청·응답은 JSON, 시각은 ISO 8601 (`2026-06-04T06:00:00`), 시간대 없음 (데이터 기준 현지 시각).
- 단위는 필드 이름 끝에 붙인다: `_kw`(kW), `_kwh`(kWh), `_c`(℃), `_krw`(원), `_min`(분), `_pct`(%).
- 오류: `422` 입력 오류(형식·범위·길이), `404` 없는 자원, `503` 의존 서비스나 모델 없음. 본문은 `{"detail": "..."}`.
- 서비스 주소는 `config/settings.yaml` 의 `services`, 환경변수 `TWIN_URL`, `FORECAST_URL`, `MPC_URL` 이 우선한다.
- 응답 시간 (실측 p95, `python scripts/benchmark_api.py`): MPC 제어 2.70초, 나머지 0.11초 이하. 기준 10초.

---

## 1. twin-service (:8001) — 가상 데이터센터

수식은 [`docs/thermodynamics.md`](thermodynamics.md).

### `GET /health`
`{"status": "ok", "service": "twin"}`

### `GET /api/v1/params`
트윈이 쓰는 상수 전체 (`config/settings.yaml` 의 `datacenter`) + 계산값.
주요 필드: `num_servers`, `server_idle_power_w`, `server_max_power_w`, `idle_power_kw`, `cpu_power_range_kw`,
`supply_temp_ref_c`(20), `supply_temp_min_c`/`max_c`(16/27), `room_temp_min_c`/`max_c`(18/27),
`air_mass_flow_kg_s`, `air_cp_kj_per_kg_k`, `air_heat_rate_kw_per_c`, `cooling_capacity_kw`, `chiller_units`,
`tes_capacity_kwh`, `tes_max_charge_kw`, `tes_max_discharge_kw`, `critical_temp_c`, `min_ride_through_min` 등.

### `POST /api/v1/simulate` — 정상 상태 한 시점 계산
"부하·외기·설정이 이러면 냉각 전력·PUE 는?" (서버실 온도 = 공급 설정 온도로 수렴한 상태)

요청
| 필드 | 타입 | 설명 |
|---|---|---|
| `points[]` | list (1~50,000) | 계산할 시점들 |
| `points[].outdoor_temp_c` | float (-40~60) | 외기온도 |
| `points[].it_power_kw` | float > 0, 선택 | IT 부하. 없으면 `cpu_utilization` 필요 |
| `points[].cpu_utilization` | float (0~1), 선택 | CPU 사용률 → SPECpower 로 IT 부하 계산 |
| `points[].supply_temp_c` | float, 선택 | 공급 공기 설정 온도. 생략 시 기준 20℃ |

응답 `results[]`: `it_power_kw`(팬 증가분 포함), `server_fan_extra_kw`, `server_heat_kw`, `air_mass_flow_kg_s`,
`return_air_temp_c`, `cooling_load_kw`(Q = ṁc_pΔT), `free_cooling_fraction`, `chiller_cop`,
`cooling_mode`(`free_cooling`|`hybrid`|`chiller`), `cooling_power_kw`, `facility_power_kw`, `pue`,
`ride_through_min`, `xfactor`. 오류: 부하 둘 다 없음 → 422.

```bash
curl -X POST :8001/api/v1/simulate -H 'Content-Type: application/json' \
  -d '{"points": [{"it_power_kw": 3000, "outdoor_temp_c": 25}]}'
# → cooling_mode "chiller", cooling_power_kw 1111.1, pue 1.420
```

### `POST /api/v1/rollout` — 시간별 계획 평가 (동적)
설정 온도·축열조 계획을 시간 순서로 적용해 서버실 온도·전력·요금을 계산한다 (5분 단위 적분).

요청
| 필드 | 타입 | 설명 |
|---|---|---|
| `it_power_kw[]` | list (1~2016) | 구간별 IT 부하 (팬 제외 기본 부하, 예측값) |
| `outdoor_temp_c[]` | list | 구간별 외기 |
| `supply_temp_c[]` | list | 구간별 공급 공기 설정 온도 (16~27℃ 로 자른다) |
| `initial_room_temp_c` | float, 선택 | 시작 서버실 온도. 생략 시 첫 설정 온도 |
| `dt_hours` | float (0~24], 기본 1 | 구간 길이 |
| `capacity_factor` | float (0~2], 기본 1 | 냉각 용량 배율 (0.75 = 칠러 4대 중 1대 고장) |
| `price_krw_per_kwh[]` | list, 선택 | 구간별 전기 단가 → 전기요금 계산 |
| `tes_kw[]` | list, 선택 | 축열조 출력 (+충전 / -방전, 냉열 kW). 생략 시 미사용 |
| `initial_tes_kwh` | float, 선택 | 시작 저장량. 생략 시 용량 × 50% |

배열 길이는 모두 같아야 한다 (다르면 422).

응답 (구간별 배열): `room_temp_c`, `return_air_temp_c`, `it_power_kw`, `server_fan_extra_kw`, `heat_removed_kw`,
`cooling_power_kw`, `facility_power_kw`, `pue`, `free_cooling_fraction`, `cooling_mode`, `ride_through_min`,
`tes_kw`, `tes_soc_kwh`, `energy_cost_krw`(단가 줬을 때).
합계: `total_energy_cost_krw`, `total_it_energy_kwh`, `total_fan_extra_energy_kwh`, `total_cooling_energy_kwh`,
`total_facility_energy_kwh`, `average_pue`, `max_room_temp_c`, `min_room_temp_c`, `min_ride_through_min`,
`average_xfactor`, `violation_steps`(18~27℃ 또는 비상 여유 15분 위반 구간 수), `capacity_limited_steps`,
`final_tes_soc_kwh`, `warnings`.

### `POST /api/v1/rollout/batch` — 계획 여러 개 한 번에 평가 (MPC 후보 비교용)
`rollout` 과 같은 입력에 `plans[][]`(계획 N개, 최대 1024), `tes_plans[][]`(선택, plans 와 같은 모양),
`initial_room_temp_c`(필수). 응답은 계획별 합계 배열 (`total_energy_cost_krw[]`, `total_facility_energy_kwh[]`,
`violation_steps[]`, `average_xfactor[]`, `final_tes_soc_kwh[]` 등). 108개 계획 × 24시간 ≈ 0.01초.

### `GET /api/v1/telemetry?start=&hours=` — 계측값
`[start, start + hours)` 의 시간별 `it_power_kw`, `outdoor_temp_c` (5분 데이터의 1시간 평균).
`start` 생략 시 데이터 끝에서 `hours` 시간. `hours` 1~2400. 응답에 `data_start`, `data_end` 포함.
범위 밖 → 404.

### `GET /api/v1/tariff?start=&hours=` — 시간별 전기 단가
한전 일반용전력(을) 고압A 선택Ⅲ (2026-06-01 시행). 응답: `timestamps[]`, `season[]`
(`summer`|`spring_fall`|`winter`), `period[]` (`off_peak` 경부하 | `mid` 중간부하 | `peak` 최대부하),
`price_krw_per_kwh[]`, `basic_charge_krw_per_kw`, `name`, `source`, `effective_date`.
토요일 최대부하 → 중간부하, 일요일·공휴일 → 경부하. `hours` 1~8784.

### `POST /api/v1/esg` — ESG 지표
요청: `it_power_kw[]`, `cooling_power_kw[]`, `cooling_mode[]`, `heat_removed_kw[]`(선택), `price_krw_per_kwh[]`(선택),
`dt_hours`. 응답: `it_energy_mwh`, `cooling_energy_mwh`, `total_energy_mwh`, `average_pue`, `carbon_tco2`
(0.459 tCO2/MWh), `cue_tco2_per_mwh`, `water_usage_l`, `wue_l_per_kwh`, `water_by_mode_l{}`,
`cooling_energy_by_mode_mwh{}`, `hours_by_mode{}`, `energy_cost_krw`, `cost_krw_per_mwh`.
알 수 없는 냉각 모드·길이 불일치 → 422.

---

## 2. forecast-service (:8002) — 예측

### `GET /health`
`{"status": "ok", "service": "forecast", "model": "ridge", "available_models": ["lgbm", "ridge"]}`

### `POST /api/v1/forecast` — IT 부하 / 냉각 수요 예측

요청
| 필드 | 타입 | 설명 |
|---|---|---|
| `horizon_hours` | int (1~168), 기본 24 | 예측 기간. 24 이하 → 24h 모델, 이상 → 168h 모델 |
| `target` | `it_load`(기본) \| `cooling_load` \| `cooling_power` | 예측 대상 (아래) |
| `model` | `ridge` \| `lgbm`, 선택 | 생략 시 `config/settings.yaml` 의 `forecast.model` (기본 ridge) |
| `history[]` | list, 선택 | 최근 IT 부하 `{timestamp, it_power_kw}`, 5분 또는 1시간 간격, **최소 168시간**. 생략 시 서버 데이터 끝 기준 |
| `outdoor_forecast_c[]` | list, 선택 | `cooling_power` 일 때 외기 예보 (`horizon_hours` 개). 생략 시 서버 데이터의 외기 실측 |

`target`
- `it_load`: IT 부하 (kW)
- `cooling_load`: 냉각 부하(열량). 데이터상 IT 부하와 같아 같은 값
- `cooling_power`: **냉각 전력(전기, kW)** = IT 부하 예측 → twin-service `/api/v1/simulate` (REST) 로 변환.
  기준 설정 온도(20℃) 운전. 구간은 IT 구간의 하한·상한을 그대로 변환 (냉각 전력은 IT 부하에 대해 증가 함수)

응답
| 필드 | 설명 |
|---|---|
| `model` | `{ridge\|lgbm}-{24h\|168h}`, `cooling_power` 는 뒤에 `+twin` |
| `origin_time` | 예측 시점 (이력의 마지막 시각) |
| `timestamps[]`, `predictions[]` | 앞으로 h 시간의 시각, 중앙값 예측 |
| `pi_lower[]`, `pi_upper[]` | 90% 예측 구간 |
| `in_training_range`, `warnings[]` | 학습 범위(2,354~3,457kW) 밖 입력·예측이면 `false` + 경고 → MPC 가 예측을 믿지 않는다 |
| `input_quality_ok`, `input_checks[]` | 학습 범위 **안**에서 생기는 입력 이상 검사 (아래). 하나라도 걸리면 `false` + 경고 `input_{rule}: …` → MPC 가 예측을 믿지 않는다 |
| `it_predictions[]`, `outdoor_temp_c[]`, `cooling_mode[]`, `free_cooling_fraction[]`, `pue[]`, `outdoor_source` | `cooling_power` 일 때만 |

입력 품질 검사 (`input_checks[]`: `rule`, `label`, `ok`, `measured`, `limit`. 한계값은 `config/settings.yaml` 의 `forecast.input_quality`)
| rule | 잡는 것 | 기준 (정상 데이터 최대·최소) |
|---|---|---|
| `stuck` | 값 고착 (센서 멈춤) | 6시간 표준편차 < 1.0kW (정상 최소 6.3kW) |
| `jump` | 1시간 사이 급변 (범위 안 스파이크 포함) | 변화 > 300kW (정상 최대 209kW) |
| `level_shift` | 계단·점진 드리프트 | 최근 24시간 평균이 1주 전 같은 시간대보다 ±80kW 초과 (정상 최대 30kW) |

`level_shift` 는 1주 전 값이 필요해 **이력이 192시간 이상일 때만** 검사한다. 더 짧으면 `ok: null`(검사 못 함)이다.
대시보드와 MPC 는 192시간을 보낸다. 범위 검사만 있을 때는 +10% 계단 입력이 그대로 통과해 예측이 약 8.6% 낮게 나왔다.

오류: 이력 168시간 미만·최근 168시간에 빈 시간 → 422, 외기 개수 틀림 → 422, 예측 구간 외기 데이터 없음 → 422,
모델 없음 → 503, twin-service 장애(`cooling_power`) → 503.

성능 (Test 2026-09-27~10-31, 35일. 학습 03-15~08-22, 검증 08-23~09-26): 아래 표, 자세한 비교는 `output/forecast/model_comparison.json`.

| 모델 | 24h MAPE | 168h MAPE | 90% 적중률 (24h/168h) |
|---|---|---|---|
| **ridge (기본)** | 0.900% | 0.907% | 89.1% / 89.0% |
| lgbm | 0.922% | 0.925% | 88.9% / 88.6% |
| 1주 전 값 (기준선) | 1.244% | 1.251% | — |

냉각 전력 예측 nMAE 1.29%, 냉각 모드 적중률 99.2% (Test 첫 14일, `output/forecast/cooling_metrics.json`).

---

## 3. mpc-service (:8003) — 최적 냉각 제어

### `GET /health`
`{"status": "ok", "service": "mpc", "forecast_url": "...", "twin_url": "...", "dependencies": {"forecast": "ok", "twin": "ok"}, "last_plan_mode": "mpc", "reasons": []}`

프로세스가 살아 있으면 항상 200 이고, `status` 로 제어가 정상 경로로 돌 수 있는지를 구분한다.
- `ok`: 예측·트윈이 응답하고, 직전 계획(`/plan`, `/control/*`)이 `mpc` 모드였거나 아직 계획한 적이 없다
- `degraded`: 예측·트윈에 닿지 않거나(`dependencies` 가 `unreachable` \| `model_unavailable`), 직전 계획이 `mpc_naive`·`fallback` 이었다.
  `reasons[]` 에 이유 (`forecast_unreachable`, `twin_unreachable`, `last_plan_mpc_naive`, `last_plan_fallback` 등)

안전하게 물러나는 동작(mpc_naive, fallback)은 그대로이고, 물러났다는 사실을 밖에서 알 수 있게 한 것이다.
폐루프 시뮬레이션(`/simulate`)은 제어기를 따로 만들어 쓰므로 이 상태에 영향을 주지 않는다.

### `POST /api/v1/control/optimize` — 최적 냉각 제어 (과제 결과물 3)
현재 상태를 받아 앞으로 24시간 **시간대별 최적 냉각 설정값과 예상 PUE** 를 돌려준다.
첫 시간 값만 적용하고 매시간 다시 부른다 (후퇴 구간).

요청 (`ControlRequest`)
| 필드 | 타입 | 설명 |
|---|---|---|
| `timestamp` | datetime | 현재 시각 = 계획 시작. `history`·`outdoor_forecast_c` 를 생략하면 필수 |
| `room_temp_c` | float (0~60) | 현재 서버실(서버 흡기) 온도 |
| `current_setpoint_c` | float, 선택 | 지금 적용 중인 설정 온도 (변화율 제약 ±2℃/h 의 기준) |
| `tes_soc_kwh` | float, 선택 | 현재 축열조 저장량. 생략 시 용량 × 50% |
| `capacity_factor` | float (0~2], 기본 1 | 냉각 용량 배율 |
| `history[]` | list, 선택 | 최근 IT 부하 (최소 168시간). 생략 시 twin `/telemetry` 에서 |
| `outdoor_forecast_c[]` | list (24), 선택 | 외기 예보. 생략 시 twin `/telemetry` 의 실측 (perfect forecast 가정) |

응답 (`ControlSchedule`)
| 필드 | 설명 |
|---|---|
| `mode` | `mpc` 정상 / `mpc_naive` 예측 불신(학습 범위 밖 또는 입력 이력 이상 → 최근 3시간 최대 부하 유지 가정) / `fallback` 트윈 장애 → 기준 20℃ |
| `method`, `objective` | `mpc`, `cost`(전기요금 최소화) |
| `timestamps[]` | 24시간 |
| `supply_temp_c[]` | **시간대별 최적 공급 공기 설정 온도** (시간당 ±2℃ 이내) |
| `tes_kw[]` | 시간대별 축열조 출력 (+충전 / -방전) |
| `cooling_mode[]`, `expected_pue[]` | **예상 냉각 모드와 PUE** |
| `expected_room_temp_c[]`, `expected_tes_soc_kwh[]`, `price_krw_per_kwh[]` | 예상 서버실 온도, 저장량, 단가 |
| `expected_energy_cost_krw`, `baseline_energy_cost_krw`, `expected_cost_saving_pct` | 24시간 예상 요금, 기준(20℃ 고정, 축열 없음), 절감률 |
| `expected_average_pue`, `computation_time_s`, `inputs_from`, `warnings[]` | 평균 PUE, 계산 시간, 입력 출처(`request`\|`twin_telemetry`) |

```bash
curl -X POST :8003/api/v1/control/optimize -H 'Content-Type: application/json' \
  -d '{"timestamp": "2026-06-04T06:00:00", "room_temp_c": 22, "current_setpoint_c": 22, "tes_soc_kwh": 6000}'
```

### `POST /api/v1/control/mpc` — MPC 제어 결과 전체 (선택 A)
요청은 `control/optimize` 와 같다. 응답 (`PlanResponse`): `mode`, `setpoint_c`(지금 적용), `plan_c[]`, `tes_kw`,
`tes_plan_kw[]`, `predicted_tes_soc_kwh[]`, `objective`, `timestamps[]`, `forecast_it_kw[]`, `price_krw_per_kwh[]`,
`predicted_room_temp_c[]`, `predicted_facility_kw[]`, `predicted_pue[]`, `predicted_cooling_mode[]`,
`expected_facility_energy_kwh`, `baseline_facility_energy_kwh`, `expected_saving_pct`,
`expected_energy_cost_krw`, `baseline_energy_cost_krw`, `expected_cost_saving_pct`, `computation_time_s`, `warnings[]`.

### `POST /api/v1/plan` — (대시보드용) MPC 결과
`control/mpc` 와 같은 계산. 요청에 `history[]`(필수, 최소 168)와 `outdoor_forecast_c[]`(필수, 24개)를 직접 넣는다.

### `POST /api/v1/simulate/jobs` — 폐루프 시뮬레이션 작업 등록 (202)
매시간 제어기가 설정을 정하고 트윈을 1시간 진행하는 실험. MPC 는 1시간당 약 2.6초라 백그라운드 작업으로 돌린다.

요청 (`SimulateRequest`)
| 필드 | 타입 | 설명 |
|---|---|---|
| `start` | datetime | 시작 시각 (데이터 시작 + 192시간 이후) |
| `hours` | int (1~168), 기본 24 | 진행 시간 |
| `controller` | `fixed_20` \| `rule_pid` \| `rule_tes` \| `reactive` \| `mpc` | 제어기 |
| `scenario` | `GET /api/v1/scenarios` 의 `name` (`normal`, `server_surge`, `triple_crisis` 등 8종) | 위기 시나리오 |
| `initial_room_temp_c` | float, 선택 | 시작 서버실 온도 |

응답 (`SimulationJob`): `job_id`, `status`(`queued`|`running`|`done`|`failed`), `progress_hours`, `total_hours`,
`elapsed_s`, `request`, `result`(done 일 때 `SimulateResponse`), `error`(failed 일 때).
같은 요청은 같은 작업을 돌려준다 (끝났으면 결과 재사용). 없는 제어기·시나리오 → 422.

### `GET /api/v1/simulate/jobs/{job_id}` — 작업 상태·결과
위와 같은 `SimulationJob`. 없는 작업 → 404 (서비스를 다시 띄우면 작업 목록이 비워진다).

### `POST /api/v1/simulate` — 폐루프 시뮬레이션 (끝날 때까지 대기)
짧은 계산·스크립트용. 요청은 위와 같고 응답 `SimulateResponse`:
`start`, `hours`, `controller`, `scenario`, `scenario_label`, `strategy`, `summary`, `timeseries{}`, `warnings[]`.
`summary`: `energy_cost_krw`, `facility_energy_kwh`, `cooling_energy_kwh`, `fan_extra_energy_kwh`, `it_energy_kwh`,
`average_pue`, `average_xfactor`, `average_setpoint_c`, `max_room_temp_c`, `min_room_temp_c`,
`temp_maintenance_pct`(18~27℃ 유지율), `temp_ok`(27℃ 이하), `min_ride_through_min`, `violation_hours`,
`capacity_limited_hours`, `mode_hours{}`, `final_tes_soc_kwh`, `diverged`, `hours_to_critical`.
서버실 온도가 위험 온도(32℃)에 닿으면 그 시각에서 멈춘다: `hours_to_critical` 에 시작부터 걸린 시간(아니면 null),
`hours` 는 실제 진행한 시간, `warnings[]` 에 `critical_temp_reached: …`. 그 뒤는 서버 보호 정지 영역이라 계산하지 않는다.
`timeseries{}`: 컬럼 → 시간별 값 (`timestamp`, `mode`, `setpoint_c`, `room_temp_c`, `it_power_kw`, `cooling_power_kw`,
`facility_power_kw`, `pue`, `cooling_mode`, `tariff_period`, `price_krw_per_kwh`, `energy_cost_krw`, `tes_kw`, `tes_soc_kwh` 등).
데이터 범위 밖 → 422, twin 장애 → 503.

### `GET /api/v1/scenarios`
위기 시나리오 목록: `name`, `label`, `strategy`(대응 전략), `it_scale`, `outdoor_min_c`, `capacity_factor`.

| name | 조건 |
|---|---|
| `normal` | 변경 없음 |
| `server_surge` | IT 부하 1.3배 |
| `chiller_failure` | 칠러 4대 중 1대 탈락 (냉각 용량 75%) |
| `heatwave` | 외기 최저 38℃ |
| `heatwave_chiller` | 복합: 외기 최저 38℃ + 칠러 1대 탈락 |
| `surge_heatwave` | 복합: IT 부하 1.3배 + 외기 최저 38℃ |
| `triple_crisis` | 복합: IT 부하 1.3배 + 외기 최저 38℃ + 칠러 1대 탈락 |
| `chiller_double_failure` | 칠러 4대 중 2대 탈락 (냉각 용량 50% < 발열). 위험 온도 도달 시 중단 |

`config/settings.yaml` 의 `scenarios` 에 `it_scale`·`outdoor_min_c`·`failed_chillers` 를 함께 적으면 복합 위기가 된다.

### `GET /api/v1/sites`
지역별 비교 (미리 계산한 결과, `output/sites/`). 같은 가상 데이터센터(같은 IT 부하)를 서울·춘천·제주의 외기(기상청 ASOS 실측)에 놓았을 때다.
- `summary{}`: 지역 → `label`, `station_id`, 외기 평균·최저·최고, 습도, 냉각 방식 비율(`free_cooling_pct`, `hybrid_pct`, `chiller_pct`),
  `average_pue`, `cooling_energy_mwh`, `monthly[]`(월별 외기·외기 냉각 비율·PUE) — `scripts/build_dataset.py`
- `comparison`: `full_period{}`(전체 기간 20℃ 고정·Rule-based PID 의 요금·PUE), `windows[]`(계절별 1주일 MPC 요금 절감률) —
  `scripts/compare_sites.py`. 아직 실행하지 않았으면 null. 요약 파일이 없으면 404.

### `GET /api/v1/results?scenario=base`
`scripts/simulate_mpc.py` 로 미리 계산한 장기 비교 실험 (`output/mpc/metrics_{scenario}.json`, `timeseries_{scenario}.csv`).
`base` = Test 14일 제어기 6종 비교, `surge` = 서버 급증 3일. 응답: `start`, `hours`, `scenario`,
`controllers{}`(제어기별 요약), `timeseries{}`(제어기별 시간별 값). 없는 결과 → 404.
