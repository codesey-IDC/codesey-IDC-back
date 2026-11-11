# codesey-IDC-back

데이터센터 냉각 제어 백엔드. 서비스 3개가 각자 FastAPI 로 떠 있고 **REST API 로만** 서로 호출한다.

| 모듈 | 폴더 | 포트 | 하는 일 |
|---|---|---|---|
| 디지털 트윈 | `services/twin/` | 8001 | 가상 데이터센터. 부하·외기·설정 온도 → 냉각 전력, 서버실 온도, PUE (물리 공식) |
| 예측 | `services/forecast/` | 8002 | 최근 IT 부하 → 앞으로 1~168시간 IT 부하 + 90% 범위 (LightGBM) |
| MPC | `services/mpc/` | 8003 | 예측 + 트윈으로 24시간 계획을 세워 지금 적용할 공급 공기 설정 온도 결정 |
| 대시보드 | `dashboard/` | 8501 | 통합 관제 화면 (Streamlit). 세 서비스를 REST 로만 호출 |

```
대시보드 ──▶ MPC (:8003) ──▶ 예측 (:8002)
    │              └───────▶ 트윈 (:8001)
    └──────────────────────▶ 트윈 / 예측 직접 호출도 가능
```

자세한 모델 설명
- 예측: `output/forecast/README_forecast.txt`
- 트윈: `services/twin/physics.py` 상단 주석 (공식, 데이터 복원값 vs 가정값)
- MPC: `services/mpc/controller.py` 상단 주석 (목적 함수, 제약, 안전장치)

---

## 1. 실행

Python 3.10 ~ 3.12.

Windows 에서는 `pip install -r requirements.txt` 후 **`start.bat`** 을 더블클릭하면 서비스 3개와 대시보드가
차례로 뜨고 브라우저가 열린다. 끌 때는 **`stop.bat`**. 직접 띄우려면 아래 명령을 쓴다.

```bash
pip install -r requirements.txt

# 터미널 3개 (저장소 루트에서). MPC 는 나머지 둘을 부르므로 마지막에 띄운다.
uvicorn services.twin.main:app --port 8001
uvicorn services.forecast.main:app --port 8002
uvicorn services.mpc.main:app --port 8003

# 대시보드 (세 서비스를 띄운 뒤) → http://127.0.0.1:8501
streamlit run dashboard/app.py
```

- 각 서비스의 요청·응답 형식과 직접 호출 화면: `http://127.0.0.1:<포트>/docs`
- MPC 가 부르는 주소는 환경변수로 바꿀 수 있다: `TWIN_URL`, `FORECAST_URL` (기본 `http://127.0.0.1:8001`, `:8002`)
- **주소는 `localhost` 대신 `127.0.0.1` 을 쓸 것.** Windows 에서 `localhost` 는 연결마다 약 2초 지연된다
  (MPC 한 번 호출이 0.3초 → 2~6초).

테스트: `python -m pytest -q` (76개)

---

## 2. 대시보드에서 쓰는 법 (Streamlit)

세 서비스 모두 **상태를 저장하지 않는다.** 요청에 필요한 값을 다 넣어 보내면 계산만 해서 돌려준다.
지나간 기록(실측 데이터, 시뮬레이션 결과)은 파일로 있고(3절), 대시보드는 2-6 의 API 로 받는다.

```python
import httpx
import pandas as pd

TWIN, FORECAST, MPC = 'http://127.0.0.1:8001', 'http://127.0.0.1:8002', 'http://127.0.0.1:8003'
client = httpx.Client(timeout=30)   # 연결 재사용: 매번 httpx.get() 보다 빠름

# 공통 준비: 1시간 단위 실측 데이터
hourly = (pd.read_parquet('data/synthetic_idc_90d.parquet').set_index('timestamp')
          [['it_power_kw', 'outdoor_temp_c']].resample('h').mean())
```

### 2-1. 상태 확인

```python
for url in (TWIN, FORECAST, MPC):
    print(client.get(f'{url}/health').json())
# {'status': 'ok', 'service': 'twin'}
# {'status': 'ok', 'service': 'forecast', 'model': 'lightgbm'}   ← 'unavailable' 이면 모델 파일 없음
# {'status': 'ok', 'service': 'mpc', 'forecast_url': ..., 'twin_url': ...}
```

### 2-2. 예측: `POST :8002/api/v1/forecast`

| 요청 | 설명 |
|---|---|
| `horizon_hours` | 1~168 (24 이하는 24h 모델, 그 이상은 168h 모델) |
| `target` | `'it_load'`(기본) 또는 `'cooling_load'` (값은 같음) |
| `history` | 선택. `[{timestamp, it_power_kw}, ...]` 5분 또는 1시간 간격, **최소 168시간**. 생략하면 학습 데이터 끝(6/12 23시) 기준 |

```python
hist = hourly.it_power_kw.loc['2026-05-27 06:00':'2026-06-03 05:00']        # 168시간
r = client.post(f'{FORECAST}/api/v1/forecast', json={
    'horizon_hours': 24,
    'history': [{'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()],
}).json()
df = pd.DataFrame({'예측': r['predictions'], '하한': r['pi_lower'], '상한': r['pi_upper']},
                  index=pd.to_datetime(r['timestamps']))
```

| 응답 | 설명 |
|---|---|
| `origin_time` | 예측 시점 (이력의 마지막 시각) |
| `timestamps`, `predictions` | 앞으로 h시간의 시각과 예측 중앙값 (kW) |
| `pi_lower`, `pi_upper` | 90% 범위 |
| `in_training_range`, `warnings` | 학습 범위 밖 입력이면 `false` + 경고 |

오류: 이력 168시간 미만·중간에 빈 시간 → 422, 모델 없음 → 503.

### 2-3. 트윈: 한 시점 계산 `POST :8001/api/v1/simulate`

"부하 X, 외기 Y, 설정 Z 이면 어떻게 되나". 정상 상태(서버실 = 설정 온도) 기준.

```python
r = client.post(f'{TWIN}/api/v1/simulate', json={'points': [
    {'it_power_kw': 3000, 'outdoor_temp_c': 25},                        # supply_temp_c 생략 = 기준 20℃
    {'it_power_kw': 3000, 'outdoor_temp_c': 25, 'supply_temp_c': 24},
    {'cpu_utilization': 0.25, 'outdoor_temp_c': 10},                    # IT 부하 대신 CPU 사용률도 가능
]}).json()['results']
```

| 응답 (점마다) | 설명 |
|---|---|
| `it_power_kw`, `server_fan_extra_kw` | 실제 IT 전력, 그중 서버실 24℃ 초과로 늘어난 팬 전력 |
| `cooling_mode`, `free_cooling_fraction`, `chiller_cop` | `free_cooling` / `hybrid` / `chiller`, 외기 냉각 비율, 칠러 효율 |
| `cooling_power_kw`, `facility_power_kw`, `pue` | 냉각 전력, 전체 전력(IT+냉각+기타), PUE |
| `ride_through_min` | 냉각이 멈추면 32℃ 까지 버티는 시간(분). 15분 미만이면 위험 |
| `xfactor` | 상대 고장률 (20℃ = 1.0) |

화면 아이디어: 외기 온도 슬라이더 + 설정 온도 슬라이더 → 냉각 전력/PUE 즉시 표시.

### 2-4. 트윈: 계획 평가 `POST :8001/api/v1/rollout`

시간별 설정 온도 계획을 넣으면 서버실 온도가 시간에 따라 어떻게 변하는지 계산.

```python
r = client.post(f'{TWIN}/api/v1/rollout', json={
    'it_power_kw':   [3000] * 24,
    'outdoor_temp_c': [22] * 24,
    'supply_temp_c': [20] * 6 + [25] * 18,      # 계획
    'initial_room_temp_c': 20,                   # 생략하면 첫 설정 온도에서 시작
    'dt_hours': 1.0,                             # 구간 길이
    'capacity_factor': 1.0,                      # 0.5 = 칠러 일부 고장 시나리오
}).json()
```

응답: 구간별 배열(`room_temp_c`, `cooling_power_kw`, `facility_power_kw`, `pue`, `cooling_mode`,
`ride_through_min`, ...) + 합계(`total_facility_energy_kwh`, `average_pue`, `max_room_temp_c`,
`violation_steps`, ...) + `warnings`.

`POST :8001/api/v1/rollout/batch` 는 계획 여러 개(`plans`)의 합계만 돌려준다 (MPC 내부용).
`GET :8001/api/v1/params` 는 트윈이 쓰는 상수 전체 (허용 온도 범위 등).

### 2-5. MPC: 설정 온도 결정 `POST :8003/api/v1/plan`

| 요청 | 설명 |
|---|---|
| `room_temp_c` | 지금 서버실 온도 |
| `history` | 최근 IT 부하 `[{timestamp, it_power_kw}, ...]`, **최소 168개** (1시간 기준 168시간) |
| `outdoor_forecast_c` | 앞으로 24시간 외기온도 예보, **정확히 24개** |
| `capacity_factor` | 선택, 냉각 용량 배율 |

```python
now = pd.Timestamp('2026-06-03 06:00')
hist = hourly.it_power_kw.loc[now - pd.Timedelta(hours=192):now - pd.Timedelta(hours=1)]
r = client.post(f'{MPC}/api/v1/plan', json={
    'room_temp_c': 20.0,
    'history': [{'timestamp': t.isoformat(), 'it_power_kw': float(v)} for t, v in hist.items()],
    'outdoor_forecast_c': hourly.outdoor_temp_c.loc[now:now + pd.Timedelta(hours=23)].tolist(),
}).json()
print(r['mode'], r['setpoint_c'], f"{r['expected_saving_pct']:.1f}%")   # mpc 24.6 7.7%
```

| 응답 | 설명 |
|---|---|
| `mode` | `mpc` 정상 / `mpc_naive` 예측을 못 믿어 "최근 3시간 최대 부하 유지" 가정 / `fallback` 트윈 장애로 기준 20℃ |
| `setpoint_c` | **지금 적용할 설정 온도** (계획의 첫 시간) |
| `plan_c`, `timestamps` | 24시간 계획 (참고용. 다음 시간에 다시 계산한다) |
| `forecast_it_kw` | MPC 가 쓴 부하 예측 |
| `predicted_room_temp_c`, `predicted_facility_kw` | 계획대로 했을 때 예상 서버실 온도·전체 전력 |
| `expected_facility_energy_kwh`, `baseline_facility_energy_kwh`, `expected_saving_pct` | 24시간 예상 전력량, 기준 20℃ 고정 시, 절감률 |
| `warnings` | `forecast.*`, `twin.*` 경고, mode 전환 이유 |

한 번 호출에 약 0.3초. 오류: 이력 부족·외기 개수 틀림 → 422.

---

### 2-6. 대시보드용 API (`dashboard/app.py` 가 쓰는 것)

대시보드는 parquet/CSV 를 직접 읽지 않고 아래 API 만 호출한다.

| API | 하는 일 |
|---|---|
| `GET :8001/api/v1/telemetry?start=&hours=` | 시간별 계측값(IT 부하·외기). `start` 생략 시 데이터 끝에서 `hours` 시간. 응답에 데이터 전체 범위(`data_start`, `data_end`) 포함 |
| `POST :8001/api/v1/esg` | 시뮬레이션 결과 시계열(`it_power_kw`, `cooling_power_kw`, `cooling_mode`) → 탄소 배출(0.459 tCO2/MWh), CUE, 물 사용량, WUE, 냉각 방식별 물 사용량 |
| `POST :8003/api/v1/simulate` | 폐루프 시뮬레이션. `start`, `hours`(최대 168), `controller`(`fixed_20`/`reactive`/`mpc`), `scenario` → 시간별 결과 + 요약 |
| `GET :8003/api/v1/scenarios` | 위기 시나리오 목록과 조건 |
| `GET :8003/api/v1/results?scenario=base` | 3절의 사전 계산 결과(`output/mpc/`)를 JSON 으로 |

위기 시나리오 (`scenario`, 시작 시각부터 구간 끝까지 적용)

| 값 | 조건 |
|---|---|
| `normal` | 변경 없음 |
| `server_surge` | IT 부하 1.3배 |
| `chiller_failure` | 냉각 용량 75% (`capacity_factor=0.75`) |
| `heatwave` | 외기온도 최저 38℃ |

```python
r = client.post(f'{MPC}/api/v1/simulate', json={
    'start': '2026-06-04T00:00:00', 'hours': 24, 'controller': 'mpc', 'scenario': 'heatwave'}, timeout=300).json()
print(r['summary']['max_room_temp_c'], r['summary']['temp_ok'], r['summary']['average_pue'])
ts = pd.DataFrame(r['timeseries'])      # timestamp, setpoint_c, room_temp_c, cooling_power_kw, pue, cooling_mode, ...
```

- `start` 는 데이터 시작 + 192시간 이후여야 한다 (예측 이력). 범위를 벗어나면 422.
- MPC 는 1시간당 약 0.3초 걸린다 (24시간 ≈ 7초).
- 물 원단위(칠러 1.8 · 혼합 1.0 · 외기 냉각 0.3 L/kWh)는 가정값이다 (`services/twin/esg.py`).

---

## 3. 파일로 있는 결과

| 파일 | 내용 | 만드는 명령 |
|---|---|---|
| `data/synthetic_idc_90d.parquet` | 90일 실측(합성) 데이터, 5분 단위 | — |
| `output/mpc/metrics_base.json` | Test 14일 제어기 4개 비교 요약 | `python simulate_mpc.py` |
| `output/mpc/timeseries_base.csv` | 같은 실험의 시간별 기록 (제어기 × 336시간) | 〃 |
| `output/mpc/metrics_surge.json`, `timeseries_surge.csv` | 서버 급증 시나리오 (6/5 10~15시 부하 1.3배) | `python simulate_mpc.py --surge --start "2026-06-04 00:00" --days 3` |
| `output/forecast/test_pred_24h.parquet`, `test_pred_168h.parquet` | 예측 모델 Test 구간 예측 vs 실제 | `python train_forecast.py` |
| `output/forecast/metrics.json` | 예측 성능 (MAPE, 적중률) | 〃 |

`timeseries_*.csv` 컬럼

| 컬럼 | 설명 |
|---|---|
| `timestamp`, `controller` | 시각, `fixed_20` / `fixed_26` / `reactive` / `mpc` |
| `mode`, `setpoint_c` | MPC 모드, 그 시간에 적용한 설정 온도 |
| `it_base_kw`, `outdoor_temp_c` | 실제 IT 부하(팬 제외), 외기 |
| `room_temp_c`, `it_power_kw`, `fan_extra_kw` | 시간 끝 서버실 온도, 실제 IT 전력, 팬 증가분 |
| `cooling_power_kw`, `facility_power_kw`, `pue` | 냉각·전체 전력, PUE |
| `ride_through_min`, `xfactor_avg`, `violation` | 비상 여유 시간, 고장률, 위반 여부 |

```python
ts = pd.read_csv('output/mpc/timeseries_base.csv', parse_dates=['timestamp'])
pivot = ts.pivot(index='timestamp', columns='controller', values='facility_power_kw')   # 제어기별 전력 비교 그래프
```

### Test 14일 결과 요약 (`metrics_base.json`)

| 제어 | 전체 전력 절감 (기준 20℃ 대비) | PUE | 위반 | 설정 변경 총량 |
|---|---|---|---|---|
| 기준 20℃ 고정 | — | 1.311 | 0 | 0℃ |
| 26℃ 고정 | 7.3% | 1.199 | 0 | 6℃ |
| 반응형 (예측 없음, 그 순간 최적) | 8.3% | 1.187 | 0 | 269℃ |
| MPC | 8.3% | 1.187 | 0 | 119℃ |

---

## 4. 주의

- **가정값**: 트윈의 열용량, 팬 전력, 비상 기준 등은 데이터에 없는 가정값이다 (`services/twin/physics.py` 의 `TwinParams`).
  화면에 결과를 보여줄 때 "가정 기반 시뮬레이션" 임을 표시할 것.
- **외기 예보**: 시뮬레이션은 실제 외기를 예보로 썼다 (예보 오차 없음).
- **데이터 범위**: 예측 모델 학습 범위는 IT 부하 2,389~3,457kW. 밖이면 예측 서비스가 경고하고 MPC 는 `mpc_naive` 로 바뀐다.
- **목표는 PUE 가 아니라 전체 전력**: 설정 온도를 너무 올리면 팬 전력(IT 로 잡힘) 때문에 PUE 는 좋아져도 전체 전력은 늘어난다.
