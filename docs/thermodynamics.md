# 열역학 모델 수식 (디지털 트윈)

구현: `services/twin/physics.py` (열·전력), `services/twin/esg.py` (ESG), `services/twin/tariff.py` (요금).
상수: `config/settings.yaml` 의 `datacenter`, `esg`, `tariff`. 라이브러리 없이 직접 구현했다.

표기: **[복원]** = 원본 데이터(`data/synthetic_idc_90d.parquet`, 90일·5분 단위·25,920행. 지금 기본 데이터 `data/idc_2026_chuncheon.parquet` 의 첫 90일)에서 역으로 찾은 값.
**[보정]** = Sinergym(EnergyPlus) 1년 결과에 맞춘 값 (11장). 냉각 효율 계수 4개가 여기에 해당한다.
**[가정]** = 데이터에 없어 문헌·상식으로 정한 값.

데이터의 냉각 전력·PUE 두 열은 [보정] 계수로 다시 계산해 두었다 (`scripts/recompute_cooling_columns.py`).
그래서 이 문서의 공식으로 계산하면 데이터의 IT 전력·냉각 전력·냉각 모드·PUE 가 **오차 없이** 나온다
(`tests/test_twin.py::test_static_reproduces_90d_data`).

## 0. 에너지 흐름

```
CPU 사용률 ─SPECpower─▶ IT 전력 ─▶ 서버 발열 ─▶ 서버실 공기 ─(CRAH, Q = ṁ c_p ΔT)─▶ 냉각 설비 ─▶ 냉각 전력
                                     ▲                                          │ 외기 냉각 / 칠러 / 축열조
                          서버 팬 증가분 (서버실 > 24℃)                          ▼
                                                                    전체 전력 = IT + 냉각 + 기타 → PUE, 탄소, 물, 요금
```

## 1. 서버 전력 (SPECpower 선형 모델)

$$P_{IT} = N \cdot \left[P_{idle} + (P_{max} - P_{idle}) \cdot u\right] / 1000 \quad [\text{kW}]$$

| 기호 | 값 | 설명 |
|---|---|---|
| $N$ | 10,000 대 [복원] | 서버 수 |
| $P_{idle}$ | 215 W [복원] | CPU 0% 전력 |
| $P_{max}$ | 575 W [복원] | CPU 100% 전력 |
| $u$ | 0~1 | CPU 사용률 (Google Cluster Trace 2019 기반) |

→ $P_{IT} = 2{,}150 + 3{,}600\,u$ kW. 규모는 한전 데이터센터 현황(국내 평균 계약전력 약 13MW)과 비교해 중소규모 IDC.

### 서버 팬·누설 전력 (설정 온도를 올릴 때의 단점) [가정]

$$r_{fan} = k_{fan} \cdot \min\big(\max(T_{room} - 24,\ 0),\ T_{crit} - 24\big)^2, \qquad P_{IT}^{actual} = P_{IT} \cdot (1 + r_{fan})$$

팬은 위험 온도 $T_{crit}$ = 32℃ 에서 최고 속도에 닿는다고 보고 그 위로는 늘지 않는다 (상한 +21.3%).
상한이 없으면 냉각 용량이 발열보다 작을 때(칠러 2대 탈락) "온도 상승 → 팬 전력 증가 → 발열 증가"가 끝없이 되먹임되어
서버실 온도가 수만 ℃ 로 발산했다. 27℃ 이하 정상 운전 결과는 상한과 무관하다.

$k_{fan} = 1/300$ → 서버실 27℃ 에서 IT 전력 +3% (문헌 범위 2~4%). 늘어난 전력도 열이 되어 냉각 부하에 더해진다.
서버 발열 $Q_{IT} = P_{IT}^{actual}$ (전기 에너지가 모두 열이 된다고 본다).

## 2. 냉각 부하 — 공기측 열수지 $Q = \dot{m}\, c_p\, \Delta T$

CRAH 가 서버를 지나 데워진 공기(환기)를 공급 설정 온도 $T_{supply}$ 까지 식힌다.

$$\Delta T_{rack} = \frac{Q_{IT}}{\dot{m}\, c_p}, \qquad T_{return} = T_{room} + \Delta T_{rack}$$

$$Q_{cool} = \dot{m}\, c_p\, (T_{return} - T_{supply}) = Q_{IT} + \dot{m}\, c_p\,(T_{room} - T_{supply})$$

| 기호 | 값 | 설명 |
|---|---|---|
| $\dot{m}$ | 250 kg/s [가정] | CRAH 공기 유량. IT 3,000kW 에서 $\Delta T_{rack}$ ≈ 12℃ (일반 10~15℃) |
| $c_p$ | 1.005 kJ/(kg·K) [물성] | 공기 정압비열 → $\dot{m} c_p$ = 251.25 kW/℃ |
| $T_{supply}$ | 16~27℃, 기준 20℃ | **제어 변수** (공급 공기 설정 온도) |

$Q_{cool}$ 은 냉각 설비 용량 $Q_{max} = 5{,}000\,\text{kW} \times$ 용량 배율 [가정] (+ 축열조 방전분)으로 자른다.
정상 상태($T_{room} = T_{supply}$)에서는 $Q_{cool} = Q_{IT}$.

## 3. 서버실 온도 (집중 열용량 모델)

$$C \frac{dT_{room}}{dt} = Q_{IT} - Q_{cool}$$

$C$ = 200 kWh/℃ [가정] (공기 + 구조체 + 장비 유효 열용량). 5분 간격 오일러 적분.
위 두 식을 합치면 $T_{room}$ 은 시상수 $C / (\dot{m} c_p)$ ≈ 0.8시간으로 $T_{supply}$ 를 따라간다
(`tests/test_twin.py::test_room_converges_to_new_setpoint` 가 해석해와 비교).

허용 범위: $18 \le T_{room} \le 27$℃ (ASHRAE TC 9.9 Class A1 권장 범위).

## 4. 냉각 방식과 효율

설정 온도를 기준(20℃)보다 $\Delta$ 올리면 외기와의 온도차가 $\Delta$ 줄어든 것과 같다고 본다 [가정]:

$$T_{eff} = T_{out} - (T_{supply} - 20)$$

### 외기 냉각(Free Cooling) 비율 — 외기 15℃ 이하 자연공조 [복원]

$$f = \text{clip}\!\left(\frac{18 - T_{eff}}{18 - 15},\ 0,\ 1\right)$$

$f = 1$ → `free_cooling`, $0 < f < 1$ → `hybrid`, $f = 0$ → `chiller`.

### 칠러 COP — 외기 온도의 함수 [보정]

$$COP_{chiller} = \max\big(5.856 - 0.459\,(T_{eff} - 18),\ 3.261\big), \qquad COP_{free} = 17.105$$

원래 데이터에서 복원한 값은 $\max(4.8 - 0.4\,(T_{eff} - 18),\ 2.7)$, $COP_{free} = 13$ 이었다.
Sinergym 과 비교하니 냉각 전력을 23% 크게 잡아, Sinergym 1년 결과에 맞춰 다시 정했다 (11장).

### 냉각 전력

$$P_{cool} = \underbrace{\frac{f\,Q_{cool}}{COP_{free}}}_{\text{외기 냉각}} + \underbrace{\frac{(1-f)\,Q_{cool} - D}{COP_{chiller}}}_{\text{칠러}} + \underbrace{\frac{C_{chg}}{COP_{chiller} \cdot \eta_{chg}}}_{\text{축열 충전}}$$

$D$, $C_{chg}$ 는 5절의 축열조 방전·충전 (쓰지 않으면 0 → 원본 데이터와 같은 식).

## 5. 냉수 축열조 (TES) [가정]

$$SOC_{t+h} = SOC_t + (C_{chg} - D - \lambda\, SOC_t)\, h$$

| 기호 | 값 | 제약 |
|---|---|---|
| 용량 $SOC_{max}$ | 12,000 kWh (IT 3,000kW 냉각 약 4시간, 물 ΔT 8℃ 기준 약 1,300㎥) | $0 \le SOC \le SOC_{max}$ |
| 충전 $C_{chg}$ | ≤ 2,000 kW | 칠러 남는 용량 $Q_{max} - (Q_{cool} - D)$ 이내 |
| 방전 $D$ | ≤ 3,000 kW | 칠러 몫 $(1-f)\,Q_{cool}$ 이내 (외기 냉각분은 대신하지 않음) |
| 충전 효율 $\eta_{chg}$ | 0.9 | 저장용 냉수를 더 차갑게 만들어야 해서 COP 저하 |
| 자연 손실 $\lambda$ | 0.2 %/h | |

## 6. 안전·신뢰성 지표

비상 여유 시간 (냉각이 멈췄을 때 위험 온도 32℃ (ASHRAE A1 허용 최대) 까지):

$$t_{ride} = \frac{C\,(32 - T_{room})}{Q_{IT}} \times 60 \quad [\text{분}] \ \ge 15 \text{분 [가정]}$$

상대 고장률 x-factor (ASHRAE TC 9.9, 20℃ = 1.00, 선형 보간): 15℃ 0.72 · 20℃ 1.00 · 25℃ 1.24 · 27.5℃ 1.34 · 30℃ 1.42 · 35℃ 1.55.

## 7. 효율 지표 PUE · WUE · CUE

$$P_{facility} = P_{IT}^{actual}\,(1 + \alpha) + P_{cool}, \qquad \alpha = 0.05 \ \text{[복원, 조명·UPS 등 기타 손실]}$$

$$PUE = \frac{E_{facility}}{E_{IT}}, \qquad CUE = \frac{CO_2}{E_{IT}}\ [\text{tCO}_2/\text{MWh}], \qquad WUE = \frac{W}{E_{IT}}\ [\text{L/kWh}]$$

$$CO_2 = E_{facility}\,[\text{MWh}] \times 0.459\ \text{tCO}_2/\text{MWh}\ \text{(과제 지정 배출계수)} \ \Rightarrow\ CUE = 0.459 \times PUE$$

$$W = \sum Q_{cool} \cdot w_{mode} \cdot \Delta t, \qquad w_{chiller} = 1.8,\ w_{hybrid} = 1.0,\ w_{free} = 0.3\ \text{L/kWh [가정, 냉각탑 증발]}$$

세 지표의 관계: 같은 IT 부하에서 냉각 전력을 줄이면 PUE·CUE 가 함께 내려가고(CUE = 0.459 × PUE),
외기 냉각 비율을 늘리면 물 사용(냉각탑 증발)이 줄어 WUE 도 내려간다.
※ 서버 팬 전력은 IT 로 잡혀서, 설정 온도를 너무 올리면 **PUE 는 좋아져도 전체 전력은 늘어날 수 있다**
(`tests/test_twin.py::test_pue_can_improve_while_total_energy_worsens`). 그래서 MPC 목표는 PUE 가 아니라 전체 전력(요금)이다.

## 8. 전기요금

$$\text{Cost} = \sum_t P_{facility,t}\ \Delta t \cdot \pi_t \quad [\text{원}]$$

$\pi_t$: 한전 **일반용전력(을) 고압A 선택Ⅲ** (2026-06-01 시행, `docs/references/kepco_tariff_general_2026.pdf`).
데이터센터는 산업용이 아니라 일반용 요금을 적용받는다.

| 계절 | 경부하 (22~08시) | 중간부하 | 최대부하 |
|---|---|---|---|
| 여름 (6~8월) | 86.4 | 139.6 (08~15, 21~22시) | 209.9 (15~21시) |
| 봄·가을 (3~5, 9~10월) | 86.4 | 108.5 | 132.2 |
| 겨울 (11~2월) | 93.7 | 139.8 (08~09, 12~16, 19~22시) | 186.7 (09~12, 16~19시) |

단위 원/kWh. 토요일 최대부하 → 중간부하, 일요일·공휴일 → 경부하. 기본요금(9,810원/kW)·부가세·전력기금은
모든 제어기에 같게 붙어 비교에서 뺐다.

## 9. 대표 값과 벤치마크

IT 3,000kW, 기준 설정 20℃ (`python scripts/evaluate_cooling_forecast.py`):

| 냉각 방식 (외기) | PUE | 냉각 전력 | 하루 전체 전력량 |
|---|---|---|---|
| 외기 냉각 (5℃) | 1.108 | 175 kW | 79.8 MWh |
| 혼합 (16.5℃) | 1.156 | 317 kW | 83.2 MWh |
| 칠러 (25℃) | 1.357 | 920 kW | 97.7 MWh |

| 벤치마크 | PUE |
|---|---|
| NAVER 각 춘천 (SASB 보고서) | 1.09 |
| 본 모델 외기 냉각 운전 (5℃) | 1.108 |
| 본 모델 Test 14일 (6월 초, 대부분 칠러) — 20℃ 고정 / MPC | 1.262 / 1.148 |
| 그린데이터센터 인증 기준 / 글로벌 평균 / 국내 민간 평균 | 1.66 / 1.58 / 2.03 |

외기 냉각만으로 운전하는 시간대의 PUE 는 $1 + 1/17.105 + 0.05 = 1.108$ 로 NAVER 각 춘천(1.09)에 근접한다.
1.09 가 되려면 $\alpha + 1/COP_{free} = 0.09$ 여야 한다 (예: 기타 손실 3%, 외기 냉각 COP 17).
Test 구간은 6월이라 칠러 운전이 많아 PUE 가 높다.

## 10. 검증

| 항목 | 방법 | 결과 |
|---|---|---|
| 데이터와 일치 | 25,920행 전체를 이 문서의 공식으로 재계산 | IT 전력·냉각 전력·냉각 모드·PUE 오차 0 (냉각 전력·PUE 열은 [보정] 계수로 다시 계산한 값) |
| 에너지 보존 | $C\,\Delta T_{room} = \sum (Q_{IT} - Q_{cool})\,\Delta t$ | 일치 (`test_energy_balance`) |
| 동적 응답 | 설정 20→24℃ 계단 응답 vs 해석해 $24 - 4e^{-\dot m c_p t / C}$ | 0.15℃ 이내 |
| 물리 상식 | 설정↑ → 냉각↓, 냉각 정지 → 온도↑, 축열 방전 → 칠러 전력↓ 등 | 테스트 통과 (`tests/test_twin.py`, `tests/test_tes.py`) |
| Sinergym 비교 (오차 10% 이내) | Sinergym `Eplus-datacenter_cw` 1년과 같은 IT 전력·외기온도에서 비교 | 냉각 전력량 **-1.5%**, PUE **-0.2%** (둘 다 통과). 보정 전에는 +23.0%, +2.8% → 11장 |

## 11. Sinergym(EnergyPlus) 비교 검증

실행 (Docker, 약 2분): `docker compose --profile validation run --rm sinergym-validate`
결과 파일: `output/validation/sinergym_samples.csv` (Sinergym 시계열), `sinergym_report.json` (비교 결과)

### 방법

| 항목 | 내용 |
|---|---|
| Sinergym | 3.12.0, `Eplus-datacenter_cw-mixed-continuous-stochastic-v1` (seed 0) |
| 환경 이름 | 과제의 `Eplus-datacenter-mixed-...` 는 Sinergym 3.x 에서 `datacenter_cw`(냉수 칠러)와 `datacenter_dx`(직접팽창)로 나뉘었다. 우리 모델이 칠러 + 외기 냉각이라 `cw` 를 쓴다 |
| 건물 | 2ZoneDataCenterHVAC_wEconomizer_CW (전산실 2개, 외기 이코노마이저 + 냉수 칠러), IT 74~117kW |
| 날씨 | 뉴욕 JFK TMY3 1년 + 외기 잡음, 외기 -13.0~35.6℃ |
| 제어 | 전산실 냉방 설정 25℃, 냉수 공급 7.5℃ 고정 (행동 공간의 가운데). 우리 모델의 기준 운전(20℃)에 대응 |
| 같은 조건 | Sinergym 의 매 시각 **IT 전력과 외기온도**를 우리 모델(`simulate_static`)에 그대로 입력 |
| 비교 | 우리 모델의 냉각 전력 vs Sinergym 냉각 설비 전체 전력 (칠러 + 송풍기 + 펌프 + 냉각탑) |

규모가 다르지만(Sinergym 약 100kW, 우리 약 3,000kW) 우리 모델의 냉각 전력은 제거열에 비례하므로 그대로 비교할 수 있다.
IT 전력은 EnergyPlus 출력 변수(서버 CPU + 서버 팬 전력)를 추가해 직접 읽었다.

### 결과 (1년 8,760시간)

| 구분 | 자체 모델 | Sinergym | 오차율 | 판정 (±10%) | 보정 전 오차율 |
|---|---|---|---|---|---|
| 냉각 전력량 (MWh) | 121.7 | 123.5 | -1.5% | 통과 | +23.0% (미달) |
| PUE = (IT + 냉각) / IT | 1.135 | 1.137 | -0.2% | 통과 | +2.8% (통과) |

| 외기온도 | 시간 | Sinergym COP | 자체 COP | 냉각 전력 오차 | 보정 전 |
|---|---|---|---|---|---|
| 5℃ 미만 | 2,287 | 15.00 | 17.10 | -12.3% | +15.4% |
| 5~15℃ | 2,733 | 19.60 | 17.11 | +14.6% | +50.8% |
| 15~18℃ | 726 | 8.15 | 9.43 | -13.6% | +7.5% |
| 18~25℃ | 1,998 | 4.05 | 4.09 | -0.9% | +22.6% |
| 25℃ 이상 | 1,016 | 3.23 | 3.26 | -1.0% | +19.5% |

(COP = IT 전력량 / 냉각 전력량. 시간별 냉각 전력 nMAE 는 24.5%, 보정 전 34.4%.
기타 손실까지 넣은 PUE 는 자체 1.185, Sinergym 1.237 로 -4.2% 차이: Sinergym 은 UPS 손실이 IT 의 10%, 우리는 기타 손실 5%)

### 보정 과정

- **보정 전에는 냉각 전력을 모든 외기 구간에서 크게 잡았다 (1년 합계 +23.0%).** 원인은 냉각 효율 계수였다.
  원래 계수(외기 냉각 COP 13, 칠러 COP 4.8 등)는 원본 데이터에서 복원한 값인데, 그 데이터를 만든 이전 작업은
  IT 전력을 "CPU 부하율 × 설계 전력 98.27kW" 로 어림해 Sinergym 에 맞췄다. EnergyPlus 의 실제 IT 전력은 그보다 크다
  (평균 79kW 가 아니라 103kW). 제거할 열을 작게 보고 맞췄으니 효율이 실제보다 나쁘게 정해졌다.
- **계수 4개를 Sinergym 의 시간별 냉각 전력에 맞춰 다시 정했다** (`scripts/validate_sinergym.py` 의 `calibrate`, 오차 제곱 최소).
  외기 냉각 COP 13→17.105, 칠러 COP(18℃) 4.8→5.856, 기울기 0.4→0.459, 최소 2.7→3.261.
  외기 냉각 전환 온도(15℃, 18℃)는 바꾸지 않았다.
- **1년 합계는 맞지만 구간별로는 ±15% 까지 어긋난다.** 계수 4개짜리 단순한 식이라 5℃ 미만(냉각탑 동결 방지 히터)과
  5~15℃(외기 냉각이 가장 효율적인 구간)의 차이를 따로 표현하지 못한다. 칠러 구간(18℃ 이상)은 1% 이내로 맞는다.
- **함께 바꾼 것**: 데이터의 냉각 전력·PUE 열 (`scripts/recompute_cooling_columns.py`), 제어기 비교·민감도 분석·냉각 수요 예측
  평가 결과 (`output/`), MPC 탐색량 (`de_popsize` 6→15, `de_maxiter` 25→60).

### 참고: 설정 온도를 바꿨을 때 (판정 대상 아님)

| 설정 온도 | 냉각 전력 (Sinergym / 자체) | IT 전력 | 전체 전력 |
|---|---|---|---|
| -3℃ | +7.0% / +16.5% | -13.2% / 0.0% | -10.9% / +1.9% |
| +3℃ | -15.3% / -16.9% | +9.9% / 0.0% | +7.1% / -1.9% |

- 설정 온도를 올리면 냉각 전력이 주는 정도는 비슷하다 (-15.3% vs -16.9%).
- **IT 전력은 크게 다르다.** Sinergym 의 서버 모델은 흡기 온도가 오르면 서버 팬 전력이 크게 늘어 IT 전력이 +9.9% 가 되고,
  전체 전력은 오히려 +7.1% 늘어난다. 우리 모델은 24℃ 까지 팬 전력 증가가 없다고 가정해 전체 전력이 1.9% 준다.
- 즉 "설정 온도를 올리면 전체 전력이 준다"는 MPC 의 절감 방향은 서버 팬 전력 가정(`fan_threshold_c`, `fan_power_coef_per_c2`)에
  달려 있다. Sinergym 건물의 서버는 설계 전력의 40% 가 팬인 모델이라 온도에 유난히 민감하다는 점은 감안해야 한다.
