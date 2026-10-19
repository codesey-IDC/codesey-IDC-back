==============================================================
 IT 부하 예측 모델 1차 학습 정리
==============================================================

이 문서는 train_forecast.py 로 학습한 결과물을 받는 사람
(예측 함수 / 시각화 담당)이 바로 이어서 작업할 수 있도록 정리한 것.


--------------------------------------------------------------
1. 한 줄 요약
--------------------------------------------------------------
과거 IT 부하만 보고 앞으로 24시간 / 168시간의 IT 부하를
"중앙값 + 90% 예측구간" 형태로 예측하는 LightGBM 모델을 만들었다.

  예) 15시 부하는 2,970kW, 90% 확률로 2,910 ~ 3,035kW 사이

냉각 부하(cooling_load_kw)는 데이터에서 IT 부하와 값이 완전히 같으므로
IT 부하 예측값 = 냉각 부하 예측값 이다. (별도 모델 없음)


--------------------------------------------------------------
2. 실행 방법
--------------------------------------------------------------
  pip install -r requirements.txt
  python train_forecast.py          # 약 30초, CPU만 사용

결과는 전부 output/forecast/ 에 저장된다.


--------------------------------------------------------------
3. 사용한 데이터
--------------------------------------------------------------
  파일   : data/synthetic_idc_90d.parquet  (5분 단위, 25,920행)
  기간   : 2026-03-15 ~ 2026-06-12 (90일)
  사용   : it_power_kw 컬럼 하나만 1시간 평균 → 2,160행

  사용하지 않은 컬럼과 이유
    - cooling_load_kw, server_heat_kw : it_power_kw 와 값이 동일 (정답 누출)
    - cpu_utilization                 : it_power_kw 를 계산한 원천 (정답 누출)
    - pue, cooling_power_kw, cooling_mode, free_cooling_fraction
                                      : 냉각 제어 결과값 (사후 산출)
    - 외기온도/습도/풍속              : Val 에서 넣으면 오히려 성능 하락
                                        (24h 0.98→1.02%, 168h 0.94→1.03%)
                                        → USE_WEATHER_FEATURES = False


--------------------------------------------------------------
4. 학습 방식
--------------------------------------------------------------
[문제 형태]
  "예측 시점 t 에 서서, h 시간 뒤(t+h)의 IT 부하를 맞힌다" = 학습 데이터 1행
  24h 모델은 h = 1~24, 168h 모델은 h = 1~168

[피처]  (모두 예측 시점 t 이전에 알 수 있는 값만 사용, 모델 입력 순서대로)
  h                 몇 시간 뒤를 예측하는지
  last_value        t 시점의 IT 부하
  rolling_mean_24h  t 까지 최근 24시간 평균
  lag_168           예측 대상 시각의 168시간 전 값
  lag_24            예측 대상 시각의 24시간 전 값   ※ 24h 모델에만 있음
  hour              예측 대상 시각의 시 (0~23)
  dayofweek         예측 대상 시각의 요일 (0=월 ~ 6=일)
  is_weekend        예측 대상 시각이 주말이면 1

  ※ 168h 모델에서 lag_24 를 뺀 이유
    h 가 24보다 크면 "대상 시각의 24시간 전"이 아직 오지 않은 미래라서.

[분할]  시간 순서 그대로, 섞지 않음
  Train : Day 1~63   (3/15 ~ 5/16)  모델 학습
  Val   : Day 64~76  (5/17 ~ 5/29)  CQR 보정, 피처 선택
  Test  : Day 77~90  (5/30 ~ 6/12)  최종 평가
  - 예측 시점과 대상 시각이 모두 같은 구간 안에 있는 행만 사용
  - 처음 168시간은 lag_168 이 없어서 제외

[모델]
  LightGBM 분위수 회귀, 분위수마다 모델 1개
    q05 = 하한, q50 = 중앙값(대표 예측), q95 = 상한
  파라미터: n_estimators 400, learning_rate 0.03, num_leaves 15,
            min_child_samples 50, subsample 0.8, colsample_bytree 0.8
  (튜닝 안 한 고정값)

[CQR 보정]
  보정 전 90% 구간 적중률이 Val 에서 83% 로 부족
  → Val 에서 "구간을 벗어난 정도"를 측정해 그만큼 위아래로 넓힘
     최종 구간 = [q05 - offset, q95 + offset]

[누수 방지]
  아래 조건을 어기면 assert 로 실행이 멈춘다.
  - lag 가 예측 시점 이후 값을 참조
  - 금지 컬럼이 피처에 포함
  - Train/Val/Test 시간이 겹침


--------------------------------------------------------------
5. 결과 (Test, 5/30 ~ 6/12)
--------------------------------------------------------------
                               24h 모델     168h 모델
  MAPE (모델 q50)               0.885%       0.881%
  MAPE (1주 전 값 그대로)       1.240%       1.210%
  MAPE (24시간 전 값 그대로)    2.022%         -
  90% 적중률 (보정 전)          83.3%        83.5%
  90% 적중률 (CQR 보정 후)      92.3%        91.1%
  평균 구간 폭 (보정 후)        127kW        119kW
  CQR offset                    14.387kW     8.891kW

  프로젝트 기준: 24h MAPE 5% 이내, 168h 8% 이내, Coverage 85% 이상 → 모두 통과

  168h 모델 예측 거리별 MAPE
    h 1~24 : 0.90%   h 25~72 : 0.87%   h 73~168 : 0.88%

[해석할 때 주의]
  - 평균 약 2,900kW 기준으로 평균 약 26kW 오차.
  - 모델은 "1주 전 같은 시각 + 시간대·요일 패턴"을 주로 쓴다
    (중요도 상위: lag_168, lag_24, hour, dayofweek).
  - 데이터의 IT 부하가 1주 프로필 반복 + 노이즈로 만들어져서
    기준선(1주 전 값)보다 약 0.35%p 밖에 안 좋다.
    발표 시 반드시 기준선과 같이 보여줄 것.
  - 예측 거리가 길어져도 오차가 거의 그대로인 것도 같은 이유.


--------------------------------------------------------------
6. 결과 파일
--------------------------------------------------------------
output/forecast/
  metrics.json                 위 5번 지표 전부 (cqr_offset_kw 포함)
  test_pred_24h.parquet        Test 구간 예측값
  test_pred_168h.parquet       Test 구간 예측값
  models/
    it_load_24h_q05.txt   it_load_24h_q50.txt   it_load_24h_q95.txt
    it_load_168h_q05.txt  it_load_168h_q50.txt  it_load_168h_q95.txt

[test_pred_*.parquet 컬럼]
  origin_time   예측을 한 시점 (t)
  target_time   예측 대상 시각 (t+h)
  h             예측 거리 (시간)
  y             실제 IT 부하 (kW)          ← 그래프의 "실제값"
  q05 / q95     CQR 보정까지 끝난 90% 구간 하한/상한 (kW)
  q50           중앙값 예측 (kW)

  ※ 같은 target_time 이 여러 번 나온다 (1시간 전, 2시간 전 ... 에서 각각 예측).
    선 그래프로 이으려면 h 를 하나로 고정해서 필터링할 것.
      예) df[df.h == 24]               → 매번 24시간 전에 한 예측
      예) df[df.origin_time == 시점]   → 한 시점에서 본 앞으로 24시간


--------------------------------------------------------------
7. 모델 불러와서 예측하기 (예시)
--------------------------------------------------------------
  import lightgbm as lgb
  import pandas as pd

  booster = lgb.Booster(model_file='output/forecast/models/it_load_24h_q50.txt')
  print(booster.feature_name())   # 피처 순서 확인

  # X 는 위 4번 피처를 같은 이름·같은 순서로 만든 DataFrame
  q50 = booster.predict(X)

  # 구간은 q05/q95 모델로 각각 예측한 뒤 offset 적용
  #   lower = q05 - 14.387   (168h 모델은 8.891)
  #   upper = q95 + 14.387


--------------------------------------------------------------
8. 예측 함수 만들 때 할 일 (다음 담당자)
--------------------------------------------------------------
  1) 피처 계산 코드를 하나로 공유
     train_forecast.py 의 build_direct_dataset 와 예측 함수가
     같은 피처 계산 함수를 쓰게 해야 한다. 따로 만들면 나중에
     한쪽만 바뀌어도 에러 없이 틀린 예측이 나온다.
  2) CQR offset 을 모델 옆에 설정 파일로 저장
     지금은 metrics.json 에만 있다. (model_config.json 같은 것 추천)
  3) 분위수 순서 정렬
     q05 / q50 / q95 를 따로 학습해서 가끔 q05 > q50 이 될 수 있다.
  4) 입력은 최근 168시간 이상 필요 (lag_168 때문)
  5) 학습 범위 밖 입력 경고
     학습 데이터 IT 부하는 2,389 ~ 3,457kW. server_surge(×1.3) 같은
     범위 밖 입력은 트리 모델이 따라가지 못하므로 경고를 붙여서
     MPC 쪽이 Fallback 여부를 판단할 수 있게 한다.


--------------------------------------------------------------
9. 한계 (보고서/발표에 명시)
--------------------------------------------------------------
  - IT 부하가 주기적 프로필 기반이라 MAPE 가 낙관적으로 나온다.
  - Test 가 14일(336시간)뿐이라 168h 예측의 독립 평가 구간이 2개 수준.
  - CQR 은 데이터 독립을 가정하지만, 여기선 같은 시각을 여러 h 에서
    겹쳐 예측하므로 완전히 성립하지 않는다 (근사).
  - 하이퍼파라미터는 튜닝하지 않았다.
  - 기상 피처를 다시 켜면 미래 외기를 실측값으로 쓰는 것이므로
    "perfect forecast 가정"을 명시해야 한다.
  - 기상 피처 제외 결정 이전에 Test 결과를 한 번 출력했다.
    결정 자체는 Val 결과로만 했다.


--------------------------------------------------------------
10. 2차: 예측 함수 / API / 시각화 (임태형)
--------------------------------------------------------------
[파일]
  services/forecast/features.py   피처 계산 (학습·서빙 공용). train_forecast.py 도 이걸 import 한다.
  services/forecast/predictor.py  Predictor: model_config.json + 모델 6개 로드, forecast(history, horizon)
  services/forecast/schemas.py    요청/응답 Pydantic 스키마
  services/forecast/main.py       FastAPI (:8002)  GET /health, POST /api/v1/forecast
  output/forecast/models/model_config.json
                                  CQR offset, 피처 순서, 학습 범위, 모델 파일명 (train_forecast.py 가 생성)
  visualize_forecast.py           output/forecast/plots/forecast_report.html (plotly, 그래프 3개 + 지표표)
  tests/                          pytest 22개 (피처 누수, 서빙=학습 재현, 안전장치, API)

[실행]
  pip install -r requirements.txt
  python train_forecast.py --config-only     # 모델은 그대로 두고 model_config.json 만 생성
  uvicorn services.forecast.main:app --port 8002   → http://localhost:8002/docs
  python visualize_forecast.py
  python -m pytest -q

[API]
  POST /api/v1/forecast
    요청  {"horizon_hours": 24, "target": "it_load" | "cooling_load",
           "history": [{"timestamp": "...", "it_power_kw": 2900.0}, ...]}   ← history 생략 시 학습 데이터 끝 시점
    응답  origin_time, timestamps[h], predictions[h](q50), pi_lower[h], pi_upper[h],
           in_training_range(bool), warnings[str]
  - horizon 1~24 → 24h 모델, 25~168 → 168h 모델
  - history 는 5분 또는 1시간 간격, 최소 168시간. 부족하거나 빈 시간 있으면 422
  - 모델 없으면 503 (가짜 값 폴백 없음, Fallback 판단은 호출 쪽)

[안전장치 (README 8번 반영)]
  1) 피처 코드 공유: features.build_direct_dataset(학습) / build_forecast_features(서빙) 가
     같은 _feature_block 을 쓴다. 서빙 결과가 test_pred_24h.parquet 와 일치하는 테스트 있음.
  2) model_config.json 로 offset 등 저장
  3) q05 <= q50 <= q95 아니면 정렬 + warnings 에 기록
  4) 168시간 미만 입력 거부
  5) 입력/예측이 학습 범위(2,389 ~ 3,457kW) 밖이면 warnings 에 기록, in_training_range=false

[환경 주의]
  - Windows 에서 pyarrow 17 + lightgbm 4.7 이면 pandas 를 먼저 import 했을 때 lightgbm predict 가
    access violation 으로 죽는다. pyarrow 26 이상으로 해결 (requirements.txt 반영).
  - Windows git(core.autocrlf=true) 이 모델 txt 를 CRLF 로 바꾸면 LightGBM 이 못 읽는다.
    .gitattributes 에 변환 금지 추가했고, predictor 도 읽을 때 LF 로 정규화한다.
