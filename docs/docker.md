# Docker 실행·확인 가이드

과제 요구사항: `docker-compose up` 명령으로 전체 시스템이 실행되어야 한다.
환경: Ubuntu 22.04 (WSL2 가능), Docker 24.0 이상, Docker Compose 2.0 이상.

## 1. 실행

```bash
git lfs install
git lfs pull                       # 데이터(parquet)·결과(csv)는 Git LFS. 안 받으면 빌드가 멈추고 안내한다
docker compose up --build          # 처음 빌드 5~10분 (패키지 설치). 이후에는 캐시로 빠르다
```

| 서비스 | 주소 | 시작 순서 |
|---|---|---|
| 디지털 트윈 | http://localhost:8001/docs | 1 |
| 예측 | http://localhost:8002/docs | 2 (트윈 healthy 후) |
| MPC | http://localhost:8003/docs | 3 (트윈·예측 healthy 후) |
| 대시보드 | http://localhost:8501 | 4 (세 서비스 healthy 후) |

끄기: `Ctrl+C` 후 `docker compose down`

## 2. 구성

- 이미지 하나(`codesey-idc:latest`)를 네 서비스가 같이 쓰고 실행 명령만 다르다 (`docker-compose.yml`).
- 서비스끼리는 REST 로만 통신하고, 주소는 환경변수로 컨테이너 이름을 가리킨다
  (`TWIN_URL=http://twin:8001`, `FORECAST_URL=http://forecast:8002`, `MPC_URL=http://mpc:8003`).
  환경변수는 `config/settings.yaml` 의 `services` 값보다 우선한다.
- 이미지를 만들 때 Ridge 예측 모델을 다시 학습한다 (`scripts/train_ridge.py`, 약 3초). joblib 파일이 scikit-learn
  버전에 묶여 있어서다. LightGBM 모델은 텍스트 파일이라 그대로 쓴다.
- 서비스가 쓰지 않는 ASHRAE 원본 데이터(`data/train.csv` 등 수백 MB)는 `.dockerignore` 로 뺐다.

## 3. 확인 체크리스트

```bash
docker compose ps                                  # 네 서비스 모두 (healthy)
curl http://localhost:8001/health                  # {"status":"ok","service":"twin"}
curl http://localhost:8002/health                  # "model":"ridge", "available_models":["lgbm","ridge"]
curl http://localhost:8003/health                  # twin_url 이 http://twin:8001 인지

# 최적 냉각 제어 API (과제 결과물 3)
curl -X POST http://localhost:8003/api/v1/control/optimize -H "Content-Type: application/json" \
     -d '{"timestamp": "2026-06-04T06:00:00", "room_temp_c": 22, "current_setpoint_c": 22}'

# 컨테이너 안에서 테스트·API 응답 시간 측정
docker compose exec mpc python -m pytest -q        # 전체 단위 테스트
docker compose exec mpc python scripts/benchmark_api.py    # 엔드포인트별 p95 (기준 10초)
```

브라우저에서 http://localhost:8501 → 서비스 상태가 모두 연결됨, 관제 탭에 PUE 게이지·전력 추이·서버실 온도·냉각 모드가 보이면 성공.

## 4. Sinergym 비교 검증 (선택)

열역학 모델을 Sinergym(EnergyPlus)과 비교한다. Sinergym 설치가 무거워 공식 이미지를 그대로 쓴다.

```bash
docker compose --profile validation run --rm sinergym-validate     # 약 2분 (이미지가 이미 있을 때)
docker compose --profile validation rm -f sinergym-collect         # 끝난 수집 컨테이너 정리
```

- 처음 실행하면 `sailugr/sinergym:latest` (약 22GB)를 내려받는다. 디스크 여유를 먼저 확인할 것.
- 두 단계로 돈다: `sinergym-collect` (Sinergym 이미지에서 1년 × 설정 온도 3개 시뮬레이션) →
  `sinergym-validate` (우리 이미지에서 같은 IT 전력·외기온도를 우리 모델에 넣어 비교).
- 결과: `output/validation/sinergym_samples.csv`, `sinergym_report.json`. 화면에 비교표가 나온다.
- 종료 코드: 0 통과 / 1 오차 10% 초과 항목 있음 / 2 Sinergym 결과 없음.
- 이미 수집한 CSV 가 있으면 Docker 없이 `python scripts/validate_sinergym.py` 로 비교만 다시 할 수 있다.
- `docker compose up` 으로는 뜨지 않는다 (`validation` 프로필을 줄 때만 실행).

결과 해석은 `docs/thermodynamics.md` 11장.

## 5. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| 빌드 중 `Git LFS 포인터입니다` | `git lfs install && git lfs pull` 후 `docker compose build` |
| `port is already allocated` | 같은 포트를 쓰는 프로그램(이전에 띄운 서비스 등) 종료 |
| 대시보드 "제어기 비교" 탭 오류 | `output/mpc/*.csv` 가 LFS 포인터 → `git lfs pull` |
| 서비스가 `unhealthy` | `docker compose logs <서비스명>` 으로 오류 확인 |

## 6. 확인 결과 기록

| 항목 | 결과 | 확인한 사람·날짜 |
|---|---|---|
| `docker compose up --build` 로 네 서비스 healthy | | |
| 대시보드 접속·화면 표시 | | |
| `control/optimize` 응답 | | |
| 컨테이너 안 `pytest` 전체 통과 | | |
| 컨테이너 안 `scripts/benchmark_api.py` 모두 10초 이내 | | |
