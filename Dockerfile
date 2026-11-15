# IDC 냉각 최적화 시스템 이미지 (트윈·예측·MPC·대시보드 공용, 실행 명령만 docker-compose.yml 에서 다르게)
# 빌드 전에 Git LFS 데이터를 받아야 한다:  git lfs install && git lfs pull
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM 이 쓰는 OpenMP 런타임
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Git LFS 포인터(수백 바이트 껍데기)로 받은 경우 빌드를 멈추고 안내한다
RUN python -c "import os, sys; p='data/synthetic_idc_90d.parquet'; \
ok = os.path.exists(p) and os.path.getsize(p) > 10000; \
sys.exit(0 if ok else 'ERROR: ' + p + ' 가 없거나 Git LFS 포인터입니다. git lfs install && git lfs pull 후 다시 빌드하세요.')"

# Ridge 모델은 scikit-learn 버전에 묶인 joblib 파일이라 이미지 안에서 다시 학습한다 (약 3초).
# LightGBM 모델(output/forecast/models/*.txt)은 텍스트라 그대로 쓴다.
RUN python scripts/train_ridge.py

EXPOSE 8001 8002 8003 8501

# 기본 명령은 twin-service. 각 서비스 명령은 docker-compose.yml 참고
CMD ["uvicorn", "services.twin.main:app", "--host", "0.0.0.0", "--port", "8001"]
