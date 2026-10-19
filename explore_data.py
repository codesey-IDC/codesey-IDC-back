import pandas as pd
import matplotlib.pyplot as plt

# 1. 깃 업로드용 샘플 파일(1만 행)을 로드해서 빠르게 확인
print("샘플 데이터 로드 중...")
df_sample = pd.read_csv('./output/cleaned_datacenter_sample.csv')

# 2. 기본 정보 및 상위 데이터 확인
print("\n--- 데이터 상위 5행 ---")
print(df_sample[['timestamp', 'building_id', 'it_power_kw', 'cooling_load_kw', 'pue']].head())

print("\n--- 주요 컬럼 통계 요약 ---")
print(df_sample[['it_power_kw', 'cooling_load_kw', 'pue', 'air_temperature']].describe())

# 3. 시각화: 시간 흐름에 따른 IT 전력 및 PUE 추이 그래프 그리기
print("\n시각화 그래프 생성 중...")
df_sample['timestamp'] = pd.to_datetime(df_sample['timestamp'])
df_sorted = df_sample.sort_values('timestamp')

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(df_sorted['timestamp'].head(500), df_sorted['it_power_kw'].head(500), label='IT Power (kW)', color='blue')
plt.title('IT Power & PUE Trend (Sample)')
plt.ylabel('Power (kW)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df_sorted['timestamp'].head(500), df_sorted['pue'].head(500), label='PUE', color='orange')
plt.xlabel('Timestamp')
plt.ylabel('PUE')
plt.legend()

plt.tight_layout()
plt.show()