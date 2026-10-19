import pandas as pd
import numpy as np
import os

def reduce_mem_usage(df, verbose=True):
    """메모리 사용량을 줄이기 위해 데이터 타입을 다운캐스팅하는 함수"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type) in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'메모리 최적화 완료: {start_mem:.2f} MB -> {end_mem:.2f} MB ({(1 - end_mem / start_mem) * 100:.1f}% 감소)')
    return df

def clean_datacenter_pipeline(data_dir='./'):
    print("1. 데이터 로드 중...")
    try:
        building_df = pd.read_csv(os.path.join(data_dir, 'building_metadata.csv'))
        weather_df = pd.read_csv(os.path.join(data_dir, 'weather_train.csv'))
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}. 경로를 확인해주세요.")
        return

    print("\n2. 메모리 최적화(다운캐스팅) 진행...")
    building_df = reduce_mem_usage(building_df)
    weather_df = reduce_mem_usage(weather_df)
    train_df = reduce_mem_usage(train_df)

    print("\n3. 타임스탬프 형식 변환...")
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])

    print("\n4. 날씨 데이터 결측치 선형 보간(Interpolation) 처리...")
    weather_df = weather_df.sort_values(['site_id', 'timestamp'])
    weather_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 
                    'precip_depth_1_hr', 'sea_level_pressure', 'wind_speed']
    
    for col in weather_cols:
        if col in weather_df.columns:
            weather_df[col] = weather_df.groupby('site_id')[col].transform(
                lambda group: group.interpolate(method='linear', limit_direction='both')
            )

    print("\n5. 이상치 제거 및 데이터센터 도메인 파생 변수 생성...")
    # 음수 meter_reading 제거
    train_df = train_df[train_df['meter_reading'] >= 0]

    print("\n6. 데이터 병합 (Train + Building + Weather)...")
    df = train_df.merge(building_df, on='building_id', how='left')
    df = df.merge(weather_df, on=['site_id', 'timestamp'], how='left')

    # [핵심 추가] 열역학 모델 및 PUE/ESG 계산 로직 적용 (미션 요구사항 반영)
    print("\n7. 열역학 모델 및 PUE, 탄소 배출량 지표 계산 중...")
    # IT 전력 (kW) 가정 (meter_reading을 IT 부하로 매핑)
    df['it_power_kw'] = df['meter_reading'] 

    # 칠러 COP 모델 (외기 온도 함수: 외기가 낮을수록 COP 상승, 최소 2.0 보장)
    df['cop'] = np.maximum(2.0, 6.0 - 0.1 * (df['air_temperature'] - 15))

    # 냉각 부하 및 칠러 전력 추정 (단순화된 열역학 공식 적용)
    # Q_cooling = IT 전력의 일정 비율(예: 95%가 열로 전환된다 가정)
    df['cooling_load_kw'] = df['it_power_kw'] * 0.95
    df['chiller_power_kw'] = df['cooling_load_kw'] / df['cop']

    # Free Cooling(자연공조) 모드 판정 (외기 15도 이하 시 활성화)
    df['free_cooling_mode'] = df['air_temperature'] <= 15
    # 자연공조 시 칠러 전력을 대폭 절감 (예: 80% 감소 효과)
    df.loc[df['free_cooling_mode'], 'chiller_power_kw'] *= 0.2

    # PUE 계산: (IT 전력 + 냉각 전력) / IT 전력
    df['total_power_kw'] = df['it_power_kw'] + df['chiller_power_kw']
    df['pue'] = df['total_power_kw'] / df['it_power_kw']
    
    # ESG 탄소 배출량 계산 (계수: 0.459 tCO2/MWh = 0.000459 tCO2/kWh)
    df['carbon_emission_tco2'] = (df['total_power_kw'] / 1000) * 0.000459

    print("\n8. 정제된 결과물을 CSV 파일로 저장...")
    os.makedirs('./output', exist_ok=True)
    
    # 전체 데이터 CSV 저장
    output_csv = './output/cleaned_datacenter_merged.csv'
    df.to_csv(output_csv, index=False)
    
    # 깃허브 업로드용 샘플 파일 (상위 1만 행)
    sample_csv = './output/cleaned_datacenter_sample.csv'
    df.head(10000).to_csv(sample_csv, index=False)
    
    print(f"모든 정제 작업이 완료되었습니다!")
    print(f"- 전체 데이터 CSV: {output_csv}")
    print(f"- 깃 업로드용 샘플 CSV(상위 1만행): {sample_csv}")
    print(f"최종 데이터 크기(행, 열): {df.shape}")

if __name__ == "__main__":
    data_path = './data'  # 본인의 CSV 파일들이 위치한 경로 지정
    clean_datacenter_pipeline(data_path)