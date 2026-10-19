import pandas as pd
import numpy as np
import os
import json

def reduce_mem_usage(df, verbose=True):
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

def clean_datacenter_to_structured_json(data_dir='./'):
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

    print("\n4. 날씨 데이터 결측치 선형 보간 처리...")
    weather_df = weather_df.sort_values(['site_id', 'timestamp'])
    weather_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 
                    'precip_depth_1_hr', 'sea_level_pressure', 'wind_speed']
    
    for col in weather_cols:
        if col in weather_df.columns:
            weather_df[col] = weather_df.groupby('site_id')[col].transform(
                lambda group: group.interpolate(method='linear', limit_direction='both')
            )

    print("\n5. 이상치 제거 및 도메인 파생 변수 생성...")
    train_df = train_df[train_df['meter_reading'] >= 0]

    print("\n6. 데이터 병합...")
    df = train_df.merge(building_df, on='building_id', how='left')
    df = df.merge(weather_df, on=['site_id', 'timestamp'], how='left')

    print("\n7. 열역학 모델 및 PUE 계산...")
    df['it_power_kw'] = df['meter_reading'] 
    df['cop'] = np.maximum(2.0, 6.0 - 0.1 * (df['air_temperature'] - 15))
    df['cooling_load_kw'] = df['it_power_kw'] * 0.95
    df['chiller_power_kw'] = df['cooling_load_kw'] / df['cop']

    df['free_cooling_mode'] = df['air_temperature'] <= 15
    df.loc[df['free_cooling_mode'], 'chiller_power_kw'] *= 0.2

    df['total_power_kw'] = df['it_power_kw'] + df['chiller_power_kw']
    df['pue'] = df['total_power_kw'] / df['it_power_kw']

    print("\n8. 전체 데이터를 시계열/시간대별 구조화된 JSON으로 변환 중...")
    os.makedirs('./output', exist_ok=True)

    # 예시 1: 전체 시간대별(또는 일별) 평균 집계 데이터 (모든 데이터의 흐름을 압축하여 반영)
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:00:00')
    hourly_agg = df.groupby('date').agg({
        'it_power_kw': 'mean',
        'cooling_load_kw': 'mean',
        'pue': 'mean',
        'air_temperature': 'mean'
    }).reset_index()

    # 보여주신 포맷 스타일을 반영한 구조화된 딕셔너리 생성
    structured_output = {
        "metadata": {
            "total_rows_processed": int(len(df)),
            "time_steps_aggregated": int(len(hourly_agg)),
            "description": "Aggregated hourly datacenter metrics from full dataset"
        },
        "metrics_summary": [
            {
                "label": "it_load",
                "unit": "kW",
                "avg_value": float(df['it_power_kw'].mean()),
                "max_value": float(df['it_power_kw'].max())
            },
            {
                "label": "cooling_load",
                "unit": "kW",
                "avg_value": float(df['cooling_load_kw'].mean()),
                "max_value": float(df['cooling_load_kw'].max())
            },
            {
                "label": "pue",
                "unit": "ratio",
                "avg_value": float(df['pue'].mean()),
                "min_value": float(df['pue'].min())
            }
        ],
        "hourly_time_series": hourly_agg.to_dict(orient='records')
    }

    json_path = './output/cleaned_datacenter_structured.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(structured_output, f, ensure_ascii=False, indent=4)

    print(f"모든 정제 작업이 완료되었습니다!")
    print(f"- 구조화된 전체 집계 JSON: {json_path}")
    print(f"- 압축된 시간대별 포인트 수: {len(hourly_agg):,}개")

if __name__ == "__main__":
    data_path = './data'
    clean_datacenter_to_structured_json(data_path)