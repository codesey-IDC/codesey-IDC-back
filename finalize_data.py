import pandas as pd
import json
import os

def finalize_training_data():
    print("1. Loading refined training data (Thermodynamic Model output)...")
    try:
        # 이 데이터는 사용자의 열역학 모델을 통해 생성되었으며, Sinergym은 검증 용도로만 활용되었습니다.
        df_train = pd.read_parquet('data/synthetic_idc_90d.parquet')
        print(f"   Loaded training data: {df_train.shape}")
    except Exception as e:
        print(f"Error reading synthetic_idc_90d.parquet: {e}")
        return

    print("\n2. Finalizing Output...")
    # 2016년 참조 데이터는 중복 계산 및 논리적 일치 문제로 제외합니다.
    # 오직 열역학 모델로 생성된 고품질 학습 데이터만 최종 파일로 생성합니다.
    
    os.makedirs('output/final', exist_ok=True)
    output_path = 'output/final/idc_train_refined_2026.parquet'
    
    # 불필요한 메타데이터 컬럼이 있다면 제거하거나 정리 (여기서는 깔끔하게 유지)
    df_train.to_parquet(output_path, index=False)
    
    print(f"   [Saved] Final Training Data: {output_path}")
    print("\n정제 완료: 열역학 모델 기반의 순수 학습 데이터셋이 준비되었습니다.")

if __name__ == "__main__":
    finalize_training_data()

