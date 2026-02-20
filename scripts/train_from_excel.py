"""Excel 파일에서 모델 학습 스크립트"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import json
from src.local_ml_service import LocalMLService
from src.config import settings

def train_model_from_excel(excel_path: str, model_name: str, target_column: str = "market_cap_usd"):
    """
    Excel 파일에서 모델 학습
    
    Args:
        excel_path: Excel 파일 경로
        model_name: 모델 이름
        target_column: 타겟 컬럼명
    """
    print(f"📂 Excel 파일 읽기: {excel_path}")
    
    # Excel 파일 읽기
    try:
        df = pd.read_excel(excel_path)
        print(f"✅ 데이터 로드 완료: {df.shape[0]}행, {df.shape[1]}컬럼")
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
        return None
    
    # 컬럼 확인
    print(f"\n📋 사용 가능한 컬럼 ({len(df.columns)}개):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # 타겟 컬럼 확인
    if target_column not in df.columns:
        print(f"\n⚠️  타겟 컬럼 '{target_column}'을 찾을 수 없습니다.")
        print("사용 가능한 숫자형 컬럼:")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in numeric_cols[:10]:  # 처음 10개만 표시
            print(f"  - {col}")
        return None
    
    # 결측치가 있는 행 제거
    initial_rows = len(df)
    df_clean = df.dropna(subset=[target_column])
    removed_rows = initial_rows - len(df_clean)
    if removed_rows > 0:
        print(f"\n⚠️  타겟 컬럼 결측치로 {removed_rows}행 제거됨")
    
    if len(df_clean) < 10:
        print(f"❌ 학습 가능한 데이터가 부족합니다. (최소 10개 필요, 현재: {len(df_clean)}개)")
        return None
    
    # 숫자형 컬럼만 선택 (타겟 제외)
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    # 범주형 컬럼도 포함 (원핫 인코딩 대신 해시값 사용)
    categorical_cols = df_clean.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    print(f"\n🔢 사용할 피처:")
    print(f"  - 숫자형: {len(numeric_cols)}개")
    print(f"  - 범주형: {len(categorical_cols)}개")
    print(f"  - 타겟: {target_column}")
    
    # 데이터를 딕셔너리 리스트로 변환
    training_data = []
    for _, row in df_clean.iterrows():
        data_dict = {}
        
        # 숫자형 피처 추가
        for col in numeric_cols:
            if pd.notna(row[col]):
                data_dict[col] = float(row[col])
        
        # 범주형 피처 추가 (문자열로 저장, 모델에서 해시 변환)
        for col in categorical_cols:
            if pd.notna(row[col]):
                data_dict[col] = str(row[col])
        
        # 타겟 추가
        if pd.notna(row[target_column]):
            data_dict[target_column] = float(row[target_column])
            training_data.append(data_dict)
    
    print(f"\n📊 학습 데이터 준비 완료: {len(training_data)}개 샘플")
    
    # 학습 데이터를 JSON으로 저장
    training_dir = project_root / "data" / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = training_dir / f"{model_name}_training_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    print(f"💾 학습 데이터 저장: {json_path}")
    
    # CSV로도 저장
    csv_path = training_dir / f"{model_name}_training_data.csv"
    df_clean.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"💾 학습 데이터 저장: {csv_path}")
    
    # 모델 학습
    print(f"\n🚀 모델 학습 시작: {model_name}")
    print(f"   타겟: {target_column}")
    
    ml_service = LocalMLService(models_dir=str(project_root / "data" / "models"))
    
    # 피처 이름 추출
    feature_names = [col for col in numeric_cols + categorical_cols]
    
    model_path = ml_service.train_model(
        model_name=model_name,
        training_data=training_data,
        target_column=target_column,
        feature_names=feature_names
    )
    
    if model_path:
        print(f"\n✅ 모델 학습 완료!")
        print(f"   모델 경로: {model_path}")
        
        # 모델 정보 출력
        model_info = ml_service.get_model_info(model_name)
        if model_info:
            print(f"\n📈 모델 성능:")
            print(f"   학습 정확도: {model_info['train_score']*100:.2f}%")
            print(f"   테스트 정확도: {model_info['test_score']*100:.2f}%")
            
            if model_info.get('feature_importance_sorted'):
                print(f"\n🔝 상위 10개 변수 중요도:")
                for i, (name, importance) in enumerate(model_info['feature_importance_sorted'][:10], 1):
                    print(f"   {i}. {name}: {importance*100:.2f}%")
        
        return model_path
    else:
        print(f"\n❌ 모델 학습 실패")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Excel 파일에서 모델 학습')
    parser.add_argument('--file', type=str, required=True, help='Excel 파일 경로')
    parser.add_argument('--model', type=str, required=True, help='모델 이름')
    parser.add_argument('--target', type=str, default='market_cap_usd', help='타겟 컬럼명 (기본값: market_cap_usd)')
    
    args = parser.parse_args()
    
    train_model_from_excel(args.file, args.model, args.target)
