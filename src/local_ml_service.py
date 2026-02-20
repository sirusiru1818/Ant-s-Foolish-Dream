"""로컬 머신러닝 서비스"""
import pickle
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime

# scikit-learn은 선택사항이므로 try-except로 처리
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None
    train_test_split = None
    StandardScaler = None


class LocalMLService:
    """로컬 머신러닝 서비스"""
    
    def __init__(self, models_dir: str = "data/models"):
        """
        ML 서비스 초기화
        
        Args:
            models_dir: 모델 저장 디렉토리
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
    
    def train_model(self, model_name: str, training_data: List[Dict], target_column: str = "target", feature_names: Optional[List[str]] = None) -> Optional[str]:
        """
        모델 학습
        
        Args:
            model_name: 모델 이름
            training_data: 학습 데이터 (리스트 of 딕셔너리)
            target_column: 타겟 컬럼명
            feature_names: 피처 이름 리스트 (None이면 자동 추출)
            
        Returns:
            모델 파일 경로 또는 None
        """
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn이 설치되지 않았습니다. pip install scikit-learn으로 설치하세요.")
            return None
        
        try:
            # 데이터 준비
            features = []
            targets = []
            
            # 피처 이름 추출 (첫 번째 샘플에서)
            if feature_names is None and len(training_data) > 0:
                feature_names = [key for key in training_data[0].keys() if key != target_column]
            
            for item in training_data:
                # 타겟 추출
                if target_column not in item:
                    continue
                
                target = item.pop(target_column)
                targets.append(float(target))
                
                # 피처 추출 (순서대로)
                feature = []
                for key in feature_names:
                    value = item.get(key)
                    if isinstance(value, (int, float)):
                        feature.append(float(value))
                    elif isinstance(value, bool):
                        feature.append(1.0 if value else 0.0)
                    else:
                        # 문자열은 해시값으로 변환
                        feature.append(float(hash(str(value)) % 10000))
                
                features.append(feature)
            
            if len(features) < 10:
                print(f"⚠️  학습 데이터가 부족합니다. 최소 10개 이상 필요합니다. (현재: {len(features)}개)")
                return None
            
            features = np.array(features)
            targets = np.array(targets)
            
            # 데이터 정규화
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # 학습/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, targets, test_size=0.2, random_state=42
            )
            
            # 모델 학습
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 정확도 평가
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # 피처 중요도 추출
            feature_importance = model.feature_importances_
            feature_importance_dict = {}
            if feature_names:
                for i, name in enumerate(feature_names):
                    feature_importance_dict[name] = float(feature_importance[i])
            else:
                for i, importance in enumerate(feature_importance):
                    feature_importance_dict[f"feature_{i}"] = float(importance)
            
            # 중요도 순으로 정렬
            sorted_importance = sorted(
                feature_importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            print(f"✅ 모델 학습 완료: {model_name}")
            print(f"   학습 정확도: {train_score:.4f}")
            print(f"   테스트 정확도: {test_score:.4f}")
            
            # 모델 저장
            model_path = self.models_dir / f"{model_name}.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # 메모리에 로드
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # 메타데이터 저장
            metadata = {
                "model_name": model_name,
                "trained_at": datetime.now().isoformat(),
                "train_score": float(train_score),
                "test_score": float(test_score),
                "n_samples": len(features),
                "n_features": features.shape[1],
                "feature_names": feature_names if feature_names else [f"feature_{i}" for i in range(features.shape[1])],
                "feature_importance": feature_importance_dict,
                "feature_importance_sorted": sorted_importance
            }
            
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return str(model_path)
            
        except Exception as e:
            print(f"❌ 모델 학습 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_model(self, model_name: str) -> bool:
        """
        모델 로드
        
        Args:
            model_name: 모델 이름
            
        Returns:
            성공 여부
        """
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            if not model_path.exists() or not scaler_path.exists():
                return False
            
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scalers[model_name] = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
    
    def predict(self, model_name: str, data: Dict[str, Any]) -> Optional[Dict]:
        """
        모델 예측
        
        Args:
            model_name: 모델 이름
            data: 예측할 데이터 (딕셔너리)
            
        Returns:
            예측 결과 또는 None
        """
        if model_name not in self.models:
            # 모델이 메모리에 없으면 로드 시도
            if not self.load_model(model_name):
                return None
        
        try:
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            # 피처 추출 (학습 시와 동일한 방식)
            feature = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    feature.append(float(value))
                elif isinstance(value, bool):
                    feature.append(1.0 if value else 0.0)
                else:
                    feature.append(float(hash(str(value)) % 10000))
            
            feature_array = np.array([feature])
            feature_scaled = scaler.transform(feature_array)
            
            prediction = model.predict(feature_scaled)[0]
            
            return {
                "prediction": float(prediction),
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"예측 실패: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """저장된 모델 목록 조회"""
        models = []
        for model_file in self.models_dir.glob("*.pkl"):
            if "_scaler" not in model_file.name:
                model_name = model_file.stem
                models.append(model_name)
        return sorted(models)
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """모델 정보 조회"""
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def delete_model(self, model_name: str) -> bool:
        """
        모델 삭제
        
        Args:
            model_name: 모델 이름
            
        Returns:
            성공 여부
        """
        try:
            # 모델 파일들 삭제
            model_path = self.models_dir / f"{model_name}.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            
            deleted = False
            
            if model_path.exists():
                model_path.unlink()
                deleted = True
            
            if scaler_path.exists():
                scaler_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            # 메모리에서도 제거
            if model_name in self.models:
                del self.models[model_name]
            
            if model_name in self.scalers:
                del self.scalers[model_name]
            
            return deleted
        except Exception as e:
            print(f"모델 삭제 실패: {e}")
            return False
