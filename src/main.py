"""FastAPI 메인 애플리케이션"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import uvicorn
import os
import csv
import json

from src.config import settings
from src.local_storage import LocalStorageService
from src.openai_service import OpenAIService
from src.local_ml_service import LocalMLService

# ML 서비스는 선택사항이므로 지연 로딩
ml_service = None
try:
    from src.ml_service import MLService
    ml_service = MLService()
except ImportError as e:
    print(f"ML 서비스 모듈을 import할 수 없습니다 (선택사항): {e}")
    ml_service = None
except Exception as e:
    print(f"ML 서비스 초기화 실패 (선택사항): {e}")
    ml_service = None


app = FastAPI(
    title="주식 성향 분석 및 종목 추천 서비스",
    description="Azure 서비스를 활용한 주식 분석 및 추천 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (프론트엔드)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_root():
    """루트 경로에서 프론트엔드 제공"""
    static_file = os.path.join(static_dir, "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {
        "message": "주식 성향 분석 및 종목 추천 서비스",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/details")
async def read_details():
    """모델 성능 테스트 페이지 제공"""
    details_file = os.path.join(static_dir, "details.html")
    if os.path.exists(details_file):
        return FileResponse(details_file)
    raise HTTPException(status_code=404, detail="Details page not found")

# 서비스 인스턴스
# 프로젝트 루트의 data 폴더에 저장
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
local_storage = LocalStorageService(base_dir=str(data_dir))
local_ml_service = LocalMLService(models_dir=str(data_dir / "models"))
openai_service = None

try:
    if settings.azure_openai_endpoint and settings.azure_openai_api_key:
        openai_service = OpenAIService()
    else:
        print("⚠️  Azure OpenAI 설정이 없습니다. 분석 기능이 비활성화됩니다.")
except Exception as e:
    print(f"⚠️  OpenAI 서비스 초기화 실패: {e}")


# Pydantic 모델
class StockData(BaseModel):
    name: str
    price: float
    news: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class UserPreference(BaseModel):
    risk_tolerance: str = "보통"  # 낮음, 보통, 높음
    investment_amount: Optional[float] = None
    investment_period: Optional[str] = None
    interests: Optional[List[str]] = None


class AnalysisRequest(BaseModel):
    stock_data: StockData
    save_to_blob: bool = True


class RecommendationRequest(BaseModel):
    user_preference: UserPreference
    save_to_blob: bool = True


# API 엔드포인트
@app.get("/api")
async def api_info():
    """API 정보 엔드포인트"""
    return {
        "message": "주식 성향 분석 및 종목 추천 서비스",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "/api/analyze",
            "recommend": "/api/recommend",
            "storage_list": "/api/storage/list",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze")
async def analyze_stock(request: AnalysisRequest):
    """
    주식 성향 분석
    
    Args:
        request: 분석 요청 데이터
        
    Returns:
        분석 결과
    """
    if not openai_service:
        raise HTTPException(status_code=503, detail="OpenAI 서비스가 설정되지 않았습니다. .env 파일에 Azure OpenAI 설정을 추가하세요.")
    
    try:
        # OpenAI를 통한 성향 분석
        stock_dict = request.stock_data.dict()
        analysis_result = openai_service.analyze_stock_sentiment(stock_dict)
        
        if not analysis_result:
            raise HTTPException(status_code=500, detail="분석 실패")
        
        # 결과 저장
        result_data = {
            "stock_name": request.stock_data.name,
            "analysis": analysis_result,
            "timestamp": datetime.now().isoformat(),
            "stock_data": stock_dict
        }
        
        if request.save_to_blob:
            file_name = f"analysis/{request.stock_data.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            local_storage.upload_json(file_name, result_data, encrypt=True)
        
        return {
            "success": True,
            "stock_name": request.stock_data.name,
            "analysis": analysis_result,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")


@app.post("/api/recommend")
async def recommend_stocks(request: RecommendationRequest):
    """
    종목 추천
    
    Args:
        request: 추천 요청 데이터
        
    Returns:
        추천 종목 리스트
    """
    if not openai_service:
        raise HTTPException(status_code=503, detail="OpenAI 서비스가 설정되지 않았습니다. .env 파일에 Azure OpenAI 설정을 추가하세요.")
    
    try:
        # OpenAI를 통한 종목 추천
        preference_dict = request.user_preference.dict()
        recommendations = openai_service.recommend_stocks(preference_dict)
        
        if not recommendations:
            raise HTTPException(status_code=500, detail="추천 실패")
        
        # 결과 저장
        result_data = {
            "user_preference": preference_dict,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        if request.save_to_blob:
            file_name = f"recommendations/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            local_storage.upload_json(file_name, result_data, encrypt=True)
        
        return {
            "success": True,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 중 오류 발생: {str(e)}")


@app.get("/api/storage/list")
async def list_stored_files(prefix: str = ""):
    """
    저장된 파일 목록 조회
    
    Args:
        prefix: 파일 경로 접두사
        
    Returns:
        파일 목록
    """
    try:
        files = local_storage.list_files(prefix=prefix)
        return {
            "success": True,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 목록 조회 실패: {str(e)}")


@app.get("/api/storage/{file_name:path}")
async def get_stored_file(file_name: str):
    """
    저장된 파일 조회
    
    Args:
        file_name: 파일 이름
        
    Returns:
        파일 내용
    """
    try:
        content = local_storage.download_json(file_name, decrypt=True)
        if not content:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
        return {
            "success": True,
            "data": content
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 조회 실패: {str(e)}")


class TrainModelRequest(BaseModel):
    model_name: str
    training_data: List[Dict[str, Any]]
    target_column: str = "target"
    save_data_file: bool = True  # 학습 데이터를 파일로 저장할지 여부


@app.post("/api/ml/train")
async def train_model(request: TrainModelRequest):
    """
    로컬에서 ML 모델 학습
    
    Args:
        request: 학습 요청 데이터
        
    Returns:
        학습 결과
    """
    try:
        # 학습 데이터를 파일로 저장 (선택사항)
        if request.save_data_file:
            training_dir = data_dir / "training"
            training_dir.mkdir(exist_ok=True)
            
            import csv
            import json
            
            # CSV 파일로 저장
            csv_path = training_dir / f"{request.model_name}_training_data.csv"
            if request.training_data:
                # 피처 이름 추출
                feature_names = [key for key in request.training_data[0].keys() if key != request.target_column]
                headers = feature_names + [request.target_column]
                
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for row in request.training_data:
                        writer.writerow(row)
            
            # JSON 파일로도 저장 (백업)
            json_path = training_dir / f"{request.model_name}_training_data.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(request.training_data, f, indent=2, ensure_ascii=False)
        
        # 피처 이름 추출
        feature_names = None
        if request.training_data:
            feature_names = [key for key in request.training_data[0].keys() if key != request.target_column]
        
        model_path = local_ml_service.train_model(
            model_name=request.model_name,
            training_data=request.training_data,
            target_column=request.target_column,
            feature_names=feature_names
        )
        
        if not model_path:
            raise HTTPException(status_code=500, detail="모델 학습 실패")
        
        model_info = local_ml_service.get_model_info(request.model_name)
        
        return {
            "success": True,
            "model_name": request.model_name,
            "model_path": model_path,
            "model_info": model_info,
            "data_saved": request.save_data_file
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"학습 중 오류 발생: {str(e)}")


@app.post("/api/ml/predict")
async def ml_predict(model_name: str, data: Dict[str, Any]):
    """
    ML 모델 예측
    
    Args:
        model_name: 모델 이름
        data: 예측 데이터
        
    Returns:
        예측 결과
    """
    try:
        result = local_ml_service.predict(model_name, data)
        if not result:
            raise HTTPException(status_code=404, detail=f"모델 '{model_name}'을 찾을 수 없습니다")
        return {
            "success": True,
            **result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")


@app.get("/api/ml/models")
async def list_models():
    """저장된 모델 목록 조회"""
    try:
        models = local_ml_service.list_models()
        models_info = []
        for model_name in models:
            info = local_ml_service.get_model_info(model_name)
            if info:
                models_info.append(info)
        
        return {
            "success": True,
            "models": models,
            "models_info": models_info,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 목록 조회 실패: {str(e)}")


@app.delete("/api/ml/models/{model_name}")
async def delete_model(model_name: str):
    """모델 삭제"""
    try:
        success = local_ml_service.delete_model(model_name)
        if not success:
            raise HTTPException(status_code=404, detail=f"모델 '{model_name}'을 찾을 수 없습니다")
        
        return {
            "success": True,
            "message": f"모델 '{model_name}'이 삭제되었습니다."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 삭제 실패: {str(e)}")


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """
    AI와 채팅
    
    Args:
        request: 채팅 메시지
        
    Returns:
        AI 응답
    """
    if not openai_service:
        raise HTTPException(status_code=503, detail="OpenAI 서비스가 설정되지 않았습니다. .env 파일에 Azure OpenAI 설정을 추가하세요.")
    
    try:
        # OpenAI를 통한 채팅
        response = openai_service.client.chat.completions.create(
            model=openai_service.deployment_name,
            messages=[
                {"role": "system", "content": "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 한국어로 자연스럽게 대화하세요."},
                {"role": "user", "content": request.message}
            ],
            max_completion_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        return {
            "success": True,
            "response": ai_response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채팅 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    from pathlib import Path
    
    # src와 scripts 디렉토리만 감시 (venv 제외)
    base_dir = Path(__file__).parent.parent
    reload_dirs = [
        str(base_dir / "src"),
        str(base_dir / "scripts"),
    ]
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=reload_dirs
    )
