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
# static 폴더에 person_image도 포함되어 있으므로 별도 마운트 불필요
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

@app.get("/personality")
async def read_personality():
    """투자 성향 S-MBTI 진단 페이지 제공"""
    personality_file = os.path.join(static_dir, "personality.html")
    if os.path.exists(personality_file):
        return FileResponse(personality_file)
    raise HTTPException(status_code=404, detail="Personality page not found")

@app.get("/fact")
async def read_fact():
    """데이터 비교 분석 페이지 제공"""
    fact_file = os.path.join(static_dir, "fact.html")
    if os.path.exists(fact_file):
        return FileResponse(fact_file)
    raise HTTPException(status_code=404, detail="Fact page not found")

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


@app.get("/api/fact")
async def get_fact_data():
    """모델 예측 결과와 CSV 데이터 비교 분석"""
    try:
        import pandas as pd
        import numpy as np
        
        # CSV 데이터 로드
        stock_data = load_csv_data_for_llm()
        if not stock_data:
            return {
                "success": False,
                "message": "주식 데이터를 찾을 수 없습니다."
            }
        
        df = pd.DataFrame(stock_data)
        
        # NaN 체크 및 변환 함수
        def safe_float(value):
            try:
                if value is None:
                    return 0.0
                if pd.isna(value):
                    return 0.0
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        return 0.0
                result = float(value)
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return result
            except (TypeError, ValueError, OverflowError):
                return 0.0
        
        # NaN 값을 0으로 채우기
        if 'market_cap_usd' in df.columns:
            df['market_cap_usd'] = df['market_cap_usd'].fillna(0)
        if 'dividend_yield' in df.columns:
            df['dividend_yield'] = df['dividend_yield'].fillna(0)
        
        # 기본 통계
        total_stocks = len(df)
        total_market_cap = safe_float((df['market_cap_usd'] / 1e12).sum()) if 'market_cap_usd' in df.columns else 0.0
        avg_dividend = safe_float((df['dividend_yield'] * 100).mean()) if 'dividend_yield' in df.columns else 0.0
        avg_market_cap = safe_float((df['market_cap_usd'] / 1e9).mean()) if 'market_cap_usd' in df.columns else 0.0
        
        # 섹터별 분석
        sector_stats = []
        if 'gics_sector' in df.columns:
            sector_groups = df.groupby('gics_sector')
            for sector, group in sector_groups:
                avg_mcap = safe_float((group['market_cap_usd'] / 1e9).mean())
                avg_div = safe_float((group['dividend_yield'] * 100).mean())
                total_mcap = safe_float((group['market_cap_usd'] / 1e12).sum())
                
                sector_stats.append({
                    "sector": str(sector) if pd.notna(sector) else "UNKNOWN",
                    "count": int(len(group)),
                    "avg_market_cap": round(avg_mcap, 2),
                    "avg_dividend": round(avg_div, 2),
                    "total_market_cap": round(total_mcap, 2)
                })
        
        # 시가총액 분포
        market_cap_buckets = {}
        if 'market_cap_bucket' in df.columns:
            bucket_counts = df['market_cap_bucket'].value_counts().to_dict()
            market_cap_buckets = bucket_counts
        
        # 배당 프로필 분석
        dividend_profile_stats = {}
        if 'dividend_profile' in df.columns:
            div_profile_counts = df['dividend_profile'].value_counts().to_dict()
            dividend_profile_stats = div_profile_counts
        
        # 상위 종목 (시가총액 기준)
        top_stocks = []
        if 'market_cap_usd' in df.columns and 'ticker_primary' in df.columns:
            top_df = df.nlargest(10, 'market_cap_usd')
            for _, row in top_df.iterrows():
                mcap = safe_float(row.get('market_cap_usd', 0) / 1e9)
                div_yield = safe_float(row.get('dividend_yield', 0) * 100)
                
                top_stocks.append({
                    "ticker": str(row.get('ticker_primary', '?')) if pd.notna(row.get('ticker_primary')) else '?',
                    "name": str(row.get('name', '?')) if pd.notna(row.get('name')) else '?',
                    "market_cap": round(mcap, 2),
                    "dividend_yield": round(div_yield, 2),
                    "sector": str(row.get('gics_sector', '?')) if pd.notna(row.get('gics_sector')) else '?'
                })
        
        # 모델 정보 (있는 경우)
        models_info = []
        try:
            models = local_ml_service.list_models()
            models_info = models if isinstance(models, list) else models.get('models', [])
        except:
            pass
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_stocks": int(total_stocks),
                "total_market_cap_trillion": round(safe_float(total_market_cap), 2),
                "avg_dividend_yield": round(safe_float(avg_dividend), 2),
                "avg_market_cap_billion": round(safe_float(avg_market_cap), 2)
            },
            "sector_analysis": sector_stats,
            "market_cap_distribution": market_cap_buckets,
            "dividend_profile_distribution": dividend_profile_stats,
            "top_stocks": top_stocks,
            "models_available": len(models_info),
            "models": models_info
        }
    except Exception as e:
        print(f"❌ Fact 데이터 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"데이터 분석 중 오류 발생: {str(e)}")


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


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    use_stock_data: bool = True
    exchange_rate: float = 1450.0
    session_id: str = "default"  # 세션 ID
    history: Optional[List[ChatMessage]] = None  # 대화 히스토리
    personality_type: Optional[str] = None  # 투자 성향 타입 (PSLD, PSLG, ... 또는 None)


# 서버 측 대화 히스토리 저장소 (세션별)
conversation_history: Dict[str, List[Dict[str, str]]] = {}


def load_csv_data_for_llm() -> list:
    """CSV 파일에서 주식 데이터를 직접 로드"""
    stock_data = []
    
    try:
        import pandas as pd
        
        # data/training 폴더에서 CSV 파일 찾기
        training_path = data_dir / "training"
        csv_files = list(training_path.glob("*.csv"))
        
        # CSV가 없으면 Excel 파일도 확인
        if not csv_files:
            excel_files = list(training_path.glob("*.xlsx"))
            if excel_files:
                latest_file = max(excel_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_excel(latest_file)
            else:
                return []
        else:
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
        
        stock_data = df.to_dict('records')
        print(f"✅ 주식 데이터 로드 완료: {len(stock_data)}개 종목")
        
    except Exception as e:
        print(f"❌ 주식 데이터 로드 실패: {e}")
    
    return stock_data


def create_full_stock_context(stock_data: list) -> str:
    """LLM이 분석할 수 있도록 전체 주식 데이터를 컨텍스트로 변환 (압축 형식)"""
    if not stock_data:
        return ""
    
    # 섹터별로 그룹화
    sector_stocks = {}
    for stock in stock_data:
        sector = stock.get("gics_sector", stock.get("gics_sector_full", "UNKNOWN"))
        if sector not in sector_stocks:
            sector_stocks[sector] = []
        sector_stocks[sector].append(stock)
    
    context_parts = [
        f"# S&P 500 데이터 ({len(stock_data)}개)",
        ""
    ]
    
    # 각 섹터별 종목 (간결한 형식)
    for sector, stocks in sorted(sector_stocks.items()):
        # 시가총액 순 정렬
        sorted_stocks = sorted(stocks, key=lambda x: x.get("market_cap_usd", 0) or 0, reverse=True)
        
        context_parts.append(f"## {sector} ({len(stocks)}개)")
        
        for s in sorted_stocks:
            ticker = s.get("ticker_primary", "?")
            name = s.get("name", "?")
            cap = (s.get("market_cap_usd", 0) or 0) / 1e9
            div = (s.get("dividend_yield", 0) or 0) * 100
            bucket = s.get("market_cap_bucket", "?")
            founded = s.get("founded", "?")
            div_profile = s.get("dividend_profile", "?")
            
            context_parts.append(f"{ticker}|{name}|${cap:.0f}B|{bucket}|배당{div:.1f}%|{div_profile}|설립{founded}")
        
        context_parts.append("")
    
    # 간단한 통계
    total_cap = sum(s.get("market_cap_usd", 0) or 0 for s in stock_data) / 1e12
    avg_div = sum(s.get("dividend_yield", 0) or 0 for s in stock_data) / len(stock_data) * 100
    
    context_parts.append(f"총시총: ${total_cap:.1f}T, 평균배당: {avg_div:.2f}%")
    
    return "\n".join(context_parts)


# 투자 성향 데이터베이스
PERSONALITY_DATABASE = {
    'PSLD': {
        'name': '완벽주의 선비 자산가',
        'desc': '안정성과 데이터를 중시하는 장기 배당 투자자',
        'strategy': '배당 성장주와 지수 ETF를 8:2 비중으로 구성하여 복리 효과를 누리세요.',
        'traits': '장기 투자, 안정성 중시, 데이터 기반 분석, 배당 선호'
    },
    'PSLG': {
        'name': '신중한 미래 설계자',
        'desc': '분석을 통해 미래 성장주를 발굴하고 길게 가져가는 스타일',
        'strategy': '현금흐름이 확실한 우량 성장주(빅테크)에 적립식으로 투자하세요.',
        'traits': '장기 투자, 데이터 분석, 성장주 선호, 신중한 접근'
    },
    'PSID': {
        'name': '따뜻한 배당 수집가',
        'desc': '직관적으로 신뢰하는 브랜드에서 배당금을 모으는 장기 투자자',
        'strategy': '우리가 실생활에서 자주 쓰는 브랜드 중 배당을 잘 주는 기업에 집중하세요.',
        'traits': '장기 투자, 직관적 선택, 브랜드 신뢰, 배당 선호'
    },
    'PSIG': {
        'name': '낭만적인 가치 투자자',
        'desc': '세상을 바꿀 아이디어에 공감하며 동행하는 장기 투자자',
        'strategy': '혁신 기술의 초기 단계에 진입하여 세상이 변할 때까지 기다리는 투자를 권합니다.',
        'traits': '장기 투자, 가치 추구, 혁신 기업 선호, 직관적 판단'
    },
    'PVLD': {
        'name': '강심장 배당 전략가',
        'desc': '위험을 감수하면서도 고수익 배당을 설계하는 분석가',
        'strategy': '커버드콜 ETF나 고배당 섹터를 분석하여 하락장에서도 현금을 창출하세요.',
        'traits': '장기 투자, 위험 감수, 데이터 분석, 고배당 추구'
    },
    'PVLG': {
        'name': '냉철한 시스템 투자자',
        'desc': '변동성을 데이터로 극복하며 위기에 베팅하는 실력파',
        'strategy': '반도체나 AI 인프라 섹터가 급락할 때 데이터를 기반으로 과감히 매수하세요.',
        'traits': '장기 투자, 위험 감수, 데이터 기반, 공격적 접근'
    },
    'PVID': {
        'name': '직관적인 모험 자산가',
        'desc': '변동성을 즐기며 자신의 안목을 믿고 끝까지 버티는 스타일',
        'strategy': '파괴적 혁신을 이끄는 기업 중 본인이 잘 아는 테마에 집중 투자하세요.',
        'traits': '장기 투자, 위험 감수, 직관적 판단, 혁신 기업 선호'
    },
    'PVIG': {
        'name': '꿈을 쫓는 불도저',
        'desc': '큰 시세 차익을 위해 모든 위험을 감수하는 공격적인 장기파',
        'strategy': '성장 잠재력이 무한한 중소형 혁신주나 레버리지 ETF를 적절히 활용하세요.',
        'traits': '장기 투자, 극도로 공격적, 고위험 고수익, 성장주 집중'
    },
    'HSLD': {
        'name': '기민한 실속파 분석가',
        'desc': '데이터를 통해 단기 저평가를 찾아 수익을 확정 짓는 스타일',
        'strategy': '실적 발표 시즌 전후의 단기 변동성을 데이터로 분석하여 매매하세요.',
        'traits': '단기 투자, 안정성 중시, 데이터 분석, 실속 추구'
    },
    'HSLG': {
        'name': '기회 포착 전략가',
        'desc': '안전하게 수익이 날 구간을 분석하여 빠르게 치고 빠지는 타입',
        'strategy': '안전 마진이 확보된 우량주 중 단기 모멘텀이 발생한 종목에 집중하세요.',
        'traits': '단기 투자, 안정성 중시, 데이터 분석, 기회 포착'
    },
    'HSID': {
        'name': '유연한 방어 투자자',
        'desc': '시장의 분위기를 읽고 짧은 배당이나 확정 수익을 챙기는 타입',
        'strategy': '시장의 소음에 민감하게 대응하되, 안전한 단기 국채와 배당주를 활용하세요.',
        'traits': '단기 투자, 안정성 중시, 직관적 판단, 방어적 접근'
    },
    'HSIG': {
        'name': '감각적인 단기 파이터',
        'desc': '트렌드 변화를 포착해 성장주로 짧고 굵게 수익을 내는 스타일',
        'strategy': '뉴스나 SNS에서 핫한 테마의 대장주를 빠르게 매매하여 수익을 확보하세요.',
        'traits': '단기 투자, 안정성 중시, 직관적 판단, 트렌드 추종'
    },
    'HVLD': {
        'name': '데이터 기반 승부사',
        'desc': '변동성 높은 장세에서 수치로 승부수를 던지는 단기파',
        'strategy': '옵션 지표나 기술적 분석을 통해 단기 고수익 배당/차익 기회를 노리세요.',
        'traits': '단기 투자, 위험 감수, 데이터 분석, 공격적 접근'
    },
    'HVLG': {
        'name': '스피드 중시 전략가',
        'desc': '공격적인 성장주 매매를 통해 복리보다 스피드를 택한 분석가',
        'strategy': '급등주 중에서도 실적 근거가 있는 종목을 골라 짧은 구간을 먹고 나옵니다.',
        'traits': '단기 투자, 위험 감수, 데이터 분석, 빠른 수익 추구'
    },
    'HVID': {
        'name': '감각적인 서핑 투자자',
        'desc': '시장의 파도를 타고 화끈하게 수익을 낸 뒤 현금화하는 타입',
        'strategy': '테마의 순환매를 직관적으로 파악하여 자금을 빠르게 이동시키세요.',
        'traits': '단기 투자, 위험 감수, 직관적 판단, 테마 투자'
    },
    'HVIG': {
        'name': '화끈한 트렌드 세터',
        'desc': '지금 이 순간 가장 뜨거운 종목에 전 재산을 던지는 승부사',
        'strategy': '손절선을 엄격히 잡되, 상승 추세가 꺾이기 전까지는 끝까지 수익을 즐기세요.',
        'traits': '단기 투자, 극도로 공격적, 직관적 판단, 고위험 고수익'
    }
}


@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """
    AI와 채팅 (CSV 데이터 기반 주식 추천)
    
    Args:
        request: 채팅 메시지
        
    Returns:
        AI 응답
    """
    if not openai_service:
        raise HTTPException(status_code=503, detail="OpenAI 서비스가 설정되지 않았습니다. .env 파일에 Azure OpenAI 설정을 추가하세요.")
    
    try:
        # CSV 주식 데이터 로드
        stock_context = ""
        stock_count = 0
        if request.use_stock_data:
            stock_data = load_csv_data_for_llm()
            stock_count = len(stock_data)
            stock_context = create_full_stock_context(stock_data)
        
        # 성향별 프롬프트 추가
        personality_prompt = ""
        if request.personality_type and request.personality_type in PERSONALITY_DATABASE:
            personality_info = PERSONALITY_DATABASE[request.personality_type]
            personality_prompt = f"""

## 투자 성향 설정: {personality_info['name']} ({request.personality_type})
**특징**: {personality_info['desc']}
**투자 특성**: {personality_info['traits']}
**권장 전략**: {personality_info['strategy']}

**중요**: 사용자의 투자 성향에 맞춰 답변하세요.
- 이 성향의 투자 스타일과 선호도를 반영
- 성향에 맞는 종목 추천 및 전략 제시
- 성향의 특성(장기/단기, 안정/공격, 데이터/직관, 배당/성장)을 고려"""
        
        # 시스템 프롬프트 구성 (데이터 사용 여부에 따라 다르게)
        base_prompt = """당신은 투자 어드바이저 AI입니다.

## 핵심 규칙
1. **간결하게 답변** - 불필요한 서론/부연 없이 핵심만
2. **환율: 1달러 = 1,450원** (이 값을 사용, "실시간 제공 불가" 같은 말 금지)

## 답변 스타일
- 짧고 명확하게
- 핵심 정보 먼저
- 필요한 것만 말하기
- 마크다운 형식: **강조**, - 리스트""" + personality_prompt

        if request.use_stock_data and stock_context:
            # 데이터 기반 모드: 제공된 CSV 데이터만 사용
            system_prompt = base_prompt + """

## 데이터 기반 모드 (활성화됨)
- **반드시 아래 제공된 S&P 500 데이터만 참조하여 답변**
- 데이터에 있는 종목만 추천 가능
- 데이터에 없는 정보는 "제공된 데이터에 없습니다"라고 답변
- 종목 추천 시 데이터의 실제 수치(시가총액, 배당률 등) 인용

## 답변 예시 (데이터 모드)

질문: "배당주 추천해줘"
→ **데이터 기반 추천**
- [데이터에서 배당률 높은 종목 선택]
- 티커, 배당률, 시가총액 명시
⚠️ 투자 결정은 개인 책임

""" + stock_context
        else:
            # 추론 모드: LLM 자체 지식으로 답변
            system_prompt = base_prompt + """

## 추론 모드 (데이터 없음)
- 제공된 주식 데이터가 없음
- **당신의 학습된 지식을 바탕으로 추론하여 답변**
- 일반적인 투자 지식, 개념 설명, 시장 트렌드 분석 가능
- 구체적 수치는 대략적인 값임을 명시
- "정확한 실시간 데이터는 확인 필요"라고 안내 가능

## 답변 예시 (추론 모드)

질문: "배당주 추천해줘"
→ **일반적으로 알려진 배당주**
- JNJ (J&J): 헬스케어, 60년+ 배당 증가
- PG (P&G): 소비재, 안정적 배당
💡 정확한 현재 배당률은 별도 확인 필요
⚠️ 투자 결정은 개인 책임

질문: "PER이 뭐야?"
→ **PER = 주가 ÷ 주당순이익**
주가가 이익의 몇 배인지 나타냄.
- 낮으면 저평가 가능성
- 높으면 성장 기대 반영"""
        
        # 세션별 대화 히스토리 관리
        session_id = request.session_id
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        # 히스토리가 너무 길면 오래된 것부터 삭제 (최근 10개 대화만 유지)
        if len(conversation_history[session_id]) > 20:
            conversation_history[session_id] = conversation_history[session_id][-20:]
        
        # 메시지 구성: 시스템 + 히스토리 + 현재 메시지
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history[session_id])
        messages.append({"role": "user", "content": request.message})
        
        # 시스템 프롬프트 길이 확인
        print(f"📝 시스템 프롬프트 길이: {len(system_prompt)} 문자")
        print(f"📊 주식 데이터: {stock_count}개 종목")
        print(f"💬 대화 히스토리: {len(conversation_history[session_id])}개 메시지")
        
        # OpenAI를 통한 채팅
        response = openai_service.client.chat.completions.create(
            model=openai_service.deployment_name,
            messages=messages,
            max_completion_tokens=3000
        )
        
        ai_response = response.choices[0].message.content
        print(f"✅ AI 응답 길이: {len(ai_response) if ai_response else 0} 문자")
        print(f"📄 AI 응답 미리보기: {ai_response[:200] if ai_response else 'None'}...")
        
        # 대화 히스토리에 현재 대화 추가
        conversation_history[session_id].append({"role": "user", "content": request.message})
        conversation_history[session_id].append({"role": "assistant", "content": ai_response})
        
        return {
            "success": True,
            "response": ai_response,
            "timestamp": datetime.now().isoformat(),
            "data_used": request.use_stock_data and bool(stock_context),
            "stocks_loaded": stock_count,
            "session_id": session_id,
            "history_count": len(conversation_history[session_id]) // 2,
            "personality_type": request.personality_type,
            "personality_name": PERSONALITY_DATABASE.get(request.personality_type, {}).get('name') if request.personality_type else None
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
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
