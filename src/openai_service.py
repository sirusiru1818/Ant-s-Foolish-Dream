"""Azure OpenAI 연동 모듈"""
from openai import AzureOpenAI
from typing import List, Dict, Optional
from src.config import settings


class OpenAIService:
    """Azure OpenAI 서비스"""
    
    def __init__(self):
        """OpenAI 클라이언트 초기화"""
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint
        )
        self.deployment_name = settings.azure_openai_deployment_name
    
    def analyze_stock_sentiment(self, stock_data: Dict) -> Optional[str]:
        """
        주식 성향 분석
        
        Args:
            stock_data: 주식 데이터 (종목명, 가격, 뉴스 등)
            
        Returns:
            분석 결과 텍스트 또는 None
        """
        try:
            prompt = f"""
            다음 주식 정보를 분석하여 투자 성향을 평가해주세요.
            
            종목명: {stock_data.get('name', 'N/A')}
            현재가: {stock_data.get('price', 'N/A')}
            뉴스/정보: {stock_data.get('news', 'N/A')}
            
            분석 항목:
            1. 기술적 분석
            2. 펀더멘털 분석
            3. 시장 심리
            4. 투자 의견 (매수/보유/매도)
            
            한국어로 상세히 분석해주세요.
            """
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 전문 주식 분석가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"성향 분석 실패: {e}")
            return None
    
    def recommend_stocks(self, user_preference: Dict) -> Optional[List[Dict]]:
        """
        종목 추천
        
        Args:
            user_preference: 사용자 선호도 (위험성향, 투자금액 등)
            
        Returns:
            추천 종목 리스트 또는 None
        """
        try:
            prompt = f"""
            다음 사용자 선호도에 맞는 주식 종목을 추천해주세요.
            
            위험 성향: {user_preference.get('risk_tolerance', '보통')}
            투자 금액: {user_preference.get('investment_amount', 'N/A')}
            투자 기간: {user_preference.get('investment_period', 'N/A')}
            관심 분야: {user_preference.get('interests', 'N/A')}
            
            JSON 형식으로 다음 정보를 포함하여 추천 종목 5개를 제시해주세요:
            {{
                "stocks": [
                    {{
                        "name": "종목명",
                        "reason": "추천 이유",
                        "risk_level": "위험도",
                        "expected_return": "예상 수익률"
                    }}
                ]
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 전문 투자 자문가입니다. JSON 형식으로만 응답하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            import json
            result_text = response.choices[0].message.content
            # JSON 추출 (마크다운 코드 블록 제거)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text)
            return result.get("stocks", [])
        except Exception as e:
            print(f"종목 추천 실패: {e}")
            return None
    
    def generate_report(self, analysis_results: List[Dict]) -> Optional[str]:
        """
        분석 리포트 생성
        
        Args:
            analysis_results: 분석 결과 리스트
            
        Returns:
            리포트 텍스트 또는 None
        """
        try:
            prompt = f"""
            다음 주식 분석 결과를 바탕으로 종합 리포트를 작성해주세요.
            
            {analysis_results}
            
            리포트에는 다음 내용을 포함해주세요:
            1. 전체 시장 동향
            2. 주요 종목 분석 요약
            3. 투자 전략 제안
            4. 리스크 관리 방안
            
            한국어로 전문적이고 읽기 쉽게 작성해주세요.
            """
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 전문 금융 리포트 작성자입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"리포트 생성 실패: {e}")
            return None
