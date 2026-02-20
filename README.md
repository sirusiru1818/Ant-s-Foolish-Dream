# Ant's Foolish Dream

Azure 서비스를 활용한 주식 성향 분석 및 종목 추천 서비스

## 주요 기능

- 📊 주식 성향 분석: Azure OpenAI를 활용한 종목별 투자 성향 분석
- 🎯 종목 추천: 사용자 선호도 기반 맞춤형 종목 추천
- 🔒 데이터 암호화: Blob Storage에 저장되는 모든 데이터 암호화
- 💾 Azure Blob Storage: 텍스트 파일 형식으로 데이터 저장
- 🤖 Azure Machine Learning: ML 모델 예측 지원

## 사용된 Azure 서비스

1. **Azure Blob Storage**: 주식 데이터 및 분석 결과 저장 (텍스트 파일)
2. **Azure OpenAI Service**: 주식 분석 및 종목 추천
3. **Azure Machine Learning**: ML 모델 학습 및 배포 (선택사항)
4. **데이터 암호화**: cryptography 라이브러리를 사용한 데이터 암호화

## 프로젝트 구조

```
.
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI 메인 애플리케이션
│   ├── config.py            # 설정 관리
│   ├── blob_storage.py      # Azure Blob Storage 연동
│   ├── ml_service.py        # Azure ML 연동
│   ├── openai_service.py    # Azure OpenAI 연동
│   └── encryption.py        # 데이터 암호화
├── scripts/
│   └── generate_key.py      # 암호화 키 생성 스크립트
├── .env.example             # 환경 변수 예시
├── requirements.txt         # Python 패키지 의존성
└── README.md
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

1. `.env.example` 파일을 복사하여 `.env` 파일을 생성:
   ```bash
   cp .env.example .env
   ```

2. 암호화 키 생성:
   ```bash
   python scripts/generate_key.py
   ```

3. `.env` 파일에 다음 정보를 입력:
   - `AZURE_OPENAI_ENDPOINT`: Azure OpenAI 엔드포인트 URL
   - `AZURE_OPENAI_API_KEY`: Azure OpenAI API 키
   - `AZURE_STORAGE_ACCOUNT_NAME`: Azure Storage 계정 이름
   - `AZURE_STORAGE_ACCOUNT_KEY`: Azure Storage 계정 키
   - `ENCRYPTION_KEY`: 생성한 암호화 키

### 3. 애플리케이션 실행

**기본 실행 (자동 리로드):**
```bash
python run.py
```

**리로드 없이 실행 (venv 변경사항 무시):**
```bash
RELOAD=false python run.py
```

**또는 직접 uvicorn 실행:**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

> **참고**: 개발 중 `venv` 디렉토리의 변경사항으로 인한 불필요한 리로드를 피하려면 `RELOAD=false` 옵션을 사용하거나, 코드 변경 후 수동으로 재시작하세요.

서버가 실행되면 `http://localhost:8000`에서 API를 사용할 수 있습니다.

## API 엔드포인트

### 주식 성향 분석
```bash
POST /api/analyze
Content-Type: application/json

{
  "stock_data": {
    "name": "삼성전자",
    "price": 75000,
    "news": "최근 실적 발표..."
  },
  "save_to_blob": true
}
```

### 종목 추천
```bash
POST /api/recommend
Content-Type: application/json

{
  "user_preference": {
    "risk_tolerance": "보통",
    "investment_amount": 1000000,
    "investment_period": "1년",
    "interests": ["반도체", "IT"]
  },
  "save_to_blob": true
}
```

### 저장된 파일 목록 조회
```bash
GET /api/storage/list?prefix=analysis/
```

### 저장된 파일 조회
```bash
GET /api/storage/{blob_name}
```

### API 문서
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 보안

- 모든 Blob Storage에 저장되는 데이터는 암호화됩니다
- 암호화 키는 `.env` 파일에 저장되며, 절대 공개 저장소에 커밋하지 마세요
- 프로덕션 환경에서는 Azure Key Vault 사용을 권장합니다

## 참고사항

- Azure Machine Learning 설정은 선택사항입니다
- Blob Storage 컨테이너는 자동으로 생성됩니다
- 모든 데이터는 텍스트 파일(JSON 형식)로 저장됩니다
