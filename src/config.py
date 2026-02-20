"""설정 관리 모듈"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_deployment_name: str = "gpt-4"
    
    # Azure Blob Storage
    azure_storage_account_name: str
    azure_storage_account_key: str
    azure_storage_container_name: str = "stock-data"
    
    # Azure Machine Learning
    azure_ml_workspace_name: Optional[str] = None
    azure_ml_resource_group: Optional[str] = None
    azure_ml_subscription_id: Optional[str] = None
    
    # 암호화
    encryption_key: str  # 32바이트 키 (base64 인코딩 권장)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
